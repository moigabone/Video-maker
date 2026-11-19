"""Indexation visuelle : keyframes → planche contact horodatée → Claude Vision → analyse structurée (cache SHA-256)."""

from __future__ import annotations

import base64
import math
import shutil
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont
from pydantic import Field

from .config import Workspace, vision_model
from .ffmpeg_utils import run_ffmpeg
from .llm import LLMError, structured_call
from .schemas import BestSegment, ClipAnalysis, MediaCatalog, MotionType, StrictModel, VideoAsset, save_model, utc_now

Log = Callable[[str], None]

THUMB_BOX = 256          # chaque vignette tient dans 256x256
SHEET_COLS = 6           # 6*256 + marges ≈ 1564 px : sous la limite de redimensionnement côté API (1568)
MAX_FRAMES_PER_SHEET = 30
MAX_SHEETS = 3
GAP = 4
HEADER_H = 34


def _noop(_: str) -> None:
    pass


# ---------------------------------------------------------------------------- keyframes

def interval_times(duration: float, fps: float, base_interval: float | None = None) -> tuple[list[float], float]:
    """Instants échantillonnés au centre de fenêtres régulières (1s pour les clips courts, 1.5s sinon)."""
    base = base_interval or (1.0 if duration <= 20 else 1.5)
    interval = max(base, duration / (MAX_FRAMES_PER_SHEET * MAX_SHEETS))
    n = max(1, math.ceil(duration / interval - 1e-9))
    last = max(0.0, duration - 1.0 / max(fps, 1.0))
    times = [round(min(last, (i + 0.5) * interval), 3) for i in range(n)]
    return sorted(set(times)), interval


def scene_times(path: str, duration: float, fps: float) -> list[float] | None:
    """Milieu de chaque plan détecté par PySceneDetect (None si indisponible ou un seul plan)."""
    try:
        from scenedetect import ContentDetector, detect  # import paresseux (OpenCV)
    except Exception:
        return None
    try:
        scenes = detect(path, ContentDetector(threshold=27.0), show_progress=False)
    except Exception:
        return None
    if len(scenes) < 2:
        return None
    times: list[float] = []
    for start, end in scenes:
        s, e = start.get_seconds(), end.get_seconds()
        times.append(round((s + e) / 2, 3))
    cap = MAX_FRAMES_PER_SHEET * MAX_SHEETS
    if len(times) > cap:
        step = len(times) / cap
        times = [times[int(i * step)] for i in range(cap)]
    last = max(0.0, duration - 1.0 / max(fps, 1.0))
    return sorted({min(t, last) for t in times})


def extract_frames(src: str, times: list[float], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for i, t in enumerate(times):
        out = out_dir / f"f{i:03d}.jpg"
        run_ffmpeg([
            "-ss", f"{t:.3f}", "-i", src, "-frames:v", "1",
            "-vf", f"scale={THUMB_BOX}:{THUMB_BOX}:force_original_aspect_ratio=decrease",
            "-q:v", "3", str(out),
        ])
        if out.exists():
            frames.append(out)
    return frames


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def build_contact_sheet(frames: list[Path], times: list[float], out_path: Path, title: str) -> Path:
    """Grille de vignettes avec timestamp incrusté (fond noir) + bandeau titre."""
    thumbs = [Image.open(f).convert("RGB") for f in frames]
    cell_w = max(t.width for t in thumbs)
    cell_h = max(t.height for t in thumbs)
    cols = min(SHEET_COLS, len(thumbs))
    rows = math.ceil(len(thumbs) / cols)
    W = cols * cell_w + (cols + 1) * GAP
    H = HEADER_H + rows * cell_h + (rows + 1) * GAP
    sheet = Image.new("RGB", (W, H), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    draw.text((GAP * 2, 8), title, fill=(235, 235, 235), font=_font(16))
    label_font = _font(15)
    for i, (thumb, t) in enumerate(zip(thumbs, times)):
        r, c = divmod(i, cols)
        x = GAP + c * (cell_w + GAP) + (cell_w - thumb.width) // 2
        y = HEADER_H + GAP + r * (cell_h + GAP) + (cell_h - thumb.height) // 2
        sheet.paste(thumb, (x, y))
        label = f"{t:.1f}s"
        bbox = draw.textbbox((0, 0), label, font=label_font)
        lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x + 3, y + 3, x + 3 + lw + 8, y + 3 + lh + 8], fill=(0, 0, 0))
        draw.text((x + 7, y + 6 - bbox[1]), label, fill=(255, 220, 0), font=label_font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, "JPEG", quality=85, optimize=True)
    for t in thumbs:
        t.close()
    return out_path


def make_contact_sheets(asset: VideoAsset, ws: Workspace, mode: Literal["interval", "scene"] = "interval") -> tuple[list[Path], list[float], float]:
    src = asset.proxy_path if asset.proxy_path and Path(asset.proxy_path).exists() else asset.path
    times, interval = interval_times(asset.duration, asset.fps)
    if mode == "scene":
        st = scene_times(src, asset.duration, asset.fps)
        if st:
            times = st
    tmp = Path(tempfile.mkdtemp(prefix=f"kf_{asset.clip_id}_"))
    try:
        frames = extract_frames(src, times, tmp)
        times = times[: len(frames)]
        if not frames:
            raise RuntimeError(f"Aucune keyframe extraite pour {asset.rel_path}")
        sheets: list[Path] = []
        n_sheets = math.ceil(len(frames) / MAX_FRAMES_PER_SHEET)
        for k in range(n_sheets):
            sl = slice(k * MAX_FRAMES_PER_SHEET, (k + 1) * MAX_FRAMES_PER_SHEET)
            title = f"{asset.clip_id} | {Path(asset.rel_path).name} | {asset.duration:.1f}s | {asset.width}x{asset.height}@{asset.fps:g} | sheet {k + 1}/{n_sheets}"
            sheets.append(build_contact_sheet(frames[sl], times[sl], ws.contact_sheets / f"{asset.clip_id}_{k + 1}.jpg", title))
        return sheets, times, interval
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------- VLM

class _SegmentRaw(StrictModel):
    start: float = Field(description="Début (s) dans le rush, lu sur les timestamps des vignettes")
    end: float = Field(description="Fin (s) dans le rush, > start")
    description: str = Field(description="Ce qui se passe, en français, une phrase")
    motion_type: MotionType = Field(description="Mouvement dominant")


class _VisionRaw(StrictModel):
    scene_description: str = Field(description="Description de la scène en français (1 à 3 phrases)")
    subjects: list[str] = Field(description="Sujets principaux (personnes, objets, lieux)")
    tags: list[str] = Field(description="5 à 12 mots-clés en français (lieu, action, style, couleurs)")
    lighting: str = Field(description="Ambiance lumineuse (ex: golden hour, néon, contre-jour, nuit bleue)")
    energy_score: int = Field(description="Énergie visuelle de 1 (contemplatif) à 10 (explosif)")
    best_segments: list[_SegmentRaw] = Field(description="1 à 6 meilleurs sous-segments exploitables")


VISION_SYSTEM = """Tu es l'assistant de dérushage d'un monteur vidéo professionnel spécialisé dans les formats courts rythmés.
On te montre une ou plusieurs planches contact d'UN SEUL rush vidéo : chaque vignette est la frame à l'instant
incrusté en jaune (en secondes depuis le début du rush) et représente environ ± la moitié de l'intervalle d'échantillonnage.

Ta mission : produire une fiche de dérushage structurée.
- scene_description : ce qu'on voit, cadrage, lieu, action.
- subjects / tags : utiles pour retrouver le plan au montage.
- lighting : l'ambiance lumineuse.
- energy_score (1-10) : mouvement du sujet + mouvement caméra + intensité visuelle. 1-3 calme/contemplatif,
  4-6 modéré, 7-8 dynamique, 9-10 action intense.
- best_segments : les passages les plus exploitables (beau cadre, action lisible, mouvement fluide, pas de flou
  de mise au point ni de tremblement raté). Chaque segment dure en général 0.8 à 6 s, avec start/end strictement
  dans [0, durée du rush], déduits des timestamps des vignettes. Ne déborde jamais de la durée du rush.
Réponds uniquement avec le JSON demandé."""


def _sanitize(raw: _VisionRaw, duration: float) -> tuple[int, list[BestSegment]]:
    energy = int(min(10, max(1, round(raw.energy_score))))
    segs: list[BestSegment] = []
    for s in raw.best_segments:
        start = min(max(0.0, float(s.start)), duration)
        end = min(max(0.0, float(s.end)), duration)
        if end - start < 0.2:
            continue
        segs.append(BestSegment(start=round(start, 3), end=round(end, 3), description=s.description, motion_type=s.motion_type))
    segs.sort(key=lambda x: x.start)
    if not segs:
        segs = [BestSegment(start=0.0, end=round(duration, 3), description="rush complet", motion_type="other")]
    return energy, segs


def analyze_clip(asset: VideoAsset, ws: Workspace, *, mode: Literal["interval", "scene"] = "interval", client=None) -> ClipAnalysis:
    sheets, times, interval = make_contact_sheets(asset, ws, mode)
    content: list[dict] = []
    for sheet in sheets:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(sheet.read_bytes()).decode()},
        })
    content.append({
        "type": "text",
        "text": (
            f"Rush : {Path(asset.rel_path).name} (clip_id {asset.clip_id})\n"
            f"Durée : {asset.duration:.2f} s | {asset.width}x{asset.height} ({asset.orientation}) | {asset.fps:g} fps | codec {asset.codec}\n"
            f"{len(times)} vignettes, intervalle ≈ {interval:.2f} s, timestamps : {', '.join(f'{t:.1f}' for t in times)}\n"
            "Produis la fiche de dérushage JSON."
        ),
    })
    model = vision_model()
    raw, _ = structured_call(model=model, system=VISION_SYSTEM, content=content, output_model=_VisionRaw,
                             max_tokens=16000, effort="medium", client=client)
    energy, segs = _sanitize(raw, asset.duration)
    return ClipAnalysis(
        scene_description=raw.scene_description,
        subjects=raw.subjects,
        tags=raw.tags,
        lighting=raw.lighting,
        energy_score=energy,
        best_segments=segs,
        model=model,
        analyzed_at=utc_now(),
        contact_sheets=[str(s) for s in sheets],
        frame_times=times,
    )


def index_catalog(
    catalog: MediaCatalog,
    ws: Workspace,
    *,
    force: bool = False,
    mode: Literal["interval", "scene"] = "interval",
    max_workers: int = 4,
    log: Log = _noop,
    client=None,
) -> MediaCatalog:
    """Analyse chaque rush non encore indexé (cache par SHA-256) et sauvegarde le catalogue au fil de l'eau."""
    todo: list[VideoAsset] = []
    for v in catalog.videos:
        cached = catalog.analysis_cache.get(v.sha256)
        if cached and not force:
            v.analysis = cached
            log(f"cache ✓ {v.rel_path}")
        else:
            todo.append(v)
    if not todo:
        save_model(catalog, ws.catalog)
        return catalog

    lock = threading.Lock()
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {pool.submit(analyze_clip, v, ws, mode=mode, client=client): v for v in todo}
        for fut in as_completed(futures):
            v = futures[fut]
            try:
                analysis = fut.result()
            except LLMError as e:
                errors.append(f"{v.rel_path}: {e}")
                log(f"échec analyse {v.rel_path} : {e}")
                continue
            with lock:
                v.analysis = analysis
                catalog.analysis_cache[v.sha256] = analysis
                save_model(catalog, ws.catalog)
            log(f"analysé {v.rel_path} → énergie {analysis.energy_score}/10, {len(analysis.best_segments)} segment(s)")
    save_model(catalog, ws.catalog)
    if errors and len(errors) == len(todo):
        raise LLMError("Toutes les analyses vision ont échoué :\n" + "\n".join(errors))
    return catalog
