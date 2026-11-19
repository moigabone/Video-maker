"""Apprentissage de style : mesure le rythme et le look d'un montage d'exemple → styles/<nom>.json.

Mesures déterministes (sans API) : fin utile (écran de fin TikTok/Reels exclu), coupes (PySceneDetect),
tempo et beats (librosa), alignement coupes/beats, durées de plans, luminance, saturation, chaleur, flashs.
Avec une clé API, Claude Vision ajoute une lecture qualitative (structure, titres, transitions, étalonnage)
et des consignes concrètes pour le directeur.
"""

from __future__ import annotations

import base64
import re
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
from pydantic import Field

from .config import PROJECT_ROOT, vision_model
from .ffmpeg_utils import probe_video, run_ffmpeg
from .schemas import StrictModel, StyleLook, StylePacing, StyleProfile, load_model, save_model

Log = Callable[[str], None]

STYLES_DIR = PROJECT_ROOT / "styles"
SAMPLE_FPS = 10
BEAT_TOLERANCE = 0.080


def _noop(_: str) -> None:
    pass


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "style"


# ---------------------------------------------------------------------------- mesures vidéo

def sample_frames(path: str | Path, fps: int = SAMPLE_FPS) -> np.ndarray:
    """Frames RGB réduites (64x36) échantillonnées à `fps` : [n, 36, 64, 3] float."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vf", f"fps={fps},scale=64:36",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 36, 64, 3).astype(np.float64)


def detect_content_end(frames: np.ndarray, fps: int = SAMPLE_FPS, min_outro_sec: float = 1.0) -> float | None:
    """Début d'un écran de fin (frames finales sombres et quasi fixes, ex. carte TikTok), sinon None."""
    if len(frames) < 3:
        return None
    luma = frames.mean(axis=(1, 2, 3))
    diff = np.r_[0.0, np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2, 3))]
    end = len(frames)
    while end > 0 and luma[end - 1] < 45 and diff[end - 1] < 4:
        end -= 1
    if (len(frames) - end) / fps >= min_outro_sec and end > 0:
        return round(end / fps, 3)
    return None


def detect_cuts(path: str | Path, end: float) -> list[float]:
    """Coupes franches et rapides (AdaptiveDetector : robuste aux mouvements de caméra)."""
    from scenedetect import AdaptiveDetector, detect

    scenes = detect(str(path), AdaptiveDetector(adaptive_threshold=2.5, min_scene_len=8), show_progress=False)
    cuts = []
    for start, _ in scenes[1:]:
        t = float(getattr(start, "seconds", None) or start.get_seconds())
        if 0.05 < t < end - 0.05:
            cuts.append(round(t, 3))
    return cuts


def measure_look(frames: np.ndarray, fps: int = SAMPLE_FPS) -> StyleLook:
    rgb = frames.reshape(len(frames), -1, 3)
    luma_px = rgb @ np.array([0.299, 0.587, 0.114])
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    luma = luma_px.mean(axis=1)
    flashes = [round(i / fps, 2) for i in range(1, len(luma) - 1) if luma[i] > luma[i - 1] + 25 and luma[i] > 150]
    return StyleLook(
        mean_luma=round(float(luma.mean()), 1),
        contrast=round(float(luma_px.std()), 1),
        saturation=round(float(sat.mean()), 3),
        warmth=round(float((rgb[..., 0] - rgb[..., 2]).mean() / 255), 3),
        flash_times=flashes,
    )


# ---------------------------------------------------------------------------- rythme

def beat_alignment(cuts: list[float], beats: list[float]) -> tuple[float | None, float | None, float | None]:
    """(décalage médian ms, part sur beat, part sur beat ou demi-beat), décalage compensé."""
    if not cuts or len(beats) < 2:
        return None, None, None
    b = np.asarray(beats)
    halves = (b[:-1] + b[1:]) / 2
    grid_half = np.sort(np.r_[b, halves])
    signed = np.array([c - b[np.argmin(np.abs(b - c))] for c in cuts])
    offset = float(np.median(signed))
    if abs(offset) > 0.1:  # décalage trop grand pour être une latence : on ne compense pas
        offset = 0.0
    on_beat = np.mean([np.min(np.abs(b - (c - offset))) <= BEAT_TOLERANCE for c in cuts])
    on_half = np.mean([np.min(np.abs(grid_half - (c - offset))) <= BEAT_TOLERANCE for c in cuts])
    return round(offset * 1000, 1), round(float(on_beat), 3), round(float(on_half), 3)


def measure_pacing(cuts: list[float], end: float, bpm: float | None, beats: list[float]) -> StylePacing:
    bounds = [0.0, *cuts, end]
    shots = [round(b - a, 3) for a, b in zip(bounds, bounds[1:]) if b - a > 1e-3]
    offset_ms, on_beat, on_half = beat_alignment(cuts, beats)
    median = float(np.median(shots))
    return StylePacing(
        content_duration=round(end, 3),
        shot_count=len(shots),
        shot_durations=shots,
        median_shot_sec=round(median, 3),
        mean_shot_sec=round(float(np.mean(shots)), 3),
        hook_shot_sec=shots[0],
        cuts_per_10s=round(len(cuts) / end * 10, 2) if end > 0 else 0.0,
        bpm=bpm,
        median_shot_beats=round(median * bpm / 60, 2) if bpm else None,
        cut_offset_ms=offset_ms,
        cuts_on_beat_ratio=on_beat,
        cuts_on_half_beat_ratio=on_half,
    )


# ---------------------------------------------------------------------------- lecture qualitative (Claude)

class _StyleQualitative(StrictModel):
    summary: str = Field(description="Le style en 2-3 phrases (ambiance, sujet, énergie)")
    structure: str = Field(description="Structure : accroche, corps, fin (boucle ? plan calme ?)")
    title_style: str = Field(description="Titres : texte, typographie, placement, apparition ; 'aucun' sinon")
    transitions: str = Field(description="Transitions : cuts secs, whip pans, flashs, light leaks, zooms…")
    color_grade: str = Field(description="Étalonnage : température, contraste, saturation, grain")
    editing_rules: list[str] = Field(description="5 à 10 consignes concrètes pour reproduire ce style avec les outils disponibles")


STYLE_SYSTEM = """Tu es chef monteur. On te montre la planche contact d'un montage court (vignettes horodatées)
et des mesures objectives de son rythme. Décris son STYLE de façon réutilisable pour monter d'AUTRES rushs
dans le même esprit (ne décris pas le contenu précis à recopier).
Les consignes (editing_rules) doivent utiliser les outils du monteur automatique : plans `single` (avec
zoom_punch), mosaïque `nine_grid`, `camera_hud`, `shutter_flash` (white / black / warm = light leak, 2-4 frames),
`kinetic_title` (positions center / top / bottom / lower_third / split = deux mots écartés gauche-droite ;
animations spring_up / scale_in / letter_stagger / fade ; styles minimal / bold / outline), vitesse de lecture,
et cuts calés sur les beats. Réponds en français, uniquement avec le JSON demandé."""


def _vision_read(video: Path, end: float, pacing: StylePacing, look: StyleLook, client=None) -> _StyleQualitative:
    from .llm import structured_call
    from .video_indexer import build_contact_sheet, extract_frames

    step = max(0.5, end / 30)
    times = [round(t, 2) for t in np.arange(0.25, end, step)][:30]
    with tempfile.TemporaryDirectory() as td:
        frames = extract_frames(str(video), times, Path(td))
        sheet = build_contact_sheet(frames, times[: len(frames)], Path(td) / "sheet.jpg", f"{video.name} | {end:.1f}s")
        data = base64.standard_b64encode(sheet.read_bytes()).decode()
    metrics = pacing.model_dump_json(indent=1) + "\n" + look.model_dump_json(indent=1)
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
        {"type": "text", "text": f"Mesures objectives du montage :\n{metrics}\nDécris ce style."},
    ]
    result, _ = structured_call(model=vision_model(), system=STYLE_SYSTEM, content=content,
                                output_model=_StyleQualitative, max_tokens=16000, effort="medium", client=client)
    return result


def _metrics_only_reading(pacing: StylePacing, look: StyleLook) -> _StyleQualitative:
    tempo = f"{pacing.bpm:.0f} BPM, " if pacing.bpm else ""
    beats = f"≈ {pacing.median_shot_beats:g} beat(s) par plan, " if pacing.median_shot_beats else ""
    warm = "chaud" if look.warmth > 0.02 else "froid" if look.warmth < -0.02 else "neutre"
    return _StyleQualitative(
        summary=f"Montage de {pacing.content_duration:.1f} s, {tempo}{pacing.shot_count} plans ({beats}plan médian {pacing.median_shot_sec:.2f} s).",
        structure=f"Plan d'accroche de {pacing.hook_shot_sec:.2f} s puis coupes toutes les {pacing.median_shot_sec:.2f} s environ.",
        title_style="non analysé (sans clé API)",
        transitions=f"{len(look.flash_times)} flash(s) détecté(s)." if look.flash_times else "cuts secs",
        color_grade=f"rendu {warm}, luminance moyenne {look.mean_luma:.0f}/255, saturation {look.saturation:.2f}",
        editing_rules=[
            f"Plan d'accroche d'environ {pacing.hook_shot_sec:.1f} s.",
            f"Durée de plan médiane ≈ {pacing.median_shot_sec:.2f} s" + (f" (≈ {pacing.median_shot_beats:g} beat)." if pacing.median_shot_beats else "."),
        ],
    )


# ---------------------------------------------------------------------------- API publique

def learn_style(
    video: str | Path,
    name: str,
    *,
    styles_dir: Path = STYLES_DIR,
    end: float | None = None,
    use_vision: bool = True,
    log: Log = _noop,
    client=None,
) -> StyleProfile:
    """Analyse un montage d'exemple et enregistre styles/<nom>.json."""
    from .audio_analyzer import analyze_file

    path = Path(video).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {path}")
    probe = probe_video(path)
    frames = sample_frames(path)
    if end is None:
        end = detect_content_end(frames) or probe.duration
        if end < probe.duration - 0.2:
            log(f"écran de fin détecté : analyse limitée à 0-{end:.2f}s (sur {probe.duration:.2f}s)")
    end = min(end, probe.duration)

    log("détection des coupes")
    cuts = detect_cuts(path, end)
    bpm, beats = None, []
    if probe.has_audio:
        log("analyse de la musique")
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "audio.wav"
            run_ffmpeg(["-i", str(path), "-t", f"{end:.3f}", "-vn", "-ac", "1", "-ar", "22050", str(wav)])
            audio = analyze_file(wav)
        bpm, beats = audio.bpm or None, audio.beats

    pacing = measure_pacing(cuts, end, bpm, beats)
    look = measure_look(frames[: max(1, int(end * SAMPLE_FPS))])
    if use_vision:
        log("lecture du style par Claude Vision")
        reading, source = _vision_read(path, end, pacing, look, client=client), "claude"
    else:
        reading, source = _metrics_only_reading(pacing, look), "metrics"

    profile = StyleProfile(
        name=slugify(name),
        source=path.name,
        format_hint="16:9" if probe.width > probe.height else "9:16",
        cut_times=cuts,
        pacing=pacing,
        look=look,
        qualitative_source=source,
        **reading.model_dump(),
    )
    save_model(profile, Path(styles_dir) / f"{profile.name}.json")
    return profile


def list_styles(styles_dir: Path = STYLES_DIR) -> list[StyleProfile]:
    return [load_model(p, StyleProfile) for p in sorted(Path(styles_dir).glob("*.json"))]


def load_style(name_or_path: str, styles_dir: Path = STYLES_DIR) -> StyleProfile:
    p = Path(name_or_path)
    if not p.suffix:
        p = Path(styles_dir) / f"{slugify(name_or_path)}.json"
    if not p.is_file():
        available = ", ".join(s.name for s in list_styles(styles_dir)) or "aucun"
        raise FileNotFoundError(f"Style introuvable : {name_or_path} (disponibles : {available})")
    return load_model(p, StyleProfile)


def style_prompt_section(style: StyleProfile) -> str:
    """Section « STYLE REFERENCE » injectée dans le prompt du directeur."""
    pc = style.pacing
    lines = [
        f"## STYLE REFERENCE « {style.name} » (learned from an example edit — reproduce its feel, not its content)",
        f"summary: {style.summary}",
        f"structure: {style.structure}",
        f"titles: {style.title_style}",
        f"transitions: {style.transitions}",
        f"color grade: {style.color_grade}",
        "measured pacing: "
        + f"hook shot {pc.hook_shot_sec:.2f}s, median shot {pc.median_shot_sec:.2f}s"
        + (f" (≈ {pc.median_shot_beats:g} beats at {pc.bpm:.0f} BPM)" if pc.median_shot_beats and pc.bpm else "")
        + f", {pc.cuts_per_10s:g} cuts / 10 s"
        + (f", {pc.cuts_on_half_beat_ratio:.0%} of cuts on a beat or half-beat" if pc.cuts_on_half_beat_ratio is not None else ""),
        "style rules:",
        *[f"- {r}" for r in style.editing_rules],
        "Follow this style for rhythm, titles, transitions and structure. It overrides the generic rhythm rule 3, "
        "but never the timeline constraints (contiguous scenes, beats, clip ranges).",
    ]
    return "\n".join(lines)
