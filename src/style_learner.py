"""Apprentissage de style : un style = l'agrégat de un ou plusieurs montages d'exemple → styles/<nom>.json.

Pour chaque montage d'exemple (mesures déterministes, sans API) : fin utile (écran de fin TikTok/Reels
exclu), coupes (PySceneDetect), tempo et beats (librosa), alignement coupes/beats, durées de plans,
luminance, saturation, chaleur, flashs. Avec une clé API, Claude Vision lit chaque exemple, puis Claude
fait la synthèse de tous les exemples en un seul style. On peut ajouter des exemples au fil du temps :
les mesures de chaque exemple sont conservées dans la fiche (les vidéos, elles, ne sont jamais versionnées).
"""

from __future__ import annotations

import base64
import hashlib
import re
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT, director_model, vision_model
from .ffmpeg_utils import probe_video, run_ffmpeg
from .schemas import (
    StyleExample,
    StyleLook,
    StylePacing,
    StyleProfile,
    StyleReading,
    load_model,
    save_model,
    utc_now,
)

Log = Callable[[str], None]

STYLES_DIR = PROJECT_ROOT / "styles"
SAMPLE_FPS = 10
BEAT_TOLERANCE = 0.080

TOOLBOX = """Outils du monteur automatique : plans `single` (avec zoom_punch), mosaïque `nine_grid`, `camera_hud`,
`shutter_flash` (white / black / warm = light leak, 2-4 frames), `kinetic_title` (positions center / top /
bottom / lower_third / split = deux mots écartés gauche-droite ; animations spring_up / scale_in /
letter_stagger / fade ; styles minimal / bold / outline), vitesse de lecture, cuts calés sur les beats."""


def _noop(_: str) -> None:
    pass


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "style"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


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
    duration = len(frames) / fps
    return StyleLook(
        mean_luma=round(float(luma.mean()), 1),
        contrast=round(float(luma_px.std()), 1),
        saturation=round(float(sat.mean()), 3),
        warmth=round(float((rgb[..., 0] - rgb[..., 2]).mean() / 255), 3),
        flashes_per_10s=round(len(flashes) / duration * 10, 2) if duration > 0 else 0.0,
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


# ---------------------------------------------------------------------------- agrégation multi-exemples

def _weighted(values: list[float | None], weights: list[float]) -> float | None:
    pairs = [(v, w) for v, w in zip(values, weights) if v is not None and w > 0]
    if not pairs:
        return None
    return float(sum(v * w for v, w in pairs) / sum(w for _, w in pairs))


def aggregate_pacing(examples: Sequence[StyleExample]) -> StylePacing:
    if len(examples) == 1:
        return examples[0].pacing
    pcs = [e.pacing for e in examples]
    shots = [d for p in pcs for d in p.shot_durations]
    duration = sum(p.content_duration for p in pcs)
    cuts = [max(0, p.shot_count - 1) for p in pcs]
    bpms = [p.bpm for p in pcs if p.bpm]
    shot_beats = [d * p.bpm / 60 for p in pcs if p.bpm for d in p.shot_durations]
    offsets = [p.cut_offset_ms for p in pcs if p.cut_offset_ms is not None]
    r = lambda x, n=3: None if x is None else round(x, n)  # noqa: E731
    return StylePacing(
        content_duration=round(duration, 3),
        shot_count=len(shots),
        shot_durations=shots,
        median_shot_sec=round(float(np.median(shots)), 3),
        mean_shot_sec=round(float(np.mean(shots)), 3),
        hook_shot_sec=round(float(np.median([p.hook_shot_sec for p in pcs])), 3),
        cuts_per_10s=round(sum(cuts) / duration * 10, 2) if duration > 0 else 0.0,
        bpm=r(float(np.median(bpms)), 2) if bpms else None,
        median_shot_beats=r(float(np.median(shot_beats)), 2) if shot_beats else None,
        cut_offset_ms=r(float(np.median(offsets)), 1) if offsets else None,
        cuts_on_beat_ratio=r(_weighted([p.cuts_on_beat_ratio for p in pcs], cuts)),
        cuts_on_half_beat_ratio=r(_weighted([p.cuts_on_half_beat_ratio for p in pcs], cuts)),
    )


def aggregate_look(examples: Sequence[StyleExample]) -> StyleLook:
    if len(examples) == 1:
        return examples[0].look
    w = [e.pacing.content_duration for e in examples]
    looks = [e.look for e in examples]
    return StyleLook(
        mean_luma=round(_weighted([lk.mean_luma for lk in looks], w), 1),
        contrast=round(_weighted([lk.contrast for lk in looks], w), 1),
        saturation=round(_weighted([lk.saturation for lk in looks], w), 3),
        warmth=round(_weighted([lk.warmth for lk in looks], w), 3),
        flashes_per_10s=round(_weighted([lk.flashes_per_10s for lk in looks], w), 2),
        flash_times=[],
    )


def metrics_reading(pacing: StylePacing, look: StyleLook) -> StyleReading:
    tempo = f"{pacing.bpm:.0f} BPM, " if pacing.bpm else ""
    beats = f"≈ {pacing.median_shot_beats:g} beat(s) par plan, " if pacing.median_shot_beats else ""
    warm = "chaud" if look.warmth > 0.02 else "froid" if look.warmth < -0.02 else "neutre"
    return StyleReading(
        summary=f"{tempo}{pacing.shot_count} plans en {pacing.content_duration:.1f} s ({beats}plan médian {pacing.median_shot_sec:.2f} s).",
        structure=f"Plan d'accroche d'environ {pacing.hook_shot_sec:.2f} s puis coupes toutes les {pacing.median_shot_sec:.2f} s environ.",
        title_style="non analysé (sans clé API)",
        transitions=f"≈ {look.flashes_per_10s:g} flash(s) / 10 s." if look.flashes_per_10s else "cuts secs",
        color_grade=f"rendu {warm}, luminance moyenne {look.mean_luma:.0f}/255, saturation {look.saturation:.2f}",
        editing_rules=[
            f"Plan d'accroche d'environ {pacing.hook_shot_sec:.1f} s.",
            f"Durée de plan médiane ≈ {pacing.median_shot_sec:.2f} s"
            + (f" (≈ {pacing.median_shot_beats:g} beat)." if pacing.median_shot_beats else "."),
        ],
        source="metrics",
    )


# ---------------------------------------------------------------------------- lectures Claude

_READING_FIELDS = {"summary", "structure", "title_style", "transitions", "color_grade", "editing_rules"}

EXAMPLE_SYSTEM = f"""Tu es chef monteur. On te montre la planche contact d'un montage court (vignettes horodatées)
et des mesures objectives de son rythme. Décris son STYLE de façon réutilisable pour monter d'AUTRES rushs
dans le même esprit (ne décris pas le contenu précis à recopier).
{TOOLBOX}
Les editing_rules sont 5 à 10 consignes concrètes utilisant ces outils. Réponds en français, uniquement en JSON."""

SYNTHESIS_SYSTEM = f"""Tu es chef monteur. On te donne les fiches de plusieurs montages d'exemple qu'un même
créateur considère « dans le même délire », et les mesures agrégées de leur rythme. Écris UNE fiche de style
qui capture ce qu'ils ont en commun (le fil rouge), en signalant les variantes acceptables (ex. accroche plus
ou moins longue). Ne recopie pas le contenu des exemples.
{TOOLBOX}
Les editing_rules sont 6 à 12 consignes concrètes utilisant ces outils. Réponds en français, uniquement en JSON."""


def _reading_schema_model():
    from pydantic import create_model

    from .schemas import StrictModel

    fields = {k: (v.annotation, v) for k, v in StyleReading.model_fields.items() if k in _READING_FIELDS}
    return create_model("StyleReadingOut", __base__=StrictModel, **fields)


def read_example(video: Path, end: float, example: StyleExample, client=None) -> StyleReading:
    from .llm import structured_call
    from .video_indexer import build_contact_sheet, extract_frames

    step = max(0.5, end / 30)
    times = [round(float(t), 2) for t in np.arange(0.25, end, step)][:30]
    with tempfile.TemporaryDirectory() as td:
        frames = extract_frames(str(video), times, Path(td))
        sheet = build_contact_sheet(frames, times[: len(frames)], Path(td) / "sheet.jpg", f"{video.name} | {end:.1f}s")
        data = base64.standard_b64encode(sheet.read_bytes()).decode()
    metrics = example.pacing.model_dump_json(indent=1) + "\n" + example.look.model_dump_json(indent=1)
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
        {"type": "text", "text": f"Mesures objectives du montage :\n{metrics}\nDécris ce style."},
    ]
    out, _ = structured_call(model=vision_model(), system=EXAMPLE_SYSTEM, content=content,
                             output_model=_reading_schema_model(), max_tokens=16000, effort="medium", client=client)
    return StyleReading(**out.model_dump(), source="claude")


def synthesize_reading(examples: Sequence[StyleExample], pacing: StylePacing, look: StyleLook, client=None) -> StyleReading:
    from .llm import structured_call

    cards = []
    for i, e in enumerate(examples, 1):
        rd = e.reading
        cards.append(
            f"### Exemple {i} — {e.source}\n"
            f"rythme : {e.pacing.shot_count} plans en {e.pacing.content_duration:.1f}s, plan médian {e.pacing.median_shot_sec:.2f}s, "
            f"accroche {e.pacing.hook_shot_sec:.2f}s, bpm {e.pacing.bpm}\n"
            f"résumé : {rd.summary}\nstructure : {rd.structure}\ntitres : {rd.title_style}\n"
            f"transitions : {rd.transitions}\nétalonnage : {rd.color_grade}\nconsignes : " + " | ".join(rd.editing_rules)
        )
    text = "\n\n".join(cards) + "\n\n### Mesures agrégées\n" + pacing.model_dump_json(exclude={"shot_durations"}) \
        + "\n" + look.model_dump_json() + "\n\nÉcris la fiche de style unique."
    out, _ = structured_call(model=director_model(), system=SYNTHESIS_SYSTEM, content=[{"type": "text", "text": text}],
                             output_model=_reading_schema_model(), max_tokens=16000, effort="medium", client=client)
    return StyleReading(**out.model_dump(), source="claude")


# ---------------------------------------------------------------------------- API publique

def measure_example(video: str | Path, *, end: float | None = None, source: str | None = None, log: Log = _noop) -> tuple[StyleExample, Path, float]:
    """Mesures déterministes d'un montage d'exemple → (StyleExample, chemin, fin utile)."""
    from .audio_analyzer import analyze_file

    path = Path(video).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {path}")
    probe = probe_video(path)
    frames = sample_frames(path)
    if end is None:
        end = detect_content_end(frames) or probe.duration
        if end < probe.duration - 0.2:
            log(f"{path.name} : écran de fin détecté, analyse limitée à 0-{end:.2f}s (sur {probe.duration:.2f}s)")
    end = min(end, probe.duration)
    cuts = detect_cuts(path, end)
    bpm, beats = None, []
    if probe.has_audio:
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "audio.wav"
            run_ffmpeg(["-i", str(path), "-t", f"{end:.3f}", "-vn", "-ac", "1", "-ar", "22050", str(wav)])
            audio = analyze_file(wav)
        bpm, beats = audio.bpm or None, audio.beats
    pacing = measure_pacing(cuts, end, bpm, beats)
    look = measure_look(frames[: max(1, int(end * SAMPLE_FPS))])
    example = StyleExample(
        source=source or path.name,
        sha256=_sha256(path),
        format_hint="16:9" if probe.width > probe.height else "9:16",
        cut_times=cuts,
        pacing=pacing,
        look=look,
        reading=metrics_reading(pacing, look),
    )
    return example, path, end


def rebuild_profile(profile: StyleProfile, *, use_api: bool = False, client=None, log: Log = _noop) -> StyleProfile:
    """Recalcule l'agrégat (rythme, look, format) et la lecture de synthèse à partir des exemples."""
    pacing = aggregate_pacing(profile.examples)
    look = aggregate_look(profile.examples)
    formats = [e.format_hint for e in profile.examples if e.format_hint]
    fmt = max(set(formats), key=formats.count) if formats else None
    if use_api:
        log(f"synthèse du style à partir de {len(profile.examples)} exemple(s)")
        reading = synthesize_reading(profile.examples, pacing, look, client=client)
    elif profile.reading.source in ("claude", "manual"):
        reading = profile.reading  # garde la lecture existante (à resynthétiser avec une clé API)
    elif len(profile.examples) == 1:
        reading = profile.examples[0].reading
    else:
        reading = metrics_reading(pacing, look)
    return profile.model_copy(update={"pacing": pacing, "look": look, "format_hint": fmt, "reading": reading,
                                      "updated_at": utc_now()})


def learn_style(
    videos: Sequence[str | Path] | str | Path,
    name: str,
    *,
    styles_dir: Path = STYLES_DIR,
    end: float | None = None,
    use_vision: bool = True,
    replace: bool = False,
    log: Log = _noop,
    client=None,
) -> StyleProfile:
    """Crée le style `name` ou lui ajoute des montages d'exemple, puis enregistre styles/<nom>.json."""
    if isinstance(videos, (str, Path)):
        videos = [videos]
    if end is not None and len(videos) != 1:
        raise ValueError("--end ne s'utilise qu'avec une seule vidéo")
    slug = slugify(name)
    path = Path(styles_dir) / f"{slug}.json"
    existing = load_model(path, StyleProfile) if path.exists() and not replace else None
    examples = list(existing.examples) if existing else []
    known = {e.sha256 for e in examples}

    added = 0
    for video in videos:
        example, vpath, vend = measure_example(video, end=end, log=log)
        if example.sha256 in known:
            log(f"{vpath.name} : déjà dans le style « {slug} », ignoré")
            continue
        if use_vision:
            log(f"{vpath.name} : lecture du style par Claude Vision")
            example = example.model_copy(update={"reading": read_example(vpath, vend, example, client=client)})
        log(f"{vpath.name} : {example.pacing.shot_count} plans, plan médian {example.pacing.median_shot_sec:.2f}s")
        examples.append(example)
        known.add(example.sha256)
        added += 1
    if not examples:
        raise ValueError("Aucun montage d'exemple exploitable.")

    base = existing or StyleProfile(name=slug, examples=examples, pacing=examples[0].pacing,
                                    look=examples[0].look, reading=examples[0].reading)
    profile = base.model_copy(update={"examples": examples})
    if added or existing is None:
        profile = rebuild_profile(profile, use_api=use_vision and (len(examples) > 1), client=client, log=log)
    save_model(StyleProfile.model_validate(profile.model_dump()), path)
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
    pc, rd = style.pacing, style.reading
    hooks = [e.pacing.hook_shot_sec for e in style.examples]
    hook = f"{min(hooks):.1f}-{max(hooks):.1f}s" if len(hooks) > 1 else f"{hooks[0]:.2f}s"
    lines = [
        f"## STYLE REFERENCE « {style.name} » (learned from {len(style.examples)} example edit(s) — reproduce the feel, not the content)",
        f"summary: {rd.summary}",
        f"structure: {rd.structure}",
        f"titles: {rd.title_style}",
        f"transitions: {rd.transitions}",
        f"color grade: {rd.color_grade}",
        "measured pacing: "
        + f"hook shot {hook}, median shot {pc.median_shot_sec:.2f}s"
        + (f" (≈ {pc.median_shot_beats:g} beats at ~{pc.bpm:.0f} BPM)" if pc.median_shot_beats and pc.bpm else "")
        + f", {pc.cuts_per_10s:g} cuts / 10 s"
        + (f", {pc.cuts_on_half_beat_ratio:.0%} of cuts on a beat or half-beat" if pc.cuts_on_half_beat_ratio is not None else "")
        + (f", {style.look.flashes_per_10s:g} flashes / 10 s" if style.look.flashes_per_10s else ""),
        "style rules:",
        *[f"- {r}" for r in rd.editing_rules],
        "Follow this style for rhythm, titles, transitions and structure. It overrides the generic rhythm rule 3, "
        "but never the timeline constraints (contiguous scenes, beats, clip ranges).",
    ]
    return "\n".join(lines)
