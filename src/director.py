"""Directeur : brief + catalogue + analyse audio → TimelineDraft (Claude) → Timeline → validateur déterministe."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

from .config import (
    FORMAT_SIZES,
    MAX_SPEED,
    MIN_OVERLAY_SEC,
    MIN_SCENE_SEC,
    MIN_SPEED,
    SNAP_TOLERANCE_SEC,
    Workspace,
    director_model,
)
from .schemas import (
    AudioAnalysis,
    AudioTrackRef,
    CameraHudOverlay,
    ClipRef,
    KineticTitle,
    MediaCatalog,
    NineGridScene,
    ShutterFlash,
    SingleScene,
    Timeline,
    TimelineDraft,
    ValidationReport,
    VideoAsset,
    save_json,
    save_model,
)

Log = Callable[[str], None]


class TimelineValidationError(ValueError):
    pass


def _noop(_: str) -> None:
    pass


# =============================================================================================
# 1. Génération (LLM)
# =============================================================================================

DIRECTOR_SYSTEM = """You are an elite short-form video editor (Reels / TikTok / Shorts) and motion designer.
You receive a creative brief, a catalog of analysed rushes and a beat-level analysis of the music.
You output ONE edit decision list as JSON matching the provided schema (TimelineDraft). A deterministic
validator will check your output, but every fix it has to make degrades the edit, so be precise.

TIME MODEL
- Timeline time t starts at 0.0. Music time = t + audio_start_offset.
- audio_start_offset must be a beat (ideally a downbeat) of the song. Choose it so the strongest part of the
  music serves the edit (you may start right before the climax for an instant hook if the brief is short).
- Beat timestamps are given in MUSIC time; convert them to timeline time by subtracting audio_start_offset.

STRUCTURE RULES
1. Hook: the scene at 0.0 must be the most striking shot (highest energy_score / strongest subject), with
   zoom_punch=true, and a shutter flash at 0.0. Grab attention in the first 1.5 s.
2. Cut on the beat: EVERY scene start/end lands on a beat timestamp (timeline time). Scenes are contiguous:
   scene[i].end == scene[i+1].start, first start = 0.0, last end = duration_sec. No gaps, no overlaps.
3. Rhythm follows energy: intro = longer shots (2-4 beats), buildup = 2 beats, climax = 1-2 beats (cuts on
   strong beats), outro = longer, calmer shots. Match clip energy_score to the section energy.
4. Alternate full-screen `single` scenes with a few `nine_grid` 3x3 mosaics (typically 1-2 per 30 s, at a
   drop or as a recap). A nine_grid lists 9 clips (diverse shots). Use grid_reveal="beats" when the grid lasts
   at least ~9 beats (cells appear one per beat), otherwise "all".
5. Source ranges: take in_sec inside a best_segment of the clip; the clip must contain
   in_sec + (end - start) * speed <= clip duration. Avoid reusing the same segment twice; vary motion types
   and subjects between consecutive shots. speed: 1.0 by default, 0.5-0.8 for slow-motion moments.
6. Overlays (all in timeline time, inside [0, duration_sec]):
   - camera_hud: a viewfinder look (REC, ISO, aperture, shutter, timecode) over "behind the camera" or
     handheld moments — a few seconds, not the entire edit unless the brief asks for it. Plausible exposure
     values (aperture like "F 2.8", iso like "ISO 800", shutter like "1/1000").
   - shutter_flashes: 2-4 frames, on strong beats / downbeats at hard cuts and section changes; at most
     one every ~2 s. White by default, black for darker moods.
   - titles: very short (<= 5 words), in the brief's language, readable (>= 1.2 s), never overlapping each
     other; use subtitle "" when none.
7. duration_sec: honour the target duration given in the request (it must also fit in the music after
   audio_start_offset).
Write intent / editing_notes briefly, in the brief's language."""


def _fmt_times(values: list[float], lo: float, hi: float) -> str:
    return ", ".join(f"{v:.3f}" for v in values if lo <= v <= hi)


def build_director_prompt(
    brief: str,
    catalog: MediaCatalog,
    audio: AudioAnalysis | None,
    *,
    fmt: str,
    fps: int,
    target_duration: float,
) -> str:
    clips = []
    for v in catalog.videos:
        a = v.analysis
        clips.append({
            "clip_id": v.clip_id,
            "file": Path(v.rel_path).name,
            "duration": round(v.duration, 2),
            "fps": v.fps,
            "resolution": f"{v.width}x{v.height}",
            "orientation": v.orientation,
            "energy_score": a.energy_score if a else None,
            "lighting": a.lighting if a else None,
            "description": a.scene_description if a else "(non analysé visuellement)",
            "subjects": a.subjects if a else [],
            "tags": a.tags if a else [],
            "best_segments": [
                {"start": s.start, "end": s.end, "motion": s.motion_type, "what": s.description} for s in a.best_segments
            ] if a else [{"start": 0.0, "end": round(v.duration, 2), "motion": "other", "what": "rush complet"}],
        })
    w, h = FORMAT_SIZES[fmt]
    parts = [
        "## BRIEF\n" + brief.strip(),
        f"## OUTPUT FORMAT\nformat {fmt} ({w}x{h}), {fps} fps, target duration ≈ {target_duration:.1f} s",
        "## RUSHES CATALOG\n" + json.dumps(clips, ensure_ascii=False, indent=1),
    ]
    if audio:
        horizon = audio.duration
        parts.append(
            "## MUSIC ANALYSIS (music time, seconds)\n"
            f"duration {audio.duration:.2f} s, bpm {audio.bpm:.1f}\n"
            f"sections: {json.dumps([s.model_dump() for s in audio.sections])}\n"
            f"downbeats: {_fmt_times(audio.downbeats, 0, horizon)}\n"
            f"strong_beats: {_fmt_times(audio.strong_beats, 0, horizon)}\n"
            f"beats: {_fmt_times(audio.beats, 0, horizon)}"
        )
    else:
        parts.append("## MUSIC\nNo music track: set audio_start_offset to 0 and cut on a regular 0.5 s grid.")
    parts.append("Return the TimelineDraft JSON.")
    return "\n\n".join(parts)


def generate_draft(
    brief: str,
    catalog: MediaCatalog,
    audio: AudioAnalysis | None,
    *,
    fmt: str,
    fps: int,
    target_duration: float,
    client=None,
) -> TimelineDraft:
    from .llm import structured_call

    prompt = build_director_prompt(brief, catalog, audio, fmt=fmt, fps=fps, target_duration=target_duration)
    draft, _ = structured_call(
        model=director_model(),
        system=DIRECTOR_SYSTEM,
        content=[{"type": "text", "text": prompt}],
        output_model=TimelineDraft,
        max_tokens=64000,
        effort="high",
        client=client,
    )
    return draft


def heuristic_draft(catalog: MediaCatalog, audio: AudioAnalysis | None, *, target_duration: float, title: str = "Auto edit") -> TimelineDraft:
    """Brouillon déterministe sans API (mode --offline) : cuts sur beats, plans triés par énergie."""
    from .schemas import DraftClip, DraftFlash, DraftHud, DraftScene, DraftTitle

    if not catalog.videos:
        raise TimelineValidationError("Catalogue vide : aucun rush à monter.")
    beats = audio.beats if audio and len(audio.beats) > 1 else [i * 0.5 for i in range(int(target_duration * 2) + 2)]
    duration = min(target_duration, audio.duration) if audio else target_duration
    ranked = sorted(catalog.videos, key=lambda v: -(v.analysis.energy_score if v.analysis else 5))
    cuts = [0.0]
    step = 2
    i = step
    while i < len(beats) and beats[i] < duration - 0.3:
        cuts.append(beats[i])
        i += step
    cuts.append(duration)
    scenes = []
    usage: dict[str, float] = {}
    for k in range(len(cuts) - 1):
        v = ranked[k % len(ranked)]
        in_sec = usage.get(v.clip_id, 0.0)
        usage[v.clip_id] = in_sec + (cuts[k + 1] - cuts[k])
        is_grid = len(cuts) > 6 and k == (len(cuts) - 1) // 2
        clips = [DraftClip(clip_id=ranked[j % len(ranked)].clip_id, in_sec=0.0, speed=1.0) for j in range(9)] if is_grid \
            else [DraftClip(clip_id=v.clip_id, in_sec=in_sec, speed=1.0)]
        section = audio.section_at(cuts[k]).label if audio and audio.sections else "climax"
        scenes.append(DraftScene(type="nine_grid" if is_grid else "single", start=cuts[k], end=cuts[k + 1], clips=clips,
                                 zoom_punch=k == 0, grid_reveal="beats" if is_grid else "all", section=section, intent="auto"))
    return TimelineDraft(
        title=title,
        duration_sec=duration,
        audio_start_offset=0.0,
        scenes=scenes,
        camera_hud=[DraftHud(start=0.0, end=min(duration, 2.5), aperture="F 2.8", iso="ISO 800", shutter="1/1000")],
        shutter_flashes=[DraftFlash(at=0.0, color="white", frames=3)],
        titles=[DraftTitle(start=0.2, end=min(duration, 2.0), text=title, subtitle="", position="center", animation="spring_up", style="minimal")],
        editing_notes="Brouillon heuristique hors-ligne (sans LLM).",
    )


def draft_to_timeline(draft: TimelineDraft, catalog: MediaCatalog, audio: AudioAnalysis | None, *, fmt: str, fps: int) -> Timeline:
    """Convertit la projection plate du LLM en Timeline canonique (sans encore valider le contexte)."""
    scenes = []
    for s in draft.scenes:
        start, end = float(s.start), float(s.end)
        if end <= start:
            end = start + MIN_SCENE_SEC  # sera réparé par le validateur
        refs = []
        for c in s.clips or []:
            speed = float(c.speed) if c.speed and c.speed > 0 else 1.0
            in_sec = max(0.0, float(c.in_sec))
            refs.append(ClipRef(clip_id=c.clip_id, in_sec=in_sec, out_sec=in_sec + max(0.05, (end - start) * speed), speed=speed))
        if not refs:
            continue
        if s.type == "nine_grid":
            scenes.append(NineGridScene(start=max(0.0, start), end=end, cells=refs[:9], reveal=s.grid_reveal, section=s.section, note=s.intent))
        else:
            scenes.append(SingleScene(start=max(0.0, start), end=end, clip=refs[0], zoom_punch=s.zoom_punch, section=s.section, note=s.intent))
    overlays: list = []
    for h in draft.camera_hud:
        if h.end > h.start:
            overlays.append(CameraHudOverlay(start=max(0.0, h.start), end=h.end, aperture=h.aperture, iso=h.iso, shutter=h.shutter))
    for f in draft.shutter_flashes:
        overlays.append(ShutterFlash(at=max(0.0, f.at), frames=int(min(4, max(2, f.frames))), color=f.color))
    for t in draft.titles:
        if t.end > t.start and t.text.strip():
            overlays.append(KineticTitle(start=max(0.0, t.start), end=t.end, text=t.text.strip(), subtitle=t.subtitle.strip() or None,
                                         position=t.position, animation=t.animation, style=t.style))
    audio_ref = None
    if audio:
        asset = catalog.audio_asset(audio_id=audio.audio_id, path=audio.source)
        audio_ref = AudioTrackRef(audio_id=asset.audio_id if asset else audio.audio_id, path=audio.source,
                                  audio_start_offset=max(0.0, float(draft.audio_start_offset)))
    if not scenes:
        raise TimelineValidationError("Le brouillon ne contient aucune scène exploitable.")
    return Timeline(title=draft.title, format=fmt, fps=fps, duration_sec=max(float(draft.duration_sec), MIN_SCENE_SEC),
                    audio=audio_ref, scenes=scenes, overlays=overlays, generator=f"director:{director_model()}")


# =============================================================================================
# 2. Validateur déterministe post-LLM
# =============================================================================================


def _q(t: float, fps: int) -> float:
    """Quantifie un temps sur la grille de frames de la composition."""
    return round(round(t * fps) / fps, 6)


def _nearest(candidates: list[float], t: float, tol: float) -> float | None:
    best, best_d = None, tol + 1e-9
    for c in candidates:
        d = abs(c - t)
        if d <= best_d:
            best, best_d = c, d
    return best


class _ClipPicker:
    """Choisit un rush de remplacement valide (énergie décroissante, rotation)."""

    def __init__(self, valid: list[VideoAsset]):
        self.pool = sorted(valid, key=lambda v: -(v.analysis.energy_score if v.analysis else 5))
        self.i = 0

    def next(self) -> VideoAsset:
        v = self.pool[self.i % len(self.pool)]
        self.i += 1
        return v


def _fix_clip_ref(ref: ClipRef, scene_dur: float, fps: int, catalog: MediaCatalog, valid_ids: set[str],
                  picker: _ClipPicker, report: ValidationReport, where: str) -> ClipRef:
    asset = catalog.video(ref.clip_id) if ref.clip_id in valid_ids else None
    if asset is None and ref.path:
        cand = catalog.video_by_path(ref.path)
        asset = cand if cand and cand.clip_id in valid_ids else None
    if asset is None:
        replacement = picker.next()
        report.add("missing_clip", f"{where}: rush '{ref.clip_id}' introuvable → remplacé par {replacement.clip_id}",
                   target=where, before=ref.clip_id, after=replacement.clip_id)
        asset = replacement
        ref = ClipRef(clip_id=asset.clip_id, in_sec=ref.in_sec, out_sec=ref.in_sec + scene_dur, speed=ref.speed)

    usable = max(1.0 / fps, asset.duration - 1.0 / asset.fps)  # marge d'une frame source
    speed = min(MAX_SPEED, max(MIN_SPEED, ref.speed))
    if speed != ref.speed:
        report.add("speed_clamped", f"{where}: vitesse {ref.speed} → {speed}", target=where, before=ref.speed, after=speed)
    needed = scene_dur * speed
    if needed > usable:
        new_speed = usable / scene_dur * 0.999
        report.add("speed_fit", f"{where}: rush trop court ({asset.duration:.2f}s) pour {scene_dur:.2f}s à x{speed:g} → x{new_speed:.3f}",
                   target=where, before=speed, after=round(new_speed, 4))
        speed = new_speed
        needed = scene_dur * speed
    in_sec = min(max(0.0, ref.in_sec), usable - needed)
    in_sec = math.floor(in_sec * fps + 1e-6) / fps  # trimBefore entier en frames de composition
    in_sec = max(0.0, in_sec)
    out_sec = in_sec + needed
    if abs(in_sec - ref.in_sec) > 1e-3:
        report.add("clamp_in", f"{where}: in_sec {ref.in_sec:.3f} → {in_sec:.3f}", target=where, before=round(ref.in_sec, 3), after=round(in_sec, 3))
    if not (0 <= in_sec < out_sec <= asset.duration + 1e-6):
        raise TimelineValidationError(f"{where}: plage source invalide après clamp ({in_sec}, {out_sec}, {asset.duration})")
    return ClipRef(clip_id=asset.clip_id, in_sec=round(in_sec, 6), out_sec=round(out_sec, 6), speed=round(speed, 6), path=asset.path)


def _grid_reveal_times(scene: NineGridScene, beats_tl: list[float], onsets_tl: list[float], fps: int) -> list[float]:
    """9 instants d'apparition (relatifs à la scène) : beats, puis beats + onsets, sinon répartition régulière."""
    dur = scene.end - scene.start
    min_gap = 2.0 / fps

    def pick(cands: list[float]) -> list[float]:
        inner = sorted({_q(c - scene.start, fps) for c in cands if scene.start - 1e-6 <= c < scene.end - 2.0 / fps})
        out = [0.0]
        for t in inner:
            if t - out[-1] >= min_gap:
                out.append(t)
        return out

    for cands in (beats_tl, beats_tl + onsets_tl):
        times = pick(cands)
        if len(times) >= 9:
            return times[:9]
    span = dur * 0.6
    return [_q(span * i / 8, fps) for i in range(9)]


def validate_timeline(
    timeline: Timeline,
    catalog: MediaCatalog,
    audio: AudioAnalysis | None,
    *,
    tolerance: float = SNAP_TOLERANCE_SEC,
    log: Log = _noop,
) -> tuple[Timeline, ValidationReport]:
    """Rend une timeline exécutable et musicale. Ne lève que si rien n'est récupérable.

    1. fichiers référencés existants (sinon substitution) ;
    2. clamp 0 <= in < out <= durée du clip (vitesse ajustée si besoin) ;
    3. scènes contiguës (ni trou ni chevauchement) et cuts snappés au beat/onset le plus proche (<= tolérance),
       puis quantifiés à la frame ;
    4. overlays bornés, flashs snappés, reveal de grille calé sur les beats.
    """
    report = ValidationReport()
    tl = timeline.model_copy(deep=True)
    fps = tl.fps
    frame = 1.0 / fps
    min_scene = max(MIN_SCENE_SEC, 2 * frame)

    valid = [v for v in catalog.videos if Path(v.path).is_file()]
    for v in catalog.videos:
        if v not in valid:
            report.add("missing_file", f"fichier original introuvable : {v.path}", target=v.clip_id)
    if not valid:
        raise TimelineValidationError("Aucun rush valide (fichiers originaux introuvables).")
    valid_ids = {v.clip_id for v in valid}
    picker = _ClipPicker(valid)

    # --- audio
    available = math.inf
    if tl.audio is not None:
        asset = catalog.audio_asset(audio_id=tl.audio.audio_id, path=tl.audio.path)
        path = asset.path if asset else tl.audio.path
        if not path or not Path(path).is_file():
            fallback = next((a for a in catalog.audio if Path(a.path).is_file()), None)
            if fallback:
                report.add("missing_audio", f"musique introuvable → {fallback.rel_path}", before=str(path), after=fallback.path)
                asset, path = fallback, fallback.path
            else:
                report.add("missing_audio", "musique introuvable → montage sans musique", before=str(path))
                tl.audio = None
        if tl.audio is not None:
            audio_dur = asset.duration if asset else (audio.duration if audio else math.inf)
            tl.audio.path = path
            tl.audio.audio_id = asset.audio_id if asset else tl.audio.audio_id
            offset = min(max(0.0, tl.audio.audio_start_offset), max(0.0, audio_dur - 1.0))
            if audio and audio.beats:
                snapped = _nearest(audio.beats, offset, tolerance)  # t=0 calé sur un beat
                if snapped is not None:
                    offset = snapped
            if abs(offset - tl.audio.audio_start_offset) > 1e-4:
                report.add("audio_offset", f"audio_start_offset {tl.audio.audio_start_offset:.3f} → {offset:.3f}",
                           before=round(tl.audio.audio_start_offset, 3), after=round(offset, 3))
            tl.audio.audio_start_offset = round(offset, 4)
            available = audio_dur - offset

    # --- candidats de snap (temps timeline), par priorité
    offset = tl.audio.audio_start_offset if tl.audio else 0.0
    tiers: list[list[float]] = []
    beats_tl: list[float] = []
    onsets_tl: list[float] = []
    if audio and tl.audio is not None:
        to_tl = lambda arr: sorted({round(t - offset, 6) for t in arr if t - offset > -1e-6})  # noqa: E731
        beats_tl = to_tl(audio.beats)
        onsets_tl = to_tl(audio.onsets)
        tiers = [to_tl(audio.strong_beats + audio.downbeats), beats_tl, onsets_tl]

    # --- structure : tri, contiguïté, durée totale
    scenes = sorted(tl.scenes, key=lambda s: (s.start, s.end))
    duration = tl.duration_sec
    if scenes[-1].end < duration - 1.0:
        report.add("duration_scenes", f"duration_sec {duration:.3f}s > fin de la dernière scène {scenes[-1].end:.3f}s → alignée",
                   before=round(duration, 3), after=round(scenes[-1].end, 3))
        duration = scenes[-1].end
    if duration > available:
        report.add("duration_audio", f"durée {duration:.3f}s > musique disponible {available:.3f}s → tronquée",
                   before=round(duration, 3), after=round(available, 3))
        duration = available
    if not math.isfinite(duration) or duration < min_scene:
        raise TimelineValidationError(f"Durée de montage invalide : {duration}")

    # cuts proposés = frontières entre scènes (moyenne si trou/chevauchement)
    kept = [s for s in scenes if s.start < duration - min_scene]
    if len(kept) < len(scenes):
        report.add("scene_dropped", f"{len(scenes) - len(kept)} scène(s) au-delà de la durée supprimée(s)")
    scenes = kept or scenes[:1]
    raw_cuts = []
    for prev, cur in zip(scenes, scenes[1:]):
        if abs(prev.end - cur.start) > 1e-3:
            kind = "gap" if cur.start > prev.end else "overlap"
            report.add(kind, f"{kind} {abs(cur.start - prev.end):.3f}s entre {prev.end:.3f} et {cur.start:.3f} → recollé")
            raw_cuts.append((prev.end + cur.start) / 2)
        else:
            raw_cuts.append(cur.start)

    # snap + quantification + garde-fous de durée minimale
    end_q = _q(duration, fps)
    if end_q > duration + 1e-9:  # ne jamais dépasser la musique disponible
        end_q = round(end_q - frame, 6)
    cuts: list[float] = [0.0]
    pending_scenes = [scenes[0]]
    for i, c in enumerate(raw_cuts):
        c = min(max(c, 0.0), duration)
        nxt = raw_cuts[i + 1] if i + 1 < len(raw_cuts) else end_q
        target = None
        for tier in tiers:
            cand = _nearest(tier, c, tolerance)
            if cand is not None and cand - cuts[-1] >= min_scene and min(nxt, end_q) - cand >= min_scene:
                target = cand
                break
        snapped = target is not None
        cq = _q(target if snapped else c, fps)
        if cq - cuts[-1] < min_scene - 1e-9 or end_q - cq < min_scene - 1e-9:
            report.add("scene_merged", f"scène trop courte autour de {c:.3f}s fusionnée avec la précédente")
            continue
        if snapped:
            report.snapped_cuts += 1
            delta_ms = abs(cq - c) * 1000
            report.max_snap_delta_ms = round(max(report.max_snap_delta_ms, delta_ms), 1)
            if delta_ms > 0.5:
                report.add("snap", f"cut {c:.3f}s → {cq:.3f}s (beat)", before=round(c, 3), after=cq)
        else:
            report.unsnapped_cuts += 1
            if tiers:
                report.add("unsnapped", f"cut {c:.3f}s : aucun beat/onset à ≤ {tolerance * 1000:.0f} ms (quantifié {cq:.3f}s)")
        cuts.append(cq)
        pending_scenes.append(scenes[i + 1])

    # fin du montage : snap éventuel sur un beat sans dépasser la musique
    if tiers:
        cand = _nearest(tiers[0] + tiers[1], end_q, tolerance)
        if cand is not None and cand <= duration + 1e-9 and cand - cuts[-1] >= min_scene:
            end_q = _q(cand, fps) if _q(cand, fps) <= duration + 1e-9 else end_q
    cuts.append(end_q)

    # --- reconstruction des scènes (contiguës) + plages source
    new_scenes = []
    for k, scene in enumerate(pending_scenes):
        start, end = cuts[k], cuts[k + 1]
        dur = end - start
        where = f"scene[{k}]"
        if isinstance(scene, SingleScene):
            clip = _fix_clip_ref(scene.clip, dur, fps, catalog, valid_ids, picker, report, where)
            new_scenes.append(scene.model_copy(update={"start": start, "end": end, "clip": clip}))
        else:
            cells = list(scene.cells)
            if len(cells) < 9:
                report.add("grid_fill", f"{where}: {len(cells)} case(s) → complétée(s) à 9")
                base = list(cells)
                j = 0
                while len(cells) < 9:
                    src = base[j % len(base)]
                    asset = catalog.video(src.clip_id)
                    span = max(1e-3, (asset.duration if asset else dur) - dur * src.speed)
                    shifted = (src.in_sec + dur * src.speed * (1 + j // len(base))) % span
                    cells.append(src.model_copy(update={"in_sec": shifted, "out_sec": shifted + dur * src.speed}))
                    j += 1
            fixed = [_fix_clip_ref(c, dur, fps, catalog, valid_ids, picker, report, f"{where}.cell[{n}]") for n, c in enumerate(cells[:9])]
            grid = scene.model_copy(update={"start": start, "end": end, "cells": fixed})
            grid.reveal_times = _grid_reveal_times(grid, beats_tl, onsets_tl, fps) if grid.reveal == "beats" else [0.0] * 9
            new_scenes.append(grid)
    tl.scenes = new_scenes
    tl.duration_sec = end_q

    # --- overlays
    cut_points = list(cuts)  # débuts de scènes (dont 0.0) + fin du montage
    flash_tiers = [cuts[:-1]] + tiers[:2]
    overlays = []
    for ov in tl.overlays:
        if isinstance(ov, ShutterFlash):
            if ov.at > end_q - 2 * frame + tolerance:
                report.add("overlay_dropped", f"flash à {ov.at:.3f}s hors montage supprimé")
                continue
            at = ov.at
            for tier in flash_tiers:
                cand = _nearest(tier, at, tolerance)
                if cand is not None:
                    at = cand
                    break
            at = _q(min(max(0.0, at), end_q - ov.frames * frame), fps)
            if abs(at - ov.at) > 1e-3:
                report.add("flash_snapped", f"flash {ov.at:.3f}s → {at:.3f}s", before=round(ov.at, 3), after=at)
            overlays.append(ov.model_copy(update={"at": at}))
            continue
        start, end = max(0.0, ov.start), min(end_q, ov.end)
        snap_s, snap_e = _nearest(cut_points, start, tolerance), _nearest(cut_points, end, tolerance)
        start = snap_s if snap_s is not None else start
        end = snap_e if snap_e is not None else end
        start, end = _q(start, fps), min(_q(end, fps), end_q)
        min_len = 0.6 if isinstance(ov, KineticTitle) else MIN_OVERLAY_SEC
        if end - start < min_len:
            if isinstance(ov, KineticTitle) and end_q - start >= min_len:
                end = _q(start + min_len, fps)
                report.add("title_extended", f"titre « {ov.text} » prolongé à {min_len:.1f}s pour la lisibilité")
            else:
                report.add("overlay_dropped", f"overlay {ov.type} [{ov.start:.2f}-{ov.end:.2f}] trop court/hors montage supprimé")
                continue
        if abs(start - ov.start) > 1e-3 or abs(end - ov.end) > 1e-3:
            report.add("overlay_clamped", f"{ov.type} [{ov.start:.3f}-{ov.end:.3f}] → [{start:.3f}-{end:.3f}]")
        overlays.append(ov.model_copy(update={"start": start, "end": end}))
    overlays.sort(key=lambda o: o.at if isinstance(o, ShutterFlash) else o.start)
    tl.overlays = overlays

    tl = Timeline.model_validate(tl.model_dump())  # revalidation Pydantic stricte
    assert_timeline_invariants(tl, catalog)
    report.scene_count = len(tl.scenes)
    report.duration_sec = tl.duration_sec
    log(f"validation : {report.scene_count} scènes, {report.snapped_cuts} cut(s) snappé(s), "
        f"{report.unsnapped_cuts} non snappé(s), {len(report.fixes)} correction(s)")
    return tl, report


def assert_timeline_invariants(tl: Timeline, catalog: MediaCatalog) -> None:
    """Invariants durs consommés par le renderer — lève TimelineValidationError si violés."""
    eps = 1e-6
    if abs(tl.scenes[0].start) > eps:
        raise TimelineValidationError("la première scène doit commencer à 0.0")
    for a, b in zip(tl.scenes, tl.scenes[1:]):
        if abs(a.end - b.start) > eps:
            raise TimelineValidationError(f"trou/chevauchement entre {a.end} et {b.start}")
    if abs(tl.scenes[-1].end - tl.duration_sec) > eps:
        raise TimelineValidationError("la dernière scène doit finir à duration_sec")
    for k, s in enumerate(tl.scenes):
        if s.end - s.start <= 0:
            raise TimelineValidationError(f"scene[{k}] de durée nulle")
        for ref in s.clip_refs():
            asset = catalog.video(ref.clip_id)
            if asset is None or not Path(asset.path).is_file():
                raise TimelineValidationError(f"scene[{k}]: rush manquant {ref.clip_id}")
            if not (0 <= ref.in_sec < ref.out_sec <= asset.duration + eps):
                raise TimelineValidationError(f"scene[{k}]: plage {ref.in_sec}-{ref.out_sec} hors [0, {asset.duration}]")
            if abs((ref.out_sec - ref.in_sec) - (s.end - s.start) * ref.speed) > 1e-3:
                raise TimelineValidationError(f"scene[{k}]: out-in != durée*speed")
        if isinstance(s, NineGridScene) and len(s.cells) != 9:
            raise TimelineValidationError(f"scene[{k}]: une grille doit avoir 9 cases")
    for ov in tl.overlays:
        if isinstance(ov, ShutterFlash):
            if not (0 <= ov.at <= tl.duration_sec):
                raise TimelineValidationError(f"flash hors montage : {ov.at}")
        elif not (0 <= ov.start < ov.end <= tl.duration_sec + eps):
            raise TimelineValidationError(f"overlay hors montage : {ov}")


# =============================================================================================
# 3. Orchestration
# =============================================================================================


def direct(
    brief: str,
    catalog: MediaCatalog,
    audio: AudioAnalysis | None,
    ws: Workspace,
    *,
    fmt: str = "9:16",
    fps: int = 30,
    target_duration: float | None = None,
    offline: bool = False,
    log: Log = _noop,
    client=None,
) -> tuple[Timeline, ValidationReport]:
    if not catalog.videos:
        raise TimelineValidationError("Aucun rush dans le catalogue : lancez d'abord `analyze`.")
    if target_duration is None:
        target_duration = min(30.0, audio.duration) if audio else 15.0
    if offline:
        log("mode hors-ligne : brouillon heuristique (sans LLM)")
        draft = heuristic_draft(catalog, audio, target_duration=target_duration)
    else:
        log(f"génération de la timeline avec {director_model()}…")
        draft = generate_draft(brief, catalog, audio, fmt=fmt, fps=fps, target_duration=target_duration, client=client)
    save_json(draft.model_dump(), ws.timeline_draft)
    timeline = draft_to_timeline(draft, catalog, audio, fmt=fmt, fps=fps)
    if offline:
        timeline.generator = "director:heuristic"
    timeline, report = validate_timeline(timeline, catalog, audio, log=log)
    save_model(timeline, ws.timeline)
    save_model(report, ws.validation_report)
    return timeline, report
