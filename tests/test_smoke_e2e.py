"""Smoke test end-to-end SANS clé API : fixtures ffmpeg → ingest → audio → (vision/directeur simulés)
→ validateur sur une timeline mock → vrai rendu Remotion → vérification ffprobe.

Les tests s'exécutent dans l'ordre du fichier et partagent tests/workspace.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from src.config import Workspace, vision_model
from src.director import assert_timeline_invariants, direct, validate_timeline
from src.ffmpeg_utils import probe_video
from src.render import find_browser, prepare_render, run_remotion_render, verify_video
from src.schemas import (
    AudioAnalysis,
    MediaCatalog,
    NineGridScene,
    ShutterFlash,
    SingleScene,
    Timeline,
    load_model,
    save_json,
    save_model,
)
from tests.conftest import SMOKE_OUTPUT, TEST_WS

TEST_WS_TIMELINE = TEST_WS / "timeline.json"

FPS = 30
HALF_FRAME = 0.5 / FPS + 1e-6


def _catalog(ws: Workspace) -> MediaCatalog:
    return load_model(ws.catalog, MediaCatalog)


def _audio(ws: Workspace) -> AudioAnalysis:
    return load_model(ws.audio_analysis, AudioAnalysis)


def _by_name(catalog: MediaCatalog, fragment: str):
    return next(v for v in catalog.videos if fragment in v.rel_path)


# --------------------------------------------------------------------------------------------- 1. fixtures

def test_fixtures_generated(input_dir: Path):
    videos = sorted((input_dir / "videos").rglob("*.*"))
    assert len(videos) == 4
    for v in videos:
        assert abs(probe_video(v).duration - 3.0) < 0.1, v
    music = input_dir / "audio" / "beat_120bpm.wav"
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(music)],
                         capture_output=True, text=True, check=True)
    assert abs(float(out.stdout) - 6.0) < 0.05


# --------------------------------------------------------------------------------------------- 2. ingest + audio (via la CLI)

def test_cli_analyze_ingest_and_audio(input_dir: Path, ws: Workspace, monkeypatch):
    import main

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = CliRunner().invoke(main.app, ["--workspace", str(ws.root), "analyze", "--input", str(input_dir), "--skip-vision"])
    assert result.exit_code == 0, result.output

    catalog = _catalog(ws)
    assert len(catalog.videos) == 4 and len(catalog.audio) == 1
    a = _by_name(catalog, "clip_a")
    assert (a.width, a.height, round(a.fps)) == (1920, 1080, 60) and a.codec == "h264"
    b = _by_name(catalog, "clip_b")
    assert (b.width, b.height) == (1080, 1920) and b.orientation == "portrait"
    c = _by_name(catalog, "clip_c")
    assert c.codec == "hevc" and c.bit_depth == 10 and c.rel_path.endswith(".mkv")
    for v in catalog.videos:
        assert len(v.sha256) == 64 and v.clip_id == f"clip_{v.sha256[:10]}"
        assert Path(v.path).is_file()
        proxy = probe_video(v.proxy_path)
        assert min(proxy.width, proxy.height) <= 720 and round(proxy.fps) == 30 and proxy.codec == "h264"
        assert abs(proxy.duration - v.duration) < 0.15

    audio = _audio(ws)
    assert 115 <= audio.bpm <= 125, audio.bpm
    assert len(audio.beats) >= 10
    assert np.allclose(np.diff(audio.beats[1:-1]), 0.5, atol=0.05)
    assert set(audio.strong_beats) <= set(audio.beats) and audio.strong_beats
    assert set(audio.downbeats) <= set(audio.beats) and audio.downbeats
    assert len(audio.onsets) >= len(audio.beats) - 2
    labels = [s.label for s in audio.sections]
    assert labels == ["intro", "buildup", "climax", "outro"], audio.sections
    assert audio.sections[0].start == 0.0 and audio.sections[-1].end == pytest.approx(6.0, abs=0.01)
    for s1, s2 in zip(audio.sections, audio.sections[1:]):
        assert s1.end == s2.start
    climax = audio.sections[2]
    assert climax.start == pytest.approx(3.0, abs=0.1) and climax.end == pytest.approx(5.0, abs=0.1)
    assert climax.mean_energy == max(s.mean_energy for s in audio.sections)


def test_ingest_is_cached_and_lut_applies(input_dir: Path, fixtures_dir: Path, ws: Workspace, tmp_path: Path):
    from src.ingest import ingest

    before = {v.clip_id: Path(v.proxy_path).stat().st_mtime for v in _catalog(ws).videos}
    again = ingest(input_dir, ws)
    assert {v.clip_id: Path(v.proxy_path).stat().st_mtime for v in again.videos} == before  # proxies réutilisés

    lut_ws = Workspace.at(tmp_path / "ws_lut")
    graded = ingest(input_dir, lut_ws, lut=fixtures_dir / "warm.cube")
    assert graded.lut and all(v.proxy_lut == graded.lut and "_lut" in v.proxy_path for v in graded.videos)

    def mean_rgb(video: str) -> np.ndarray:
        png = tmp_path / f"{Path(video).stem}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1", "-i", video, "-frames:v", "1", str(png)], check=True)
        return np.asarray(Image.open(png).convert("RGB"), dtype=np.float64).reshape(-1, 3).mean(0)

    plain = mean_rgb(_by_name(_catalog(ws), "clip_a").proxy_path)
    warm = mean_rgb(_by_name(graded, "clip_a").proxy_path)
    assert warm[0] > plain[0] + 3 and warm[2] < plain[2] - 3, (plain, warm)


# --------------------------------------------------------------------------------------------- 3. vision (VLM simulé)

def test_contact_sheet_and_vision_indexer_with_mocked_claude(ws: Workspace, fake_claude):
    from src.video_indexer import index_catalog, make_contact_sheets

    catalog = _catalog(ws)
    a = _by_name(catalog, "clip_a")
    sheets, times, interval = make_contact_sheets(a, ws)
    assert len(sheets) == 1 and times == [0.5, 1.5, 2.5] and interval == 1.0
    with Image.open(sheets[0]) as im:
        assert max(im.size) <= 1568

    def responder(body: dict) -> str:
        return json.dumps({
            "scene_description": "Mire de test animée", "subjects": ["mire"], "tags": ["test", "couleurs"],
            "lighting": "studio", "energy_score": 12,
            "best_segments": [
                {"start": 0.2, "end": 1.4, "description": "début", "motion_type": "static"},
                {"start": 2.0, "end": 9.0, "description": "déborde → clampé", "motion_type": "pan"},
                {"start": 1.0, "end": 1.05, "description": "trop court → ignoré", "motion_type": "other"},
            ],
        })

    fake = fake_claude(responder)
    index_catalog(catalog, ws, client=fake.client, max_workers=2)
    assert len(fake.requests) == 4
    req = fake.requests[0]
    assert req["model"] == vision_model()
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert req["fallbacks"] == "default"
    assert "server-side-fallback" in fake.headers[0].get("anthropic-beta", "")
    images = [blk for blk in req["messages"][0]["content"] if blk["type"] == "image"]
    assert images and images[0]["source"]["media_type"] == "image/jpeg"

    reloaded = _catalog(ws)
    for v in reloaded.videos:
        assert v.analysis is not None and v.analysis.energy_score == 10  # clampé à 10
        assert v.sha256 in reloaded.analysis_cache
        segs = v.analysis.best_segments
        assert len(segs) == 2 and segs[-1].end <= v.duration
    # Deuxième passage : tout vient du cache SHA-256, aucun appel API
    fake2 = fake_claude(responder)
    index_catalog(reloaded, ws, client=fake2.client)
    assert fake2.requests == []


# --------------------------------------------------------------------------------------------- 4. directeur (LLM simulé + hors-ligne)

def test_director_with_mocked_claude(ws: Workspace, fake_claude, tmp_path: Path):
    catalog, audio = _catalog(ws), _audio(ws)
    ids = [v.clip_id for v in catalog.videos]

    def responder(body: dict) -> str:
        prompt = body["messages"][0]["content"][0]["text"]
        assert "## BRIEF" in prompt and "strong_beats" in prompt and ids[0] in prompt
        clip = lambda i, t=0.0: {"clip_id": ids[i % 4], "in_sec": t, "speed": 1.0}  # noqa: E731
        scene = lambda typ, s, e, clips, sec: {"type": typ, "start": s, "end": e, "clips": clips, "zoom_punch": s == 0,  # noqa: E731
                                               "grid_reveal": "beats" if typ == "nine_grid" else "all", "section": sec, "intent": "test"}
        return json.dumps({
            "title": "Mock", "duration_sec": 6.0, "audio_start_offset": 0.0,
            "scenes": [
                scene("single", 0.0, 1.55, [clip(0, 0.3)], "intro"),       # 1.55 → snap 1.533 (strong beat)
                scene("single", 1.55, 3.06, [clip(1, 0.5)], "buildup"),     # 3.06 → snap 3.019
                scene("nine_grid", 3.06, 5.0, [clip(i) for i in range(9)], "climax"),
                scene("single", 5.0, 6.0, [clip(3, 1.0)], "outro"),
            ],
            "camera_hud": [{"start": 0.0, "end": 1.5, "aperture": "F 2.8", "iso": "ISO 800", "shutter": "1/1000"}],
            "shutter_flashes": [{"at": 0.0, "color": "white", "frames": 3}, {"at": 3.0, "color": "white", "frames": 9}],
            "titles": [{"start": 0.1, "end": 1.4, "text": "MOCK", "subtitle": "", "position": "center",
                        "animation": "spring_up", "style": "minimal"}],
            "editing_notes": "test",
        })

    fake = fake_claude(responder)
    mock_ws = Workspace.at(tmp_path / "ws_director")
    for f in (ws.catalog, ws.audio_analysis):
        (mock_ws.root / f.name).write_bytes(f.read_bytes())
    tl, report = direct("Brief de test", catalog, audio, mock_ws, fmt="9:16", fps=FPS, target_duration=6.0, client=fake.client)
    req = fake.requests[0]
    assert req["output_config"]["format"]["schema"]["properties"]["scenes"]["type"] == "array"
    assert mock_ws.timeline.exists() and mock_ws.timeline_draft.exists() and mock_ws.validation_report.exists()
    assert [s.type for s in tl.scenes] == ["single", "single", "nine_grid", "single"]
    offset = tl.audio.audio_start_offset  # le validateur cale t=0 sur le premier beat (0.023 s)
    assert offset == pytest.approx(0.023, abs=0.002)
    assert tl.scenes[1].start == pytest.approx(1.533 - offset, abs=HALF_FRAME)
    flashes = [o for o in tl.overlays if isinstance(o, ShutterFlash)]
    assert [f.frames for f in flashes] == [3, 4]  # frames clampées à 2-4
    assert report.snapped_cuts >= 2


def test_director_offline_heuristic(ws: Workspace, tmp_path: Path):
    mock_ws = Workspace.at(tmp_path / "ws_offline")
    tl, report = direct("", _catalog(ws), _audio(ws), mock_ws, fmt="16:9", fps=60, target_duration=6.0, offline=True)
    assert (tl.width, tl.height, tl.fps) == (1920, 1080, 60)
    assert_timeline_invariants(tl, _catalog(ws))


# --------------------------------------------------------------------------------------------- 5. validateur sur timeline mock

def _mock_timeline(catalog: MediaCatalog) -> dict:
    a, b, c, d = (_by_name(catalog, k).clip_id for k in ("clip_a", "clip_b", "clip_c", "clip_d"))
    audio = catalog.audio[0]
    ref = lambda cid, i, o, speed=1.0: {"clip_id": cid, "in_sec": i, "out_sec": o, "speed": speed}  # noqa: E731
    return {
        "title": "Smoke test", "format": "9:16", "fps": FPS, "duration_sec": 6.0,
        "audio": {"audio_id": audio.audio_id, "path": audio.path, "audio_start_offset": 0.0, "volume": 1.0,
                  "fade_in_sec": 0.0, "fade_out_sec": 0.5},
        "scenes": [
            # cuts volontairement décalés des beats (≤ 120 ms), avec trou, chevauchement et plages hors bornes
            {"type": "single", "start": 0.0, "end": 1.08, "clip": ref(a, 0.2, 1.28), "zoom_punch": True},
            {"type": "single", "start": 1.10, "end": 2.10, "clip": ref(b, 2.7, 3.2, 0.5)},              # trou + in trop loin
            {"type": "nine_grid", "start": 2.05, "end": 4.05, "reveal": "beats",                          # chevauchement
             "cells": [ref(a, 0, 2), ref(b, 0, 2), ref(c, 0, 2), ref(d, 0, 2)]},                          # 4 cases → 9
            {"type": "single", "start": 4.05, "end": 5.0, "clip": ref(c, 5.0, 6.0), "zoom_punch": True},  # in > durée
            {"type": "single", "start": 5.0, "end": 6.2, "clip": ref("clip_does_not_exist", 0, 1.2)},     # rush absent
        ],
        "overlays": [
            {"type": "camera_hud", "start": 0.0, "end": 2.1, "aperture": "F 4.0", "iso": "ISO 800", "shutter": "1/1000"},
            {"type": "camera_hud", "start": 5.5, "end": 9.0},
            {"type": "kinetic_title", "start": 0.15, "end": 1.9, "text": "SMOKE TEST", "subtitle": "AI VIDEO EDITOR",
             "position": "center", "animation": "spring_up", "style": "minimal"},
            {"type": "kinetic_title", "start": 4.05, "end": 4.95, "text": "BEAT SYNC", "position": "split",
             "animation": "spring_up", "style": "bold"},
            {"type": "kinetic_title", "start": 5.05, "end": 7.2, "text": "FIN", "position": "lower_third",
             "animation": "letter_stagger", "style": "outline"},
            {"type": "shutter_flash", "at": 0.0, "frames": 3, "color": "white"},
            {"type": "shutter_flash", "at": 2.06, "frames": 2, "color": "black"},
            {"type": "shutter_flash", "at": 4.0, "frames": 4, "color": "warm"},
            {"type": "shutter_flash", "at": 7.5, "frames": 3, "color": "white"},                            # hors montage
        ],
    }


def test_validator_on_mock_timeline(ws: Workspace):
    catalog, audio = _catalog(ws), _audio(ws)
    mock_path = save_json(_mock_timeline(catalog), ws.root / "timeline.mock.json")
    raw = load_model(mock_path, Timeline)
    assert {type(s).__name__ for s in raw.scenes} == {"SingleScene", "NineGridScene"}
    assert {o.type for o in raw.overlays} == {"camera_hud", "shutter_flash", "kinetic_title"}

    tl, report = validate_timeline(raw, catalog, audio)
    save_model(tl, ws.timeline)
    save_model(report, ws.validation_report)

    # Audio : t=0 calé sur le premier beat → la musique disponible fait 6.0 - offset
    offset = tl.audio.audio_start_offset
    assert offset == pytest.approx(audio.beats[0], abs=1e-3)
    assert tl.duration_sec <= 6.0 - offset + 1e-9 and tl.duration_sec == pytest.approx(6.0 - offset, abs=1.0 / FPS)

    # Structure : contiguë, de 0 à duration_sec, sans trou ni chevauchement
    assert tl.scenes[0].start == 0.0 and tl.scenes[-1].end == tl.duration_sec
    for s1, s2 in zip(tl.scenes, tl.scenes[1:]):
        assert s1.end == s2.start
    assert len(tl.scenes) == 5

    # Chaque cut est sur un beat (à la quantification frame près) et sur la grille de frames
    candidates = np.array(sorted(set(audio.beats + audio.strong_beats + audio.onsets))) - offset
    for s in tl.scenes[1:]:
        assert np.min(np.abs(candidates - s.start)) <= HALF_FRAME, s.start
        assert abs(s.start * FPS - round(s.start * FPS)) < 1e-6
    assert report.snapped_cuts == 4 and report.unsnapped_cuts == 0
    assert report.max_snap_delta_ms <= 120 + 1000 * HALF_FRAME

    # Plages source clampées dans [0, durée du rush]
    for s in tl.scenes:
        for r in s.clip_refs():
            asset = catalog.video(r.clip_id)
            assert 0 <= r.in_sec < r.out_sec <= asset.duration
            assert r.path == asset.path
    kinds = {f.kind for f in report.fixes}
    assert {"missing_clip", "clamp_in", "gap", "overlap", "grid_fill", "snap", "overlay_dropped"} <= kinds, kinds
    assert tl.scenes[4].clip.clip_id in {v.clip_id for v in catalog.videos}
    assert tl.scenes[1].clip.speed == 0.5

    grid = tl.scenes[2]
    assert isinstance(grid, NineGridScene) and len(grid.cells) == 9
    assert len(grid.reveal_times) == 9 and grid.reveal_times[0] == 0.0
    assert all(t2 > t1 for t1, t2 in zip(grid.reveal_times, grid.reveal_times[1:]))
    assert grid.reveal_times[-1] < grid.end - grid.start

    # Overlays bornés ; flash hors montage supprimé ; flashs sur les cuts
    flashes = [o for o in tl.overlays if isinstance(o, ShutterFlash)]
    assert len(flashes) == 3
    cuts = {round(s.start, 6) for s in tl.scenes}
    assert all(round(f.at, 6) in cuts for f in flashes)
    for o in tl.overlays:
        if not isinstance(o, ShutterFlash):
            assert 0 <= o.start < o.end <= tl.duration_sec
    assert sum(1 for o in tl.overlays if o.type == "kinetic_title") == 3
    assert isinstance(tl.scenes[0], SingleScene) and tl.scenes[0].zoom_punch

    # Idempotence : re-valider une timeline valide ne change plus les cuts
    tl2, _ = validate_timeline(tl, catalog, audio)
    assert [s.start for s in tl2.scenes] == [s.start for s in tl.scenes]


def test_cli_validate_command(ws: Workspace):
    import main

    result = CliRunner().invoke(main.app, ["--workspace", str(ws.root), "validate"])
    assert result.exit_code == 0, result.output
    assert "timeline validée" in result.output


# --------------------------------------------------------------------------------------------- 6. rendu Remotion réel

def test_remotion_render_smoke(ws: Workspace):
    catalog, tl = _catalog(ws), load_model(ws.timeline, Timeline)
    props_path = prepare_render(tl, catalog, ws, mode="final", conform=True)
    props = json.loads(props_path.read_text())
    refs = [s["clip"] for s in props["timeline"]["scenes"] if s["type"] == "single"] + \
           [c for s in props["timeline"]["scenes"] if s["type"] == "nine_grid" for c in s["cells"]]
    for r in refs:
        assert (ws.render_public / r["src"]).exists()
    assert props["timeline"]["audio"]["src"].startswith("media/")

    if SMOKE_OUTPUT.exists():
        SMOKE_OUTPUT.unlink()
    stages: list[str] = []
    info = run_remotion_render(ws, SMOKE_OUTPUT, quality="high", on_progress=lambda st, p: stages.append(st))
    assert "rendering" in stages

    info = verify_video(SMOKE_OUTPUT)
    assert info["size_bytes"] > 50_000
    assert (info["width"], info["height"]) == (1080, 1920)
    assert info["video_codec"] == "h264" and info["audio_codec"] == "aac"
    assert info["duration"] == pytest.approx(tl.duration_sec, abs=0.1)
    assert abs(info["nb_frames"] - round(tl.duration_sec * FPS)) <= 1

    # Le contenu n'est pas noir : plan plein écran (0.5 s) et mosaïque (3.6 s, toutes les cases révélées)
    def frame_at(t: float) -> np.ndarray:
        png = ws.root / f"frame_{t:.1f}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t}", "-i", str(SMOKE_OUTPUT), "-frames:v", "1", str(png)], check=True)
        return np.asarray(Image.open(png).convert("L"), dtype=np.float64)

    single = frame_at(0.5)
    assert single.mean() > 25
    grid_frame = frame_at(3.6)
    assert grid_frame.mean() > 25
    # bordures noires de la mosaïque 3x3 : colonne au tiers de la largeur
    border_col = grid_frame[200:-200, 358:362].mean()
    assert border_col < grid_frame.mean() * 0.6, (border_col, grid_frame.mean())


# --------------------------------------------------------------------------------------------- 7. styles

def test_learn_style_on_rendered_smoke(tmp_path: Path, fake_claude):
    """learn-style : un style agrège plusieurs exemples, ajoutables au fil du temps (Claude simulé)."""
    from src.style_learner import learn_style, load_style, style_prompt_section

    tl = load_model(TEST_WS_TIMELINE, Timeline)
    true_cuts = [s.start for s in tl.scenes[1:]]
    calls = {"vision": 0, "synthesis": 0}

    def responder(body: dict) -> str:
        has_image = any(b["type"] == "image" for b in body["messages"][0]["content"])
        calls["vision" if has_image else "synthesis"] += 1
        tag = "exemple" if has_image else "synthèse"
        return json.dumps({"summary": f"lecture {tag}", "structure": "accroche puis cuts", "title_style": "split",
                           "transitions": "flashs", "color_grade": "saturé", "editing_rules": [f"règle {tag}"]})

    fake = fake_claude(responder)
    profile = learn_style(SMOKE_OUTPUT, "Smoke Test!", styles_dir=tmp_path, client=fake.client)
    assert profile.name == "smoke-test" and (tmp_path / "smoke-test.json").exists()
    assert calls == {"vision": 1, "synthesis": 0} and len(profile.examples) == 1
    ex = profile.examples[0]
    assert ex.reading.source == "claude" and profile.reading.editing_rules == ["règle exemple"]
    assert ex.format_hint == profile.format_hint == "9:16"
    assert ex.pacing.content_duration == pytest.approx(tl.duration_sec, abs=0.1)  # pas d'écran de fin à exclure
    for c in true_cuts:
        assert min(abs(c - d) for d in ex.cut_times) <= 0.1, (c, ex.cut_times)
    assert ex.pacing.bpm == pytest.approx(120, abs=5)
    assert ex.pacing.cuts_on_half_beat_ratio is not None and ex.pacing.cuts_on_half_beat_ratio >= 0.5
    assert ex.look.flash_times

    # Deuxième exemple « dans le même délire » ajouté plus tard → agrégat + synthèse
    second = tmp_path / "second.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(SMOKE_OUTPUT), "-t", "4", "-c:v", "libx264",
                    "-c:a", "aac", str(second)], check=True)
    profile = learn_style([second, SMOKE_OUTPUT], "smoke-test", styles_dir=tmp_path, client=fake.client)
    assert len(profile.examples) == 2  # la vidéo déjà connue est ignorée (même SHA-256)
    assert calls == {"vision": 2, "synthesis": 1}
    assert profile.reading.editing_rules == ["règle synthèse"]
    assert profile.pacing.shot_count == sum(e.pacing.shot_count for e in profile.examples)
    assert profile.pacing.content_duration == pytest.approx(sum(e.pacing.content_duration for e in profile.examples))

    reloaded = load_style("smoke-test", styles_dir=tmp_path)
    assert len(reloaded.examples) == 2
    section = style_prompt_section(reloaded)
    assert "STYLE REFERENCE" in section and "2 example edit(s)" in section


def test_bundled_styles_and_director_with_style(ws: Workspace, tmp_path: Path):
    import main
    from src.director import build_director_prompt
    from src.style_learner import STYLES_DIR, list_styles, load_style

    names = {st.name for st in list_styles(STYLES_DIR)}
    assert "cinematic-travel" in names
    result = CliRunner().invoke(main.app, ["styles"])
    assert result.exit_code == 0 and "cinematic-travel" in result.output

    style = load_style("cinematic-travel")
    assert len(style.examples) >= 2 and style.reading.source in ("manual", "claude")
    catalog, audio = _catalog(ws), _audio(ws)
    prompt = build_director_prompt("brief", catalog, audio, fmt="16:9", fps=FPS, target_duration=6.0, style=style)
    assert "STYLE REFERENCE « cinematic-travel »" in prompt and style.reading.editing_rules[0] in prompt

    tl, _ = direct("", catalog, audio, Workspace.at(tmp_path / "ws_style"), fmt="16:9", fps=FPS,
                   target_duration=6.0, offline=True, style=style)
    assert_timeline_invariants(tl, catalog)
    titles = [o for o in tl.overlays if o.type == "kinetic_title"]
    assert titles and titles[0].position == "split"
    assert any(o.type == "shutter_flash" and o.color == "warm" for o in tl.overlays)
    # rythme du style (≈ 1 beat par plan) : nettement plus de plans que le brouillon par défaut (2 beats)
    default_tl, _ = direct("", catalog, audio, Workspace.at(tmp_path / "ws_default"), fmt="16:9", fps=FPS,
                           target_duration=6.0, offline=True)
    assert len(tl.scenes) > len(default_tl.scenes)


def test_browser_detection():
    # Informations de diagnostic : un navigateur local évite le téléchargement par Remotion
    browser = find_browser()
    assert browser is None or Path(browser).exists()
