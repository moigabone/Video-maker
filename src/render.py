"""Pont Python → Remotion : préparation des médias (conform/LUT ou proxies), props, rendu et preview."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import RENDERER_DIR, Workspace
from .ffmpeg_utils import ffprobe_json, lut3d_filter, run_ffmpeg
from .ingest import short_hash
from .schemas import MediaCatalog, Timeline, save_json

Log = Callable[[str], None]
Progress = Callable[[str, float], None]

QUALITY_PRESETS = ("draft", "medium", "high")
CONFORM_PAD_SEC = 0.5
CONFORM_MERGE_GAP_SEC = 2.0


def _noop(*_: object) -> None:
    pass


class RenderError(RuntimeError):
    pass


def find_browser() -> str | None:
    """Chrome Headless Shell pour Remotion : env, puis installation Playwright locale, sinon None (téléchargement Remotion)."""
    env = os.environ.get("REMOTION_BROWSER_EXECUTABLE")
    if env and Path(env).exists():
        return env
    candidates = [Path("/opt/pw-browsers/chromium_headless_shell")]
    for root in (Path("/opt/pw-browsers"), Path.home() / ".cache/ms-playwright"):
        if root.is_dir():
            candidates += sorted(root.glob("chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell"), reverse=True)
            candidates += sorted(root.glob("chromium_headless_shell-*/chrome-linux/headless_shell"), reverse=True)
    for c in candidates:
        if c.exists() and os.access(c.resolve(), os.X_OK):
            return str(c.resolve())
    return None


def node_available() -> bool:
    return shutil.which("node") is not None and (RENDERER_DIR / "node_modules" / "remotion").is_dir()


def _link(target: str | Path, link: Path) -> None:
    """Place `target` dans le dossier public Remotion.

    Hard link (instantané, sans duplication) : le serveur de fichiers de Remotion refuse de servir
    les symlinks (lstat → 404). Repli sur une copie si la cible est sur un autre volume.
    """
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    src = Path(target).resolve()
    try:
        os.link(src, link)
    except OSError:
        shutil.copy2(src, link)


@dataclass
class _Range:
    start: float
    end: float


def _merge_ranges(ranges: list[_Range]) -> list[_Range]:
    merged: list[_Range] = []
    for r in sorted(ranges, key=lambda x: x.start):
        if merged and r.start - merged[-1].end <= CONFORM_MERGE_GAP_SEC:
            merged[-1].end = max(merged[-1].end, r.end)
        else:
            merged.append(_Range(r.start, r.end))
    return merged


def conform_segment(src: str, start: float, end: float, out: Path, lut: str | None, log: Log = _noop) -> Path:
    """Extrait [start, end] de l'original en résolution/fps natifs (+ LUT), intermédiaire H.264 quasi sans perte."""
    if out.exists() and out.stat().st_size > 0:
        return out
    filters = [f for f in (lut3d_filter(lut), "format=yuv420p") if f]
    tmp = out.with_suffix(".tmp.mp4")
    log(f"conform {Path(src).name} [{start:.2f}-{end:.2f}s]{' + LUT' if lut else ''}")
    run_ffmpeg([
        "-ss", f"{start:.3f}", "-i", src, "-t", f"{end - start:.3f}",
        "-map", "0:v:0", "-vf", ",".join(filters),
        "-c:v", "libx264", "-preset", "fast", "-crf", "12", "-an",
        "-movflags", "+faststart", str(tmp),
    ])
    tmp.replace(out)
    return out


def prepare_render(
    timeline: Timeline,
    catalog: MediaCatalog,
    ws: Workspace,
    *,
    mode: Literal["final", "preview"] = "final",
    conform: bool = True,
    lut: str | None = None,
    log: Log = _noop,
) -> Path:
    """Construit workspace/render_public (hard links) + workspace/render_props.json pour Remotion.

    - preview : proxies 720p (déjà étalonnés si l'ingest avait une LUT) ;
    - final   : originaux ; avec `conform` (défaut) seuls les segments utilisés sont extraits en qualité
      quasi sans perte (LUT appliquée à ce moment), sinon liens directs vers les originaux.
    """
    lut = lut or catalog.lut
    if mode == "final" and lut and not conform:
        log("LUT demandée : conform activé automatiquement (Remotion ne sait pas appliquer une LUT)")
        conform = True
    media_dir = ws.render_public / "media"
    if media_dir.exists():
        shutil.rmtree(media_dir)
    media_dir.mkdir(parents=True)

    data = timeline.model_dump(mode="json")
    refs = []  # dicts ClipRef de la copie JSON (modifiés en place : src + in/out recalés)
    for scene in data["scenes"]:
        for ref in ([scene["clip"]] if scene["type"] == "single" else scene["cells"]):
            refs.append(ref)

    by_clip: dict[str, list[dict]] = {}
    for ref in refs:
        by_clip.setdefault(ref["clip_id"], []).append(ref)

    for clip_id, clip_refs in by_clip.items():
        asset = catalog.video(clip_id)
        if asset is None:
            raise RenderError(f"clip_id inconnu dans la timeline : {clip_id} (relancez `direct` ou `validate`)")
        if mode == "preview":
            src = asset.proxy_path if asset.proxy_path and Path(asset.proxy_path).exists() else asset.path
            name = f"{clip_id}{Path(src).suffix.lower()}"
            _link(src, media_dir / name)
            for ref in clip_refs:
                ref["src"] = f"media/{name}"
            continue
        if not conform:
            name = f"{clip_id}{Path(asset.path).suffix.lower()}"
            _link(asset.path, media_dir / name)
            for ref in clip_refs:
                ref["src"] = f"media/{name}"
            continue
        ranges = _merge_ranges([
            _Range(max(0.0, r["in_sec"] - CONFORM_PAD_SEC), min(asset.duration, r["out_sec"] + CONFORM_PAD_SEC)) for r in clip_refs
        ])
        lut_tag = f"_lut{short_hash(str(Path(lut).resolve()))}" if lut else ""
        for rg in ranges:
            seg_name = f"{clip_id}_{int(rg.start * 1000)}_{int(rg.end * 1000)}{lut_tag}.mp4"
            seg = conform_segment(asset.path, rg.start, rg.end, ws.conform / seg_name, lut, log)
            _link(seg, media_dir / seg_name)
            for ref in clip_refs:
                if rg.start - 1e-6 <= ref["in_sec"] and ref["out_sec"] <= rg.end + 1e-6:
                    ref["in_sec"] = round(ref["in_sec"] - rg.start, 6)
                    ref["out_sec"] = round(ref["out_sec"] - rg.start, 6)
                    ref["src"] = f"media/{seg_name}"

    missing = [r["clip_id"] for r in refs if "src" not in r]
    if missing:
        raise RenderError(f"Média non résolu pour : {sorted(set(missing))}")

    if data.get("audio"):
        a = data["audio"]
        asset = catalog.audio_asset(audio_id=a.get("audio_id"), path=a.get("path"))
        path = asset.path if asset else a.get("path")
        if path and Path(path).is_file():
            name = f"{(asset.audio_id if asset else 'music')}{Path(path).suffix.lower()}"
            _link(path, media_dir / name)
            a["src"] = f"media/{name}"
        else:
            log("musique introuvable : rendu sans audio")
            data["audio"] = None

    save_json({"timeline": data, "previewMode": mode == "preview"}, ws.render_props)
    return ws.render_props


def _node_cmd(props: Path, public_dir: Path, output: Path, quality: str, concurrency: int | None) -> list[str]:
    cmd = ["node", str(RENDERER_DIR / "scripts" / "render.mjs"),
           "--props", str(props), "--public-dir", str(public_dir), "--output", str(output), "--quality", quality]
    browser = find_browser()
    if browser:
        cmd += ["--browser", browser]
    if concurrency:
        cmd += ["--concurrency", str(concurrency)]
    return cmd


def run_remotion_render(
    ws: Workspace,
    output: str | Path,
    *,
    quality: str = "high",
    concurrency: int | None = None,
    on_progress: Progress = _noop,
    log: Log = _noop,
) -> dict:
    """Lance renderer/scripts/render.mjs et renvoie les infos ffprobe du MP4 produit."""
    if quality not in QUALITY_PRESETS:
        raise RenderError(f"qualité inconnue : {quality} (choix : {', '.join(QUALITY_PRESETS)})")
    if not node_available():
        raise RenderError("Remotion non installé : lancez `cd renderer && npm install`.")
    out = Path(output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = _node_cmd(ws.render_props, ws.render_public, out, quality, concurrency)
    proc = subprocess.Popen(cmd, cwd=RENDERER_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                evt = None
            if evt and evt.get("event") == "progress":
                on_progress(evt.get("stage", "render"), float(evt.get("progress", 0)))
                continue
            if evt and evt.get("event") == "log":
                log(evt.get("message", ""))
                continue
        tail.append(line)
        tail = tail[-40:]
    code = proc.wait()
    if code != 0:
        raise RenderError("Le rendu Remotion a échoué :\n" + "\n".join(tail))
    return verify_video(out)


def verify_video(path: str | Path) -> dict:
    """Vérifie via ffprobe que le fichier est un MP4 lisible, non vide, avec un flux vidéo."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise RenderError(f"Fichier de sortie absent ou vide : {p}")
    info = ffprobe_json(p)
    streams = info.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RenderError(f"Aucun flux vidéo dans {p}")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return {
        "path": str(p),
        "size_bytes": p.stat().st_size,
        "duration": float((info.get("format") or {}).get("duration") or 0),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": video.get("avg_frame_rate"),
        "video_codec": video.get("codec_name"),
        "nb_frames": int(video.get("nb_frames") or 0),
        "audio_codec": audio.get("codec_name") if audio else None,
    }


def launch_studio(ws: Workspace, port: int | None = None) -> int:
    """Ouvre Remotion Studio sur render_props.json (bloquant jusqu'à Ctrl+C)."""
    if not node_available():
        raise RenderError("Remotion non installé : lancez `cd renderer && npm install`.")
    cmd = ["npx", "remotion", "studio", "src/index.ts", f"--props={ws.render_props}", f"--public-dir={ws.render_public}"]
    if port:
        cmd.append(f"--port={port}")
    browser = find_browser()
    env = dict(os.environ)
    if browser:
        env.setdefault("REMOTION_BROWSER_EXECUTABLE", browser)
    return subprocess.call(cmd, cwd=RENDERER_DIR, env=env)
