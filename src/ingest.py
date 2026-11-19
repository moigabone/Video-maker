"""Ingestion des rushs : scan récursif, SHA-256, métadonnées ffprobe, proxies 720p/30fps (+ LUT optionnelle)."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path

from .config import AUDIO_EXTENSIONS, PROXY_FPS, PROXY_SHORT_EDGE, VIDEO_EXTENSIONS, Workspace
from .ffmpeg_utils import lut3d_filter, probe_audio, probe_video, require_binaries, run_ffmpeg
from .schemas import AudioAsset, MediaCatalog, VideoAsset, load_model, save_model, utc_now

Log = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def short_hash(text: str, n: int = 8) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:n]


def scan_media(input_dir: Path) -> tuple[list[Path], list[Path]]:
    """Retourne (vidéos, audios) trouvés récursivement dans input/videos et input/audio.

    Si ces sous-dossiers n'existent pas, `input_dir` est scanné en entier (tri par extension).
    """
    videos_dir, audio_dir = input_dir / "videos", input_dir / "audio"
    video_roots = [videos_dir] if videos_dir.is_dir() else [input_dir]
    audio_roots = [audio_dir] if audio_dir.is_dir() else [input_dir]

    def walk(roots: list[Path], exts: set[str]) -> list[Path]:
        found: list[Path] = []
        for root in roots:
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("."):
                    found.append(p.resolve())
        return sorted(set(found))

    return walk(video_roots, VIDEO_EXTENSIONS), walk(audio_roots, AUDIO_EXTENSIONS)


def proxy_size(width: int, height: int, short_edge: int = PROXY_SHORT_EDGE) -> tuple[int, int]:
    """Dimensions paires du proxy : petit côté ramené à `short_edge` (jamais d'upscale)."""
    short = min(width, height)
    factor = min(1.0, short_edge / short) if short else 1.0
    tw = max(2, int(round(width * factor / 2)) * 2)
    th = max(2, int(round(height * factor / 2)) * 2)
    return tw, th


def build_proxy(asset: VideoAsset, proxies_dir: Path, lut: str | None, log: Log = _noop, force: bool = False) -> Path:
    lut_tag = f"_lut{short_hash(str(Path(lut).resolve()))}" if lut else ""
    out = proxies_dir / f"{asset.clip_id}{lut_tag}.mp4"
    if out.exists() and out.stat().st_size > 0 and not force:
        return out
    tw, th = proxy_size(asset.width, asset.height)
    filters = [f"scale={tw}:{th}:flags=lanczos"]
    lut_f = lut3d_filter(lut)
    if lut_f:
        filters.append(lut_f)
    filters += [f"fps={PROXY_FPS}", "format=yuv420p"]
    tmp = out.with_suffix(".tmp.mp4")
    log(f"proxy {asset.rel_path} → {out.name} ({tw}x{th}@{PROXY_FPS}{' + LUT' if lut else ''})")
    run_ffmpeg([
        "-i", asset.path,
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", ",".join(filters),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-g", str(PROXY_FPS), "-keyint_min", str(PROXY_FPS), "-sc_threshold", "0",
        "-c:a", "aac", "-b:a", "128k", "-ac", "2",
        "-movflags", "+faststart",
        str(tmp),
    ])
    tmp.replace(out)
    return out


def _load_previous(ws: Workspace) -> MediaCatalog | None:
    if ws.catalog.exists():
        try:
            return load_model(ws.catalog, MediaCatalog)
        except Exception:  # catalogue corrompu ou d'un ancien format → on repart de zéro
            return None
    return None


def ingest(
    input_dir: str | Path,
    ws: Workspace,
    *,
    lut: str | Path | None = None,
    make_proxies: bool = True,
    log: Log = _noop,
) -> MediaCatalog:
    """Scanne `input_dir`, met à jour `workspace/media_catalog.json` et génère les proxies.

    Les analyses VLM déjà présentes (cache par SHA-256) sont conservées.
    """
    require_binaries("ffmpeg", "ffprobe")
    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"Dossier d'entrée introuvable : {input_path}")
    lut_str = str(Path(lut).expanduser().resolve()) if lut else None
    if lut_str:
        lut3d_filter(lut_str)  # valide l'existence / le format tôt

    previous = _load_previous(ws)
    prev_by_path: dict[str, VideoAsset | AudioAsset] = {}
    analysis_cache = {}
    if previous:
        prev_by_path.update({v.path: v for v in previous.videos})
        prev_by_path.update({a.path: a for a in previous.audio})
        analysis_cache = dict(previous.analysis_cache)
        for v in previous.videos:
            if v.analysis and v.sha256 not in analysis_cache:
                analysis_cache[v.sha256] = v.analysis

    video_files, audio_files = scan_media(input_path)
    log(f"{len(video_files)} vidéo(s), {len(audio_files)} audio(s) trouvés dans {input_path}")

    def file_hash(p: Path) -> tuple[str, int, float]:
        st = p.stat()
        prev = prev_by_path.get(str(p))
        if prev and prev.size_bytes == st.st_size and abs(prev.mtime - st.st_mtime) < 1e-3:
            return prev.sha256, st.st_size, st.st_mtime
        log(f"sha256 {p.name}")
        return sha256_file(p), st.st_size, st.st_mtime

    videos: list[VideoAsset] = []
    seen_ids: set[str] = set()
    for p in video_files:
        sha, size, mtime = file_hash(p)
        clip_id = f"clip_{sha[:10]}"
        if clip_id in seen_ids:
            log(f"doublon ignoré (même contenu) : {p}")
            continue
        seen_ids.add(clip_id)
        pr = probe_video(p)
        asset = VideoAsset(
            clip_id=clip_id,
            path=str(p),
            rel_path=os.path.relpath(p, input_path),
            sha256=sha,
            size_bytes=size,
            mtime=mtime,
            duration=pr.duration,
            fps=pr.fps,
            width=pr.width,
            height=pr.height,
            codec=pr.codec,
            pix_fmt=pr.pix_fmt,
            bit_depth=pr.bit_depth,
            color_transfer=pr.color_transfer,
            has_audio=pr.has_audio,
            rotation=pr.rotation,
            analysis=analysis_cache.get(sha),
        )
        if make_proxies:
            asset.proxy_path = str(build_proxy(asset, ws.proxies, lut_str, log))
            asset.proxy_lut = lut_str
        videos.append(asset)

    audios: list[AudioAsset] = []
    for p in audio_files:
        sha, size, mtime = file_hash(p)
        pa = probe_audio(p)
        audios.append(
            AudioAsset(
                audio_id=f"audio_{sha[:10]}",
                path=str(p),
                rel_path=os.path.relpath(p, input_path),
                sha256=sha,
                size_bytes=size,
                mtime=mtime,
                duration=pa.duration,
                sample_rate=pa.sample_rate,
                channels=pa.channels,
                codec=pa.codec,
            )
        )

    catalog = MediaCatalog(
        input_dir=str(input_path),
        updated_at=utc_now(),
        lut=lut_str,
        videos=videos,
        audio=audios,
        analysis_cache=analysis_cache,
    )
    save_model(catalog, ws.catalog)
    return catalog
