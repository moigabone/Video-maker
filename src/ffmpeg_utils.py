"""Wrappers ffprobe / ffmpeg (jamais de shell=True)."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def require_binaries(*names: str) -> None:
    missing = [n for n in names if shutil.which(n) is None]
    if missing:
        raise FFmpegError(f"Binaire(s) introuvable(s) dans le PATH : {', '.join(missing)}")


def run_ffmpeg(args: list[str], *, timeout: float | None = None) -> None:
    """Lance ffmpeg avec des arguments déjà découpés. Lève FFmpegError avec la fin de stderr."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise FFmpegError(f"ffmpeg a échoué ({proc.returncode}) : {' '.join(cmd)}\n{tail}")


def ffprobe_json(path: str | Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe a échoué sur {path}: {proc.stderr.strip()}")
    return json.loads(proc.stdout or "{}")


def _parse_rate(rate: str | None) -> float:
    if not rate or rate in ("0/0", "0"):
        return 0.0
    try:
        return float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _rotation(stream: dict) -> int:
    rot = 0
    tags = stream.get("tags") or {}
    if "rotate" in tags:
        try:
            rot = int(float(tags["rotate"]))
        except ValueError:
            rot = 0
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                rot = int(float(sd["rotation"]))
            except (TypeError, ValueError):
                pass
    return rot % 360


@dataclass
class VideoProbe:
    duration: float
    fps: float
    width: int
    height: int
    codec: str
    pix_fmt: str | None
    bit_depth: int | None
    color_transfer: str | None
    rotation: int
    has_audio: bool


@dataclass
class AudioProbe:
    duration: float
    sample_rate: int
    channels: int
    codec: str


def probe_video(path: str | Path) -> VideoProbe:
    data = ffprobe_json(path)
    streams = data.get("streams") or []
    v = next((s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")), None)
    if v is None:
        raise FFmpegError(f"Aucun flux vidéo dans {path}")
    fps = _parse_rate(v.get("avg_frame_rate")) or _parse_rate(v.get("r_frame_rate"))
    duration = float(v.get("duration") or 0) or float((data.get("format") or {}).get("duration") or 0)
    if duration <= 0 and v.get("nb_frames") and fps:
        duration = int(v["nb_frames"]) / fps
    rotation = _rotation(v)
    w, h = int(v.get("width") or 0), int(v.get("height") or 0)
    if rotation in (90, 270):
        w, h = h, w
    bit_depth = v.get("bits_per_raw_sample")
    if not bit_depth:
        pf = v.get("pix_fmt") or ""
        bit_depth = 10 if "10" in pf else 12 if "12" in pf else 8 if pf else None
    return VideoProbe(
        duration=round(duration, 4),
        fps=round(fps, 4) if fps else 30.0,
        width=w,
        height=h,
        codec=v.get("codec_name") or "unknown",
        pix_fmt=v.get("pix_fmt"),
        bit_depth=int(bit_depth) if bit_depth else None,
        color_transfer=v.get("color_transfer"),
        rotation=rotation,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def probe_audio(path: str | Path) -> AudioProbe:
    data = ffprobe_json(path)
    a = next((s for s in data.get("streams") or [] if s.get("codec_type") == "audio"), None)
    if a is None:
        raise FFmpegError(f"Aucun flux audio dans {path}")
    duration = float(a.get("duration") or 0) or float((data.get("format") or {}).get("duration") or 0)
    return AudioProbe(
        duration=round(duration, 4),
        sample_rate=int(a.get("sample_rate") or 0),
        channels=int(a.get("channels") or 0),
        codec=a.get("codec_name") or "unknown",
    )


def escape_filter_path(path: str | Path) -> str:
    """Échappe un chemin pour l'utiliser comme valeur d'option dans un filtergraph ffmpeg."""
    s = str(Path(path).resolve())
    for ch in ("\\", ":", "'", ",", ";", "[", "]", "="):
        s = s.replace(ch, "\\" + ch)
    return s


def lut3d_filter(lut_path: str | Path | None) -> str | None:
    if not lut_path:
        return None
    p = Path(lut_path)
    if not p.is_file():
        raise FFmpegError(f"LUT introuvable : {p}")
    if p.suffix.lower() not in {".cube", ".3dl", ".dat", ".m3d", ".csp"}:
        raise FFmpegError(f"Format de LUT non supporté par ffmpeg lut3d : {p.suffix}")
    return f"lut3d=file={escape_filter_path(p)}:interp=tetrahedral"
