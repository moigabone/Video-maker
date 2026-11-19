"""Génère les médias synthétiques du smoke test avec ffmpeg (4 clips colorés de 3 s + musique rythmique de 6 s)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ffmpeg_utils import run_ffmpeg  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"

# (nom relatif, source lavfi, fps, args d'encodage) — formats / conteneurs / codecs variés à dessein
VIDEOS = [
    ("videos/clip_a_testsrc_1080p60.mp4", "testsrc2=size=1920x1080:rate=60", ["-c:v", "libx264", "-pix_fmt", "yuv420p"]),
    ("videos/vertical/clip_b_mandelbrot_portrait.mov", "mandelbrot=size=1080x1920:rate=30:end_scale=0.05", ["-c:v", "libx264", "-pix_fmt", "yuv420p"]),
    ("videos/clip_c_bars_hevc10.mkv",
     "smptehdbars=size=1280x720:rate=25,drawbox=x='mod(t*400\\,1100)':y=260:w=180:h=200:color=yellow@0.9:t=fill",
     ["-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", "log-level=error"]),
    ("videos/clip_d_life_orange.mp4",
     "life=size=640x360:rate=30:mold=10:ratio=0.3:life_color=#ff8a00:death_color=#2a0845,scale=1280:720:flags=neighbor",
     ["-c:v", "libx264", "-pix_fmt", "yuv420p"]),
]

# Musique 120 BPM : kick sur chaque temps, clap sur 2 et 4, charleston en croches ;
# enveloppe intro (calme) → buildup → climax (3-5 s) → outro (décroissance).
AUDIO_EXPR = (
    "(0.22+0.28*gte(t,1.5)+0.5*gte(t,3)-0.7*gte(t,5)*(t-5))*("
    "0.9*sin(2*PI*(50+90*exp(-35*mod(t,0.5)))*mod(t,0.5))*exp(-9*mod(t,0.5))"
    "+0.45*gte(t,1.5)*(2*random(0)-1)*exp(-28*mod(t-0.5,1))*gte(mod(t,1),0.5)"
    "+0.12*gte(t,1.5)*(2*random(1)-1)*exp(-70*mod(t,0.25))"
    "+0.35*gte(t,3)*lt(t,5)*sin(2*PI*110*t)*exp(-4*mod(t,0.5)))"
)


def write_lut(path: Path, size: int = 9) -> Path:
    """LUT .cube « chaude » (rouge +, bleu −) pour tester l'option --lut."""
    lines = ["TITLE \"warm test\"", f"LUT_3D_SIZE {size}"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                rr, gg, bb = r / (size - 1), g / (size - 1), b / (size - 1)
                lines.append(f"{min(1.0, rr * 1.15 + 0.03):.6f} {gg:.6f} {max(0.0, bb * 0.75):.6f}")
    path.write_text("\n".join(lines) + "\n")
    return path


def make_fixtures(root: Path = FIXTURES, force: bool = False) -> Path:
    inp = root / "input"
    for rel, src, enc in VIDEOS:
        out = inp / rel
        if out.exists() and not force:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        run_ffmpeg([
            "-f", "lavfi", "-i", src,
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
            "-t", "3", *enc, "-c:a", "aac", "-shortest", str(out),
        ])
    music = inp / "audio" / "beat_120bpm.wav"
    if not music.exists() or force:
        music.parent.mkdir(parents=True, exist_ok=True)
        run_ffmpeg(["-f", "lavfi", "-i", f"aevalsrc='{AUDIO_EXPR}':s=44100:d=6", "-c:a", "pcm_s16le", str(music)])
    lut = root / "warm.cube"
    if not lut.exists() or force:
        write_lut(lut)
    return root


if __name__ == "__main__":
    print(make_fixtures(force="--force" in sys.argv))
