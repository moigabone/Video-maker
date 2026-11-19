"""Analyse musicale (librosa) : BPM, beats, downbeats, strong beats, onsets percussifs, sections d'énergie."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .config import Workspace
from .ffmpeg_utils import run_ffmpeg
from .ingest import sha256_file
from .schemas import AudioAnalysis, EnergySection, MediaCatalog, load_model, save_model

Log = Callable[[str], None]

SR = 22050
HOP = 512
CURVE_HZ = 10.0


def _noop(_: str) -> None:
    pass


def _decode_mono(path: Path, sr: int = SR) -> np.ndarray:
    """Décode n'importe quel format via ffmpeg → wav mono float, puis charge avec librosa."""
    import librosa

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "mono.wav"
        run_ffmpeg(["-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_f32le", str(wav)])
        y, _ = librosa.load(wav, sr=sr, mono=True)
    return y.astype(np.float32)


def _round_times(values, ndigits: int = 3) -> list[float]:
    return [round(float(v), ndigits) for v in values]


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(x) < 3:
        return x.copy()
    win = min(win, len(x))
    kernel = np.hanning(win + 2)[1:-1]
    kernel /= kernel.sum()
    pad = win // 2
    xp = np.pad(x, (pad, win - 1 - pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def _accent(env: np.ndarray, frames: np.ndarray, radius: int = 2) -> np.ndarray:
    """Force d'attaque max autour de chaque frame (±radius)."""
    out = np.zeros(len(frames), dtype=np.float64)
    for i, f in enumerate(frames):
        lo, hi = max(0, int(f) - radius), min(len(env), int(f) + radius + 1)
        out[i] = float(env[lo:hi].max()) if hi > lo else 0.0
    return out


def beat_sync_energy(rms: np.ndarray, rms_t: np.ndarray, beats: np.ndarray, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Énergie RMS moyenne par intervalle inter-beats (supprime l'oscillation intra-beat).

    Retourne (bornes [n+1], énergies [n]) ; sans beats exploitables, une grille de 0.5 s est utilisée.
    """
    grid = beats[(beats > 0.05) & (beats < duration - 0.05)] if beats.size else np.array([])
    if grid.size < 2:
        grid = np.arange(0.5, duration, 0.5)
    bounds = np.concatenate([[0.0], grid, [duration]])
    energies = np.zeros(len(bounds) - 1)
    for k in range(len(bounds) - 1):
        mask = (rms_t >= bounds[k]) & (rms_t < bounds[k + 1])
        energies[k] = float(rms[mask].mean()) if mask.any() else 0.0
    return bounds, energies


def segment_energy(bounds: np.ndarray, energies: np.ndarray, duration: float) -> list[EnergySection]:
    """Découpe l'énergie par beat en intro / buildup / climax / outro (frontières toujours sur des beats).

    - climax  : bloc contigu autour du pic où l'énergie >= lo + 65 % de la dynamique ;
    - intro   : jusqu'au premier beat dépassant lo + 15 % (sortie du plateau bas) ;
    - buildup : entre l'intro et le climax ; outro : après le climax.
    Les énergies sont lissées sur ~1/16 du morceau (macro-structure) et normalisées 0-1.
    """
    n = len(energies)
    if n == 0:
        return [EnergySection(label="climax", start=0.0, end=round(duration, 3), mean_energy=0.0)]
    win = max(1, int(round(n / 16)))
    e = _smooth(energies, win) if win > 1 else energies.astype(np.float64)
    e = e / (e.max() + 1e-12)
    lo, hi = float(e.min()), float(e.max())
    if hi - lo < 1e-6:
        return [EnergySection(label="climax", start=0.0, end=round(duration, 3), mean_energy=round(hi, 3))]
    thr_hi = lo + 0.65 * (hi - lo)
    thr_mid = lo + 0.15 * (hi - lo)
    peak = int(np.argmax(e))
    a = peak
    while a > 0 and e[a - 1] >= thr_hi:
        a -= 1
    b = peak
    while b < n - 1 and e[b + 1] >= thr_hi:
        b += 1
    above_mid = np.nonzero(e[: a + 1] >= thr_mid)[0]
    i_intro = int(above_mid[0]) if above_mid.size else a
    # indices de segments : intro [0, i_intro), buildup [i_intro, a), climax [a, b], outro (b, n)
    spans = [("intro", 0, i_intro), ("buildup", i_intro, a), ("climax", a, b + 1), ("outro", b + 1, n)]
    out: list[EnergySection] = []
    for label, i0, i1 in spans:
        if i1 <= i0:
            continue
        out.append(EnergySection(
            label=label,
            start=round(float(bounds[i0]), 3),
            end=round(float(bounds[i1]), 3),
            mean_energy=round(float(np.clip(e[i0:i1].mean(), 0.0, 1.0)), 3),
        ))
    out[0].start = 0.0
    out[-1].end = round(duration, 3)
    return out


def analyze_file(path: str | Path, log: Log = _noop, audio_id: str | None = None, sha256: str | None = None) -> AudioAnalysis:
    import librosa

    p = Path(path).expanduser().resolve()
    log(f"décodage {p.name}")
    y = _decode_mono(p)
    duration = len(y) / SR
    if duration <= 0:
        raise ValueError(f"Audio vide : {p}")

    log("HPSS + enveloppes d'attaque")
    _, y_perc = librosa.effects.hpss(y)
    oenv = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    oenv_perc = librosa.onset.onset_strength(y=y_perc, sr=SR, hop_length=HOP)

    log("suivi de tempo / beats")
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=oenv, sr=SR, hop_length=HOP, trim=False)
    bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    beat_frames = np.asarray(beat_frames, dtype=int)
    if beat_frames.size < 2 and bpm > 0:
        period_frames = 60.0 / bpm * SR / HOP
        beat_frames = np.arange(0, len(oenv), period_frames).astype(int)
    beats = librosa.frames_to_time(beat_frames, sr=SR, hop_length=HOP) if beat_frames.size else np.array([])
    if beats.size >= 4:  # tempo affiné par l'intervalle médian réel (moins quantifié que l'estimation globale)
        bpm = 60.0 / float(np.median(np.diff(beats)))

    # Onsets percussifs (pics d'attaque de la composante percussive)
    onset_frames = librosa.onset.onset_detect(onset_envelope=oenv_perc, sr=SR, hop_length=HOP, backtrack=False)
    onset_frames = np.asarray(onset_frames, dtype=int)
    onsets = librosa.frames_to_time(onset_frames, sr=SR, hop_length=HOP) if onset_frames.size else np.array([])

    # Strong beats : beats dont l'accent percussif est dans le haut de la distribution
    strong_beats = np.array([])
    downbeats = np.array([])
    if beat_frames.size:
        acc = _accent(oenv_perc / (oenv_perc.max() + 1e-9), beat_frames)
        thr = max(0.35, float(np.quantile(acc, 0.6)))
        strong_beats = beats[acc >= thr]
        # Downbeats : phase (mod 4) qui maximise l'accent cumulé
        scores = [acc[ph::4].mean() if acc[ph::4].size else 0.0 for ph in range(4)]
        phase = int(np.argmax(scores))
        downbeats = beats[phase::4]

    # Énergie RMS lissée, normalisée 0-1, échantillonnée à CURVE_HZ
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0].astype(np.float64)
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=SR, hop_length=HOP)
    grid_t = np.arange(0, duration, 1.0 / CURVE_HZ)
    curve = np.interp(grid_t, rms_t, rms)
    win = int(round(max(0.5, min(4.0, duration / 12.0)) * CURVE_HZ))
    curve = _smooth(curve, win)
    curve = curve / (curve.max() + 1e-12)

    bounds, beat_energy = beat_sync_energy(rms, rms_t, beats, duration)
    sections = segment_energy(bounds, beat_energy, duration)

    return AudioAnalysis(
        source=str(p),
        audio_id=audio_id,
        sha256=sha256 or sha256_file(p),
        duration=round(duration, 3),
        sample_rate=SR,
        bpm=round(bpm, 2),
        beats=_round_times(beats),
        downbeats=_round_times(downbeats),
        strong_beats=_round_times(strong_beats),
        onsets=_round_times(onsets),
        sections=sections,
        energy_curve_hz=CURVE_HZ,
        energy_curve=[round(float(v), 4) for v in curve],
    )


def analyze_catalog_audio(
    catalog: MediaCatalog, ws: Workspace, *, audio_id: str | None = None, force: bool = False, log: Log = _noop
) -> AudioAnalysis | None:
    """Analyse la piste musicale du catalogue (la première, ou `audio_id`) → workspace/audio_analysis.json."""
    if not catalog.audio:
        log("aucune piste audio dans input/audio : analyse musicale ignorée")
        return None
    asset = catalog.audio_asset(audio_id=audio_id) if audio_id else catalog.audio[0]
    if asset is None:
        raise ValueError(f"audio_id inconnu : {audio_id}")
    if ws.audio_analysis.exists() and not force:
        try:
            cached = load_model(ws.audio_analysis, AudioAnalysis)
            if cached.sha256 == asset.sha256:
                log(f"analyse audio en cache ({asset.rel_path})")
                return cached
        except Exception:
            pass
    analysis = analyze_file(asset.path, log=log, audio_id=asset.audio_id, sha256=asset.sha256)
    save_model(analysis, ws.audio_analysis)
    return analysis
