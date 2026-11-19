#!/usr/bin/env python3
"""CLI unifiée du pipeline de montage IA : analyze → direct → preview / render (ou run pour tout enchaîner)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.config import Workspace, director_model, has_api_credentials, vision_model

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="🎬 Montage vidéo automatique piloté par IA (rushs + beat sync + timeline LLM + Remotion).",
)
console = Console()
STATE: dict[str, Workspace] = {}


def ws() -> Workspace:
    return STATE["ws"]


def log(msg: str) -> None:
    console.print(f"  [dim]›[/dim] {msg}")


def fail(msg: str, code: int = 1) -> None:
    console.print(Panel(msg, title="Erreur", border_style="red"))
    raise typer.Exit(code)


@app.callback()
def _main(
    workspace: Path = typer.Option(Path("workspace"), "--workspace", "-w", help="Dossier de travail (catalogue, proxies, timeline…)"),
) -> None:
    STATE["ws"] = Workspace.at(workspace)


# ------------------------------------------------------------------------------------- analyze

def _analyze(input_dir: Path, lut: Optional[Path], skip_vision: bool, force_vision: bool, keyframes: str, workers: int) -> None:
    from src.audio_analyzer import analyze_catalog_audio
    from src.ingest import ingest
    from src.llm import LLMError
    from src.video_indexer import index_catalog

    console.rule("[bold cyan]1/3 Ingestion des rushs")
    catalog = ingest(input_dir, ws(), lut=lut, log=log)
    if not catalog.videos:
        fail(f"Aucune vidéo (.mp4/.mov/.mkv) trouvée dans {input_dir / 'videos'}")
    table = Table(show_header=True, header_style="bold")
    for col in ("clip_id", "fichier", "durée", "résolution", "fps", "codec"):
        table.add_column(col)
    for v in catalog.videos:
        table.add_row(v.clip_id, v.rel_path, f"{v.duration:.2f}s", f"{v.width}x{v.height}", f"{v.fps:g}", f"{v.codec} {v.bit_depth or ''}bit")
    console.print(table)

    console.rule("[bold cyan]2/3 Analyse musicale")
    audio = analyze_catalog_audio(catalog, ws(), log=log)
    if audio:
        labels = " → ".join(f"{s.label} [{s.start:.1f}-{s.end:.1f}s]" for s in audio.sections)
        console.print(f"  BPM [bold]{audio.bpm:.1f}[/bold] · {len(audio.beats)} beats · {len(audio.strong_beats)} strong · "
                      f"{len(audio.onsets)} onsets\n  {labels}")

    console.rule("[bold cyan]3/3 Indexation visuelle (VLM)")
    if skip_vision:
        log("indexation visuelle ignorée (--skip-vision)")
    elif not has_api_credentials():
        log("[yellow]pas de ANTHROPIC_API_KEY : indexation visuelle ignorée (le directeur utilisera les métadonnées)[/yellow]")
    else:
        try:
            index_catalog(catalog, ws(), force=force_vision, mode=keyframes, max_workers=workers, log=log)
        except LLMError as e:
            fail(str(e))
    console.print(f"[green]✓[/green] catalogue → {ws().catalog}")


@app.command()
def analyze(
    input: Path = typer.Option(Path("input"), "--input", "-i", help="Dossier contenant videos/ et audio/"),
    lut: Optional[Path] = typer.Option(None, "--lut", help="LUT .cube (ex: D-Log M → Rec.709) appliquée aux proxies et au rendu"),
    skip_vision: bool = typer.Option(False, "--skip-vision", help="Ne pas appeler le VLM"),
    force_vision: bool = typer.Option(False, "--force-vision", help="Ré-analyser même les rushs en cache"),
    keyframes: str = typer.Option("interval", "--keyframes", help="interval (1-1.5s) ou scene (PySceneDetect)"),
    workers: int = typer.Option(4, "--workers", help="Appels VLM en parallèle"),
) -> None:
    """Ingestion (ffprobe + proxies) + analyse audio (librosa) + indexation VLM des rushs."""
    if keyframes not in ("interval", "scene"):
        fail("--keyframes doit valoir 'interval' ou 'scene'")
    try:
        _analyze(input, lut, skip_vision, force_vision, keyframes, workers)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        fail(str(e))


# ------------------------------------------------------------------------------------- direct

def _load_inputs():
    from src.schemas import AudioAnalysis, MediaCatalog, load_model

    if not ws().catalog.exists():
        fail("Catalogue absent : lancez d'abord `python main.py analyze --input ./input`.")
    catalog = load_model(ws().catalog, MediaCatalog)
    audio = load_model(ws().audio_analysis, AudioAnalysis) if ws().audio_analysis.exists() else None
    return catalog, audio


def _print_report(timeline, report) -> None:
    table = Table(title=f"Timeline « {timeline.title} » — {timeline.format} {timeline.width}x{timeline.height} @ {timeline.fps}fps",
                  header_style="bold")
    for col in ("#", "type", "début", "fin", "clips", "section"):
        table.add_column(col)
    for i, s in enumerate(timeline.scenes):
        clips = s.clip.clip_id if s.type == "single" else f"9 cases ({s.reveal})"
        table.add_row(str(i), s.type + (" ⚡" if getattr(s, "zoom_punch", False) else ""), f"{s.start:.3f}", f"{s.end:.3f}", clips, s.section or "")
    console.print(table)
    if timeline.overlays:
        console.print("  overlays : " + ", ".join(
            f"{o.type}@{o.at:.2f}" if o.type == "shutter_flash" else f"{o.type}[{o.start:.2f}-{o.end:.2f}]" for o in timeline.overlays))
    console.print(f"  validateur : {report.snapped_cuts} cut(s) snappé(s) (max {report.max_snap_delta_ms:.0f} ms), "
                  f"{report.unsnapped_cuts} non snappé(s), {len(report.fixes)} correction(s) → {ws().validation_report}")


def _direct(brief: Path, fmt: str, fps: int, duration: Optional[float], offline: bool) -> None:
    from src.director import TimelineValidationError, direct
    from src.llm import LLMError

    if fmt not in ("9:16", "16:9"):
        fail("--format doit valoir 9:16 ou 16:9")
    if fps not in (30, 60):
        fail("--fps doit valoir 30 ou 60")
    if not brief.exists():
        fail(f"Brief introuvable : {brief} (voir brief.example.txt)")
    if not offline and not has_api_credentials():
        fail("ANTHROPIC_API_KEY absente : renseignez .env ou utilisez --offline pour un brouillon heuristique.")
    catalog, audio = _load_inputs()
    console.rule("[bold cyan]Directeur (LLM) + validateur")
    try:
        timeline, report = direct(brief.read_text(encoding="utf-8"), catalog, audio, ws(), fmt=fmt, fps=fps,
                                  target_duration=duration, offline=offline, log=log)
    except (LLMError, TimelineValidationError) as e:
        fail(str(e))
    _print_report(timeline, report)
    console.print(f"[green]✓[/green] timeline → {ws().timeline}")


@app.command()
def direct(
    brief: Path = typer.Option(Path("brief.txt"), "--brief", "-b", help="Brief créatif en texte libre"),
    format: str = typer.Option("9:16", "--format", "-f", help="9:16 (1080x1920) ou 16:9 (1920x1080)"),
    fps: int = typer.Option(30, "--fps", help="30 ou 60"),
    duration: Optional[float] = typer.Option(None, "--duration", "-d", help="Durée cible en secondes (défaut : 30s ou longueur de la musique)"),
    offline: bool = typer.Option(False, "--offline", help="Brouillon heuristique sans appel API"),
) -> None:
    """Génère et valide workspace/timeline.json à partir du brief."""
    _direct(brief, format, fps, duration, offline)


@app.command()
def validate(
    timeline: Optional[Path] = typer.Option(None, "--timeline", "-t", help="Timeline à valider (défaut : workspace/timeline.json)"),
) -> None:
    """Re-passe le validateur déterministe sur une timeline (ex : éditée à la main)."""
    from src.director import TimelineValidationError, validate_timeline
    from src.schemas import Timeline, load_model, save_model

    path = timeline or ws().timeline
    if not path.exists():
        fail(f"Timeline introuvable : {path}")
    catalog, audio = _load_inputs()
    try:
        tl, report = validate_timeline(load_model(path, Timeline), catalog, audio, log=log)
    except (TimelineValidationError, ValueError) as e:
        fail(str(e))
    save_model(tl, ws().timeline)
    save_model(report, ws().validation_report)
    _print_report(tl, report)
    console.print(f"[green]✓[/green] timeline validée → {ws().timeline}")


# ------------------------------------------------------------------------------------- preview / render

def _load_timeline():
    from src.director import TimelineValidationError, validate_timeline
    from src.schemas import Timeline, load_model

    if not ws().timeline.exists():
        fail("timeline.json absente : lancez d'abord `python main.py direct --brief ./brief.txt`.")
    catalog, audio = _load_inputs()
    try:  # re-validation légère : les fichiers ont pu bouger depuis `direct`
        tl, _ = validate_timeline(load_model(ws().timeline, Timeline), catalog, audio)
    except TimelineValidationError as e:
        fail(f"Timeline invalide : {e}")
    return tl, catalog


@app.command()
def preview(port: Optional[int] = typer.Option(None, "--port", help="Port de Remotion Studio")) -> None:
    """Lance Remotion Studio sur la timeline courante (lecture fluide via les proxies 720p)."""
    from src.render import RenderError, launch_studio, prepare_render

    tl, catalog = _load_timeline()
    try:
        prepare_render(tl, catalog, ws(), mode="preview", log=log)
        console.print("[cyan]Remotion Studio[/cyan] — Ctrl+C pour quitter")
        launch_studio(ws(), port)
    except RenderError as e:
        fail(str(e))


def _render(output: Path, quality: str, lut: Optional[Path], conform: bool, concurrency: Optional[int]) -> dict:
    from src.ffmpeg_utils import FFmpegError
    from src.render import RenderError, prepare_render, run_remotion_render

    tl, catalog = _load_timeline()
    console.rule("[bold cyan]Rendu Remotion")
    try:
        prepare_render(tl, catalog, ws(), mode="final", conform=conform, lut=str(lut.resolve()) if lut else None, log=log)
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(), TimeElapsedColumn(),
                      console=console) as progress:
            task = progress.add_task("bundling", total=1.0)

            def on_progress(stage: str, value: float) -> None:
                progress.update(task, description=stage, completed=value)

            info = run_remotion_render(ws(), output, quality=quality, concurrency=concurrency, on_progress=on_progress, log=log)
    except (RenderError, FFmpegError) as e:
        fail(str(e))
    console.print(Panel(
        f"{info['path']}\n{info['width']}x{info['height']} · {info['duration']:.2f}s · {info['video_codec']}/{info['audio_codec']} · "
        f"{info['size_bytes'] / 1e6:.1f} Mo", title="✓ Rendu terminé", border_style="green"))
    return info


@app.command()
def render(
    quality: str = typer.Option("high", "--quality", "-q", help="draft | medium | high"),
    output: Path = typer.Option(Path("output/final.mp4"), "--output", "-o", help="Fichier MP4 de sortie"),
    lut: Optional[Path] = typer.Option(None, "--lut", help="LUT .cube (défaut : celle de l'ingest)"),
    conform: bool = typer.Option(True, "--conform/--no-conform", help="Extraire les segments utilisés des originaux (recommandé)"),
    concurrency: Optional[int] = typer.Option(None, "--concurrency", help="Onglets de rendu en parallèle"),
) -> None:
    """Rendu MP4 final sur les rushs originaux."""
    _render(output, quality, lut, conform, concurrency)


@app.command()
def run(
    input: Path = typer.Option(Path("input"), "--input", "-i"),
    brief: Path = typer.Option(Path("brief.txt"), "--brief", "-b"),
    output: Path = typer.Option(Path("output/final.mp4"), "--output", "-o"),
    format: str = typer.Option("9:16", "--format", "-f"),
    fps: int = typer.Option(30, "--fps"),
    duration: Optional[float] = typer.Option(None, "--duration", "-d"),
    quality: str = typer.Option("high", "--quality", "-q"),
    lut: Optional[Path] = typer.Option(None, "--lut"),
    skip_vision: bool = typer.Option(False, "--skip-vision"),
    offline: bool = typer.Option(False, "--offline", help="Sans API : pas de VLM, brouillon heuristique"),
) -> None:
    """Pipeline complet : analyze → direct → render."""
    console.print(Panel(f"vision : {vision_model()} · directeur : {director_model()} · workspace : {ws().root}", title="🎬 run"))
    try:
        _analyze(input, lut, skip_vision or offline, False, "interval", 4)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        fail(str(e))
    _direct(brief, format, fps, duration, offline)
    _render(output, quality, lut, True, None)


if __name__ == "__main__":
    app()
