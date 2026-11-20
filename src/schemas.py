"""Modèles Pydantic v2 stricts : MediaCatalog, AudioAnalysis, Timeline (+ TimelineDraft côté LLM).

Conventions :
- tous les temps sont en secondes (float) ;
- `extra="forbid"` partout : un champ inconnu dans un JSON est une erreur, pas un silence ;
- les invariants *structurels* (bornes, ordre start<end) sont vérifiés ici, les invariants
  *contextuels* (fichier existe, durée du clip, snap sur les beats) dans `director.validate_timeline`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import FORMAT_SIZES


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------------
# Media catalog
# --------------------------------------------------------------------------------------------

MotionType = Literal[
    "static", "pan", "tilt", "tracking", "handheld", "zoom", "dolly", "drone", "action", "timelapse", "other"
]


class BestSegment(StrictModel):
    start: float = Field(ge=0, description="Début du sous-segment dans le rush (secondes)")
    end: float = Field(gt=0, description="Fin du sous-segment dans le rush (secondes)")
    description: str = Field(description="Ce qui se passe dans ce sous-segment")
    motion_type: MotionType = Field(description="Type de mouvement caméra/sujet dominant")

    @model_validator(mode="after")
    def _check_order(self) -> "BestSegment":
        if self.end <= self.start:
            raise ValueError(f"best_segment end ({self.end}) doit être > start ({self.start})")
        return self


class VisionAnalysis(StrictModel):
    """Sortie structurée attendue du VLM pour un rush (planche contact)."""

    scene_description: str = Field(description="Description concise de la scène (1-3 phrases)")
    subjects: list[str] = Field(description="Sujets principaux visibles (personnes, objets, lieux)")
    tags: list[str] = Field(description="Mots-clés de recherche (lieu, action, style, couleur...)")
    lighting: str = Field(description="Ambiance lumineuse (ex: golden hour, néon nocturne, contre-jour...)")
    energy_score: int = Field(ge=1, le=10, description="Énergie visuelle perçue, 1=calme, 10=explosif")
    best_segments: list[BestSegment] = Field(description="Meilleurs sous-segments exploitables au montage")


class ClipAnalysis(VisionAnalysis):
    model: str
    analyzed_at: str
    contact_sheets: list[str] = Field(default_factory=list)
    frame_times: list[float] = Field(default_factory=list)


class VideoAsset(StrictModel):
    clip_id: str
    path: str = Field(description="Chemin absolu du rush original")
    rel_path: str
    sha256: str
    size_bytes: int
    mtime: float
    duration: float = Field(gt=0)
    fps: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    codec: str
    pix_fmt: str | None = None
    bit_depth: int | None = None
    color_transfer: str | None = None
    has_audio: bool = False
    rotation: int = 0
    proxy_path: str | None = None
    proxy_lut: str | None = None
    analysis: ClipAnalysis | None = None

    @property
    def orientation(self) -> str:
        if self.width == self.height:
            return "square"
        return "landscape" if self.width > self.height else "portrait"


class AudioAsset(StrictModel):
    audio_id: str
    path: str
    rel_path: str
    sha256: str
    size_bytes: int
    mtime: float
    duration: float = Field(gt=0)
    sample_rate: int
    channels: int
    codec: str


class MediaCatalog(StrictModel):
    version: Literal[1] = 1
    input_dir: str
    updated_at: str = Field(default_factory=utc_now)
    lut: str | None = Field(default=None, description="LUT .cube appliquée aux proxies et au conform final")
    videos: list[VideoAsset] = Field(default_factory=list)
    audio: list[AudioAsset] = Field(default_factory=list)
    analysis_cache: dict[str, ClipAnalysis] = Field(
        default_factory=dict, description="Analyses VLM indexées par SHA-256 du fichier"
    )

    def video(self, clip_id: str) -> VideoAsset | None:
        return next((v for v in self.videos if v.clip_id == clip_id), None)

    def video_by_path(self, path: str) -> VideoAsset | None:
        p = str(Path(path).expanduser())
        for v in self.videos:
            if v.path == p or v.rel_path == path or Path(v.path).name == Path(path).name:
                return v
        return None

    def audio_asset(self, audio_id: str | None = None, path: str | None = None) -> AudioAsset | None:
        for a in self.audio:
            if audio_id and a.audio_id == audio_id:
                return a
            if path and (a.path == path or a.rel_path == path or Path(a.path).name == Path(path).name):
                return a
        return None


# --------------------------------------------------------------------------------------------
# Audio analysis
# --------------------------------------------------------------------------------------------

SectionLabel = Literal["intro", "buildup", "climax", "outro"]


class EnergySection(StrictModel):
    label: SectionLabel
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    mean_energy: float = Field(ge=0, le=1)


class AudioAnalysis(StrictModel):
    version: Literal[1] = 1
    source: str
    audio_id: str | None = None
    sha256: str
    duration: float = Field(gt=0)
    sample_rate: int
    bpm: float = Field(ge=0)
    beats: list[float]
    downbeats: list[float]
    strong_beats: list[float]
    onsets: list[float]
    sections: list[EnergySection]
    energy_curve_hz: float = 10.0
    energy_curve: list[float] = Field(description="RMS normalisé 0-1 échantillonné à energy_curve_hz")
    analyzed_at: str = Field(default_factory=utc_now)

    def section_at(self, t: float) -> EnergySection | None:
        return next((s for s in self.sections if s.start <= t < s.end), self.sections[-1] if self.sections else None)


# --------------------------------------------------------------------------------------------
# Timeline (source de vérité consommée par Remotion)
# --------------------------------------------------------------------------------------------

FormatLiteral = Literal["9:16", "16:9"]


class ClipRef(StrictModel):
    clip_id: str
    in_sec: float = Field(ge=0, description="Point d'entrée dans le rush original (s)")
    out_sec: float = Field(gt=0, description="Point de sortie dans le rush original (s)")
    speed: float = Field(default=1.0, gt=0, description="Vitesse de lecture (0.5 = ralenti x2)")
    path: str | None = Field(default=None, description="Chemin absolu de l'original (rempli par le validateur)")

    @model_validator(mode="after")
    def _check_range(self) -> "ClipRef":
        if self.out_sec <= self.in_sec:
            raise ValueError(f"{self.clip_id}: out_sec ({self.out_sec}) doit être > in_sec ({self.in_sec})")
        return self


class _SceneBase(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    section: SectionLabel | None = None
    note: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @model_validator(mode="after")
    def _check_order(self) -> "_SceneBase":
        if self.end <= self.start:
            raise ValueError(f"scène: end ({self.end}) doit être > start ({self.start})")
        return self


class SingleScene(_SceneBase):
    type: Literal["single"] = "single"
    clip: ClipRef
    zoom_punch: bool = False

    def clip_refs(self) -> list[ClipRef]:
        return [self.clip]


class NineGridScene(_SceneBase):
    type: Literal["nine_grid"] = "nine_grid"
    cells: list[ClipRef] = Field(min_length=1, max_length=9)
    reveal: Literal["all", "beats"] = "all"
    reveal_times: list[float] = Field(
        default_factory=list, description="Apparition de chaque case, en secondes relatives au début de la scène"
    )
    border_px: int = Field(default=6, ge=0, le=40)

    def clip_refs(self) -> list[ClipRef]:
        return list(self.cells)


Scene = Annotated[Union[SingleScene, NineGridScene], Field(discriminator="type")]


class _OverlayWindow(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)

    @model_validator(mode="after")
    def _check_order(self) -> "_OverlayWindow":
        if self.end <= self.start:
            raise ValueError(f"overlay: end ({self.end}) doit être > start ({self.start})")
        return self


class CameraHudOverlay(_OverlayWindow):
    type: Literal["camera_hud"] = "camera_hud"
    aperture: str = "F 4.0"
    iso: str = "ISO 800"
    shutter: str = "1/1000"
    white_balance: str = "5600K"
    resolution_label: str = "4K 60P"
    show_rec: bool = True
    show_timecode: bool = True
    timecode_offset_sec: float = Field(default=0.0, ge=0)


class ShutterFlash(StrictModel):
    type: Literal["shutter_flash"] = "shutter_flash"
    at: float = Field(ge=0)
    frames: int = Field(default=3, ge=2, le=4)
    color: Literal["white", "black", "warm"] = Field(default="white", description="warm = light leak orangé")


class KineticTitle(_OverlayWindow):
    type: Literal["kinetic_title"] = "kinetic_title"
    text: str = Field(min_length=1)
    subtitle: str | None = None
    position: Literal["center", "top", "bottom", "lower_third", "split"] = Field(
        default="center", description="split = mots écartés à gauche et à droite du cadre"
    )
    animation: Literal["spring_up", "scale_in", "letter_stagger", "fade"] = "spring_up"
    style: Literal["minimal", "bold", "outline"] = "minimal"
    color: str = "#FFFFFF"


Overlay = Annotated[Union[CameraHudOverlay, ShutterFlash, KineticTitle], Field(discriminator="type")]


class AudioTrackRef(StrictModel):
    audio_id: str | None = None
    path: str | None = None
    audio_start_offset: float = Field(default=0.0, ge=0, description="Position dans la musique au temps 0 (s)")
    volume: float = Field(default=1.0, ge=0, le=2)
    fade_in_sec: float = Field(default=0.0, ge=0)
    fade_out_sec: float = Field(default=0.6, ge=0)


class Timeline(StrictModel):
    version: Literal[1] = 1
    title: str = "Untitled edit"
    format: FormatLiteral = "9:16"
    fps: Literal[30, 60] = 30
    width: int = 1080
    height: int = 1920
    duration_sec: float = Field(gt=0)
    audio: AudioTrackRef | None = None
    scenes: list[Scene] = Field(min_length=1)
    overlays: list[Overlay] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    generator: str | None = None

    @model_validator(mode="after")
    def _apply_format(self) -> "Timeline":
        self.width, self.height = FORMAT_SIZES[self.format]
        return self


# --------------------------------------------------------------------------------------------
# TimelineDraft : projection plate de Timeline pour les Structured Outputs du LLM
# (tous les champs requis, aucune union → schéma simple et robuste)
# --------------------------------------------------------------------------------------------


class DraftClip(StrictModel):
    clip_id: str = Field(description="clip_id exact d'un rush du catalogue")
    in_sec: float = Field(description="Point d'entrée dans le rush (s), idéalement dans un best_segment")
    speed: float = Field(description="Vitesse de lecture : 1.0 normal, 0.5 ralenti, 2.0 accéléré")


class DraftScene(StrictModel):
    type: Literal["single", "nine_grid"] = Field(description="single = plein écran, nine_grid = mosaïque 3x3")
    start: float = Field(description="Début de la scène sur la timeline (s) — doit tomber sur un beat")
    end: float = Field(description="Fin de la scène sur la timeline (s) — doit tomber sur un beat")
    clips: list[DraftClip] = Field(description="1 clip pour single, 9 clips (ou moins, complétés) pour nine_grid")
    zoom_punch: bool = Field(description="Micro punch-in à l'entrée du plan (single uniquement)")
    grid_reveal: Literal["all", "beats"] = Field(description="nine_grid : tout d'un coup ou case par case sur les beats")
    section: SectionLabel = Field(description="Section musicale couverte par la scène")
    intent: str = Field(description="Pourquoi ce plan ici (court)")


class DraftHud(StrictModel):
    start: float
    end: float
    aperture: str = Field(description="ex: F 2.8")
    iso: str = Field(description="ex: ISO 800")
    shutter: str = Field(description="ex: 1/1000")


class DraftFlash(StrictModel):
    at: float = Field(description="Instant du flash sur la timeline (s), sur un temps fort")
    color: Literal["white", "black", "warm"] = Field(description="warm = light leak orangé (transition chaude)")
    frames: int = Field(description="Durée en frames, entre 2 et 4")


class DraftTitle(StrictModel):
    start: float
    end: float
    text: str = Field(description="Titre très court (<= 5 mots)")
    subtitle: str = Field(description="Sous-titre optionnel, chaîne vide si aucun")
    position: Literal["center", "top", "bottom", "lower_third", "split"] = Field(
        description="split = 2 mots écartés gauche/droite, apparition mot par mot"
    )
    animation: Literal["spring_up", "scale_in", "letter_stagger", "fade"]
    style: Literal["minimal", "bold", "outline"]


class TimelineDraft(StrictModel):
    title: str
    duration_sec: float = Field(description="Durée totale du montage (s)")
    audio_start_offset: float = Field(description="Position dans la musique au temps 0 du montage (s), sur un beat")
    scenes: list[DraftScene]
    camera_hud: list[DraftHud]
    shutter_flashes: list[DraftFlash]
    titles: list[DraftTitle]
    editing_notes: str = Field(description="Résumé de la logique de montage")


# --------------------------------------------------------------------------------------------
# StyleProfile : style appris d'un montage d'exemple (learn-style)
# --------------------------------------------------------------------------------------------


class StylePacing(StrictModel):
    content_duration: float = Field(description="Durée utile analysée (écran de fin exclu), s")
    shot_count: int
    shot_durations: list[float]
    median_shot_sec: float
    mean_shot_sec: float
    hook_shot_sec: float = Field(description="Durée du premier plan (accroche)")
    cuts_per_10s: float
    bpm: float | None = None
    median_shot_beats: float | None = Field(default=None, description="Durée médiane d'un plan, en beats")
    cut_offset_ms: float | None = Field(default=None, description="Décalage médian coupe − beat le plus proche")
    cuts_on_beat_ratio: float | None = Field(default=None, description="Part des coupes à ≤ 80 ms d'un beat")
    cuts_on_half_beat_ratio: float | None = Field(default=None, description="Part à ≤ 80 ms d'un beat ou demi-beat")


class StyleLook(StrictModel):
    mean_luma: float = Field(description="Luminance moyenne 0-255")
    contrast: float = Field(description="Écart-type de luminance 0-255")
    saturation: float = Field(description="Saturation moyenne 0-1")
    warmth: float = Field(description="(R - B) moyen / 255 : > 0 chaud, < 0 froid")
    flashes_per_10s: float = Field(default=0.0, description="Fréquence des flashs / light leaks")
    flash_times: list[float] = Field(default_factory=list, description="Pics de luminance (par exemple uniquement)")


class StyleReading(StrictModel):
    """Lecture qualitative d'un style (Claude Vision, rédigée à la main, ou déduite des mesures)."""

    summary: str = ""
    structure: str = ""
    title_style: str = ""
    transitions: str = ""
    color_grade: str = ""
    editing_rules: list[str] = Field(default_factory=list, description="Consignes concrètes données au directeur")
    source: Literal["claude", "manual", "metrics"] = "metrics"


class StyleExample(StrictModel):
    """Mesures d'UN montage d'exemple (la vidéo elle-même n'est jamais versionnée)."""

    source: str = Field(description="Description ou nom du montage d'exemple")
    sha256: str
    added_at: str = Field(default_factory=utc_now)
    format_hint: FormatLiteral | None = None
    cut_times: list[float] = Field(default_factory=list)
    pacing: StylePacing
    look: StyleLook
    reading: StyleReading = Field(default_factory=StyleReading)


class StyleProfile(StrictModel):
    """Un style = l'agrégat de un ou plusieurs montages d'exemple « dans le même délire »."""

    version: Literal[2] = 2
    name: str
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    format_hint: FormatLiteral | None = None
    examples: list[StyleExample] = Field(min_length=1)
    pacing: StylePacing = Field(description="Rythme agrégé sur tous les exemples")
    look: StyleLook = Field(description="Look agrégé (moyenne pondérée par la durée)")
    reading: StyleReading = Field(description="Lecture de synthèse du style, donnée au directeur")


# --------------------------------------------------------------------------------------------
# Rapport de validation
# --------------------------------------------------------------------------------------------


class ValidationFix(StrictModel):
    kind: str
    message: str
    target: str | None = None
    before: float | str | None = None
    after: float | str | None = None


class ValidationReport(StrictModel):
    ok: bool = True
    fixes: list[ValidationFix] = Field(default_factory=list)
    snapped_cuts: int = 0
    unsnapped_cuts: int = 0
    max_snap_delta_ms: float = 0.0
    scene_count: int = 0
    duration_sec: float = 0.0
    created_at: str = Field(default_factory=utc_now)

    def add(self, kind: str, message: str, target: str | None = None, before=None, after=None) -> None:
        self.fixes.append(ValidationFix(kind=kind, message=message, target=target, before=before, after=after))


# --------------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------------

M = TypeVar("M", bound=BaseModel)


def load_model(path: str | Path, model: type[M]) -> M:
    return model.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_model(obj: BaseModel, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(obj.model_dump_json(indent=2, exclude_none=False), encoding="utf-8")
    tmp.replace(p)
    return p


def save_json(data: object, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return p
