"""Configuration centrale : chemins du projet, variables d'environnement, constantes de montage."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RENDERER_DIR = PROJECT_ROOT / "renderer"

load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_MODEL = "claude-opus-5-5"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac"}

FORMAT_SIZES: dict[str, tuple[int, int]] = {"9:16": (1080, 1920), "16:9": (1920, 1080)}

# Validateur
SNAP_TOLERANCE_SEC = 0.120
MIN_SCENE_SEC = 0.20
MIN_OVERLAY_SEC = 0.30
MIN_SPEED = 0.25
MAX_SPEED = 4.0

# Proxies
PROXY_SHORT_EDGE = 720
PROXY_FPS = 30


def vision_model() -> str:
    return os.environ.get("VISION_MODEL") or DEFAULT_MODEL


def director_model() -> str:
    return os.environ.get("DIRECTOR_MODEL") or DEFAULT_MODEL


def has_api_credentials() -> bool:
    """True si une clé (ou un jeton) Anthropic est disponible dans l'environnement."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


@dataclass(frozen=True)
class Workspace:
    """Arborescence de travail (tout ce qui est dérivé des rushs)."""

    root: Path

    @classmethod
    def at(cls, root: str | Path) -> "Workspace":
        ws = cls(Path(root).expanduser().resolve())
        ws.ensure()
        return ws

    def ensure(self) -> None:
        for d in (self.root, self.proxies, self.contact_sheets, self.conform):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def catalog(self) -> Path:
        return self.root / "media_catalog.json"

    @property
    def audio_analysis(self) -> Path:
        return self.root / "audio_analysis.json"

    @property
    def timeline(self) -> Path:
        return self.root / "timeline.json"

    @property
    def timeline_draft(self) -> Path:
        return self.root / "timeline.draft.json"

    @property
    def validation_report(self) -> Path:
        return self.root / "validation_report.json"

    @property
    def proxies(self) -> Path:
        return self.root / "proxies"

    @property
    def contact_sheets(self) -> Path:
        return self.root / "contact_sheets"

    @property
    def conform(self) -> Path:
        return self.root / "conform"

    @property
    def render_public(self) -> Path:
        return self.root / "render_public"

    @property
    def render_props(self) -> Path:
        return self.root / "render_props.json"
