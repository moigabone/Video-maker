"""Appel Claude avec Structured Outputs (JSON schema) — partagé par l'indexeur vision et le directeur."""

from __future__ import annotations

from typing import Any, TypeVar

import anthropic
from anthropic import transform_schema
from pydantic import BaseModel, ValidationError

from .config import has_api_credentials

T = TypeVar("T", bound=BaseModel)

# Fallback serveur : si le modèle décline pour raison de politique, l'API rejoue la requête sur le
# modèle de repli par défaut (sans effet sur un modèle qui ne supporte pas les fallbacks).
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    pass


def get_client() -> anthropic.Anthropic:
    if not has_api_credentials():
        raise LLMError(
            "Aucune clé API Anthropic : copiez .env.example vers .env et renseignez ANTHROPIC_API_KEY "
            "(ou utilisez --offline / --skip-vision)."
        )
    return anthropic.Anthropic(max_retries=3)


def output_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema compatible Structured Outputs (contraintes numériques déplacées en description)."""
    return transform_schema(model)


def structured_call(
    *,
    model: str,
    system: str,
    content: list[dict[str, Any]],
    output_model: type[T],
    max_tokens: int = 32000,
    effort: str = "high",
    client: anthropic.Anthropic | None = None,
) -> tuple[T, Any]:
    """Envoie un message utilisateur et renvoie (objet validé, message brut).

    Streaming (évite les timeouts HTTP sur les longues réponses) + `output_config.format` pour
    garantir un JSON conforme, puis validation Pydantic stricte côté client.
    """
    client = client or get_client()
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": content}],
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": output_schema(output_model)},
        },
        "betas": [FALLBACK_BETA],
        "fallbacks": "default",
    }
    try:
        with client.beta.messages.stream(**request) as stream:
            message = stream.get_final_message()
    except anthropic.AuthenticationError as e:
        raise LLMError("Clé API Anthropic invalide (401). Vérifiez ANTHROPIC_API_KEY dans .env.") from e
    except anthropic.PermissionDeniedError as e:
        raise LLMError(f"Accès refusé au modèle {model} (403) : {e.message}") from e
    except anthropic.NotFoundError as e:
        raise LLMError(f"Modèle introuvable : {model} (404). Vérifiez VISION_MODEL / DIRECTOR_MODEL.") from e
    except anthropic.BadRequestError as e:
        raise LLMError(f"Requête refusée par l'API (400) : {e.message}") from e
    except anthropic.RateLimitError as e:
        raise LLMError("Limite de débit atteinte (429) après plusieurs tentatives ; réessayez plus tard.") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"Erreur API {e.status_code} : {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError(f"Impossible de joindre l'API Anthropic : {e}") from e

    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        raise LLMError(f"Le modèle a décliné la requête (refusal) : {getattr(details, 'explanation', '') or ''}")
    if message.stop_reason == "max_tokens":
        raise LLMError(f"Réponse tronquée (max_tokens={max_tokens}) ; augmentez max_tokens.")

    text = "".join(block.text for block in message.content if block.type == "text").strip()
    if not text:
        raise LLMError(f"Réponse vide du modèle (stop_reason={message.stop_reason}).")
    try:
        parsed = output_model.model_validate_json(text)
    except ValidationError as e:
        raise LLMError(f"JSON du modèle non conforme au schéma {output_model.__name__} :\n{e}") from e
    return parsed, message
