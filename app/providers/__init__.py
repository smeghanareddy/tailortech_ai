"""Provider registry.

`SKETCH_PROVIDER` / `TRYON_PROVIDER` pick a provider explicitly; `auto` (the
default) takes the first configured one in preference order.
"""

from __future__ import annotations

from ..config import settings
from ..errors import ProviderUnavailable
from . import huggingface, local_compose, local_cv, pose, replicate_api

SKETCH_PROVIDERS = {
    "replicate": replicate_api,
    "huggingface": huggingface,
}

TRYON_PROVIDERS = {
    "replicate": replicate_api,
}

# Order tried when the setting is "auto".
SKETCH_ORDER = ["replicate", "huggingface"]
TRYON_ORDER = ["replicate"]


def _resolve(configured: str, registry: dict, order: list[str], endpoint: str):
    if configured == "none":
        raise ProviderUnavailable(f"{endpoint} is disabled by configuration")

    if configured != "auto":
        provider = registry.get(configured)
        if provider is None:
            raise ProviderUnavailable(
                f"Unknown provider {configured!r} for {endpoint}",
                detail=f"valid options: {', '.join(registry)}, auto, none",
            )
        if not provider.available():
            raise ProviderUnavailable(
                f"Provider {configured!r} is not configured",
                detail=f"set its API token to enable {endpoint}",
            )
        return provider

    for key in order:
        provider = registry[key]
        if provider.available():
            return provider

    raise ProviderUnavailable(
        f"No provider is configured for {endpoint}",
        detail=(
            "set REPLICATE_API_TOKEN or HUGGINGFACE_API_TOKEN in the AI "
            "service .env, then restart it"
        ),
    )


def sketch_provider():
    return _resolve(settings.sketch_provider, SKETCH_PROVIDERS, SKETCH_ORDER, "/sketch")


def tryon_provider():
    return _resolve(settings.tryon_provider, TRYON_PROVIDERS, TRYON_ORDER, "/tryon")


def status() -> dict:
    """Provider availability, surfaced by /health."""
    return {
        "local_cv": {"available": local_cv.available(), "model": settings.rembg_model},
        "local_compose": {"available": local_compose.available(), "canvas": list(local_compose.CANVAS)},
        "pose": {
            "available": pose.available(),
            "mediapipe": pose.mediapipe_available(),
        },
        "replicate": {
            "available": replicate_api.available(),
            "sketch_model": settings.replicate_sketch_model,
            "tryon_model": settings.replicate_tryon_model,
        },
        "huggingface": {
            "available": huggingface.available(),
            "sketch_model": settings.huggingface_sketch_model,
        },
    }


def capabilities() -> dict[str, bool]:
    """Which endpoints can actually do their job right now."""

    def can(resolver) -> bool:
        try:
            resolver()
        except ProviderUnavailable:
            return False
        return True

    return {
        "sketch": can(sketch_provider),
        "measurements": pose.available(),
        "assets": local_cv.available(),
        # Local compositing needs nothing configured, and try-on now degrades
        # to a geometric overlay rather than failing outright.
        "compose": local_compose.available(),
        "tryon": can(tryon_provider) or pose.available(),
        "tryon_photorealistic": can(tryon_provider),
    }
