"""Replicate-hosted diffusion models.

Sketching and virtual try-on are diffusion-scale jobs, so they run on hosted
GPUs rather than this process. Two models:

  stability-ai/sdxl  → design sketch from the component tree
  cuuupid/idm-vton   → garment warped onto the user's photo
"""

from __future__ import annotations

import asyncio
import logging

from ..config import settings
from ..errors import ProviderFailed, ProviderUnavailable
from ..imaging import encode_bytes
from .base import NEGATIVE_PROMPT, build_prompt

logger = logging.getLogger(__name__)

name = "replicate"


def available() -> bool:
    if not settings.replicate_api_token:
        return False
    try:
        import replicate  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def _client():
    try:
        import replicate  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ProviderUnavailable(
            "The replicate package is not installed", detail=str(exc)
        ) from exc

    if not settings.replicate_api_token:
        raise ProviderUnavailable("REPLICATE_API_TOKEN is not set")

    return replicate.Client(api_token=settings.replicate_api_token)


async def _run(model: str, payload: dict) -> list:
    """Run a model off the event loop; the SDK is synchronous and blocking."""
    client = _client()

    def call():
        return client.run(model, input=payload)

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(call), timeout=settings.provider_timeout
        )
    except asyncio.TimeoutError as exc:
        raise ProviderFailed(
            f"{model} did not finish within {settings.provider_timeout}s"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - SDK raises provider-specific types
        raise ProviderFailed(f"{model} failed", detail=str(exc)) from exc

    return result if isinstance(result, list) else [result]


def _first_output(outputs: list) -> dict:
    """Normalise Replicate's output into what Rails' ImageIngestor accepts.

    Depending on the model this is a URL string, a FileOutput object, or raw
    bytes — all three end up as either image_url or image_base64.
    """
    if not outputs:
        raise ProviderFailed("The model returned no output")

    first = outputs[0]

    if isinstance(first, str):
        return {"image_url": first}
    if isinstance(first, (bytes, bytearray)):
        return {"image_base64": encode_bytes(bytes(first))}
    if hasattr(first, "url"):
        return {"image_url": first.url}
    if hasattr(first, "read"):
        return {"image_base64": encode_bytes(first.read())}

    raise ProviderFailed(f"Unrecognised output type: {type(first).__name__}")


async def sketch(design: dict) -> dict:
    outputs = await _run(
        settings.replicate_sketch_model,
        {
            "prompt": build_prompt(design),
            "negative_prompt": NEGATIVE_PROMPT,
            "width": 768,
            "height": 1024,
            "num_outputs": 1,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "apply_watermark": False,
        },
    )
    return {**_first_output(outputs), "model": settings.replicate_sketch_model}


async def tryon(photo: bytes, design: dict) -> dict:
    garment_url = design.get("sketch_url")
    if not garment_url:
        raise ProviderFailed(
            "Try-on needs a garment image",
            detail="generate the design sketch first, then retry the try-on",
        )

    import io  # noqa: PLC0415

    outputs = await _run(
        settings.replicate_tryon_model,
        {
            "human_img": io.BytesIO(photo),
            "garm_img": garment_url,
            "garment_des": build_prompt(design),
            "is_checked": True,
            "is_checked_crop": True,
            "steps": 30,
        },
    )
    return {**_first_output(outputs), "model": settings.replicate_tryon_model}
