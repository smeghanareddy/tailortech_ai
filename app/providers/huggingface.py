"""Hugging Face Inference API — the fallback sketch provider.

Used when Replicate is not configured. The Inference API returns raw image
bytes rather than a URL, so results come back as base64.

Note there is no try-on model here: the hosted Inference API does not expose a
virtual try-on endpoint, so try-on requires Replicate.
"""

from __future__ import annotations

import logging

import httpx

from ..config import settings
from ..errors import ProviderFailed, ProviderUnavailable
from ..imaging import encode_bytes
from .base import NEGATIVE_PROMPT, build_prompt

logger = logging.getLogger(__name__)

name = "huggingface"

API_ROOT = "https://api-inference.huggingface.co/models"


def available() -> bool:
    return bool(settings.huggingface_api_token)


async def sketch(design: dict) -> dict:
    if not available():
        raise ProviderUnavailable("HUGGINGFACE_API_TOKEN is not set")

    model = settings.huggingface_sketch_model
    payload = {
        "inputs": build_prompt(design),
        "parameters": {
            "negative_prompt": NEGATIVE_PROMPT,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
        },
        "options": {"wait_for_model": True},
    }

    async with httpx.AsyncClient(timeout=settings.provider_timeout) as client:
        try:
            response = await client.post(
                f"{API_ROOT}/{model}",
                json=payload,
                headers={"Authorization": f"Bearer {settings.huggingface_api_token}"},
            )
        except httpx.TimeoutException as exc:
            raise ProviderFailed(
                f"{model} did not respond within {settings.provider_timeout}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderFailed(f"Could not reach {model}", detail=str(exc)) from exc

    if response.status_code == 503:
        raise ProviderFailed(
            f"{model} is still loading on Hugging Face — retry shortly",
            detail=response.text[:300],
        )
    if response.status_code >= 400:
        raise ProviderFailed(
            f"{model} returned {response.status_code}", detail=response.text[:300]
        )

    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise ProviderFailed(
            f"{model} returned {content_type or 'an unknown type'} instead of an image",
            detail=response.text[:300],
        )

    return {
        "image_base64": encode_bytes(response.content),
        "content_type": content_type,
        "model": model,
    }
