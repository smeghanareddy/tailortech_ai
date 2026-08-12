from typing import Any

from fastapi import APIRouter, Body, Depends
from fastapi.concurrency import run_in_threadpool

from ..deps import require_token
from ..errors import InvalidInput
from ..imaging import encode_png
from ..providers import local_compose
from ..schemas import ComposeResponse

router = APIRouter()


@router.post("/compose", response_model=ComposeResponse)
async def compose(
    payload: dict[str, Any] = Body(...),
    _: None = Depends(require_token),
) -> ComposeResponse:
    """Flatten a layer manifest into a single preview image.

    This is the spec's "Layer Merge → Final Preview" step, done locally. It
    needs no API key and no model, and unlike a generated sketch the output is
    guaranteed to match the customer's actual selections.

    Body: { layers: [...], width?, height?, background? } — `layers` is exactly
    what Rails' LayerComposer returns.
    """
    layers = payload.get("layers")
    if not isinstance(layers, list) or not layers:
        raise InvalidInput("`layers` must be a non-empty array")

    image, report = await run_in_threadpool(
        local_compose.compose,
        layers,
        payload.get("width"),
        payload.get("height"),
        payload.get("background"),
    )

    if not report["layers_painted"]:
        # Nothing had usable art — say so rather than returning a blank PNG
        # that would look like a successful render.
        raise InvalidInput(
            "None of the layers had a usable image",
            detail="upload assets for these styles in the Design Library first",
        )

    return ComposeResponse(
        image_base64=encode_png(image),
        width=image.width,
        height=image.height,
        provider=local_compose.name,
        **report,
    )
