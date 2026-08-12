from typing import Any

from fastapi import APIRouter, Body, Depends

from ..deps import require_token
from ..providers import sketch_provider
from ..schemas import SketchResponse

router = APIRouter()


@router.post("/sketch", response_model=SketchResponse)
async def sketch(
    design: dict[str, Any] = Body(...),
    _: None = Depends(require_token),
) -> SketchResponse:
    """Generate a design sketch from the component tree.

    Rails posts the whole design — neck front/back, sleeve, silhouette,
    fabric, colour zones, embroidery and lace placement, plus the resolved
    layer stack — rather than a prose prompt. Wording is assembled in
    providers/base.py so it can be tuned without a backend deploy.
    """
    provider = sketch_provider()
    result = await provider.sketch(design)

    return SketchResponse(
        image_base64=result.get("image_base64"),
        image_url=result.get("image_url"),
        content_type=result.get("content_type", "image/png"),
        provider=provider.name,
        model=result.get("model"),
    )
