from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from ..deps import read_image, require_token
from ..providers import local_cv
from ..schemas import AssetResponse

router = APIRouter()


@router.post("/assets/process", response_model=AssetResponse)
async def process(
    image: UploadFile = File(...),
    category: str = Form(default="neck"),
    mirror_back: bool = Form(default=False),
    _: None = Depends(require_token),
) -> AssetResponse:
    """Design Library upload pipeline.

    Boundary extraction, background removal and shadow derivation, all local.
    `back_base64` is null unless `mirror_back` is set and the category is
    laterally symmetric — a back view cannot honestly be recovered from a
    front photo, so Rails leaves that layer unattached and an admin uploads
    the real one.
    """
    data = await read_image(image, "image")

    result = await run_in_threadpool(
        local_cv.process_asset, data, category, mirror_back=mirror_back
    )

    return AssetResponse(**result)
