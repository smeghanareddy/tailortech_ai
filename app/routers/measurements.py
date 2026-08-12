from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from ..deps import read_image, require_token
from ..providers import pose
from ..schemas import MeasurementsResponse

router = APIRouter()


@router.post("/measurements/estimate", response_model=MeasurementsResponse)
async def estimate(
    photo: UploadFile = File(...),
    height_hint: str | None = Form(default=None),
    _: None = Depends(require_token),
) -> MeasurementsResponse:
    """Estimate body measurements from one front-view photo.

    Pass `height_hint` (real height in cm) for absolute sizing; without it the
    result is proportional only and comes back with low confidence. Rails does
    not persist any of this — the user reviews and adjusts before saving.
    """
    data = await read_image(photo, "photo")

    hint: float | None = None
    if height_hint:
        try:
            hint = float(height_hint)
        except ValueError:
            hint = None

    # Segmentation and pose are CPU-bound; keep them off the event loop.
    result = await run_in_threadpool(pose.estimate, data, hint)

    return MeasurementsResponse(**result)
