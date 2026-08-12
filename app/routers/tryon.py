import json
import logging

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from ..config import settings
from ..deps import read_image, require_token
from ..errors import InvalidInput, ProviderFailed, ProviderUnavailable
from ..imaging import encode_png
from ..providers import local_compose, local_cv, pose, tryon_provider
from ..schemas import TryonResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/tryon", response_model=TryonResponse)
async def tryon(
    photo: UploadFile = File(...),
    design: str = Form(...),
    _: None = Depends(require_token),
) -> TryonResponse:
    """Put the design on the customer's photo.

    Two paths:

      * a configured diffusion provider (Replicate IDM-VTON) — photorealistic;
      * otherwise a local geometric overlay: the garment is composed from its
        layers and warped onto the torso landmarks. That gives correct scale
        and placement on the real body but is explicitly NOT photorealistic —
        no fabric drape, no lighting match, no occlusion by the arms. The
        response says which ran via `provider`.
    """
    try:
        design_data = json.loads(design)
    except json.JSONDecodeError as exc:
        raise InvalidInput("`design` was not valid JSON", detail=str(exc)) from exc

    data = await read_image(photo, "photo")

    # Body analysis runs either way — it is local and cheap.
    landmarks: dict = {}
    measurements: dict = {}
    frame: dict = {}
    try:
        estimate = await run_in_threadpool(pose.estimate, data, None)
        landmarks = estimate["landmarks"]
        measurements = estimate["measurements"]
        frame = estimate.get("frame") or {}
    except ProviderUnavailable as exc:
        logger.info("body analysis skipped: %s", exc.message)

    try:
        provider = tryon_provider()
    except ProviderUnavailable as exc:
        logger.info("no diffusion provider (%s) — using the local overlay", exc.message)
        return await _local_tryon(data, design_data, landmarks, measurements, frame)

    result = await provider.tryon(data, design_data)
    return TryonResponse(
        image_base64=result.get("image_base64"),
        image_url=result.get("image_url"),
        content_type=result.get("content_type", "image/png"),
        provider=provider.name,
        model=result.get("model"),
        landmarks=landmarks,
        measurements=measurements,
    )


def _isolate_garment(image):
    """Garment only — the model it is photographed on is removed.

    Pose is run on the *garment* photo (not the wearer's) so the shoulder and
    hip lines used to cut away the head and limbs belong to the right body.
    Returns (garment, landmarks) with the landmarks translated into the
    cropped image's coordinates, ready for shoulder-to-shoulder alignment.
    """
    import numpy as np  # noqa: PLC0415

    garment_landmarks = None
    if pose.mediapipe_available():
        try:
            garment_landmarks = pose._mediapipe_landmarks(np.array(image.convert("RGB")))
        except Exception as exc:  # noqa: BLE001
            logger.info("no pose found on the garment photo: %s", exc)

    return local_cv.extract_garment(image, garment_landmarks)


async def _local_tryon(photo: bytes, design: dict, landmarks: dict,
                       measurements: dict, frame: dict) -> TryonResponse:
    # pose.estimate already falls back to silhouette-derived torso points, so
    # an empty dict here means no subject was found at all.
    if not all(landmarks.get(k) for k in
               ("left_shoulder", "right_shoulder", "left_hip", "right_hip")):
        raise ProviderUnavailable(
            "No body could be located in the photo, so the garment cannot be placed",
            detail="use a full-body, front-facing photo against a plain background",
        )

    layers = design.get("layers") or []
    garment = None
    garment_landmarks: dict | None = None

    if layers:
        image, report = await run_in_threadpool(local_compose.compose, layers, 900, 1200)
        if report["layers_painted"]:
            garment = image

    if garment is None:
        # A single garment image: either linked from another site, or the
        # design's own sketch.
        url = design.get("garment_url") or design.get("sketch_url") or design.get("preview_url")
        if not url:
            raise ProviderUnavailable(
                "The design has no image to place on the photo",
                detail="upload Design Library assets, or generate a sketch first",
            )
        garment = await run_in_threadpool(local_compose._load, url)
        if garment is None:
            raise ProviderFailed("The design's image could not be loaded")

        # A shopping-site photo is a *model wearing* the garment. Removing the
        # background alone would paste her face and arms onto the wearer, so
        # the model has to be stripped out too.
        if local_cv.available():
            try:
                garment, garment_landmarks = await run_in_threadpool(_isolate_garment, garment)
            except Exception as exc:  # noqa: BLE001 - a failed matte must not lose the try-on
                logger.warning("garment extraction failed, using the raw image: %s", exc)

    # The wearer's own silhouette keeps the garment on the body instead of
    # letting it spill onto the background.
    person_mask = None
    if local_cv.available():
        try:
            _, _, person_mask = await run_in_threadpool(local_cv.segment_person, photo)
        except Exception as exc:  # noqa: BLE001 - falling back to no clip is fine
            logger.info("wearer silhouette unavailable: %s", exc)

    composited = await run_in_threadpool(
        local_compose.overlay_on_photo, photo, garment, landmarks, frame,
        garment_landmarks, settings.tryon_blend, person_mask
    )

    return TryonResponse(
        image_base64=encode_png(composited),
        provider="local_geometric",
        model=None,
        landmarks=landmarks,
        measurements=measurements,
    )
