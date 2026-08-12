"""Local OpenCV / rembg pipeline — no API keys, no GPU, no per-call cost.

Serves the Design Library upload step from the spec:

    Upload → extract outer boundary → remove background →
    transparent asset → front / back / shadow layers

The first call downloads the rembg segmentation model (~176 MB) to
~/.u2net and is therefore slow; every call afterwards is local and fast.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageFilter

from ..config import settings
from ..errors import ProviderUnavailable
from ..imaging import (
    alpha_mask,
    decode,
    downscale,
    encode_png,
    outer_boundary,
    trim_to_content,
)

logger = logging.getLogger(__name__)

name = "local_cv"


def _rembg():
    """Import rembg lazily so the service boots without it installed."""
    try:
        import rembg  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ProviderUnavailable(
            "Background removal is unavailable — rembg is not installed",
            detail=str(exc),
        ) from exc
    return rembg


@lru_cache(maxsize=2)
def _session(model: str):
    """Cache the ONNX session; building it is the expensive part."""
    return _rembg().new_session(model)


def available() -> bool:
    try:
        import rembg  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def remove_background(image: Image.Image) -> Image.Image:
    """RGBA cutout with the subject isolated on transparency."""
    rembg = _rembg()
    array = np.array(image)
    result = rembg.remove(array, session=_session(settings.rembg_model))
    cutout = Image.fromarray(result).convert("RGBA")
    return _clean_alpha(cutout)


def _clean_alpha(image: Image.Image) -> Image.Image:
    """Tidy the matte: drop speckles, close pinholes, soften the edge.

    Segmentation masks come back noisy at garment edges — lace, fringe and
    thin straps in particular. A morphological open/close pass plus a 1px
    feather keeps the cutout from looking cut with scissors.
    """
    alpha = np.array(image.getchannel("A"))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=1)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep only the largest connected component — removes stray background
    # fragments the model was unsure about.
    binary = (alpha > 10).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        alpha = np.where(labels == largest, alpha, 0).astype(np.uint8)

    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    channels = list(image.split())
    channels[3] = Image.fromarray(alpha)
    return Image.merge("RGBA", channels)


def build_shadow(image: Image.Image, *, offset: int = 6, blur: int = 12,
                 opacity: float = 0.35) -> Image.Image:
    """Derive the shadow layer from the cutout's own silhouette.

    This is a drop shadow computed from the alpha channel — not invented
    detail. The compositor paints it beneath the garment at z-index 95.
    """
    alpha = image.getchannel("A")

    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow.putalpha(alpha.point(lambda a: int(a * opacity)))

    canvas = Image.new("RGBA", image.size, (0, 0, 0, 0))
    canvas.paste(shadow, (offset, offset), shadow)
    return canvas.filter(ImageFilter.GaussianBlur(blur))


def process_asset(data: bytes, category: str, *, mirror_back: bool = False) -> dict:
    """The full Design Library pipeline for one uploaded asset."""
    image = downscale(decode(data), settings.asset_max_edge)

    cutout = trim_to_content(remove_background(image))
    mask = alpha_mask(cutout)
    _, bounds = outer_boundary(mask)

    if bounds is None:
        raise ProviderUnavailable(
            "No subject could be isolated in this image",
            detail="the background removal returned an empty mask",
        )

    payload = {
        "front_base64": encode_png(cutout),
        "shadow_base64": encode_png(build_shadow(cutout)),
        "bounds": {**bounds, "category": category},
        "provider": name,
    }

    # A back view cannot be recovered from a front photo. Mirroring is only
    # a sane approximation for laterally symmetric parts, so it is opt-in and
    # the result is labelled as derived.
    if mirror_back and category in {"neck", "back_neck", "sleeve", "belt", "button"}:
        payload["back_base64"] = encode_png(cutout.transpose(Image.FLIP_LEFT_RIGHT))
        payload["bounds"]["back_layer"] = "mirrored_from_front"
    else:
        payload["back_base64"] = None

    return payload


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels that look like skin, in both YCrCb and HSV.

    Requiring agreement from two colour spaces keeps deeply saturated fabrics
    (and shadowed skin) from being misread as one another.
    """
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    a = cv2.inRange(ycrcb, np.array([0, 133, 77], np.uint8), np.array([255, 173, 127], np.uint8))
    b = cv2.inRange(hsv, np.array([0, 20, 60], np.uint8), np.array([25, 170, 255], np.uint8))
    mask = cv2.bitwise_and(a, b)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.dilate(mask, kernel, iterations=2)


def extract_garment(image: Image.Image, landmarks: dict | None = None) -> Image.Image:
    """Isolate the garment from a photo of a model wearing it.

    Background removal alone is not enough: it returns "model + garment" as one
    silhouette, so compositing it pastes someone else's face and arms onto the
    wearer. This strips the person away too:

      1. cut the background (rembg)
      2. drop everything above the shoulder line — head, hair, face
      3. drop skin-coloured pixels outside the torso core, which takes the
         hands, forearms, neck and any bare leg with them
      4. keep the largest remaining region — the garment

    The torso core is protected from step 3 so a nude or beige dress is not
    eaten by its own colour.
    """
    cutout = remove_background(image)
    width, height = cutout.size
    alpha = np.array(cutout.getchannel("A"))

    shoulder_y, hip_y = None, None
    if landmarks:
        ys = [landmarks[k]["y"] for k in ("left_shoulder", "right_shoulder") if landmarks.get(k)]
        hs = [landmarks[k]["y"] for k in ("left_hip", "right_hip") if landmarks.get(k)]
        if ys:
            shoulder_y = sum(ys) / len(ys)
        if hs:
            hip_y = sum(hs) / len(hs)

    if shoulder_y is None:
        # No pose: assume the head occupies roughly the top fifth.
        rows = np.flatnonzero(alpha.any(axis=1))
        if rows.size:
            top, bottom = int(rows[0]), int(rows[-1])
            shoulder_y = top + (bottom - top) * 0.18

    # 2 — everything above the shoulders goes.
    if shoulder_y is not None:
        cut = int(max(0, shoulder_y - height * 0.015))
        alpha[:cut, :] = 0

    # 3 — skin outside the torso core.
    rgb = np.array(cutout.convert("RGB"))
    skin = _skin_mask(rgb)

    protect = np.zeros_like(skin)
    if shoulder_y is not None:
        core_top = int(shoulder_y)
        core_bottom = int(hip_y + (hip_y - shoulder_y) * 0.9) if hip_y else height
        x0, x1 = int(width * 0.30), int(width * 0.70)
        protect[core_top:core_bottom, x0:x1] = 255

    removable = cv2.bitwise_and(skin, cv2.bitwise_not(protect))
    alpha[removable > 0] = 0

    # 4 — keep only what the wearer's torso actually passes through. Props the
    # model is holding (a handbag, a shopping bag) survive step 3 because they
    # are not skin, but they sit off to the side and are dropped here.
    binary = (alpha > 10).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))
        keep = np.zeros_like(binary)

        for label in range(1, count):
            if label == largest:
                keep |= labels == label
                continue
            # A fragment is only kept if it is substantial *and* overlapping
            # the torso column — otherwise it is a prop or a stray edge.
            touches_core = protect.any() and np.any((labels == label) & (protect > 0))
            if touches_core and stats[label, cv2.CC_STAT_AREA] > areas.max() * 0.12:
                keep |= labels == label

        alpha = np.where(keep > 0, alpha, 0).astype(np.uint8)

    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    channels = list(cutout.split())
    channels[3] = Image.fromarray(alpha)
    merged = Image.merge("RGBA", channels)

    # Cropping shifts the origin, so the garment's own landmarks have to move
    # with it — otherwise aligning its shoulders to the wearer's uses stale
    # coordinates and the dress lands in the wrong place.
    box = merged.getbbox()
    if box is None:
        return merged, landmarks or {}

    pad = 2
    x0 = max(0, box[0] - pad)
    y0 = max(0, box[1] - pad)
    cropped = merged.crop((x0, y0, min(width, box[2] + pad), min(height, box[3] + pad)))

    shifted = {}
    for key, point in (landmarks or {}).items():
        if isinstance(point, dict) and "x" in point:
            shifted[key] = {**point, "x": point["x"] - x0, "y": point["y"] - y0}

    return cropped, shifted


def segment_person(data: bytes) -> tuple[Image.Image, Image.Image, np.ndarray]:
    """Source image, cutout and binary mask of a person.

    The source is returned alongside the cutout because pose detection works
    better on the original photo — background removal can clip limbs, and the
    transparent regions read as black once flattened.
    """
    image = downscale(decode(data), settings.asset_max_edge)
    cutout = remove_background(image)
    return image, cutout, alpha_mask(cutout)
