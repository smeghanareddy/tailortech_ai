"""Image plumbing shared by every provider.

Rails' ImageIngestor accepts either a raw base64 string or a URL, so
everything here returns plain base64 without a data: prefix.
"""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image

from .errors import InvalidInput


def decode(data: bytes) -> Image.Image:
    """Bytes → RGBA Pillow image, EXIF rotation applied."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # noqa: BLE001 - Pillow raises many types
        raise InvalidInput("Could not read the uploaded image", detail=str(exc)) from exc

    from PIL import ImageOps

    image = ImageOps.exif_transpose(image)
    return image.convert("RGBA")


def encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def to_cv(image: Image.Image) -> np.ndarray:
    """Pillow RGBA → OpenCV BGRA array."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)


def to_pil(array: np.ndarray) -> Image.Image:
    """OpenCV BGRA → Pillow RGBA."""
    if array.shape[2] == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2BGRA)
    return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA))


def downscale(image: Image.Image, max_edge: int) -> Image.Image:
    """Cap the longest edge, preserving aspect ratio. Never upscales."""
    width, height = image.size
    longest = max(width, height)
    if longest <= max_edge:
        return image

    scale = max_edge / longest
    return image.resize((round(width * scale), round(height * scale)), Image.LANCZOS)


def alpha_mask(image: Image.Image) -> np.ndarray:
    """Binary mask (uint8 0/255) of the non-transparent region."""
    alpha = np.array(image.getchannel("A"))
    _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    return mask


def outer_boundary(mask: np.ndarray) -> tuple[np.ndarray | None, dict | None]:
    """Largest external contour and its bounding box.

    This is the "extract outer boundary" step of the Design Library upload
    pipeline — it tells the layer compositor where the asset actually sits
    inside its canvas so the frontend can anchor it.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    height, width = mask.shape[:2]

    bounds = {
        "x": int(x),
        "y": int(y),
        "width": int(w),
        "height": int(h),
        "canvas_width": int(width),
        "canvas_height": int(height),
        # Normalised so the client can position the layer at any render size.
        "relative": {
            "x": round(x / width, 4),
            "y": round(y / height, 4),
            "width": round(w / width, 4),
            "height": round(h / height, 4),
        },
        "area_ratio": round(float(cv2.contourArea(contour)) / (width * height), 4),
    }
    return contour, bounds


def trim_to_content(image: Image.Image, padding: int = 8) -> Image.Image:
    """Crop away fully transparent margins, leaving a small padding."""
    bbox = image.getbbox()
    if bbox is None:
        return image

    left, top, right, bottom = bbox
    width, height = image.size
    return image.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(width, right + padding),
            min(height, bottom + padding),
        )
    )
