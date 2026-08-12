import secrets

from fastapi import Header, HTTPException, UploadFile, status

from .config import settings
from .errors import InvalidInput, PayloadTooLarge

IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}


async def require_token(authorization: str | None = Header(default=None)) -> None:
    """Bearer auth matching the header Rails' AiClient sends.

    Disabled when AI_SERVICE_TOKEN is empty, which is the local default.
    """
    if not settings.auth_required:
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )

    supplied = authorization.split(" ", 1)[1].strip()
    if not secrets.compare_digest(supplied, settings.ai_service_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
        )


async def read_image(upload: UploadFile, field: str = "image") -> bytes:
    """Read an uploaded image, enforcing type and size limits."""
    if upload is None:
        raise InvalidInput(f"`{field}` is required")

    content_type = (upload.content_type or "").lower().split(";")[0]
    if content_type and content_type not in IMAGE_TYPES:
        raise InvalidInput(
            f"`{field}` must be an image",
            detail=f"received content type {content_type!r}",
        )

    data = await upload.read()
    if not data:
        raise InvalidInput(f"`{field}` was empty")

    if len(data) > settings.max_upload_bytes:
        raise PayloadTooLarge(
            f"`{field}` exceeds the {settings.max_upload_mb} MB limit",
            detail=f"{len(data) / 1024 / 1024:.1f} MB uploaded",
        )

    return data
