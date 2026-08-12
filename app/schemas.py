"""Response models.

Field names are dictated by Rails: app/services/ai_client.rb reads them, and
ImageIngestor looks specifically for `image_base64` / `image_url` /
`front_base64` / `back_base64` / `shadow_base64`. Renaming anything here
breaks the Rails side silently, so they are pinned by these models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    providers: dict[str, Any]
    capabilities: dict[str, bool]


class SketchResponse(BaseModel):
    image_base64: str | None = None
    image_url: str | None = None
    content_type: str = "image/png"
    provider: str
    model: str | None = None


class TryonResponse(BaseModel):
    image_base64: str | None = None
    image_url: str | None = None
    content_type: str = "image/png"
    provider: str
    model: str | None = None
    landmarks: dict[str, Any] = Field(default_factory=dict)
    measurements: dict[str, float] = Field(default_factory=dict)


class MeasurementsResponse(BaseModel):
    measurements: dict[str, float]
    confidence: float
    provider: str
    landmarks: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ComposeResponse(BaseModel):
    image_base64: str
    content_type: str = "image/png"
    width: int
    height: int
    provider: str
    painted: list[str | None] = Field(default_factory=list)
    skipped: list[dict[str, Any]] = Field(default_factory=list)
    layers_in: int = 0
    layers_painted: int = 0


class AssetResponse(BaseModel):
    front_base64: str | None = None
    back_base64: str | None = None
    shadow_base64: str | None = None
    bounds: dict[str, Any] | None = None
    provider: str
    content_type: str = "image/png"
