from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven configuration.

    Mirrors .env.example; every value has a working default so the service
    boots with no configuration at all (local CV endpoints work, hosted ones
    report themselves as unavailable).
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ── Service ──
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    ai_service_token: str = ""
    allowed_origins: str = "http://localhost:3000"
    max_upload_mb: int = 12

    # ── Providers ──
    sketch_provider: str = "auto"
    tryon_provider: str = "auto"
    provider_timeout: int = 120

    replicate_api_token: str = ""
    replicate_sketch_model: str = "stability-ai/sdxl"
    replicate_tryon_model: str = "cuuupid/idm-vton"

    huggingface_api_token: str = ""
    huggingface_sketch_model: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # ── Local CV ──
    rembg_model: str = "u2net"
    asset_max_edge: int = 1400

    # How the garment is blended into the wearer's photo.
    #   seamless — Poisson cloning; edges disappear, but it reconstructs
    #              colour from gradients so the garment's hue can drift
    #   alpha    — feathered composite; the garment keeps its exact colour,
    #              at the cost of a slightly more visible boundary
    # Colour fidelity matters in a tailoring preview, so this is worth trying
    # both ways on your own photos.
    tryon_blend: str = "seamless"
    tryon_shading_strength: float = 0.55

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def auth_required(self) -> bool:
        return bool(self.ai_service_token)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
