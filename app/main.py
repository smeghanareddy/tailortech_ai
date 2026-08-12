"""TailorTech AI service.

FastAPI front-end for the CV and generative work the Rails API delegates:

    POST /sketch                design sketch from the component tree
    POST /tryon                 garment warped onto the user's photo
    POST /measurements/estimate body measurements from one photo
    POST /assets/process        design-library cut-out pipeline
    GET  /health                liveness + what actually works right now

Paths and response keys are fixed by tailor_tech_backend/app/services/
ai_client.rb — see schemas.py before renaming anything.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .config import settings
from .errors import ServiceError
from .routers import assets, compose, health, measurements, sketch, tryon

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TailorTech AI Service",
    version=__version__,
    description=__doc__,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(ServiceError)
async def service_error_handler(_: Request, exc: ServiceError) -> JSONResponse:
    """Structured errors.

    A 503 tells Rails the AI is unavailable, which makes it fall back to
    library assets instead of losing the user's work — so the distinction
    between 503 (nothing configured) and 502 (provider tried and failed)
    matters downstream.
    """
    if exc.status_code >= 500:
        logger.warning("%s: %s (%s)", exc.code, exc.message, exc.detail or "-")
    return JSONResponse(status_code=exc.status_code, content=exc.as_dict())


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal error in the AI service", "code": "internal_error"},
    )


app.include_router(health.router, tags=["health"])
app.include_router(sketch.router, tags=["generation"])
app.include_router(tryon.router, tags=["generation"])
app.include_router(measurements.router, tags=["analysis"])
app.include_router(assets.router, tags=["library"])
app.include_router(compose.router, tags=["rendering"])


@app.on_event("startup")
async def on_startup() -> None:
    from .providers import capabilities

    caps = capabilities()
    logger.info("TailorTech AI %s ready — capabilities: %s", __version__, caps)
    if not caps["sketch"]:
        logger.info(
            "No diffusion provider configured — /sketch returns 503 and Rails "
            "falls back to local compositing, which needs no API key. Set "
            "REPLICATE_API_TOKEN only if you want generated imagery."
        )
    if not caps["tryon_photorealistic"]:
        logger.info(
            "/tryon will use the local geometric overlay: correct fit and "
            "placement, but not photorealistic."
        )


def run() -> None:
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=True,
    )


if __name__ == "__main__":
    run()
