from fastapi import APIRouter

from .. import __version__
from ..providers import capabilities, status
from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Unauthenticated liveness + capability probe.

    Rails calls this from its own /up endpoint, so it must stay cheap and must
    never raise — it reports what works rather than failing when something
    doesn't.
    """
    return HealthResponse(
        status="ok",
        version=__version__,
        providers=status(),
        capabilities=capabilities(),
    )
