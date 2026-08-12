import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def person_png() -> bytes:
    """A crude standing figure on a plain background.

    Enough for the silhouette estimator to find a subject; not enough for a
    meaningful measurement, which is the point of the assertions in the
    measurement tests.
    """
    image = Image.new("RGB", (400, 900), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.ellipse((175, 40, 225, 100), fill=(60, 60, 70))       # head
    draw.rectangle((150, 100, 250, 420), fill=(60, 60, 70))    # torso
    draw.rectangle((110, 120, 150, 380), fill=(60, 60, 70))    # arms
    draw.rectangle((250, 120, 290, 380), fill=(60, 60, 70))
    draw.rectangle((160, 420, 195, 850), fill=(60, 60, 70))    # legs
    draw.rectangle((205, 420, 240, 850), fill=(60, 60, 70))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def swatch_png() -> bytes:
    """A small garment-ish shape for the asset pipeline."""
    image = Image.new("RGB", (300, 300), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.polygon([(150, 40), (250, 150), (200, 260), (100, 260), (50, 150)],
                 fill=(120, 40, 160))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
