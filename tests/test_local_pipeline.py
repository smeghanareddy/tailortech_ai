"""Tests for the local CV path — the part that works with no API keys.

Skipped when rembg is not installed, since the whole pipeline depends on it.
"""

import base64

import pytest

from app.providers import local_cv


requires_cv = pytest.mark.skipif(
    not local_cv.available(), reason="rembg is not installed"
)


@requires_cv
def test_assets_process_returns_a_transparent_front_layer(client, swatch_png):
    response = client.post(
        "/assets/process",
        files={"image": ("neckline.png", swatch_png, "image/png")},
        data={"category": "neck"},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["front_base64"]
    assert base64.b64decode(body["front_base64"])[:8] == b"\x89PNG\r\n\x1a\n"
    assert body["shadow_base64"]
    # Back views cannot be recovered from a front photo.
    assert body["back_base64"] is None


@requires_cv
def test_assets_process_reports_the_outer_boundary(client, swatch_png):
    body = client.post(
        "/assets/process",
        files={"image": ("neckline.png", swatch_png, "image/png")},
        data={"category": "neck"},
    ).json()

    bounds = body["bounds"]
    assert bounds["width"] > 0 and bounds["height"] > 0
    assert 0 < bounds["relative"]["width"] <= 1
    assert bounds["category"] == "neck"


@requires_cv
def test_mirror_back_is_opt_in_and_category_gated(client, swatch_png):
    body = client.post(
        "/assets/process",
        files={"image": ("sleeve.png", swatch_png, "image/png")},
        data={"category": "sleeve", "mirror_back": "true"},
    ).json()

    assert body["back_base64"]
    assert body["bounds"]["back_layer"] == "mirrored_from_front"


@requires_cv
def test_measurements_scale_from_the_supplied_height(client, person_png):
    body = client.post(
        "/measurements/estimate",
        files={"photo": ("me.png", person_png, "image/png")},
        data={"height_hint": "170"},
    ).json()

    assert body["measurements"]["height"] == 170.0
    # Every returned girth must be physically plausible; Rails rejects
    # anything outside its own ranges anyway.
    for field in ("bust", "waist", "hip"):
        if field in body["measurements"]:
            assert 30 < body["measurements"][field] < 220


@requires_cv
def test_confidence_drops_without_a_height_hint(client, person_png):
    with_hint = client.post(
        "/measurements/estimate",
        files={"photo": ("me.png", person_png, "image/png")},
        data={"height_hint": "170"},
    ).json()

    without_hint = client.post(
        "/measurements/estimate",
        files={"photo": ("me.png", person_png, "image/png")},
    ).json()

    assert without_hint["confidence"] < with_hint["confidence"]
    assert any("stature" in note for note in without_hint["notes"])
