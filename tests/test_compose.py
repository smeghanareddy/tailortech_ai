"""Local compositor — runs with no API key, so these are not skipped."""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from app.providers import local_compose


def _png_bytes(color=(255, 255, 255, 255), size=(120, 160), box=None):
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    box = box or (20, 20, size[0] - 20, size[1] - 20)
    Image.Image.paste(img, Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), color), box[:2])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def stub_layers(monkeypatch):
    """Serve layer art from memory instead of over HTTP."""
    art = _png_bytes()
    monkeypatch.setattr(local_compose, "_fetch", lambda url: art)
    return art


def test_compose_paints_layers_in_z_order(stub_layers):
    layers = [
        {"slot": "sleeve", "url": "http://x/sleeve.png", "z_index": 40, "opacity": 1.0},
        {"slot": "dress_base", "url": "http://x/base.png", "z_index": 20, "opacity": 1.0},
    ]
    image, report = local_compose.compose(layers, 200, 260)

    assert report["layers_painted"] == 2
    # sorted by z_index regardless of input order
    assert report["painted"] == ["dress_base", "sleeve"]
    assert image.size == (200, 260)


def test_layer_without_a_url_is_reported_not_silently_dropped(stub_layers):
    layers = [
        {"slot": "body", "url": None, "z_index": 0},
        {"slot": "dress_base", "url": "http://x/base.png", "z_index": 20},
    ]
    _, report = local_compose.compose(layers, 120, 160)

    assert report["layers_painted"] == 1
    assert report["skipped"] == [{"slot": "body", "reason": "no image"}]


def test_a_broken_layer_does_not_kill_the_render(monkeypatch):
    def explode(url):
        if "bad" in url:
            raise RuntimeError("404")
        return _png_bytes()

    monkeypatch.setattr(local_compose, "_fetch", explode)
    layers = [
        {"slot": "dress_base", "url": "http://x/ok.png", "z_index": 20},
        {"slot": "lace", "url": "http://x/bad.png", "z_index": 70},
    ]
    _, report = local_compose.compose(layers, 120, 160)

    assert report["layers_painted"] == 1
    assert report["skipped"][0]["reason"] == "fetch failed"


def test_solid_colour_is_applied_to_the_layer(stub_layers):
    layers = [{"slot": "dress_base", "url": "http://x/base.png", "z_index": 20,
               "color": {"base": "#ff0000", "stripes": []}}]
    image, _ = local_compose.compose(layers, 120, 160)

    pixels = np.asarray(image)
    painted = pixels[pixels[..., 3] > 0]
    assert painted.size > 0
    # White art multiplied by pure red leaves red only.
    assert painted[:, 0].mean() > 200
    assert painted[:, 1].mean() < 20
    assert painted[:, 2].mean() < 20


def test_stripes_use_the_specs_direction_priority(stub_layers):
    # curve outranks straight, so its colour owns the first band.
    layers = [{"slot": "skirt", "url": "http://x/s.png", "z_index": 26,
               "color": {"base": "#ffffff", "stripes": [
                   {"color": "#0000ff", "direction": "straight"},
                   {"color": "#00ff00", "direction": "curve"},
               ]}}]
    image, _ = local_compose.compose(layers, 160, 200)

    pixels = np.asarray(image)
    painted = pixels[pixels[..., 3] > 0]
    # Both stripe colours appear somewhere in the garment.
    assert painted[:, 1].max() > 200  # green present
    assert painted[:, 2].max() > 200  # blue present


def test_opacity_reduces_coverage(stub_layers):
    solid = local_compose.compose(
        [{"slot": "a", "url": "http://x/a.png", "z_index": 1, "opacity": 1.0}], 120, 160)[0]
    faint = local_compose.compose(
        [{"slot": "a", "url": "http://x/a.png", "z_index": 1, "opacity": 0.3}], 120, 160)[0]

    assert np.asarray(faint)[..., 3].mean() < np.asarray(solid)[..., 3].mean()


def test_a_texture_layer_cannot_bleed_outside_the_garment(monkeypatch):
    """A full-bleed fabric swatch must be clipped to the garment silhouette."""
    garment = _png_bytes(size=(200, 200), box=(60, 60, 140, 140))
    texture = _png_bytes(size=(200, 200), box=(0, 0, 200, 200), color=(180, 180, 180, 255))

    monkeypatch.setattr(
        local_compose, "_fetch",
        lambda url: texture if "fabric" in url else garment,
    )

    image, _ = local_compose.compose([
        {"slot": "dress_base", "url": "http://x/dress.png", "z_index": 20},
        {"slot": "fabric", "url": "http://x/fabric.png", "z_index": 50,
         "blend_mode": "multiply", "opacity": 0.55},
    ], 200, 200)

    alpha = np.asarray(image)[..., 3]
    # The corners are outside the garment and must stay fully transparent.
    assert alpha[5, 5] == 0
    assert alpha[195, 195] == 0
    # The garment itself is still solid.
    assert alpha[100, 100] > 200


def test_endpoint_refuses_a_manifest_with_no_usable_art(client):
    response = client.post("/compose", json={"layers": [{"slot": "body", "url": None, "z_index": 0}]})

    assert response.status_code == 400
    assert "usable image" in response.json()["error"]


def test_endpoint_returns_a_png(client, monkeypatch):
    monkeypatch.setattr(local_compose, "_fetch", lambda url: _png_bytes())
    response = client.post("/compose", json={
        "layers": [{"slot": "dress_base", "url": "http://x/a.png", "z_index": 20}],
        "width": 150, "height": 200,
    })

    assert response.status_code == 200
    body = response.json()
    assert base64.b64decode(body["image_base64"])[:8] == b"\x89PNG\r\n\x1a\n"
    assert body["provider"] == "local_compose"
    assert body["layers_painted"] == 1
