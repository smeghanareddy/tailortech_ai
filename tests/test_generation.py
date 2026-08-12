"""Contract tests for the endpoints that need a hosted provider.

With no API token configured these must return 503 — not 500 — because Rails
treats 503 as "AI unavailable" and falls back to library assets, whereas a 500
surfaces to the user as a failure.
"""

import json


def test_sketch_returns_503_without_a_provider(client):
    response = client.post("/sketch", json={"dress_type": "Anarkali", "fabric": "Silk"})

    assert response.status_code == 503
    assert response.json()["code"] == "provider_unavailable"


def test_tryon_returns_503_without_a_provider(client, person_png):
    response = client.post(
        "/tryon",
        files={"photo": ("me.png", person_png, "image/png")},
        data={"design": json.dumps({"dress_type": "A-Line"})},
    )

    assert response.status_code == 503


def test_tryon_rejects_malformed_design_json(client, person_png):
    response = client.post(
        "/tryon",
        files={"photo": ("me.png", person_png, "image/png")},
        data={"design": "not json"},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_input"


def test_non_image_upload_is_rejected(client):
    response = client.post(
        "/measurements/estimate",
        files={"photo": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
