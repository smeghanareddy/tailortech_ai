def test_health_reports_capabilities(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body["capabilities"]) == {
        "sketch", "tryon", "tryon_photorealistic", "measurements", "assets", "compose",
    }
    assert "local_cv" in body["providers"]
    assert "local_compose" in body["providers"]


def test_the_local_capabilities_need_no_api_key(client):
    """Rails polls this from /up; it must report, not raise.

    With no tokens configured the local work still reports available — that is
    the whole point of the offline path.
    """
    body = client.get("/health").json()
    caps = body["capabilities"]

    # Local, no keys required.
    assert caps["assets"] is True
    assert caps["measurements"] is True
    assert caps["compose"] is True
    # Try-on still answers, via the geometric overlay...
    assert caps["tryon"] is True
    # ...but it is explicitly not the photorealistic path.
    assert caps["tryon_photorealistic"] is False
    # Generated imagery does need a provider.
    assert caps["sketch"] is False

    assert body["status"] == "ok"
