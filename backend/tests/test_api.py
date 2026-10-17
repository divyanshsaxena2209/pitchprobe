import pytest
from httpx import ASGITransport, AsyncClient
from backend.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "classifier_loaded" in data
    assert "generator_loaded" in data
    assert "device" in data


@pytest.mark.asyncio
async def test_analyze_endpoint_success(sample_pitch_pdf):
    files = {"file": ("pitch_deck.pdf", sample_pitch_pdf, "application/pdf")}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/analyze", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "pitch_deck.pdf"
    assert data["total_sentences"] >= 3
    assert "summary" in data
    summary = data["summary"]
    assert summary["low_risk"] + summary["moderate_risk"] + summary["high_risk"] == data["total_sentences"]

    # Check sentence structure and gating in API response
    for sentence in data["sentences"]:
        assert sentence["risk"] in ("Low", "Moderate", "High")
        assert 0.0 <= sentence["score"] <= 1.0
        if sentence["risk"] == "Low":
            assert sentence["question"] is None
        else:
            assert isinstance(sentence["question"], str)
            assert sentence["question"].endswith("?")


@pytest.mark.asyncio
async def test_analyze_endpoint_invalid_file():
    files = {"file": ("test.txt", b"plain text content", "text/plain")}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/analyze", files=files)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
