import pytest
from fastapi.testclient import TestClient
from src.app import app, pipeline


@pytest.fixture(autouse=True)
def init_app():
    if not pipeline.indexed:
        pipeline.initialize()


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == "Clanker"
    assert data["is_indexed"] is True
    assert data["indexed_chunks"] > 0


def test_documents_endpoint():
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert len(data["documents"]) == 3
    sources = [doc["source"] for doc in data["documents"]]
    assert "01-getting-started.md" in sources
    assert "02-pricing-and-plans.md" in sources
    assert "03-troubleshooting.md" in sources


def test_api_query_rag_mode():
    response = client.post("/api/query", json={
        "question": "How often does NimbusNote sync while the app is in the foreground?"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "rag"
    assert data["supported"] is True
    assert data["top_similarity"] >= data["threshold"]
    assert len(data["citations"]) > 0
    assert data["citations"][0]["source"] == "01-getting-started.md"
    assert data["citations"][0]["section"] == "Sync behavior"


def test_api_query_casual_mode():
    response = client.post("/api/query", json={
        "question": "yo Clanker"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "casual"
    assert data["supported"] is True
    assert len(data["citations"]) == 0
    assert len(data["answer"]) > 0


def test_api_query_multi_turn_follow_up():
    response = client.post("/api/query", json={
        "question": "and how many collaborators?",
        "history": [
            {"role": "user", "content": "How much is the Pro plan?"},
            {"role": "assistant", "content": "The Pro plan is $6/month."}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "rag"
    assert data["supported"] is True
    assert len(data["citations"]) > 0
    assert any("20 collaborators" in c["passage"] for c in data["citations"])


def test_api_query_out_of_scope():
    response = client.post("/api/query", json={
        "question": "What is the weather in Chennai today?"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "unsupported"
    assert data["supported"] is False
    assert data["top_similarity"] < data["threshold"]
    assert len(data["citations"]) == 0
    assert "couldn't find enough information" in data["answer"].lower()
