import os
import tempfile

# Must be set before the application is imported.
os.environ.setdefault("GROQ_API_KEY", "fake")
os.environ.setdefault("VECTOR_DB_PATH", tempfile.mkdtemp(prefix="rag_test_db_"))

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes.query import get_pipeline
from app.core.config import settings


class FakePipeline:
    def ask(self, question: str) -> str:
        return f"réponse à : {question}"


class FailingPipeline:
    def ask(self, question: str) -> str:
        raise RuntimeError("Groq indisponible")


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").status_code == 200


def test_config_endpoint(client):
    assert client.get("/config").status_code == 200


def test_metrics_endpoint(client):
    assert client.get("/metrics").status_code == 200


def test_query_returns_answer(client):
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()

    response = client.post("/query", json={"question": "Bonjour ?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "réponse à : Bonjour ?"


def test_query_returns_502_when_llm_fails(client):
    app.dependency_overrides[get_pipeline] = lambda: FailingPipeline()

    response = client.post("/query", json={"question": "Bonjour ?"})

    assert response.status_code == 502


def test_query_requires_question_field(client):
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()

    assert client.post("/query", json={}).status_code == 422


def test_ingest_rejects_path_outside_uploads(client):
    response = client.post("/ingest", json={"file_path": "/etc/hosts"})
    assert response.status_code == 403


def test_ingest_missing_file(client):
    path = os.path.join(settings.uploads.directory, "inexistant.txt")
    response = client.post("/ingest", json={"file_path": path})
    assert response.status_code == 404


def test_upload_rejects_unsupported_extension(client):
    response = client.post("/upload", files={"file": ("virus.exe", b"data")})
    assert response.status_code == 415


def test_upload_rejects_empty_file(client):
    response = client.post("/upload", files={"file": ("vide.txt", b"")})
    assert response.status_code == 400


def test_upload_sanitizes_filename_and_ingests(client, monkeypatch, tmp_path):
    from app.api.routes import upload as upload_module

    class FakeIngestion:
        def ingest(self, path: str) -> int:
            assert os.path.dirname(path) == str(tmp_path)
            return 3

    monkeypatch.setattr(settings.uploads, "directory", str(tmp_path))
    monkeypatch.setattr(upload_module, "ingestion_service", FakeIngestion())

    response = client.post(
        "/upload",
        files={"file": ("../../evil.txt", b"contenu de test")},
    )

    assert response.status_code == 200
    assert response.json() == {"filename": "evil.txt", "chunks_created": 3}
    assert (tmp_path / "evil.txt").exists()
