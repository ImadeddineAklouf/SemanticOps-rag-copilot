from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.api.dependencies import embedding_service, vector_store
from app.api.schemas import IngestRequest, IngestResponse
from app.core.config import settings
from app.ingestion.ingestion_service import IngestionService

router = APIRouter()

ingestion_service = IngestionService(
    vector_store=vector_store,
    embedding_service=embedding_service,
)


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    upload_dir = Path(settings.uploads.directory).resolve()
    path = Path(request.file_path).resolve()

    if not path.is_relative_to(upload_dir):
        raise HTTPException(403, "Chemin hors du dossier d'upload.")
    if not path.is_file():
        raise HTTPException(404, "Fichier introuvable.")

    try:
        chunks_created = ingestion_service.ingest(str(path))
    except ValueError as exc:
        raise HTTPException(415, str(exc))

    return IngestResponse(chunks_created=chunks_created)
