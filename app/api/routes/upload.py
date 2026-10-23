from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.dependencies import embedding_service, vector_store
from app.core.config import settings
from app.ingestion.ingestion_service import IngestionService

router = APIRouter()

ingestion_service = IngestionService(
    vector_store=vector_store,
    embedding_service=embedding_service,
)

ALLOWED_EXTENSIONS = {".txt", ".md", ".json", ".yaml", ".yml", ".csv"}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "Nom de fichier manquant.")

    # Keep only the file name (blocks "../../" in the client-provided name).
    safe_name = Path(file.filename).name
    suffix = Path(safe_name).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"Format non supporté : {suffix or 'aucun'}")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Le fichier est vide.")

    upload_dir = Path(settings.uploads.directory)
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / safe_name
    upload_path.write_bytes(content)

    try:
        chunks_created = ingestion_service.ingest(str(upload_path))
    except ValueError as exc:
        raise HTTPException(415, str(exc))

    return {
        "filename": safe_name,
        "chunks_created": chunks_created,
    }
