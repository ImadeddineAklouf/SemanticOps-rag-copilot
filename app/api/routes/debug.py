from fastapi import APIRouter

from app.api.dependencies import vector_store

router = APIRouter()


@router.get("/debug/chunks")
def debug_chunks():
    return vector_store.collection.get()
