"""chunk 管理接口。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from app.services.chunk_curation_service import (
    ChunkCurationService,
    ChunkNotFoundError,
    ChunkPublishError,
)


router = APIRouter(prefix="/api/admin/chunks")
chunk_curation_service = ChunkCurationService()


class SaveDraftRequest(BaseModel):
    draft_text: str = Field(alias="draftText")

    model_config = {
        "populate_by_name": True,
    }


class PublishChunkRequest(BaseModel):
    editor: str = "admin"
    edit_note: str | None = Field(default=None, alias="editNote")

    model_config = {
        "populate_by_name": True,
    }


@router.get("/{chunk_key:path}/history")
async def get_chunk_history(chunk_key: str):
    try:
        history = chunk_curation_service.list_history(chunk_key)
        return {
            "code": 200,
            "message": "success",
            "data": {
                "chunkKey": chunk_key,
                "items": jsonable_encoder(history),
            },
        }
    except ChunkNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/{chunk_key:path}/draft")
async def save_chunk_draft(chunk_key: str, request: SaveDraftRequest):
    try:
        chunk = chunk_curation_service.save_draft(chunk_key, request.draft_text)
        return {
            "code": 200,
            "message": "success",
            "data": jsonable_encoder(chunk),
        }
    except ChunkNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/{chunk_key:path}/publish")
async def publish_chunk(chunk_key: str, request: PublishChunkRequest):
    try:
        chunk = chunk_curation_service.publish_chunk(
            chunk_key,
            editor=request.editor,
            edit_note=request.edit_note,
        )
        return {
            "code": 200,
            "message": "success",
            "data": jsonable_encoder(chunk),
        }
    except ChunkNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ChunkPublishError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{chunk_key:path}")
async def get_chunk(chunk_key: str):
    try:
        chunk = chunk_curation_service.get_chunk(chunk_key)
        return {
            "code": 200,
            "message": "success",
            "data": jsonable_encoder(chunk),
        }
    except ChunkNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
