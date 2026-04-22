"""低置信度聚合查询接口。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from loguru import logger

from app.core.postgres import postgres_manager
from app.repositories.low_confidence_repository import LowConfidenceRepository


router = APIRouter(prefix="/api/admin/low-confidence")
low_confidence_repository = LowConfidenceRepository()


@router.get("/fingerprints")
async def list_fingerprint_groups(limit: int = Query(default=50, ge=1, le=200)):
    try:
        with postgres_manager.session_scope() as session:
            items = low_confidence_repository.list_fingerprint_groups(session, limit=limit)

        return JSONResponse(
            status_code=200,
            content={
                "code": 200,
                "message": "success",
                "data": {
                    "items": jsonable_encoder(items),
                },
            },
        )
    except Exception as exc:
        logger.error(f"查询 low confidence 指纹聚合失败: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/fingerprints/{fingerprint}")
async def get_fingerprint_detail(fingerprint: str, limit: int = Query(default=50, ge=1, le=200)):
    try:
        with postgres_manager.session_scope() as session:
            events = low_confidence_repository.list_events_by_fingerprint(
                session,
                fingerprint,
                limit=limit,
            )
            if not events:
                raise HTTPException(status_code=404, detail="fingerprint not found")

            event_items = []
            for event in events:
                chunks = low_confidence_repository.list_event_chunks(session, int(event["id"]))
                event_items.append(
                    {
                        **event,
                        "chunks": chunks,
                    }
                )

        return JSONResponse(
            status_code=200,
            content={
                "code": 200,
                "message": "success",
                "data": {
                    "fingerprint": fingerprint,
                    "eventCount": len(event_items),
                    "events": jsonable_encoder(event_items),
                },
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"查询 low confidence 指纹详情失败: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
