"""chunk 编辑历史 repository。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


class ChunkEditHistoryRepository:
    """负责 chunk_edit_history 表的数据访问。"""

    def create_history(
        self,
        session: Session,
        *,
        chunk_id: int,
        version_no: int,
        old_text: str,
        new_text: str,
        editor: str,
        edit_note: str | None = None,
        publish_status: str = "published",
    ) -> dict[str, Any]:
        result = session.execute(
            text(
                """
                insert into chunk_edit_history (
                    chunk_id, version_no, old_text, new_text, editor, edit_note, publish_status
                ) values (
                    :chunk_id, :version_no, :old_text, :new_text, :editor, :edit_note, :publish_status
                )
                returning *
                """
            ),
            {
                "chunk_id": chunk_id,
                "version_no": version_no,
                "old_text": old_text,
                "new_text": new_text,
                "editor": editor,
                "edit_note": edit_note,
                "publish_status": publish_status,
            },
        )
        return dict(result.mappings().one())

    def list_by_chunk_id(self, session: Session, chunk_id: int) -> list[dict[str, Any]]:
        result = session.execute(
            text(
                """
                select *
                from chunk_edit_history
                where chunk_id = :chunk_id
                order by version_no desc, created_at desc
                """
            ),
            {"chunk_id": chunk_id},
        )
        return [dict(row) for row in result.mappings().all()]
