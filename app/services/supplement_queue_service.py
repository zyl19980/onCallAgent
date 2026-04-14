"""低置信度知识补充队列。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from loguru import logger

from app.config import config


class SupplementQueueService:
    """将低置信度问题写入本地 JSONL 队列。"""

    def __init__(self):
        self.queue_path = Path(config.supplement_queue_path)
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)

    def enqueue(self, payload: Dict[str, Any]) -> bool:
        record = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        try:
            with self.queue_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("低置信度问题已加入待补充队列")
            return True
        except Exception as exc:
            logger.error(f"写入待补充队列失败: {exc}")
            return False


supplement_queue_service = SupplementQueueService()
