"""PostgreSQL 连接层。"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import config


class PostgresManager:
    """封装 PostgreSQL engine 和 Session 工厂。"""

    def __init__(self, dsn: str | None = None, *, echo: bool = False):
        self._dsn = dsn
        self._echo = echo
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None

    @property
    def dsn(self) -> str:
        return self._dsn or config.postgres_effective_dsn

    def get_engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(
                self.dsn,
                echo=self._echo,
                future=True,
                pool_pre_ping=True,
            )
            logger.info("PostgreSQL engine 初始化完成")
        return self._engine

    def get_session_factory(self) -> sessionmaker[Session]:
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.get_engine(),
                autoflush=False,
                autocommit=False,
                expire_on_commit=False,
                class_=Session,
            )
        return self._session_factory

    def get_session(self) -> Session:
        return self.get_session_factory()()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def health_check(self) -> bool:
        try:
            with self.get_engine().connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.warning(f"PostgreSQL 健康检查失败: {exc}")
            return False

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
        self._engine = None
        self._session_factory = None


postgres_manager = PostgresManager()


def get_postgres_session() -> Generator[Session, None, None]:
    """获取 PostgreSQL Session，可直接用于依赖注入。"""
    session = postgres_manager.get_session()
    try:
        yield session
    finally:
        session.close()
