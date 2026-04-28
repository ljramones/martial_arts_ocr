"""Explicit database context for isolated runtime and test databases."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.pool import StaticPool


@dataclass(frozen=True)
class DatabaseConfig:
    """Database location for a runtime context."""

    database_path: Path | None = None
    database_url: str | None = None
    echo: bool = False

    @classmethod
    def from_data_dir(cls, data_dir: str | Path, filename: str = "martial_arts_ocr.db") -> "DatabaseConfig":
        return cls(database_path=Path(data_dir) / filename)

    @classmethod
    def from_url(cls, database_url: str, echo: bool = False) -> "DatabaseConfig":
        return cls(database_url=database_url, echo=echo)

    @property
    def url(self) -> str:
        if self.database_url:
            return self.database_url
        if self.database_path is None:
            return "sqlite:///:memory:"
        return f"sqlite:///{self.database_path}"


class DatabaseContext:
    """Owns a SQLAlchemy engine and session factory for one database."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        if self.config.database_path is not None:
            self.config.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = self._create_engine(self.config.url, echo=self.config.echo)
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            future=True,
        )
        self.Session = scoped_session(self.SessionLocal)

    @staticmethod
    def _create_engine(database_url: str, echo: bool = False):
        engine = create_engine(
            database_url,
            echo=echo,
            poolclass=StaticPool,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False, "timeout": 30},
        )

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            if database_url != "sqlite:///:memory:":
                cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.close()

        return engine

    def init_db(self) -> None:
        from martial_arts_ocr.db.database import Base
        from martial_arts_ocr.db import models  # noqa: F401

        Base.metadata.create_all(bind=self.engine)

    def drop_all_tables(self) -> None:
        from martial_arts_ocr.db.database import Base

        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def get_db_session(self) -> Iterator:
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_session(self):
        return self.SessionLocal()
