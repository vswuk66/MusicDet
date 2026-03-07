import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Column, DateTime, String, create_engine, select, text
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


def _is_enabled(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "128"))


class TrackEmbedding(Base):
    __tablename__ = "track_embeddings"

    id = Column(String(36), primary_key=True, index=True)
    track_name = Column(String(255), nullable=False, index=True)
    genre = Column(String(64), nullable=True, index=True)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)
    extra_metadata = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, index=True)


class VectorStoreService:
    def __init__(self) -> None:
        self.enabled = _is_enabled(os.getenv("ENABLE_VECTOR_DB", "false"))
        self.database_url = os.getenv("DATABASE_URL", "").strip()
        self.embedding_dim = EMBEDDING_DIM
        self.ready = False
        self.last_error: Optional[str] = None

        self._engine = None
        self._session_factory = None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return

        self._initialized = True
        if not self.enabled:
            logger.info("Vector DB is disabled (ENABLE_VECTOR_DB=false).")
            return

        if not self.database_url:
            self.last_error = "DATABASE_URL is empty while ENABLE_VECTOR_DB=true."
            logger.error(self.last_error)
            return

        try:
            self._engine = create_engine(self.database_url, pool_pre_ping=True, future=True)
            self._session_factory = sessionmaker(
                bind=self._engine, autoflush=False, autocommit=False, expire_on_commit=False
            )

            with self._engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            Base.metadata.create_all(self._engine)
            self.ready = True
            logger.info("Vector DB initialized successfully.")
        except Exception as exc:
            self.ready = False
            self.last_error = str(exc)
            logger.error("Failed to initialize vector DB: %s", exc, exc_info=True)

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ready": self.ready,
            "embedding_dim": self.embedding_dim,
            "error": self.last_error,
        }

    def _ensure_ready(self) -> None:
        if not self.enabled:
            raise RuntimeError("Vector DB is disabled.")
        if not self.ready or self._session_factory is None:
            raise RuntimeError(f"Vector DB is not ready. Last error: {self.last_error}")

    def _validate_embedding(self, embedding: List[float]) -> None:
        if len(embedding) != self.embedding_dim:
            raise ValueError(
                f"Embedding must have exactly {self.embedding_dim} values, got {len(embedding)}."
            )

    def add_embedding(
        self,
        track_name: str,
        embedding: List[float],
        genre: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._ensure_ready()
        self._validate_embedding(embedding)

        if self._session_factory is None:
            raise RuntimeError("Vector DB session factory is not initialized.")

        embedding_record = TrackEmbedding(
            id=str(uuid.uuid4()),
            track_name=track_name,
            genre=genre,
            embedding=[float(value) for value in embedding],
            extra_metadata=metadata or {},
            created_at=datetime.now(timezone.utc),
        )

        with self._session_factory() as session:
            session.add(embedding_record)
            session.commit()

        return {
            "id": embedding_record.id,
            "track_name": embedding_record.track_name,
            "genre": embedding_record.genre,
            "metadata": embedding_record.extra_metadata,
            "created_at": embedding_record.created_at.isoformat(),
        }

    def search_similar(
        self, query_embedding: List[float], top_k: int = 5, genre: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        self._ensure_ready()
        self._validate_embedding(query_embedding)

        if self._session_factory is None:
            raise RuntimeError("Vector DB session factory is not initialized.")

        top_k = max(1, min(top_k, 50))
        distance_expr = TrackEmbedding.embedding.cosine_distance(query_embedding).label("distance")
        statement = (
            select(TrackEmbedding, distance_expr).order_by(distance_expr.asc()).limit(top_k)
        )
        if genre:
            statement = statement.where(TrackEmbedding.genre == genre)

        with self._session_factory() as session:
            rows = session.execute(statement).all()

        results: List[Dict[str, Any]] = []
        for embedding_record, distance in rows:
            similarity_score = max(0.0, 1.0 - float(distance))
            results.append(
                {
                    "id": embedding_record.id,
                    "track_name": embedding_record.track_name,
                    "genre": embedding_record.genre,
                    "metadata": embedding_record.extra_metadata,
                    "distance": float(distance),
                    "similarity": similarity_score,
                    "created_at": embedding_record.created_at.isoformat(),
                }
            )
        return results


vector_store_service = VectorStoreService()
