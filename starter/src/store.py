"""Persistent local Qdrant vector store with cosine search and payload filters."""
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models
from .models import Chunk


class QdrantVectorStore:
    def __init__(self, path: Path, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))
        self.dimensions = dimensions

    @staticmethod
    def _collection_name(strategy: str) -> str:
        if strategy not in {"A", "B", "C"}:
            raise ValueError(f"Unknown strategy: {strategy!r}")
        return f"hust_{strategy}"

    def create_collection(self, strategy: str) -> None:
        collection_name = self._collection_name(strategy)
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimensions,
                    distance=models.Distance.COSINE,
                ),
            )

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one embedding vector.")
        self.create_collection(strategy)
        collection_name = self._collection_name(strategy)
        if any(len(vector) != self.dimensions for vector in vectors):
            raise ValueError(f"Embedding dimension does not match configured size {self.dimensions}.")

        existing_ids: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=["chunk_id"],
                with_vectors=False,
            )
            existing_ids.update(str(point.payload["chunk_id"]) for point in points if point.payload)
            if offset is None:
                break

        new_ids = {chunk.chunk_id for chunk in chunks}
        stale_ids = existing_ids - new_ids
        if stale_ids:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="chunk_id",
                                match=models.MatchAny(any=list(stale_ids)),
                            )
                        ]
                    )
                ),
            )

        points = []
        for chunk, vector in zip(chunks, vectors):
            if any(not isinstance(value, (int, float)) for value in vector):
                raise ValueError(f"Non-numeric embedding value for chunk {chunk.chunk_id}.")
            payload = {
                **chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source": chunk.source,
                "page": str(chunk.page),
                "text": chunk.text,
            }
            points.append(
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, f"{strategy}:{chunk.chunk_id}")),
                    vector=vector,
                    payload=payload,
                )
            )
        if points:
            self.client.upsert(collection_name=collection_name, points=points, wait=True)

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if len(query_vector) != self.dimensions:
            raise ValueError(f"Query dimension does not match configured size {self.dimensions}.")
        query_filter = None
        if filters:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                    for key, value in filters.items()
                ]
            )
        result = self.client.query_points(
            collection_name=self._collection_name(strategy),
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        hits: list[dict[str, Any]] = []
        for point in result.points:
            payload = dict(point.payload or {})
            text = payload.pop("text", "")
            chunk_id = payload.pop("chunk_id", str(point.id))
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "score": float(point.score),
                    "metadata": payload,
                }
            )
        return hits

    def close(self) -> None:
        self.client.close()
