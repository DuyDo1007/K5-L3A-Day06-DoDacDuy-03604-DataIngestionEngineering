"""Local multilingual sentence-transformer embeddings."""
import os
from typing import Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MultilingualEmbedder:
    """Real local embedding model; model-loading failures are surfaced to the caller."""

    def __init__(self, model_name: str, revision: str | None = None, max_tokens: int | None = None) -> None:
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        model_path = snapshot_download(
            model_name,
            revision=revision,
            local_files_only=os.getenv("HF_HUB_OFFLINE") == "1",
            allow_patterns=["*.json", "*.safetensors", "tokenizer.*", "*.txt", "*.model"],
        )
        self._model = SentenceTransformer(model_path, local_files_only=True)
        self.revision = revision
        if max_tokens is not None:
            architecture_limit = self._model[0].auto_model.config.max_position_embeddings
            if not 0 < max_tokens <= architecture_limit:
                raise ValueError(f"Invalid token limit {max_tokens}; architecture limit is {architecture_limit}.")
            self._model.max_seq_length = max_tokens
            self._model.tokenizer.model_max_length = max_tokens
        self.max_tokens = self._model.max_seq_length
        dimensions = self._model.get_sentence_embedding_dimension()
        if dimensions is None or dimensions <= 0:
            raise RuntimeError(f"Embedding model returned invalid dimensions: {dimensions}")
        self.dimensions = dimensions

    def token_counts(self, texts: list[str]) -> list[int]:
        return self._model.tokenizer(texts, truncation=False, return_length=True)["length"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 1,
        )
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding provider returned a different number of vectors than inputs.")
        return vectors.tolist()

    def embed_query(self, question: str) -> list[float]:
        return self.embed_documents([question])[0]
