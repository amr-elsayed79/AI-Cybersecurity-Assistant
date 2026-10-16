"""Embedding service wrapping SentenceTransformers with multilingual support."""

from typing import List, Optional
from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingService:
    """Manages sentence embedding generation using a multilingual model."""

    _instance: Optional["EmbeddingService"] = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.device = device or settings.EMBEDDING_DEVICE
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
        return self._model

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        return settings.EMBEDDING_DIMENSION

    def embed_text(self, text: str) -> List[float]:
        """Generate an embedding vector for a single string.

        Raises:
            ValueError: if text is empty or only whitespace.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed an empty or whitespace-only text.")
        vector = self.model.encode(text.strip(), convert_to_numpy=True)
        return vector.tolist()

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> List[List[float]]:
        """Generate embedding vectors for a batch of strings."""
        if not texts:
            return []
        cleaned_texts = [t.strip() if t.strip() else " " for t in texts]
        vectors = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]


# Module-level shared singleton for reuse across requests
embedding_service = EmbeddingService()
