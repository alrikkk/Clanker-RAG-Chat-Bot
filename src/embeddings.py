import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME


class EmbeddingModel:
    """
    Local embedding generator using sentence-transformers (all-MiniLM-L6-v2).
    Generates 384-dimensional dense vectors locally without external API keys.
    """
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            # Lazy load model on first usage
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Generates embeddings for a single text or list of texts.
        If normalize=True, outputs unit-length vectors (L2 norm = 1.0),
        allowing cosine similarity to be computed via dot product.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes a single query string into a 1D numpy array."""
        embeddings = self.encode(query, normalize=True)
        return embeddings[0]
