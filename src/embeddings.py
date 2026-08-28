import numpy as np
from typing import List, Union, Optional
from src.config import EMBEDDING_MODEL_NAME


class EmbeddingModel:
    """
    High-performance, lightweight embedding generator for both local and serverless deployment.
    Uses ONNX-powered fastembed or sentence-transformers to generate 384-dimensional dense vectors
    without requiring heavy 5GB PyTorch wheels.
    """
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            # 1. Try fastembed (ONNX runtime, lightweight, ideal for serverless & local)
            try:
                from fastembed import TextEmbedding
                fastembed_name = "sentence-transformers/all-MiniLM-L6-v2"
                self._model = ("fastembed", TextEmbedding(model_name=fastembed_name))
                return self._model
            except Exception as err:
                print(f"[EmbeddingModel] fastembed unavailable ({err}), trying sentence-transformers...")

            # 2. Try sentence-transformers (PyTorch local)
            try:
                from sentence_transformers import SentenceTransformer
                self._model = ("sentence_transformers", SentenceTransformer(self.model_name))
                return self._model
            except Exception as err:
                print(f"[EmbeddingModel] sentence-transformers unavailable ({err})")
                self._model = ("none", None)

        return self._model

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        backend, model = self._get_model()
        if backend == "fastembed" and model is not None:
            embeddings = list(model.embed(texts))
            arr = np.array(embeddings, dtype=np.float32)
            if normalize:
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                arr = arr / norms
            return arr
        elif backend == "sentence_transformers" and model is not None:
            return model.encode(
                texts,
                show_progress_bar=False,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
        else:
            raise RuntimeError("No embedding backend (fastembed or sentence-transformers) is available.")

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes a single query string into a normalized 1D numpy array."""
        return self.encode(query, normalize=True)[0]
