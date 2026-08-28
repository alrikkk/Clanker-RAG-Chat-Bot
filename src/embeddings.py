import os
import json
import urllib.request
import numpy as np
from typing import List, Union, Optional
from src.config import EMBEDDING_MODEL_NAME


class EmbeddingModel:
    """
    Dual-mode embedding generator:
    1. Local mode: uses sentence-transformers (all-MiniLM-L6-v2) if installed (CPU/MPS).
    2. Serverless/Vercel mode: uses fast inference or lightweight projection when
       running in size-constrained serverless environments (avoiding 5GB PyTorch bundle).
    """
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._local_model: Optional[Any] = None
        self._local_available: Optional[bool] = None

    def _get_local_model(self):
        if self._local_available is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._local_model = SentenceTransformer(self.model_name)
                self._local_available = True
            except Exception:
                self._local_model = None
                self._local_available = False
        return self._local_model

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        local_model = self._get_local_model()
        if local_model is not None:
            return local_model.encode(
                texts,
                show_progress_bar=False,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )

        # Batch encode fallback using inference
        results = [self.encode_query(t) for t in texts]
        return np.array(results, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encodes a single query string into a 384-dimensional dense vector.
        """
        local_model = self._get_local_model()
        if local_model is not None:
            emb = local_model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
            return emb

        # In serverless environments, query HuggingFace free inference endpoint for all-MiniLM-L6-v2
        try:
            api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/{self.model_name}"
            payload = json.dumps({"inputs": query, "options": {"wait_for_model": True}}).encode("utf-8")
            req = urllib.request.Request(
                api_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=6) as response:
                raw_data = json.loads(response.read().decode("utf-8"))
                if isinstance(raw_data, list):
                    arr = np.array(raw_data, dtype=np.float32).flatten()
                    norm = np.linalg.norm(arr)
                    return arr / norm if norm > 0 else arr
        except Exception as err:
            print(f"[EmbeddingModel] Serverless remote inference failed ({err}), using lexical projection.")

        # Pure python deterministic lexical vector projection (fallback)
        return self._lexical_embedding_projection(query)

    def _lexical_embedding_projection(self, query: str) -> np.ndarray:
        """
        Deterministic 384-d pseudo-vector based on query token hashing for extreme fallback.
        """
        vec = np.zeros(384, dtype=np.float32)
        words = query.lower().split()
        for w in words:
            h = hash(w)
            idx = abs(h) % 384
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
