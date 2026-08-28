import numpy as np
from typing import List, Dict, Any, Tuple


class InMemoryVectorStore:
    """
    A lightweight, transparent in-memory vector store that stores chunk metadata
    alongside embedding vectors and performs cosine similarity search.
    """
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Indexes chunks and their corresponding embedding vectors.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Number of chunks ({len(chunks)}) must match number of embeddings ({len(embeddings)})")
            
        self.chunks = list(chunks)
        self.embeddings = np.array(embeddings, dtype=np.float32)

    def is_empty(self) -> bool:
        return len(self.chunks) == 0 or self.embeddings is None

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Computes cosine similarity between the query embedding and all indexed chunks,
        then returns the top_k most similar chunks sorted in descending order of similarity.
        
        Returns a list of dicts containing chunk metadata + 'similarity': float (0.0 to 1.0).
        """
        if self.is_empty():
            return []

        # Ensure query is 1D float32 array
        q_vec = np.array(query_embedding, dtype=np.float32).flatten()
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_unit = q_vec / q_norm

        # Normalize stored chunk embeddings along axis 1 if not already normalized
        doc_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        doc_norms = np.where(doc_norms == 0, 1e-10, doc_norms)
        doc_units = self.embeddings / doc_norms

        # Cosine similarity is the dot product of normalized vectors: u . v / (||u|| * ||v||)
        similarities = np.dot(doc_units, q_unit)

        # Sort indices in descending order of similarity
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk_data = dict(self.chunks[idx])
            chunk_data["similarity"] = float(round(similarities[idx], 4))
            results.append(chunk_data)

        return results
