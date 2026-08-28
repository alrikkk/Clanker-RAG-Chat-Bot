import pytest
import numpy as np
from src.vector_store import InMemoryVectorStore


def test_in_memory_vector_store_cosine_similarity():
    store = InMemoryVectorStore()
    chunks = [
        {"source": "doc1.md", "section": "Intro", "text": "Hello world"},
        {"source": "doc2.md", "section": "Pricing", "text": "Pro plan pricing"},
        {"source": "doc3.md", "section": "Sync", "text": "Syncing note offline"}
    ]
    # Synthetic 3D embeddings
    # Vector 0: [1, 0, 0]
    # Vector 1: [0, 1, 0]
    # Vector 2: [0, 0, 1]
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    store.add_chunks(chunks, embeddings)

    # Query aligned with Vector 1 [0, 1, 0]
    query_vec = np.array([0.0, 1.0, 0.0])
    results = store.search(query_vec, top_k=2)

    assert len(results) == 2
    assert results[0]["source"] == "doc2.md"
    assert pytest.approx(results[0]["similarity"], 0.01) == 1.0
    assert pytest.approx(results[1]["similarity"], 0.01) == 0.0


def test_empty_vector_store():
    store = InMemoryVectorStore()
    results = store.search(np.array([1.0, 0.0, 0.0]))
    assert results == []
