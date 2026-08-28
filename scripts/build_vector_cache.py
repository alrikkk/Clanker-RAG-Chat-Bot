import json
from pathlib import Path
from src.config import DOCS_DIR
from src.chunker import load_and_chunk_documents
from src.embeddings import EmbeddingModel


def build_cache():
    print("Building static vector cache for NimbusNote chunks...")
    chunks = load_and_chunk_documents(DOCS_DIR)
    embedder = EmbeddingModel()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, normalize=True)

    cache_data = []
    for chunk, vec in zip(chunks, embeddings):
        item = dict(chunk)
        item["embedding"] = vec.tolist()
        cache_data.append(item)

    out_file = DOCS_DIR / "embedded_chunks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2)

    print(f"Successfully cached {len(cache_data)} embedded chunks to {out_file}")


if __name__ == "__main__":
    build_cache()
