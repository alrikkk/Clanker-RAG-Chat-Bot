import pytest
from pathlib import Path
from src.chunker import parse_markdown_document, load_and_chunk_documents
from src.config import DOCS_DIR


def test_chunker_loads_all_three_documents():
    chunks = load_and_chunk_documents(DOCS_DIR)
    assert len(chunks) > 0

    sources = {c["source"] for c in chunks}
    assert "01-getting-started.md" in sources
    assert "02-pricing-and-plans.md" in sources
    assert "03-troubleshooting.md" in sources


def test_chunk_metadata_preservation():
    chunks = load_and_chunk_documents(DOCS_DIR)
    for chunk in chunks:
        assert "source" in chunk and chunk["source"].endswith(".md")
        assert "doc_title" in chunk and len(chunk["doc_title"]) > 0
        assert "section" in chunk and len(chunk["section"]) > 0
        assert "chunk_index" in chunk and isinstance(chunk["chunk_index"], int)
        assert "text" in chunk and len(chunk["text"].strip()) > 0


def test_sync_behavior_chunk_exists():
    chunks = load_and_chunk_documents(DOCS_DIR)
    sync_chunks = [c for c in chunks if c["section"] == "Sync behavior"]
    assert len(sync_chunks) >= 1
    assert "15 seconds" in sync_chunks[0]["text"]
    assert sync_chunks[0]["source"] == "01-getting-started.md"
