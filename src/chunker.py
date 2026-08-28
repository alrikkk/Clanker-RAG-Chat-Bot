import re
from pathlib import Path
from typing import List, Dict, Any


def parse_markdown_document(filepath: Path) -> List[Dict[str, Any]]:
    """
    Parses a markdown document into deterministic semantic chunks based on
    Markdown headings (# and ##) and paragraphs, preserving rich metadata.
    
    Returns a list of dicts:
    {
        "source": "01-getting-started.md",
        "doc_title": "NimbusNote — Getting Started",
        "section": "Sync behavior",
        "chunk_index": 3,
        "text": "NimbusNote syncs every 15 seconds while the app is in the foreground..."
    }
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Document file not found: {filepath}")
    
    content = filepath.read_text(encoding="utf-8")
    lines = content.splitlines()
    
    filename = filepath.name
    doc_title = filename
    current_section = "Overview"
    current_lines: List[str] = []
    chunks: List[Dict[str, Any]] = []
    chunk_counter = 0

    def flush_chunk(section_name: str, lines_to_flush: List[str]):
        nonlocal chunk_counter
        text_body = "\n".join(lines_to_flush).strip()
        if text_body:
            # Include section heading in chunk text for complete context and searchability
            if section_name and section_name != "Overview":
                full_text = f"## {section_name}\n\n{text_body}"
            else:
                full_text = text_body
                
            chunks.append({
                "source": filename,
                "doc_title": doc_title,
                "section": section_name,
                "chunk_index": chunk_counter,
                "text": full_text
            })
            chunk_counter += 1

    for line in lines:
        stripped = line.strip()
        
        # Top-level heading (# Title)
        if stripped.startswith("# ") and not stripped.startswith("## "):
            flush_chunk(current_section, current_lines)
            current_lines = []
            doc_title = stripped[2:].strip()
            current_section = "Overview"
            
        # Section-level heading (## Section)
        elif stripped.startswith("## "):
            flush_chunk(current_section, current_lines)
            current_lines = []
            # Strip quotes or punctuation from section headers if any (e.g. ## "My note didn't sync")
            current_section = stripped[3:].strip().strip('"').strip("'")
            
        else:
            current_lines.append(line)

    # Flush any remaining lines
    flush_chunk(current_section, current_lines)
    
    return chunks


def load_and_chunk_documents(docs_dir: Path) -> List[Dict[str, Any]]:
    """
    Loads all markdown documents from the given directory and chunks them deterministically.
    """
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")
        
    all_chunks: List[Dict[str, Any]] = []
    # Sort files deterministically
    md_files = sorted(list(docs_dir.glob("*.md")))
    
    for md_file in md_files:
        # Ignore readme files if any
        if md_file.name.lower() == "readme.md":
            continue
        file_chunks = parse_markdown_document(md_file)
        all_chunks.extend(file_chunks)
        
    # Re-index globally for unique global identification while preserving per-doc metadata
    for idx, chunk in enumerate(all_chunks):
        chunk["global_index"] = idx
        
    return all_chunks
