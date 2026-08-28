from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from src.config import BASE_DIR, HOST, PORT, EMBEDDING_MODEL_NAME, RELEVANCE_THRESHOLD
from src.rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize RAG pipeline and pre-index documents at startup
    if not pipeline.indexed:
        pipeline.initialize()
    yield


app = FastAPI(
    title="Clanker — RAG Mini Q&A Bot",
    description="Skeuomorphic notebook-powered RAG assistant for NimbusNote documentation.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageItem(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: Optional[str] = Field(default=None, description="The user's question.")
    message: Optional[str] = Field(default=None, description="Alternative alias for question.")
    history: Optional[List[MessageItem]] = Field(default_factory=list, description="Prior conversation turns.")
    conversation_history: Optional[List[MessageItem]] = Field(default_factory=list, description="Alternative alias for history.")


class CitationItem(BaseModel):
    source: str
    doc_title: str
    section: str
    similarity: float
    chunk_index: int
    passage: str


class QueryResponse(BaseModel):
    question: str
    message: Optional[str] = None
    answer: str
    response: Optional[str] = None
    mode: str = "rag"  # "rag" | "general" | "unsupported"
    supported: bool
    top_similarity: float
    threshold: float
    citations: List[CitationItem]
    sources: Optional[List[CitationItem]] = None
    retrieved_count: int


@app.post("/api/query", response_model=QueryResponse)
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Submits a query to Clanker, intelligently routing between General AI
    and RAG Knowledge Mode with Groq LLM generation and source citations.
    """
    user_input = (request.question or request.message or "").strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Missing question or message in request body.")

    try:
        raw_history = [{"role": m.role, "content": m.content} for m in (request.history or request.conversation_history or [])]
        result = pipeline.query(user_input, history=raw_history)
        result["message"] = user_input
        result["response"] = result["answer"]
        result["sources"] = result["citations"]
        return result
    except Exception as e:
        print(f"[clanker] Query processing exception: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/api/health")
@app.get("/health")
async def health_endpoint():
    """Returns application health and indexing metadata."""
    if not pipeline.indexed:
        pipeline.initialize()
    return {
        "status": "healthy",
        "app_name": "Clanker",
        "tagline": "Your tiny notebook-powered RAG assistant.",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "indexed_chunks": len(pipeline.chunks),
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "is_indexed": pipeline.indexed
    }


@app.get("/api/documents")
@app.get("/documents")
async def documents_endpoint():
    """Returns the list of indexed documents and their section breakdown."""
    if not pipeline.indexed:
        pipeline.initialize()
    docs_summary: Dict[str, Any] = {}
    for chunk in pipeline.chunks:
        source = chunk["source"]
        if source not in docs_summary:
            docs_summary[source] = {
                "source": source,
                "doc_title": chunk.get("doc_title", source),
                "sections": set(),
                "chunk_count": 0
            }
        docs_summary[source]["sections"].add(chunk.get("section", "Overview"))
        docs_summary[source]["chunk_count"] += 1

    # Convert sets to sorted lists for JSON serialization
    serialized = []
    for doc in docs_summary.values():
        serialized.append({
            "source": doc["source"],
            "doc_title": doc["doc_title"],
            "sections": sorted(list(doc["sections"])),
            "chunk_count": doc["chunk_count"]
        })

    return {"documents": serialized, "total_chunks": len(pipeline.chunks)}


# Mount static directory for frontend
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host=HOST, port=PORT, reload=True)
