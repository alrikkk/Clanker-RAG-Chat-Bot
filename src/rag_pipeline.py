from pathlib import Path
from typing import List, Dict, Any, Optional
from src.config import DOCS_DIR, RELEVANCE_THRESHOLD, TOP_K
from src.chunker import load_and_chunk_documents
from src.embeddings import EmbeddingModel
from src.vector_store import InMemoryVectorStore
from src.generator import AnswerGenerator
from src.intent_router import is_casual_conversation, contextualize_query_with_history


class RAGPipeline:
    """
    Intelligent Dual-Path RAG & Conversational Pipeline:
    1. Casual Conversation Mode: greetings, pleasantries, jokes, banter -> natural conversational AI
    2. Knowledge / RAG Mode: document questions -> retrieval, threshold gating, grounded answer, citations
    3. Calm Out-of-Scope Mode: external factual questions -> clear refusal without hallucinations
    """
    def __init__(
        self,
        docs_dir: Path = DOCS_DIR,
        threshold: float = RELEVANCE_THRESHOLD,
        top_k: int = TOP_K,
        embedding_model: Optional[EmbeddingModel] = None,
        generator: Optional[AnswerGenerator] = None
    ):
        self.docs_dir = Path(docs_dir)
        self.threshold = threshold
        self.top_k = top_k
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = InMemoryVectorStore()
        self.generator = generator or AnswerGenerator()
        self.indexed = False
        self.chunks: List[Dict[str, Any]] = []

    def initialize(self):
        """Loads, chunks, embeds, and indexes the document corpus in memory."""
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Documentation directory not found at {self.docs_dir}")

        # Check for precomputed static embedding cache for ultra-fast startup
        cache_file = self.docs_dir / "embedded_chunks.json"
        if cache_file.exists():
            try:
                import json
                import numpy as np
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached:
                    self.chunks = []
                    vectors = []
                    for item in cached:
                        chunk_dict = {k: v for k, v in item.items() if k != "embedding"}
                        self.chunks.append(chunk_dict)
                        vectors.append(item["embedding"])
                    self.vector_store.add_chunks(self.chunks, np.array(vectors, dtype=np.float32))
                    self.indexed = True
                    print(f"[RAGPipeline] Loaded {len(self.chunks)} precomputed embedded chunks from cache.")
                    return
            except Exception as err:
                print(f"[RAGPipeline] Note: Could not load cache ({err}), re-embedding...")

        print(f"[RAGPipeline] Loading documents from {self.docs_dir}...")
        self.chunks = load_and_chunk_documents(self.docs_dir)

        if not self.chunks:
            print("[RAGPipeline] Warning: No document chunks were loaded.")
            self.indexed = True
            return

        texts = [chunk["text"] for chunk in self.chunks]
        print(f"[RAGPipeline] Generating local embeddings for {len(texts)} chunks...")
        chunk_embeddings = self.embedding_model.encode(texts, normalize=True)

        self.vector_store.add_chunks(self.chunks, chunk_embeddings)
        self.indexed = True
        print(f"[RAGPipeline] Successfully indexed {len(self.chunks)} chunks in memory.")

    def query(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Executes query routing and processing:
        - Casual Mode: natural AI small talk without triggering retrieval cards
        - RAG Mode: retrieval -> similarity ranking -> threshold check -> grounded generation + citations
        - Out-of-Scope Mode: calm refusal without hallucinations
        """
        if not self.indexed:
            self.initialize()

        cleaned_question = question.strip()
        if not cleaned_question:
            return {
                "question": question,
                "answer": "Please write a question or message in the notebook.",
                "mode": "casual",
                "supported": True,
                "top_similarity": 0.0,
                "threshold": self.threshold,
                "citations": [],
                "retrieved_count": 0
            }

        # Path 1: Check for Casual Conversation Mode
        if is_casual_conversation(cleaned_question):
            casual_answer = self.generator.generate_casual_response(cleaned_question, history=history)
            return {
                "question": question,
                "answer": casual_answer,
                "mode": "casual",
                "supported": True,
                "top_similarity": 0.0,
                "threshold": self.threshold,
                "citations": [],
                "retrieved_count": 0
            }

        # Path 2: Document / Factual Inquiries -> Perform Retrieval
        # Contextualize follow-up questions with conversation history if applicable
        search_query = contextualize_query_with_history(cleaned_question, history=history)
        
        # Local query embedding
        query_vec = self.embedding_model.encode_query(search_query)

        # In-memory cosine similarity search
        candidates = self.vector_store.search(query_vec, top_k=self.top_k)

        top_similarity = candidates[0].get("similarity", 0.0) if candidates else 0.0

        # Check against relevance threshold
        if top_similarity < self.threshold:
            # Out-of-scope factual query
            return {
                "question": question,
                "answer": "I couldn't find enough information to answer that from the provided documents.",
                "mode": "unsupported",
                "supported": False,
                "top_similarity": top_similarity,
                "threshold": self.threshold,
                "citations": [],
                "retrieved_count": 0
            }

        # Filter chunks that meet the relevance threshold
        relevant_chunks = [c for c in candidates if c.get("similarity", 0.0) >= self.threshold]

        # Generate strictly grounded answer with subtle tone adaptation
        answer = self.generator.generate_grounded_answer(relevant_chunks, cleaned_question, history=history)

        # Build citations
        citations = []
        for chunk in relevant_chunks:
            citations.append({
                "source": chunk.get("source", "unknown"),
                "doc_title": chunk.get("doc_title", "NimbusNote Docs"),
                "section": chunk.get("section", "Overview"),
                "similarity": chunk.get("similarity", 0.0),
                "chunk_index": chunk.get("chunk_index", 0),
                "passage": chunk.get("text", "")
            })

        return {
            "question": question,
            "answer": answer,
            "mode": "rag",
            "supported": True,
            "top_similarity": top_similarity,
            "threshold": self.threshold,
            "citations": citations,
            "retrieved_count": len(citations)
        }
