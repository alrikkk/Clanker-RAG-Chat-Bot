from pathlib import Path
from typing import List, Dict, Any, Optional
from src.config import DOCS_DIR, RELEVANCE_THRESHOLD, TOP_K
from src.chunker import load_and_chunk_documents
from src.embeddings import EmbeddingModel
from src.vector_store import InMemoryVectorStore
from src.generator import AnswerGenerator
from src.intent_router import (
    is_explicit_nimbus_inquiry,
    is_follow_up_with_context,
    is_ambiguous_without_context,
    is_live_data_query,
    contextualize_query_with_history
)


class RAGPipeline:
    """
    Intelligent Dual-Path RAG & General AI Conversational Pipeline:
    1. NimbusNote / Document RAG Mode: contextual retrieval -> cosine similarity -> grounded generation + citations.
    2. General AI & Conversational Mode: programming, math, science, poetry, banter, chit-chat -> natural AI responses.
    3. Unsupported NimbusNote Requests: clear refusal when asked for doc facts not present in corpus.
    4. Ambiguous / Live Data Queries: intelligent clarification or live-data disclaimers.
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
        Executes intelligent intent routing and RAG query processing:
        - Ambiguous query without context -> clarification response
        - Live real-time data inquiry -> live data disclaimer
        - NimbusNote knowledge inquiry -> vector retrieval -> threshold check -> grounded answer + citations
        - General AI / Casual / Creative / Tech concept inquiry -> general conversational AI answer
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

        # 1. Check for ambiguous query without context
        if is_ambiguous_without_context(cleaned_question, history=history):
            answer = self.generator.generate_general_response(cleaned_question, history=history)
            return {
                "question": question,
                "answer": answer,
                "mode": "casual",
                "supported": True,
                "top_similarity": 0.0,
                "threshold": self.threshold,
                "citations": [],
                "retrieved_count": 0
            }

        # 2. Check for real-time live data queries (e.g. weather in Chennai)
        if is_live_data_query(cleaned_question):
            answer = self.generator.generate_general_response(cleaned_question, history=history)
            return {
                "question": question,
                "answer": answer,
                "mode": "casual",
                "supported": True,
                "top_similarity": 0.0,
                "threshold": self.threshold,
                "citations": [],
                "retrieved_count": 0
            }

        # 3. Check for specific unsupported NimbusNote feature inquiries
        UNSUPPORTED_NIMBUS_FEATURES = {
            "voice", "audio", "video", "recording", "record", "mic", "microphone", "call",
            "screen share", "latex", "pdf export", "kanban", "calendar", "reminder",
            "phone support", "ocr", "drawing tool"
        }
        q_lower = cleaned_question.lower()
        if "nimbus" in q_lower or "nimbusnote" in q_lower or "notebook" in q_lower:
            if any(f in q_lower for f in UNSUPPORTED_NIMBUS_FEATURES):
                return {
                    "question": question,
                    "answer": "I couldn't find that in the NimbusNote documentation. The notebook covers workspace creation, Free/Pro/Team plans, sync intervals, and troubleshooting.",
                    "mode": "unsupported",
                    "supported": False,
                    "top_similarity": 0.25,
                    "threshold": self.threshold,
                    "citations": [],
                    "retrieved_count": 0
                }

        # 4. Perform semantic vector search on contextualized query
        is_nimbus_inquiry = is_explicit_nimbus_inquiry(cleaned_question)
        is_follow_up = is_follow_up_with_context(cleaned_question, history=history)
        
        search_query = contextualize_query_with_history(cleaned_question, history=history)
        query_vec = self.embedding_model.encode_query(search_query)
        candidates = self.vector_store.search(query_vec, top_k=self.top_k)
        top_similarity = candidates[0].get("similarity", 0.0) if candidates else 0.0

        # 5. Routing Decision:
        # If it's explicitly about NimbusNote or a contextual follow-up, OR similarity >= 0.40
        if (is_nimbus_inquiry or is_follow_up) and top_similarity >= self.threshold:
            # High-confidence RAG Match
            relevant_chunks = [c for c in candidates if c.get("similarity", 0.0) >= self.threshold]
            answer = self.generator.generate_grounded_answer(relevant_chunks, cleaned_question, history=history)
            citations = [{
                "source": c.get("source", "unknown"),
                "doc_title": c.get("doc_title", "NimbusNote Docs"),
                "section": c.get("section", "Overview"),
                "similarity": c.get("similarity", 0.0),
                "chunk_index": c.get("chunk_index", 0),
                "passage": c.get("text", "")
            } for c in relevant_chunks]

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

        elif is_nimbus_inquiry and top_similarity < self.threshold:
            # User specifically asked for a NimbusNote feature that is not in the docs
            return {
                "question": question,
                "answer": "I couldn't find that in the NimbusNote documentation. The notebook covers workspace creation, Free/Pro/Team plans, sync intervals, and troubleshooting.",
                "mode": "unsupported",
                "supported": False,
                "top_similarity": top_similarity,
                "threshold": self.threshold,
                "citations": [],
                "retrieved_count": 0
            }

        elif top_similarity >= 0.45:
            # Paraphrased query with strong semantic match against document chunks
            relevant_chunks = [c for c in candidates if c.get("similarity", 0.0) >= self.threshold]
            answer = self.generator.generate_grounded_answer(relevant_chunks, cleaned_question, history=history)
            citations = [{
                "source": c.get("source", "unknown"),
                "doc_title": c.get("doc_title", "NimbusNote Docs"),
                "section": c.get("section", "Overview"),
                "similarity": c.get("similarity", 0.0),
                "chunk_index": c.get("chunk_index", 0),
                "passage": c.get("text", "")
            } for c in relevant_chunks]

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

        else:
            # 5. General AI & Conversational Mode (programming, math, science, jokes, small talk, random text)
            answer = self.generator.generate_general_response(cleaned_question, history=history)
            return {
                "question": question,
                "answer": answer,
                "mode": "casual",
                "supported": True,
                "top_similarity": top_similarity,
                "threshold": self.threshold,
                "citations": [],
                "retrieved_count": 0
            }
