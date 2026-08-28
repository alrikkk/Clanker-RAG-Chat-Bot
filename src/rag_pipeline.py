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
    contextualize_query_with_history
)


class RAGPipeline:
    """
    Intelligent Clanker Brain Architecture:
    The LLM is Clanker's brain and ALWAYS generates the final response.
    RAG is an authoritative knowledge source consulted when NimbusNote information is needed.
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

        # Check for precomputed static embedding cache for instant startup
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
                    print(f"[clanker] Loaded {len(self.chunks)} precomputed embedded chunks from cache.")
                    return
            except Exception as err:
                print(f"[clanker] Note: Could not load cache ({err}), re-embedding...")

        print(f"[clanker] Loading documents from {self.docs_dir}...")
        self.chunks = load_and_chunk_documents(self.docs_dir)

        if not self.chunks:
            print("[clanker] Warning: No document chunks were loaded.")
            self.indexed = True
            return

        texts = [chunk["text"] for chunk in self.chunks]
        print(f"[clanker] Generating local embeddings for {len(texts)} chunks...")
        chunk_embeddings = self.embedding_model.encode(texts, normalize=True)

        self.vector_store.add_chunks(self.chunks, chunk_embeddings)
        self.indexed = True
        print(f"[clanker] Successfully indexed {len(self.chunks)} chunks in memory.")

    def query(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Processes user query through Clanker:
        1. Contextualize query with recent conversation history.
        2. Determine if NimbusNote knowledge is relevant.
        3. If relevant, retrieve chunks and pass to LLM as authoritative context.
        4. If not, pass question directly to LLM.
        5. LLM always produces the final response.
        """
        if not self.indexed:
            self.initialize()

        cleaned_question = question.strip()
        if not cleaned_question:
            return {
                "question": question,
                "answer": "Please write a question or message in the notebook.",
                "mode": "general",
                "supported": True,
                "top_similarity": 0.0,
                "threshold": self.threshold,
                "citations": [],
                "retrieved_count": 0
            }

        print(f"[clanker] Received message: '{cleaned_question}'")

        # Check for explicit NimbusNote keywords or follow-up in context
        is_nimbus_inquiry = is_explicit_nimbus_inquiry(cleaned_question)
        is_follow_up = is_follow_up_with_context(cleaned_question, history=history)

        # Contextualize query for semantic vector search
        search_query = contextualize_query_with_history(cleaned_question, history=history)
        query_vec = self.embedding_model.encode_query(search_query)
        candidates = self.vector_store.search(query_vec, top_k=self.top_k)
        top_similarity = candidates[0].get("similarity", 0.0) if candidates else 0.0

        # Check for explicit unsupported NimbusNote feature inquiries
        UNSUPPORTED_NIMBUS_FEATURES = {
            "voice", "audio", "video", "recording", "record", "mic", "microphone", "call",
            "screen share", "latex", "pdf export", "kanban", "calendar", "reminder",
            "phone support", "ocr", "drawing tool", "quantum", "bluetooth"
        }
        q_lower = cleaned_question.lower()
        if ("nimbus" in q_lower or "nimbusnote" in q_lower or "notebook" in q_lower) and any(f in q_lower for f in UNSUPPORTED_NIMBUS_FEATURES):
            print(f"[clanker] Response mode: Unsupported NimbusNote query (unsupported feature)")
            llm_answer = self.generator.generate_response(
                cleaned_question,
                retrieved_chunks=None,
                history=history,
                unsupported_note=True
            )
            return {
                "question": question,
                "answer": llm_answer,
                "mode": "unsupported",
                "supported": False,
                "top_similarity": 0.25,
                "threshold": self.threshold,
                "citations": [],
                "retrieved_count": 0
            }

        # RAG Decision:
        # If it's explicitly about NimbusNote or a follow-up, OR has strong semantic match >= 0.50
        requires_nimbus_rag = (is_nimbus_inquiry or is_follow_up) and (top_similarity >= self.threshold)
        is_strong_semantic_rag = (top_similarity >= 0.50)

        if requires_nimbus_rag or is_strong_semantic_rag:
            # 1. RAG KNOWLEDGE MODE — Consult the notebook and pass to LLM
            relevant_chunks = [c for c in candidates if c.get("similarity", 0.0) >= self.threshold]
            print(f"[clanker] Response mode: RAG (retrieved {len(relevant_chunks)} chunks, top_sim={top_similarity:.3f})")

            llm_answer = self.generator.generate_response(
                cleaned_question,
                retrieved_chunks=relevant_chunks,
                history=history
            )

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
                "answer": llm_answer,
                "mode": "rag",
                "supported": True,
                "top_similarity": top_similarity,
                "threshold": self.threshold,
                "citations": citations,
                "retrieved_count": len(citations)
            }

        elif is_nimbus_inquiry and top_similarity < self.threshold:
            # 2. UNSUPPORTED NIMBUSNOTE QUERY — LLM explains documentation does not contain feature
            print(f"[clanker] Response mode: Unsupported NimbusNote query (top_sim={top_similarity:.3f} < {self.threshold})")

            llm_answer = self.generator.generate_response(
                cleaned_question,
                retrieved_chunks=None,
                history=history,
                unsupported_note=True
            )

            return {
                "question": question,
                "answer": llm_answer,
                "mode": "unsupported",
                "supported": False,
                "top_similarity": top_similarity,
                "threshold": self.threshold,
                "citations": [],
                "retrieved_count": 0
            }

        else:
            # 3. GENERAL AI CONVERSATION MODE — Normal LLM generation with no document card
            print(f"[clanker] Response mode: General AI (top_sim={top_similarity:.3f})")

            llm_answer = self.generator.generate_response(
                cleaned_question,
                retrieved_chunks=None,
                history=history,
                unsupported_note=False
            )

            return {
                "question": question,
                "answer": llm_answer,
                "mode": "general",
                "supported": True,
                "top_similarity": top_similarity,
                "threshold": self.threshold,
                "citations": [],
                "retrieved_count": 0
            }
