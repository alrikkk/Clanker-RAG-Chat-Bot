import pytest
from src.rag_pipeline import RAGPipeline
from src.config import RELEVANCE_THRESHOLD


@pytest.fixture(scope="module")
def pipeline():
    pipe = RAGPipeline()
    pipe.initialize()
    return pipe


# =========================================================================
# 1. RAG KNOWLEDGE MODE TESTS
# =========================================================================

def test_in_scope_sync_behavior_retrieval(pipeline: RAGPipeline):
    """
    Test 1: Standard in-scope document query
    'How often does NimbusNote sync while the app is in the foreground?'
    Must trigger RAG mode, retrieve 01-getting-started.md, and cite Sync behavior section.
    """
    question = "How often does NimbusNote sync while the app is in the foreground?"
    result = pipeline.query(question)

    assert result["mode"] == "rag"
    assert result["supported"] is True
    assert result["top_similarity"] >= pipeline.threshold
    assert len(result["citations"]) > 0

    top_citation = result["citations"][0]
    assert top_citation["source"] == "01-getting-started.md"
    assert top_citation["section"] == "Sync behavior"
    assert "15 seconds" in top_citation["passage"]
    assert "15 seconds" in result["answer"] or "foreground" in result["answer"]


def test_casual_rag_query_still_triggers_retrieval(pipeline: RAGPipeline):
    """
    Test 2: Casual phrasing for a document query
    'yo bro how often does NimbusNote sync?'
    Must STILL use RAG, return citations, and ground the answer.
    """
    question = "yo bro how often does NimbusNote sync?"
    result = pipeline.query(question)

    assert result["mode"] == "rag"
    assert result["supported"] is True
    assert len(result["citations"]) > 0
    assert result["citations"][0]["source"] == "01-getting-started.md"
    assert "15 seconds" in result["citations"][0]["passage"]


def test_citation_metadata_structure(pipeline: RAGPipeline):
    """
    Test 3: Citation metadata preservation
    Verifies all citations retain source, section, similarity, and passage text.
    """
    question = "What are the limitations of the Free plan?"
    result = pipeline.query(question)

    assert result["mode"] == "rag"
    assert result["supported"] is True
    assert len(result["citations"]) > 0

    for citation in result["citations"]:
        assert "source" in citation and citation["source"].endswith(".md")
        assert "section" in citation and len(citation["section"]) > 0
        assert "similarity" in citation and isinstance(citation["similarity"], float)
        assert citation["similarity"] >= pipeline.threshold
        assert "passage" in citation and len(citation["passage"]) > 0


def test_troubleshooting_image_upload_query(pipeline: RAGPipeline):
    """
    Test 4: Technical troubleshooting inquiry
    'Why can't I upload an image?'
    """
    question = "Why can't I upload an image?"
    result = pipeline.query(question)

    assert result["mode"] == "rag"
    assert result["supported"] is True
    assert any(c["source"] in ["03-troubleshooting.md", "01-getting-started.md", "02-pricing-and-plans.md"] for c in result["citations"])
    assert any("20MB" in c["passage"] or "Free plan" in c["passage"] for c in result["citations"])


# =========================================================================
# 2. GENERAL CONVERSATION MODE TESTS
# =========================================================================

def test_casual_greeting_no_retrieval(pipeline: RAGPipeline):
    """
    Test 5: Casual greeting
    'yo Clanker'
    Must respond in casual mode WITHOUT triggering retrieval cards or errors.
    """
    question = "yo Clanker"
    result = pipeline.query(question)

    assert result["mode"] == "casual"
    assert result["supported"] is True
    assert len(result["citations"]) == 0
    assert len(result["answer"]) > 0
    assert "not found" not in result["answer"].lower()


def test_casual_small_talk_and_gratitude(pipeline: RAGPipeline):
    """
    Test 6: Gratitude and small talk
    'lol thanks bro'
    """
    question = "lol thanks bro"
    result = pipeline.query(question)

    assert result["mode"] == "casual"
    assert result["supported"] is True
    assert len(result["citations"]) == 0
    assert len(result["answer"]) > 0


def test_casual_joke_request(pipeline: RAGPipeline):
    """
    Test 7: Joke request
    'tell me a joke'
    """
    question = "tell me a joke"
    result = pipeline.query(question)

    assert result["mode"] == "casual"
    assert result["supported"] is True
    assert len(result["citations"]) == 0
    assert len(result["answer"]) > 0


def test_identity_inquiry(pipeline: RAGPipeline):
    """
    Test 8: Bot identity inquiry
    'who are you?'
    """
    question = "who are you?"
    result = pipeline.query(question)

    assert result["mode"] == "casual"
    assert result["supported"] is True
    assert "Clanker" in result["answer"]


# =========================================================================
# 3. MULTI-TURN CONTEXT & FOLLOW-UP TESTS
# =========================================================================

def test_multi_turn_follow_up_retrieval(pipeline: RAGPipeline):
    """
    Test 9: Multi-turn conversation follow-up
    Turn 1: 'how much is Pro?'
    Turn 2: 'and how many collaborators?'
    Must resolve 'Pro' context and retrieve the Pro collaborator limit.
    """
    # Turn 1
    t1_q = "how much is Pro?"
    t1_res = pipeline.query(t1_q)
    assert t1_res["mode"] == "rag"
    assert any("$6" in c["passage"] for c in t1_res["citations"])

    # Turn 2 with history
    history = [
        {"role": "user", "content": t1_q},
        {"role": "assistant", "content": t1_res["answer"]}
    ]
    t2_q = "and how many collaborators?"
    t2_res = pipeline.query(t2_q, history=history)

    assert t2_res["mode"] == "rag"
    assert t2_res["supported"] is True
    assert len(t2_res["citations"]) > 0
    # Must retrieve Pro plan chunk with 20 collaborators
    assert any("20 collaborators" in c["passage"] for c in t2_res["citations"])


# =========================================================================
# 4. OUT-OF-SCOPE FACTUAL QUERY TESTS
# =========================================================================

def test_out_of_scope_unsupported_query(pipeline: RAGPipeline):
    """
    Test 10: Out-of-scope factual question
    'What is the weather in Chennai today?'
    Must fail threshold, return mode='unsupported', and refuse to hallucinate.
    """
    question = "What is the weather in Chennai today?"
    result = pipeline.query(question)

    assert result["mode"] == "unsupported"
    assert result["supported"] is False
    assert result["top_similarity"] < pipeline.threshold
    assert len(result["citations"]) == 0
    assert "couldn't find enough information" in result["answer"].lower()
