import pytest
from src.rag_pipeline import RAGPipeline


@pytest.fixture(scope="module")
def pipeline():
    pipe = RAGPipeline()
    pipe.initialize()
    return pipe


# =========================================================================
# 1. RAG KNOWLEDGE MODE TESTS
# =========================================================================

def test_in_scope_sync_behavior_retrieval(pipeline: RAGPipeline):
    """TEST: Standard in-scope document query (sync timing)."""
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
    """TEST: Casual phrasing for a document query ('yo bro how much is Pro?')."""
    question = "yo bro how much is Pro?"
    result = pipeline.query(question)

    assert result["mode"] == "rag"
    assert result["supported"] is True
    assert len(result["citations"]) > 0
    assert result["citations"][0]["source"] == "02-pricing-and-plans.md"
    assert any("$6" in c["passage"] or "Pro plan" in c["passage"] for c in result["citations"])


def test_citation_metadata_structure(pipeline: RAGPipeline):
    """TEST: Citation metadata structure and exact passage text."""
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
    """TEST: Troubleshooting image upload query."""
    question = "Why can't I upload an image?"
    result = pipeline.query(question)

    assert result["mode"] == "rag"
    assert result["supported"] is True
    assert any(c["source"] in ["03-troubleshooting.md", "01-getting-started.md", "02-pricing-and-plans.md"] for c in result["citations"])
    assert any("20MB" in c["passage"] or "Free plan" in c["passage"] for c in result["citations"])


# =========================================================================
# 2. GENERAL AI & CONVERSATIONAL MODE TESTS
# =========================================================================

def test_casual_greeting_no_retrieval(pipeline: RAGPipeline):
    """TEST A: Casual greeting ('yo Clanker')."""
    question = "yo Clanker"
    result = pipeline.query(question)

    assert result["mode"] == "casual"
    assert result["supported"] is True
    assert len(result["citations"]) == 0
    assert len(result["answer"]) > 0


def test_casual_joke_request(pipeline: RAGPipeline):
    """TEST B: Joke request ('tell me a joke')."""
    question = "tell me a joke"
    result = pipeline.query(question)

    assert result["mode"] == "casual"
    assert result["supported"] is True
    assert len(result["citations"]) == 0
    assert len(result["answer"]) > 0


def test_general_programming_explanation(pipeline: RAGPipeline):
    """TEST C: General programming concept ('explain recursion like I'm new to programming')."""
    question = "explain recursion like I'm new to programming"
    result = pipeline.query(question)

    assert result["mode"] == "casual"
    assert result["supported"] is True
    assert len(result["citations"]) == 0
    assert "recursion" in result["answer"].lower() or "base case" in result["answer"].lower()


def test_general_science_and_math(pipeline: RAGPipeline):
    """TEST: General science & math questions ('what is a black hole?', 'what is 2+2?')."""
    q1 = "what is a black hole?"
    r1 = pipeline.query(q1)
    assert r1["mode"] == "casual"
    assert "gravity" in r1["answer"].lower() or "light" in r1["answer"].lower()

    q2 = "what is 2+2?"
    r2 = pipeline.query(q2)
    assert r2["mode"] == "casual"
    assert "4" in r2["answer"]


def test_identity_inquiry(pipeline: RAGPipeline):
    """TEST: Identity inquiry ('who are you?')."""
    question = "who are you?"
    result = pipeline.query(question)

    assert result["mode"] == "casual"
    assert result["supported"] is True
    assert "Clanker" in result["answer"]


def test_creative_story_or_poem(pipeline: RAGPipeline):
    """TEST: Creative poetry request."""
    question = "write a poem about robots"
    result = pipeline.query(question)

    assert result["mode"] == "casual"
    assert len(result["citations"]) == 0
    assert len(result["answer"]) > 20


def test_random_keyboard_mash(pipeline: RAGPipeline):
    """TEST I: Random input does not crash ('asdfgh')."""
    question = "asdfgh"
    result = pipeline.query(question)

    assert result["mode"] == "casual"
    assert result["supported"] is True
    assert len(result["citations"]) == 0
    assert len(result["answer"]) > 0


def test_ambiguous_question_clarification(pipeline: RAGPipeline):
    """TEST J: Ambiguous follow-up without context ('what about that?')."""
    question = "what about that?"
    result = pipeline.query(question, history=[])

    assert result["mode"] == "casual"
    assert "referring to" in result["answer"].lower() or "detail" in result["answer"].lower()


def test_live_weather_disclaimer(pipeline: RAGPipeline):
    """TEST H: Real-time weather inquiry ('what is the weather in Chennai today?')."""
    question = "what is the weather in Chennai today?"
    result = pipeline.query(question)

    assert result["mode"] == "casual"
    assert len(result["citations"]) == 0
    assert "live" in result["answer"].lower() or "weather" in result["answer"].lower()


# =========================================================================
# 3. MULTI-TURN CONTEXT & FOLLOW-UP TESTS
# =========================================================================

def test_multi_turn_follow_up_collaborators(pipeline: RAGPipeline):
    """
    TEST F: Multi-turn follow-up
    Turn 1: 'how much is Pro?'
    Turn 2: 'and how many people can use it?'
    """
    # Turn 1
    t1_q = "how much is Pro?"
    t1_res = pipeline.query(t1_q)
    assert t1_res["mode"] == "rag"

    # Turn 2 with history
    history = [
        {"role": "user", "content": t1_q},
        {"role": "assistant", "content": t1_res["answer"]}
    ]
    t2_q = "and how many people can use it?"
    t2_res = pipeline.query(t2_q, history=history)

    assert t2_res["mode"] == "rag"
    assert t2_res["supported"] is True
    assert len(t2_res["citations"]) > 0
    assert any("20 collaborators" in c["passage"] or "collaborators" in c["passage"] for c in t2_res["citations"])


def test_multi_turn_follow_up_image_upload_pro(pipeline: RAGPipeline):
    """
    TEST G: Follow-up troubleshooting
    Turn 1: 'why can't I upload images?'
    Turn 2: 'does Pro change that?'
    """
    t1_q = "why can't I upload images?"
    t1_res = pipeline.query(t1_q)
    assert t1_res["mode"] == "rag"

    history = [
        {"role": "user", "content": t1_q},
        {"role": "assistant", "content": t1_res["answer"]}
    ]
    t2_q = "does Pro change that?"
    t2_res = pipeline.query(t2_q, history=history)

    assert t2_res["mode"] == "rag"
    assert t2_res["supported"] is True
    assert any("20MB" in c["passage"] or "Pro plan" in c["passage"] for c in t2_res["citations"])


# =========================================================================
# 4. UNSUPPORTED NIMBUSNOTE QUESTIONS
# =========================================================================

def test_unsupported_nimbus_feature_request(pipeline: RAGPipeline):
    """
    TEST: Unsupported NimbusNote question
    Asking specifically for a NimbusNote feature not present in docs.
    """
    question = "Does NimbusNote support voice notes and audio recording?"
    result = pipeline.query(question)

    assert result["mode"] == "unsupported"
    assert result["supported"] is False
    assert len(result["citations"]) == 0
    assert "couldn't find that in the nimbusnote" in result["answer"].lower()
