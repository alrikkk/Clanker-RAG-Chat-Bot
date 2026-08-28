import re
from typing import List, Dict, Any, Optional, Tuple

# Explicit keywords related to the NimbusNote domain
NIMBUS_KEYWORDS = {
    "nimbus", "nimbusnote", "workspace", "workspaces", "notebook", "notebooks",
    "sync", "syncing", "synced", "offline", "note", "notes", "revision", "revisions",
    "pro", "team", "free", "free plan", "pro plan", "team plan", "price", "pricing", "cost", "month", "seat",
    "collaborator", "collaborators", "image", "images", "attachment", "attachments",
    "20mb", "upgrade", "downgrade", "refund", "refunds", "student discount", "discount",
    "troubleshoot", "troubleshooting", "icon", "grey cloud", "red cloud", "green cloud",
    "compare versions", "forgot password", "reset email", "recovery", "sso", "audit log",
    "export", "priority sync", "5 seconds", "15 seconds", "5 minutes", "20 collaborators",
    "unlimited notebooks", "inbox", "markdown"
}

# Live / real-time data indicators (things that require live internet/weather/stocks)
LIVE_DATA_PATTERNS = [
    r"\b(weather|temperature|forecast|climate)\b",
    r"\b(stock price|stock market|bitcoin price|crypto price|exchange rate)\b",
    r"\b(who won|game score|match score|live score)\b",
    r"\b(today's news|breaking news|current events|what happened today|what happened this morning)\b"
]

# Ambiguous short follow-ups with pronouns
AMBIGUOUS_PATTERNS = [
    r"^(what about that|what about it|and that|why is that|how about that|can it|does it|is that so|what do you mean by that|how come)(\??)$"
]


def is_live_data_query(question: str) -> bool:
    """Detects if user is asking for real-time external data like live weather or stock prices."""
    q = question.strip().lower()
    for pat in LIVE_DATA_PATTERNS:
        if re.search(pat, q):
            return True
    return False


def is_ambiguous_without_context(question: str, history: Optional[List[Dict[str, str]]] = None) -> bool:
    """Checks if a user is asking a vague follow-up when there is no prior context."""
    if history and len(history) >= 1:
        return False
    q = question.strip().lower()
    for pat in AMBIGUOUS_PATTERNS:
        if re.match(pat, q):
            return True
    return False


def is_explicit_nimbus_inquiry(question: str) -> bool:
    """Checks if question explicitly mentions NimbusNote domain topics."""
    cleaned = question.strip().lower()
    cleaned_no_punct = re.sub(r"[^\w\s]", " ", cleaned).strip()
    words = set(cleaned_no_punct.split())

    # Check direct word/phrase intersection
    if words.intersection(NIMBUS_KEYWORDS):
        return True

    for kw in ["free plan", "pro plan", "team plan", "student discount", "password reset", "red cloud", "grey cloud", "green cloud", "20mb", "image upload"]:
        if kw in cleaned:
            return True

    return False


def is_follow_up_with_context(question: str, history: Optional[List[Dict[str, str]]] = None) -> bool:
    """Checks if a message is a follow-up that refers to recent conversation context."""
    if not history or len(history) < 1:
        return False

    q = question.strip().lower()
    follow_up_triggers = [
        "how much", "how many", "what about", "and", "does it", "can they", "can i",
        "does pro", "does team", "is it", "why", "what is included", "how does that work",
        "people can use", "collaborators", "limit", "cost", "price", "sync"
    ]

    # Check if recent history was discussing NimbusNote topics
    recent_history_text = " ".join([m.get("content", "").lower() for m in history[-3:]])
    has_nimbus_context = any(kw in recent_history_text for kw in ["pro", "team", "free", "workspace", "sync", "note", "image", "upload", "plan", "limit", "nimbus"])

    if has_nimbus_context:
        if any(q.startswith(t) for t in follow_up_triggers) or any(t in q for t in follow_up_triggers) or len(q.split()) <= 6:
            return True

    return False


def contextualize_query_with_history(question: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Augments follow-up questions (e.g. 'how many collaborators?' or 'does Pro fix that?')
    with topic antecedents from recent turns for rich semantic vector retrieval.
    """
    if not history or len(history) < 1:
        return question

    q = question.strip().lower()
    
    # Extract topics from previous user and assistant turns
    recent_topics = []
    for msg in reversed(history[-4:]):
        text = msg.get("content", "").lower()
        if "pro" in text and "pro plan" not in recent_topics:
            recent_topics.append("pro plan")
        if "team" in text and "team plan" not in recent_topics:
            recent_topics.append("team plan")
        if "free" in text and "free plan" not in recent_topics:
            recent_topics.append("free plan")
        if any(w in text for w in ["image", "upload", "attach", "photo"]) and "image upload" not in recent_topics:
            recent_topics.append("image upload limit")
        if any(w in text for w in ["sync", "cloud", "offline"]) and "sync behavior" not in recent_topics:
            recent_topics.append("sync behavior")
        if "workspace" in text and "workspace" not in recent_topics:
            recent_topics.append("workspace")

    if recent_topics:
        # If question contains pronouns or is a short follow-up, prepend discovered context
        follow_up_tokens = {"it", "that", "this", "they", "them", "and", "how", "what", "does", "can", "why", "how much", "how many"}
        q_words = set(re.findall(r"\w+", q))
        if len(q.split()) <= 6 or q_words.intersection(follow_up_tokens):
            context_str = " ".join(recent_topics[:2])
            return f"{context_str} {question}"

    return question
