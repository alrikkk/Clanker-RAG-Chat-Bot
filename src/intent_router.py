import re
from typing import List, Dict, Any, Optional

# Regular expressions for casual greetings, small talk, gratitude, jokes, and bot identity
CASUAL_PATTERNS = [
    # Greetings
    r"^(hey|hi|hello|yo|sup|hiya|howdy|what'?s up|good (morning|afternoon|evening|day)|greetings)(\s+.*)?$",
    # Gratitude & pleasantries
    r"^(thanks|thank you|thx|ty|appreciate it|cheers|cool|nice|awesome|great|sounds good|bet|fire|lit|ok|okay|got it)(\s+.*)?$",
    # Laughter & reactions
    r"^(lol|haha|hahaha|lmao|rofl|hehe)(\s+.*)?$",
    # Bot identity & capability queries (not NimbusNote docs)
    r"^(who are you|what is your name|what are you|what can you do|how are you|are you (an ai|a bot|alive|real|human)|introduce yourself)(\??)$",
    # Jokes & small talk
    r"^(tell me a joke|make me laugh|say something( funny)?|what do you think)(\??)$",
    # Compliments & friendly banter
    r"^(you('re| are)? (cool|awesome|great|smart|funny|the best)|i like you|love you)(\s+.*)?$",
    # Goodbyes
    r"^(bye|goodbye|see (ya|you)|cya|later|have a good (day|one))(\s+.*)?$"
]

# Keywords indicating specific NimbusNote domain inquiries
NIMBUS_KEYWORDS = {
    "nimbus", "nimbusnote", "workspace", "workspaces", "notebook", "notebooks",
    "sync", "syncing", "synced", "offline", "note", "notes", "revision", "revisions",
    "free plan", "pro plan", "team plan", "price", "pricing", "cost", "month", "seat",
    "collaborator", "collaborators", "image", "images", "attachment", "attachments",
    "20mb", "50", "history", "upgrade", "downgrade", "refund", "refunds", "student",
    "discount", "troubleshoot", "troubleshooting", "icon", "grey cloud", "red cloud",
    "green cloud", "compare versions", "forgot password", "reset email", "recovery", "sso", "audit"
}


def is_casual_conversation(question: str) -> bool:
    """
    Determines whether a query is casual conversation / small talk rather than
    a document knowledge inquiry or factual question.
    """
    cleaned = question.strip().lower()
    cleaned_no_punct = re.sub(r"[^\w\s]", "", cleaned).strip()

    # If explicitly mentioning NimbusNote domain topics, it's a knowledge inquiry
    words = set(cleaned_no_punct.split())
    if words.intersection(NIMBUS_KEYWORDS):
        return False

    # Check for direct matches against casual conversational regexes
    for pattern in CASUAL_PATTERNS:
        if re.match(pattern, cleaned, re.IGNORECASE) or re.match(pattern, cleaned_no_punct, re.IGNORECASE):
            return True

    # Very short conversational pleasantries
    if len(words) <= 3 and any(w in {"yo", "hey", "sup", "thanks", "lol", "cool", "nice", "fire", "bet", "ok", "okay", "bye"} for w in words):
        return True

    return False


def contextualize_query_with_history(question: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    If the question is a follow-up (e.g. 'and how many collaborators?' or 'how much is it?'),
    augments the query string with key topic context from recent turns to enable accurate retrieval.
    """
    if not history or len(history) < 2:
        return question

    cleaned = question.strip().lower()
    words = cleaned.split()

    # Follow-up indicators (pronouns, conjunctions, short queries)
    is_follow_up = False
    follow_up_triggers = {"and", "what about", "how about", "how many", "does it", "can they", "is it", "why", "how much"}

    if any(cleaned.startswith(t) for t in follow_up_triggers) or len(words) <= 4:
        is_follow_up = True

    if is_follow_up:
        # Extract keywords/topics from the last user turn and assistant response
        recent_context_words = []
        for msg in reversed(history[-2:]):
            text = msg.get("content", "").lower()
            # Look for plan names or features in recent context
            for kw in ["pro plan", "free plan", "team plan", "pro", "team", "free", "workspace", "sync", "image", "upload"]:
                if kw in text and kw not in recent_context_words:
                    recent_context_words.append(kw)

        if recent_context_words:
            # Prepend the discovered context to the question for vector search
            context_prefix = " ".join(recent_context_words[:2])
            return f"{context_prefix} {question}"

    return question
