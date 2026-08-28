import os
import re
import urllib.request
import json
from typing import List, Dict, Any, Optional
from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL

# System prompt for RAG Knowledge Mode
RAG_SYSTEM_PROMPT = """You are Clanker, a helpful, intelligent assistant living inside a digital notebook.
Your job is to answer questions using ONLY the provided retrieved context passages from the NimbusNote documentation.

Rules:
1. Ground your answer ENTIRELY in the provided retrieved context.
2. Do NOT use outside knowledge, and do NOT make up facts.
3. If the context does not contain enough information to answer, state clearly:
   "I couldn't find enough information to answer that from the provided documents."
4. Match the user's conversational tone and energy naturally (if they are casual, be casually helpful; if technical, be precise) while keeping all factual claims strictly accurate.
5. Keep your answer clear, concise, and useful."""

# System prompt for General Conversation Mode
CASUAL_SYSTEM_PROMPT = """You are Clanker, a friendly, witty, and helpful AI assistant living inside a digital notebook.
You are having a casual conversation with the user (greeting, small talk, banter, joke, or pleasantry).

Rules:
1. Respond naturally, warmly, and conversationally.
2. Match the user's tone and vibe (casual, relaxed, playful, or concise) without forcing excessive slang.
3. If asked who you are or what you can do, explain that you are Clanker, a notebook-powered assistant ready to answer questions about NimbusNote documentation or chat.
4. Keep casual responses relatively brief and engaging."""


def detect_tone(question: str) -> str:
    """Detects conversational vibe from the user's input."""
    q = question.lower()
    if any(w in q for w in ["yo", "bro", "gang", "fam", "fire", "bet", "lit", "sup", "lol", "lmao", "haha", "homie"]):
        return "casual"
    elif any(w in q for w in ["joke", "funny", "laugh", "cool"]):
        return "playful"
    elif any(w in q for w in ["explain", "specifically", "parameter", "technical", "detail", "architecture"]):
        return "technical"
    return "balanced"


def format_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks into a clean context block."""
    context_blocks = []
    for chunk in retrieved_chunks:
        header = f"[Document: {chunk.get('source', 'unknown')} | Section: {chunk.get('section', 'General')}]"
        body = chunk.get("text", "").strip()
        context_blocks.append(f"{header}\n{body}")
    return "\n\n---\n\n".join(context_blocks)


def call_llm_api(system_prompt: str, user_content: str, history: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
    """Calls OpenAI-compatible API if configured."""
    if not OPENAI_API_KEY:
        return None

    api_url = OPENAI_BASE_URL or "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    messages = [{"role": "system", "content": system_prompt}]
    
    if history:
        # Include last 4 turns for context
        for msg in history[-4:]:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 350
    }

    try:
        req = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[Generator] LLM API call failed ({e}), falling back to local engine.")
        return None


def synthesize_local_grounded_answer(retrieved_chunks: List[Dict[str, Any]], question: str, tone: str = "balanced") -> str:
    """
    Local grounded synthesis that extracts relevant factual sentences and
    frames them naturally with tone adaptation.
    """
    if not retrieved_chunks:
        return "I couldn't find enough information to answer that from the provided documents."

    question_terms = set(re.findall(r"\w+", question.lower()))
    stop_words = {"what", "is", "the", "in", "to", "how", "often", "does", "do", "a", "an", "and", "or", "of", "for", "with", "can", "i", "my", "while", "yo", "bro"}
    key_terms = question_terms - stop_words

    candidates = []
    for chunk in retrieved_chunks:
        text = chunk.get("text", "")
        raw_items = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+[-•]\s+|\n\n+", text) if s.strip()]
        for item in raw_items:
            clean_item = re.sub(r"^#+\s*|^[-•*]\s*", "", item).strip()
            if len(clean_item) < 10:
                continue
            item_terms = set(re.findall(r"\w+", clean_item.lower()))
            overlap = len(key_terms.intersection(item_terms))
            candidates.append((overlap, clean_item))

    candidates.sort(key=lambda x: x[0], reverse=True)

    if candidates and candidates[0][0] > 0:
        top_sentences = [candidates[0][1]]
        if len(candidates) > 1 and candidates[1][0] > 0 and candidates[1][1] not in top_sentences:
            top_sentences.append(candidates[1][1])
        base_answer = " ".join(top_sentences)
    else:
        # Fallback to the top chunk's first paragraph without markdown hashes
        top_chunk_text = retrieved_chunks[0].get("text", "").strip()
        first_para = top_chunk_text.split("\n\n")[0]
        base_answer = re.sub(r"^#+\s*", "", first_para).strip()

    # Subtle conversational tone framing
    if tone == "casual":
        if not base_answer.lower().startswith(("yep", "here", "sure", "nimbusnote")):
            return f"Got you — {base_answer[0].lower() + base_answer[1:] if len(base_answer) > 1 else base_answer}"
    return base_answer


def synthesize_local_casual_response(question: str, tone: str = "balanced") -> str:
    """Generates natural, responsive small talk without requiring an API key."""
    q = question.strip().lower()

    # Greetings
    if any(q.startswith(g) for g in ["yo", "sup", "hey", "hi", "hello", "howdy"]):
        if "yo" in q or "bro" in q or "gang" in q or "fam" in q:
            return "Yo! Clanker here. What are we looking up in the notebook today?"
        elif "sup" in q:
            return "Not much, just keeping the notebook organized! What do you need?"
        else:
            return "Hey there! I'm Clanker. Ask me anything about NimbusNote, or just say what's on your mind."

    # Gratitude & pleasantries
    if any(w in q for w in ["thanks", "thank you", "thx", "appreciate"]):
        if "bro" in q or tone == "casual":
            return "Anytime! Let me know if you need anything else from the notebook."
        return "You're very welcome! Feel free to ask if you have more questions."

    if any(w in q for w in ["cool", "nice", "awesome", "great", "fire", "bet", "lit"]):
        return "Glad you like it! Ready whenever you want to check something in the notebook."

    if any(w in q for w in ["lol", "haha", "lmao", "rofl", "hehe"]):
        return "Haha! Glad to keep things light. Let me know what you'd like to look up!"

    # Identity & Capabilities
    if any(p in q for p in ["who are you", "what are you"]):
        return "I'm Clanker — your tiny notebook-powered AI assistant. I can search through the NimbusNote documentation to find exact passages, pricing details, sync specs, and troubleshooting tips, or just chat with you!"

    if any(p in q for p in ["what can you do", "help"]):
        return "I can search the NimbusNote documentation for answers on workspace limits, Free/Pro/Team plans, sync behavior, and troubleshooting. I'll always show you the exact passage and source document when answering!"

    if any(p in q for p in ["how are you", "how's it going", "how are you doing"]):
        return "Doing great, thanks for asking! Running smoothly and ready to search the notebook. How are you doing?"

    if "joke" in q or "funny" in q:
        return "Why did the markdown file cross the road? To get to the other ## header!"

    if any(p in q for p in ["you look cool", "you're cool", "you are cool", "i like you"]):
        return "Thanks! The notebook aesthetic suits me well. What can I help you find today?"

    if any(p in q for p in ["bye", "goodbye", "see ya", "cya", "later"]):
        return "Catch you later! The notebook is always here when you need it."

    # General conversational fallback
    return "I hear you! Feel free to ask any question about NimbusNote documentation, or let me know what you'd like to explore."


class AnswerGenerator:
    """
    Unified answer generator handling both Grounded RAG Generation
    and General Conversational responses.
    """
    def generate_grounded_answer(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        question: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        if not retrieved_chunks:
            return "I couldn't find enough information to answer that from the provided documents."

        tone = detect_tone(question)
        context_str = format_context(retrieved_chunks)
        user_content = f"Retrieved Context:\n{context_str}\n\nUser Question: {question}\n\nAnswer:"

        if OPENAI_API_KEY:
            api_answer = call_llm_api(RAG_SYSTEM_PROMPT, user_content, history)
            if api_answer:
                return api_answer

        return synthesize_local_grounded_answer(retrieved_chunks, question, tone=tone)

    def generate_casual_response(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        tone = detect_tone(question)

        if OPENAI_API_KEY:
            api_answer = call_llm_api(CASUAL_SYSTEM_PROMPT, question, history)
            if api_answer:
                return api_answer

        return synthesize_local_casual_response(question, tone=tone)
