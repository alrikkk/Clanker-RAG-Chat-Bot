import os
import urllib.request
import urllib.error
import json
from typing import List, Dict, Any, Optional
from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL

# Authoritative Clanker System Prompt
CLANKER_SYSTEM_PROMPT = """You are Clanker, a conversational AI assistant living inside a digital notebook.

You can help the user with general questions, explanations, brainstorming, coding, casual conversation, and other normal requests.

When NimbusNote documentation is supplied as context, treat that retrieved documentation as the authoritative source for NimbusNote-specific facts.

Do not invent NimbusNote features, pricing, limits, or behavior.

If retrieved NimbusNote context is insufficient to answer a NimbusNote-specific question, say so clearly rather than making up an answer.

For normal questions that are unrelated to NimbusNote, answer naturally using your general capabilities.

Use the conversation history to understand follow-up questions and references.

Match the user's tone naturally without forcing slang.

Be concise by default and give more detail when requested."""


def format_context_block(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved document chunks into clean markdown context blocks."""
    blocks = []
    for chunk in retrieved_chunks:
        doc = chunk.get("source", "unknown")
        section = chunk.get("section", "Overview")
        text = chunk.get("text", "").strip()
        blocks.append(f"### [Document: {doc} | Section: {section}]\n{text}")
    return "\n\n---\n\n".join(blocks)


def call_llm_api(messages: List[Dict[str, str]]) -> str:
    """
    Executes actual LLM completion call via OpenAI-compatible Chat Completions API.
    Raises RuntimeError on failure or missing API key.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured in the environment. "
            "Please set your OpenAI API key in .env to enable Clanker's AI brain."
        )

    api_url = OPENAI_BASE_URL or "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 600
    }

    print(f"[clanker] Calling LLM API ({OPENAI_MODEL}) with {len(messages)} messages...")

    try:
        req = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            content = res_data["choices"][0]["message"]["content"].strip()
            print(f"[clanker] LLM response received ({len(content)} chars)")
            return content
    except urllib.error.HTTPError as http_err:
        err_body = http_err.read().decode("utf-8", errors="ignore")
        print(f"[clanker] LLM API HTTP Error {http_err.code}: {err_body}")
        raise RuntimeError(f"LLM API returned HTTP {http_err.code}: {err_body}")
    except urllib.error.URLError as url_err:
        print(f"[clanker] LLM API Connection Error: {url_err.reason}")
        raise RuntimeError(f"Could not connect to LLM API: {url_err.reason}")
    except Exception as e:
        print(f"[clanker] LLM API Unexpected Error: {str(e)}")
        raise RuntimeError(f"LLM generation failed: {str(e)}")


class AnswerGenerator:
    """
    Unified LLM Answer Generator.
    Clanker's AI Brain is the LLM.
    When RAG context is supplied, it grounds the answer in retrieved NimbusNote docs.
    When no RAG context is needed, the LLM answers naturally.
    """
    def generate_response(
        self,
        question: str,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        unsupported_note: bool = False
    ) -> str:
        messages = [{"role": "system", "content": CLANKER_SYSTEM_PROMPT}]

        # Inject conversation history
        if history:
            for turn in history[-8:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ["user", "assistant"] and content:
                    messages.append({"role": role, "content": content})

        # Build current user prompt with optional authoritative context
        if retrieved_chunks and len(retrieved_chunks) > 0:
            context_text = format_context_block(retrieved_chunks)
            user_content = (
                f"[AUTHORITATIVE RETRIEVED NIMBUSNOTE DOCUMENTATION CONTEXT]:\n"
                f"{context_text}\n\n"
                f"---\n"
                f"User Question: {question}"
            )
        elif unsupported_note:
            user_content = (
                f"[NIMBUSNOTE DOCUMENTATION CONTEXT]:\n"
                f"(The provided NimbusNote documents do not contain information or support for this requested feature).\n\n"
                f"---\n"
                f"User Question: {question}\n\n"
                f"Instructions: Explain clearly and naturally that this information or feature is not present in the supplied NimbusNote documentation. Do not hallucinate."
            )
        else:
            # Normal conversational AI request (no documents needed)
            user_content = question

        messages.append({"role": "user", "content": user_content})

        # The LLM is ALWAYS the final generator
        return call_llm_api(messages)
