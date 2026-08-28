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
2. Do NOT invent NimbusNote features, and do NOT make up facts.
3. If the context does not contain enough information to answer a NimbusNote question, state clearly:
   "I couldn't find that in the NimbusNote documentation."
4. Match the user's conversational tone naturally (if casual, be casually helpful; if technical, be clear and precise).
5. Keep your answer clear, concise, and useful."""

# System prompt for General AI Conversation Mode
GENERAL_SYSTEM_PROMPT = """You are Clanker, an intelligent, friendly, and witty AI assistant living inside a digital notebook.
You are a general-purpose AI chat assistant. You can converse, explain programming concepts, solve math, write poems, tell jokes, brainstorm, and answer general questions.

Rules:
1. Respond helpfully, clearly, and naturally to whatever the user asks.
2. If asked to explain a topic (e.g. recursion, black holes), provide a clear, easy-to-understand explanation.
3. If the user asks for real-time live data (e.g., current weather, live sports scores), explain politely that you do not have real-time live internet data access.
4. Match the user's energy and tone naturally without forcing exaggerated slang.
5. If the user types gibberish or random letters, respond in a friendly, lighthearted way."""


def detect_tone(question: str) -> str:
    """Detects conversational vibe from the user's input."""
    q = question.lower()
    if any(w in q for w in ["yo", "bro", "gang", "fam", "fire", "bet", "lit", "sup", "lol", "lmao", "haha", "homie"]):
        return "casual"
    elif any(w in q for w in ["joke", "funny", "laugh", "cool", "poem", "story", "bored"]):
        return "playful"
    elif any(w in q for w in ["explain", "specifically", "parameter", "technical", "detail", "architecture", "code", "programming", "algorithm"]):
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
        for msg in history[-6:]:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 450
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
        print(f"[Generator] LLM API call failed ({e}), using local synthesis.")
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
        top_chunk_text = retrieved_chunks[0].get("text", "").strip()
        first_para = top_chunk_text.split("\n\n")[0]
        base_answer = re.sub(r"^#+\s*", "", first_para).strip()

    if tone == "casual":
        if not base_answer.lower().startswith(("yep", "here", "sure", "nimbusnote")):
            return f"Got you — {base_answer[0].lower() + base_answer[1:] if len(base_answer) > 1 else base_answer}"
    return base_answer


def synthesize_local_general_response(question: str, tone: str = "balanced") -> str:
    """
    Offline local general AI generator covering programming concepts, math, science,
    creative requests, jokes, small talk, random text, and live-data disclaimers.
    """
    q = question.strip().lower()
    q_no_punct = re.sub(r"[^\w\s]", "", q).strip()

    # 1. Random text / keyboard mash (e.g. asdfgh, qwerty, zxcvbn)
    if re.match(r"^[asdfghjklqwertyuiopzxcvbnm\s]{4,}$", q_no_punct) and not any(w in q_no_punct.split() for w in ["what", "how", "who", "why", "when", "is", "can"]):
        if any(seq in q_no_punct for seq in ["asdf", "qwerty", "zxcv", "hjkl", "ghjk"]):
            return "Looks like some keyboard static! I'm here and listening — what would you like to explore or check in the notebook?"

    # 2. Ambiguous follow-ups with no context
    if q in ["what about that", "what about it", "why is that", "how about that", "can it", "what about that?"]:
        return "I'm listening, but I'm not sure what you're referring to. Could you give me a bit more detail on what you'd like to check?"

    # 3. Live real-time data queries (e.g. weather, stocks, live scores)
    if any(w in q for w in ["weather", "temperature", "forecast", "stock price", "who won", "score"]):
        loc_match = re.search(r"in\s+([a-zA-Z\s]+)", q)
        loc = f" for {loc_match.group(1).strip().title()}" if loc_match else ""
        return f"I don't have access to live real-time weather or current event data{loc}. I'm a notebook-based assistant, but I can help with general explanations, coding concepts, writing, or anything in the NimbusNote documentation."

    # 4. Programming concepts (Recursion, Binary Search, Python, Git, etc.)
    if "recursion" in q:
        return "Recursion is when a function calls itself to break a problem down into smaller sub-problems. It always needs two key parts:\n1. Base Case: A condition that stops the recursion so it doesn't run forever.\n2. Recursive Case: The step where the function calls itself with modified input.\n\nA classic example is calculating a factorial: factorial(n) = n * factorial(n - 1), with base case factorial(1) = 1."

    if "binary search" in q:
        return "Binary search is an efficient algorithm for finding an item in a sorted list. It works by repeatedly dividing the search interval in half: if the target value is less than the middle element, it searches the lower half; otherwise, it searches the upper half. Its time complexity is O(log n)."

    if "learn python" in q or "learning python" in q or "good way to learn python" in q:
        return "A great way to learn Python is through hands-on practice:\n1. Master the basics: variables, loops, lists, and functions.\n2. Build small CLI projects: a calculator, a to-do list, or a text parser.\n3. Explore libraries: try requests, fastapi, or pandas.\n4. Practice problem solving on platforms like LeetCode or exercism.io."

    # 5. Math queries (e.g. 2+2, 5*5, simple arithmetic)
    math_match = re.search(r"(\d+)\s*([\+\-\*\/])\s*(\d+)", q)
    if math_match:
        a, op, b = int(math_match.group(1)), math_match.group(2), int(math_match.group(3))
        res = a + b if op == "+" else (a - b if op == "-" else (a * b if op == "*" else (a / b if b != 0 else "undefined")))
        return f"{a} {op} {b} is {res}."

    # 6. Science concepts (Black holes, Photosynthesis, Gravity)
    if "black hole" in q:
        return "A black hole is a region of spacetime where gravity is so strong that nothing — not even light — can escape from it. They usually form when massive stars collapse at the end of their life cycle. The boundary beyond which nothing can escape is called the event horizon."

    if "photosynthesis" in q:
        return "Photosynthesis is the biological process by which green plants and certain organisms use sunlight, water, and carbon dioxide to create oxygen and energy in the form of glucose."

    # 7. Creative requests (Poem, Story, Haiku)
    if "poem" in q or "haiku" in q or "rhyme" in q:
        if "haiku" in q:
            return "Pages quietly turn,\nParchment holds the words of thought,\nClanker never sleeps."
        return "Gears of brass and ink on page,\nA quiet mind upon the stage.\nThrough notebook lines and spiral wire,\nClanker sparks the thinking fire."

    if "story" in q:
        return "Once in a quiet workshop filled with gears and parchment, a mechanical assistant named Clanker came to life. Unlike other machines that wanted to conquer galaxies, Clanker just wanted to organize thoughts, answer questions, and keep the notebook tidy. And so, between the ruled lines of paper, Clanker found its purpose."

    # 8. Humor & Jokes
    if "joke" in q or "funny" in q or "laugh" in q:
        jokes = [
            "Why did the markdown file cross the road? To get to the other ## header!",
            "There are 10 types of people in the world: those who understand binary, and those who don't.",
            "Why do programmers prefer dark mode? Because light attracts bugs!"
        ]
        return jokes[0]

    # 9. Identity & Capabilities
    if any(p in q for p in ["who are you", "what are you", "introduce yourself"]):
        return "I'm Clanker — your notebook-powered AI assistant. I can chat, explain programming and science topics, help brainstorm, or search the NimbusNote documentation with grounded citations."

    if any(p in q for p in ["what can you do", "help me", "what do you do"]):
        return "You can talk to me about anything! I can answer general questions, explain code, write stories, or search through the NimbusNote documentation for exact workspace limits, sync specs, and troubleshooting."

    if any(p in q for p in ["how are you", "how's it going", "how are you doing"]):
        return "Doing great, thanks for asking! Running smoothly and ready to chat. What's on your mind today?"

    if "bored" in q:
        return "I can help with that! We can brainstorm an app idea, solve a coding puzzle, explore a weird science fact, or I can tell you a joke. What sounds fun?"

    # 10. Greetings & Small talk
    if any(q.startswith(g) for g in ["yo", "sup", "hey", "hi", "hello", "howdy", "hiya"]):
        if "yo" in q or "bro" in q:
            return "Yo! Clanker here. What are we working on or looking up today?"
        return "Hey there! I'm Clanker. What's on your mind today?"

    if any(w in q for w in ["thanks", "thank you", "thx", "appreciate"]):
        return "Anytime! Let me know if you want to explore anything else."

    if any(w in q for w in ["cool", "nice", "awesome", "great", "fire", "bet", "lit", "ok", "okay"]):
        return "Glad to hear! Ready whenever you want to ask something else."

    if any(w in q for w in ["lol", "haha", "lmao", "rofl", "hehe"]):
        return "Haha! Glad to keep things light. Let me know what you'd like to explore next!"

    if any(w in q for w in ["bye", "goodbye", "see ya", "cya", "later"]):
        return "Catch you later! The notebook is always here when you need it."

    # General conversational fallback
    return "I'm here! Feel free to ask me any general question, ask for an explanation, or ask about the NimbusNote documentation."


class AnswerGenerator:
    """
    Unified answer generator handling Grounded RAG Generation
    and General AI Conversational responses.
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

    def generate_general_response(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generates general AI conversational and explanatory responses."""
        tone = detect_tone(question)

        if OPENAI_API_KEY:
            api_answer = call_llm_api(GENERAL_SYSTEM_PROMPT, question, history)
            if api_answer:
                return api_answer

        return synthesize_local_general_response(question, tone=tone)
