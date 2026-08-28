import pytest
from unittest.mock import patch


def mock_call_groq_api(messages):
    """
    Deterministic Mock Groq generator for unit test suite.
    Allows pytest suite to execute offline in 0.5s without exhausting API rate limits.
    """
    system_msg = messages[0]["content"] if messages else ""
    user_msg = messages[-1]["content"] if messages else ""
    user_msg_lower = user_msg.lower()

    if "[authoritative retrieved nimbusnote documentation context]" in user_msg_lower:
        if "sync" in user_msg_lower:
            return "NimbusNote syncs every 15 seconds while the app is in the foreground, and every 5 minutes in the background."
        elif "pro" in user_msg_lower or "$6" in user_msg_lower:
            return "The Pro plan is $6/month per workspace and includes 20 collaborators, unlimited notebooks, and 20MB image attachments."
        elif "image" in user_msg_lower:
            return "Image attachments are supported up to 20MB on Pro and Team plans, but not supported on the Free plan."
        return "Based on the retrieved NimbusNote documentation: " + user_msg[:120]

    elif "[nimbusnote documentation context]" in user_msg_lower or "unsupported" in user_msg_lower:
        return "I couldn't find that in the NimbusNote documentation. The documentation covers workspace creation, Free/Pro/Team plans, sync intervals, and troubleshooting."

    # General AI requests
    if "recursion" in user_msg_lower:
        return "Recursion is a programming technique where a function calls itself. It requires a base case to stop and a recursive step."
    elif "joke" in user_msg_lower:
        return "Why did the markdown file cross the road? To get to the other ## header!"
    elif "black hole" in user_msg_lower:
        return "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape."
    elif "2+2" in user_msg_lower:
        return "2 + 2 is 4."
    elif "weather" in user_msg_lower:
        return "I don't have access to live real-time weather data. I am an AI assistant focused on general questions and the NimbusNote notebook."
    elif "calc" in user_msg_lower or "calculator" in user_msg_lower:
        return "To build a basic calculator app, you will need a user interface for number buttons and operators (+, -, *, /), and logic to calculate expressions."
    elif "yo" in user_msg_lower or "gang" in user_msg_lower:
        return "Yo! Clanker here. What's on your mind today?"
    elif "who are you" in user_msg_lower:
        return "I'm Clanker, your conversational AI assistant living inside a digital notebook."
    elif "project" in user_msg_lower:
        return "Here are some cool project ideas: 1. A markdown note app, 2. A real-time chat bot, 3. A weather dashboard."

    return f"I understand: '{user_msg}'. How can I help you further with that?"


@pytest.fixture(autouse=True)
def mock_groq_for_tests(monkeypatch):
    """Auto-mocks Groq API call for all unit tests."""
    monkeypatch.setattr("src.generator.call_groq_api", mock_call_groq_api)
