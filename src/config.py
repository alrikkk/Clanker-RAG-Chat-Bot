import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Data and Models
DOCS_DIR = Path(os.getenv("DOCS_DIR", BASE_DIR / "data"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Retrieval parameters
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.40"))
TOP_K = int(os.getenv("TOP_K", "3"))

# Server configuration
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# Groq LLM Generation Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "groq/compound").strip() or "groq/compound"
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions").strip()
