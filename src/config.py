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
# A relevance threshold of 0.40 cleanly separates in-scope semantic matches
# (typically >= 0.50-0.85) from unsupported / out-of-scope queries (typically < 0.35).
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.40"))
TOP_K = int(os.getenv("TOP_K", "3"))

# Server configuration
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# Generation configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or None
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
