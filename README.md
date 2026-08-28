# Clanker

A small RAG chatbot built around the NimbusNote documentation.

## What it does

Clanker answers questions using a small set of NimbusNote documents. For document-related questions it retrieves relevant passages first, then gives the model that context to work from. The retrieved document, section, similarity score, and exact passage are shown with each grounded answer. For casual conversation (like greetings or jokes), it chats naturally without running unnecessary document lookups.

## Reference Documents

The documentation files in `data/` (`01-getting-started.md`, `02-pricing-and-plans.md`, and `03-troubleshooting.md`) originate from the reference document repository (`https://github.com/MLSA-SRM/recruit-task-rag-docs`). NimbusNote is a fictional note-syncing service used for this capstone exercise. The documents are stored locally in `data/` and indexed by Clanker at startup.

## How it works

1. **Load documents**: Reads the Markdown files from `data/`.
2. **Chunking**: Splits files deterministically around Markdown headings (`#`, `##`) and paragraphs into structured chunks with metadata (`source`, `section`, `chunk_index`, `text`).
3. **Embeddings**: Generates 384-dimensional dense vectors locally using `sentence-transformers` (`all-MiniLM-L6-v2`).
4. **Vector search**: Stores chunks and embeddings in an in-memory vector store. When a query arrives, it computes cosine similarity:
   $$\text{similarity} = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
5. **Relevance threshold**: Queries must meet a similarity threshold of `0.40` to be considered answerable from the docs. Questions below threshold are rejected without hallucinating facts.
6. **Answer generation**: Passes only the top retrieved passages to the generator. If `OPENAI_API_KEY` is provided, it calls OpenAI; otherwise it runs fully offline using its built-in extractor.
7. **Source citation**: Displays the source document filename, section heading, similarity score, and quoted passage.

## Running locally

### Prerequisites
- Python 3.10 or 3.11

### 1. Set up virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment (optional)
Clanker runs locally and offline out-of-the-box. To use an OpenAI model:
```bash
cp .env.example .env
```
Edit `.env` and set `OPENAI_API_KEY`.

### 3. Start the app
```bash
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
```
Open `http://127.0.0.1:8000` in your browser.

## Environment variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DOCS_DIR` | `data` | Path to document directory |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformers model name |
| `RELEVANCE_THRESHOLD` | `0.40` | Minimum cosine similarity to accept a document query |
| `TOP_K` | `3` | Number of passages to retrieve |
| `HOST` | `127.0.0.1` | Local server bind host |
| `PORT` | `8000` | Local server bind port |
| `OPENAI_API_KEY` | *(empty)* | Optional OpenAI API key for generation |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name when OpenAI API is used |

## Tests

Run the test suite with `pytest`:
```bash
pytest -v
```

The test suite covers:
- Markdown chunking and heading hierarchy preservation
- In-memory vector store cosine similarity math
- In-scope retrieval (e.g. sync intervals, pricing tiers, image limits)
- Casual conversation routing (greetings, jokes, small talk)
- Multi-turn conversational follow-up retrieval
- Out-of-scope question rejection and refusal behavior
- FastAPI endpoints (`/api/health`, `/api/documents`, `/api/query`)

## Project structure

```
.
├── data/                       # Local Markdown documentation corpus
│   ├── 01-getting-started.md
│   ├── 02-pricing-and-plans.md
│   ├── 03-troubleshooting.md
│   └── embedded_chunks.json
├── src/                        # Python backend & RAG pipeline
│   ├── app.py                  # FastAPI web application
│   ├── chunker.py              # Deterministic Markdown chunker
│   ├── config.py               # Configuration & threshold settings
│   ├── embeddings.py           # Local embedding model wrapper
│   ├── generator.py            # Answer generator (OpenAI + local fallback)
│   ├── intent_router.py        # Casual vs RAG intent router & history resolver
│   ├── rag_pipeline.py         # End-to-end RAG coordinator
│   └── vector_store.py         # In-memory cosine similarity vector store
├── static/                     # Web frontend
│   ├── index.html              # HTML with "Let's Chat" page & notebook chat
│   ├── css/style.css           # Skeuomorphic notebook styles
│   ├── js/app.js               # Frontend controller & citation renderer
│   └── assets/                 # Clanker robot mascot artwork
├── tests/                      # Automated pytest suite
│   ├── test_api.py
│   ├── test_chunker.py
│   ├── test_rag_pipeline.py
│   └── test_vector_store.py
├── api/                        # Vercel serverless entrypoint
│   └── index.py
├── scripts/                    # Helper scripts
│   └── build_vector_cache.py
├── vercel.json                 # Vercel deployment routing
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment variables
└── README.md
```

## Deployment

### Vercel
1. Push your repository to GitHub.
2. Import the project in the [Vercel Dashboard](https://vercel.com).
3. If using an OpenAI key, add `OPENAI_API_KEY` under **Project Settings → Environment Variables**.
4. Deploy.

Vercel will use `vercel.json` and `api/index.py` to serve the FastAPI backend along with the static frontend files.

## Example Queries

### In-scope question
- **Question**: *"How often does NimbusNote sync while the app is in the foreground?"*
- **Source**: `01-getting-started.md`
- **Section**: `Sync behavior`
- **Similarity**: `~0.77`
- **Answer**: *"NimbusNote syncs every 15 seconds while the app is in the foreground, and every 5 minutes in the background."*

### Out-of-scope question
- **Question**: *"What is the weather in Chennai today?"*
- **Similarity**: `< 0.15` (Below `0.40` threshold)
- **Answer**: *"I couldn't find enough information to answer that from the provided documents."*
- **Citations**: None displayed.
