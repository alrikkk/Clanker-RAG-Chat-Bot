<img width="217" height="223" alt="Screenshot 2026-11-21 at 3 54 03 PM" src="https://github.com/user-attachments/assets/a3f2843e-8d8a-4d08-a443-c0dc8cb8b111" />

# Clanker

A notebook-powered AI assistant and RAG chatbot built around the NimbusNote documentation.

## What it does

Clanker is a conversational AI assistant living inside a digital notebook. The message box is general-purpose: you can ask general programming and technical questions, solve math, request poems or jokes, or chat naturally. When you ask a question related to NimbusNote, Clanker retrieves the most relevant passages from its local documentation corpus first, verifies similarity against a relevance threshold, and provides a grounded answer with exact citations (source file, section heading, similarity score, and passage quote).

## Reference Documents

The documentation files in `data/` (`01-getting-started.md`, `02-pricing-and-plans.md`, and `03-troubleshooting.md`) originate from the reference document repository (`https://github.com/MLSA-SRM/recruit-task-rag-docs`). NimbusNote is a fictional note-syncing service used for this capstone exercise. The documents are stored locally in `data/` and indexed by Clanker at startup.

## How it works

1. **Load documents**: Reads the Markdown files from `data/`.
2. **Chunking**: Splits files deterministically around Markdown headings (`#`, `##`) and paragraphs into structured chunks with metadata (`source`, `section`, `chunk_index`, `text`).
3. **Embeddings**: Generates 384-dimensional dense vectors using `fastembed` (`sentence-transformers/all-MiniLM-L6-v2` ONNX runtime).
4. **Intent & Vector search**: Evaluates whether a question is general AI/casual or a NimbusNote knowledge inquiry. For document inquiries, it computes cosine similarity against indexed chunks:
   $$\text{similarity} = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
5. **Relevance threshold**: Document queries must meet a similarity threshold of `0.40` to be considered answerable from the docs. Specific doc features not in the corpus are rejected without hallucinating facts.
6. **Answer generation**: Passes retrieved passages to the generator. If `GROQ_API_KEY` is provided, it calls Groq (`llama-3.3-70b-versatile`); for normal general questions, Groq answers directly with full conversation history.
7. **Source citation**: Displays the source document filename, section heading, similarity score, and quoted passage when RAG is used.

## Running locally

### Prerequisites
- Python 3.10 or 3.11

### 1. Set up virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
```
Edit `.env` and set `GROQ_API_KEY`.

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
| `GROQ_API_KEY` | *(empty)* | Groq API key for generation |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |

## Tests

Run the test suite with `pytest`:
```bash
pytest -v
```

The test suite covers:
- Markdown chunking and heading hierarchy preservation
- In-memory vector store cosine similarity math
- In-scope retrieval (sync intervals, pricing tiers, image limits)
- Casual conversation routing (greetings, jokes, small talk)
- General AI explanations (recursion, math, science, poetry)
- Multi-turn conversational follow-up retrieval
- Unsupported NimbusNote question rejection
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
│   ├── embeddings.py           # ONNX embedding model wrapper
│   ├── generator.py            # Answer generator (OpenAI + local fallback)
│   ├── intent_router.py        # Intent router & conversation history resolver
│   ├── rag_pipeline.py         # End-to-end RAG & conversation coordinator
│   └── vector_store.py         # In-memory cosine similarity vector store
├── static/                     # Web frontend
│   ├── index.html              # HTML with "Let's Chat" page & notebook chat
│   ├── css/style.css           # Skeuomorphic notebook styles
│   ├── js/app.js               # Frontend controller & citation renderer
│   └── assets/                 # Clanker robot mascot artwork
├── public/                     # Static assets served by Edge CDN
├── tests/                      # Automated pytest suite
│   ├── test_api.py
│   ├── test_chunker.py
│   ├── test_rag_pipeline.py
│   └── test_vector_store.py
├── api/                        # Vercel serverless entrypoint
│   └── index.py
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


## Preview 

<img width="1511" height="831" alt="Screenshot 2026-11-21 at 3 54 45 PM" src="https://github.com/user-attachments/assets/d7475228-57b5-40d0-876e-208a1c4202bf" />

<img width="1512" height="831" alt="Screenshot 2026-11-21 at 3 55 45 PM" src="https://github.com/user-attachments/assets/2767ba25-8cf6-4dd1-a8f3-717ac08e59e5" />

<img width="1203" height="768" alt="Screenshot 2026-11-21 at 3 56 33 PM" src="https://github.com/user-attachments/assets/743193a7-4182-424e-8ddd-ad9aa8b0099a" />


## note
The Bot's Ai has been disabled dude to trial API use, you may check it out and try to run your own one locally.

Thanks for checking this project out.
