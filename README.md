# SemanticOps RAG Platform

A Retrieval-Augmented Generation (RAG) platform built with **FastAPI**, **ChromaDB** and **Groq**.
Documents are ingested, chunked, embedded and stored in a vector database. Questions are answered by an LLM that is instructed to use **only the retrieved context**.

## Features

- Multi-format ingestion: `.txt`, `.md`, `.json`, `.yaml` / `.yml`, `.csv`
- Configurable chunking engine (size and overlap)
- Embedding pipeline (multilingual sentence-transformers)
- Persistent vector database (ChromaDB)
- Retrieval engine (top-k similarity search)
- RAG pipeline: context builder, prompt builder, LLM generation (Groq)
- Basic monitoring (`/metrics`) and evaluation (`/evaluation`) endpoints
- REST API with interactive documentation (Swagger UI)
- Docker support and a Pytest suite

## Architecture

```
Document
   ↓
Loader            (txt, md, json, yaml, csv)
   ↓
Chunker           (chunk_size / chunk_overlap)
   ↓
Embeddings
   ↓
ChromaDB          (persistent, data/vector_db)
   ↓
Retriever         (top_k)
   ↓
Context Builder
   ↓
Prompt Builder
   ↓
LLM (Groq)
   ↓
Answer
```

## Project structure

```
app/
├── api/            FastAPI app, shared dependencies, routes and schemas
├── chunking/       Text chunker
├── core/           Settings (settings.yaml + config.py) and logging
├── domain/         Pydantic models (Document, Chunk, Query, Answer, ...)
├── embeddings/     Embedding service
├── evaluation/     Evaluator
├── generation/     Context builder, prompt builder, LLM service, RAG pipeline
├── ingestion/      Ingestion service (load → chunk → embed → store)
├── loaders/        File loaders and loader factory
├── monitoring/     In-memory metrics
├── retrieval/      Retriever
└── vectorstore/    ChromaDB wrapper
docs/               Sample documents and evaluation set
scripts/            Helper scripts (python -m scripts.load_demo_data)
tests/              Pytest suite
```

## API

| Method | Endpoint        | Description                                              |
|--------|-----------------|----------------------------------------------------------|
| POST   | `/upload`       | Upload a file (multipart) and ingest it                  |
| POST   | `/ingest`       | Ingest a file that already exists on the server          |
| POST   | `/query`        | Ask a question, get an answer generated from the context |
| GET    | `/health`       | Health check                                             |
| GET    | `/metrics`      | Documents ingested, chunks created, query count          |
| GET    | `/evaluation`   | Evaluation of a sample answer                            |
| GET    | `/config`       | Current configuration (from `settings.yaml`)             |
| GET    | `/debug/chunks` | Dump the chunks stored in the vector database            |

Interactive documentation: <http://127.0.0.1:8000/docs>

### Request examples

```bash
# Upload and ingest a document
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@docs/sample.txt"

# Ask a question
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is SemanticOps RAG?"}'
```

If the answer is not in the retrieved context, the model is instructed to say so instead of guessing.

## Tech stack

- Python 3.13
- FastAPI + Uvicorn
- ChromaDB
- sentence-transformers (embeddings)
- Groq API (LLM generation)
- Pydantic, PyYAML
- Pytest, Ruff
- Docker

## Getting started

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/ImadeddineAklouf/semanticops-rag-copilot.git
cd semanticops-rag-copilot

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure the environment

Copy the example file and fill in your values. **Never commit `.env`.**

```bash
cp .env.example .env
```

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_groq_model_id
```

- Create a free API key at [console.groq.com](https://console.groq.com) (no credit card required; usage is rate-limited).
- Pick a chat/instruct model from the list available to your key. The Groq catalog changes regularly, which is why the model id lives in the environment and is never hard-coded.
- The free tier offers no data-privacy guarantee: do not send confidential documents.
- The server starts without a key; `POST /query` then answers `503` until `GROQ_API_KEY` is set.
- Optional: `VECTOR_DB_PATH` changes where ChromaDB stores its data (default `data/vector_db`).
- The first request that needs embeddings downloads the sentence-transformers model (about 120 MB).

### 3. Run the server

```bash
uvicorn app.api.main:app --reload --env-file .env
```

## Configuration

Application settings live in `app/core/settings.yaml`:

| Key                           | Default | Description                              |
|-------------------------------|---------|------------------------------------------|
| `chunking.chunk_size`         | `500`   | Characters per chunk                     |
| `chunking.chunk_overlap`      | `50`    | Characters shared between two chunks     |
| `retrieval.top_k`             | `5`     | Number of chunks sent to the LLM         |
| `vectorstore.collection_name` | `api`   | ChromaDB collection                      |
| `uploads.directory`           | `uploads` | Where uploaded files are stored        |
| `monitoring.enabled`          | `true`  | Enable in-memory metrics                 |

`chunk_overlap` must stay strictly lower than `chunk_size`.

Secrets and model selection come from environment variables (`GROQ_API_KEY`, `GROQ_MODEL`).

## Docker

```bash
docker compose up --build
```

The compose file mounts `uploads/` and `data/` so documents and the vector database survive restarts. Pass your environment variables with `env_file: .env` in `docker-compose.yml`.

## Tests

```bash
pytest
```

The suite runs offline: the LLM and the embeddings are replaced by fakes (see `tests/conftest.py`), and the API tests override the pipeline dependency, so no Groq quota is used.

## Security notes

- Secrets are read from `.env`, which is listed in `.gitignore`.
- Uploaded file names are sanitized before being written to disk.
- `/debug/chunks` and `/ingest` are intended for development: restrict or disable them before any public deployment.

## Roadmap

- [ ] Replace the sample evaluation with the test set in `docs/evaluation/test_set.json`
- [ ] Return the sources used for each answer
- [ ] Support PDF and DOCX ingestion
- [ ] Re-ingestion without duplicates (upsert)
- [ ] Switchable LLM backends (Groq, Ollama) behind a common interface
- [ ] Persist metrics beyond the process lifetime
- [ ] LLM latency and token usage in `/metrics`

## License

Add a license of your choice (for example MIT).
