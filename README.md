# PaperPrism — A Query-Adaptive Retrieval Pipeline for Scientific Literature

PaperPrism is a RAG (Retrieval-Augmented Generation) system purpose-built for querying scientific research papers. Upload your papers, ask questions, get cited answers with verified sources.

**What makes it different:** PaperPrism classifies each question into one of 8 types (factual, methods, summary, comparative, novelty, limitations, framing, general) and adapts its retrieval strategy — different chunk types, result counts, section filters, and diversity limits per query. Benchmarked at **13/15 wins** against direct Claude on blind evaluation.

## Quick Start

**Prerequisites:** [Docker](https://docs.docker.com/get-docker/) and API keys for:
- [Anthropic](https://console.anthropic.com/) (Claude — answer generation)
- [Voyage AI](https://dash.voyageai.com/) (embeddings)
- [Cohere](https://dashboard.cohere.com/) (reranking)

**Three commands:**

```bash
git clone https://github.com/YOUR_USERNAME/paper-prism.git && cd paper-prism
cp .env.example .env    # then add your API keys
docker compose up
```

Open **http://localhost:3000** — no login required. Upload PDFs and start asking questions.

## What You'll See

- **Chat interface** with real-time pipeline visualization (each of the 13 stages lights up as it runs)
- **Citation badges** — inline `[Source N]` markers that are color-coded by verification confidence (green/amber/red)
- **Paper library** with drag-and-drop upload, semantic search, and metadata extraction
- **Customizable prompts** per query type and response mode

## Architecture

```
User Question
  → Query Rewriting (spelling correction)
  → Entity Extraction (chemicals, proteins, methods)
  → Query Classification (8 types)
  → Query Expansion (domain synonyms)
  → Hybrid Search (dense + BM25 via Qdrant)
  → Cross-Encoder Reranking (Cohere)
  → Answer Generation (Claude Opus)
  → Citation Verification (each [Source N] checked)
  → Response with verified citations
```

Each paper is split into **6 chunk types** — abstract, section, fine-grained paragraphs, tables, captions, and a full-paper embedding — so the right granularity is available for each query type.

## Tech Stack

| Component | Technology |
|---|---|
| PDF Extraction | MinerU (layout-aware OCR, table extraction) |
| Embeddings | Voyage AI `voyage-3-large` (1024-dim) |
| Vector Database | Qdrant (hybrid dense + BM25 sparse) |
| Reranker | Cohere `rerank-v3.5` |
| LLM | Claude Opus 4.5 (answers), Haiku (fast tasks), Sonnet (classification) |
| Backend | Python FastAPI, SQLite, async streaming |
| Frontend | React 19 + TypeScript + Tailwind CSS |

## Configuration

All configuration is in `.env`. The defaults work out of the box — you only need to add API keys.

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=pa-...
COHERE_API_KEY=...

# Auth is off by default (no login needed for local use)
# Set ENABLE_AUTH=true if you expose PaperPrism to the internet
```

See `.env.example` for all available options.

## Data Persistence

Your papers, conversations, and vector embeddings are stored in Docker volumes. They survive container stops, restarts, and laptop reboots.

```bash
docker compose down       # stops everything — data is safe
docker compose up         # starts again — everything is where you left it
docker compose down -v    # ⚠️ DELETES ALL DATA (volumes removed)
```

To back up your papers:
```bash
docker compose cp backend:/app/papers ./backup_papers
```

## Development (Without Docker)

```bash
# Backend
cd backend
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env  # add API keys
docker compose up qdrant -d  # just Qdrant
uvicorn api.main:app --port 8000 --reload

# Frontend (separate terminal)
cd frontend
npm install
npm run dev   # http://localhost:5173, proxies /api to :8000
```

## License

[BSL 1.1](LICENSE) — source available, free for non-commercial and academic use. Each release converts to Apache 2.0 after 4 years. Commercial use requires a license — contact [your email].
