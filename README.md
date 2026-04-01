# PaperPrism — A Query-Adaptive Retrieval Pipeline for Scientific Literature

PaperPrism is a RAG (Retrieval-Augmented Generation) system purpose-built for querying scientific research papers. Upload your papers, ask questions, get cited answers with verified sources.

Like a prism splitting light into its spectrum, PaperPrism splits each paper into 6 distinct representations — abstracts, sections, fine-grained paragraphs, tables, captions, and a full-paper embedding — then selects the right facets to answer each question. A factual question refracts through precise table chunks. A comparative question refracts through abstracts across many papers. Same paper, different views, depending on what you ask.

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

## FAQ

### Why not just upload PDFs to ChatGPT?

You can — and for one or two papers it works fine. The problems start with a real research library:

- **Scale.** ChatGPT's context window fits ~5-10 papers. PaperPrism indexes hundreds permanently — always available, never re-uploaded.
- **Retrieval precision.** ChatGPT dumps your entire PDF into context and hopes the model finds the relevant passage. PaperPrism's 13-stage pipeline narrows thousands of chunks to the 15 most relevant before the LLM sees anything.
- **Query adaptation.** ChatGPT processes every question the same way. PaperPrism classifies your question and changes how it searches — different chunk types, result counts, section filters, and diversity limits per query type.
- **Structured extraction.** ChatGPT reads raw text. Two-column layouts get garbled, tables lose structure, captions mix into body text. PaperPrism uses MinerU for layout-aware extraction and creates separate searchable chunks for tables, captions, and sections.
- **Citation verification.** When ChatGPT says "according to the paper, the IC50 is 2.3 nM," you can't check if it actually found that or hallucinated it. PaperPrism verifies each `[Source N]` against the actual passage and shows a confidence score.
- **Persistence.** Your library stays indexed. Conversations persist. Come back tomorrow and ask a follow-up — PaperPrism resolves pronouns like "it" and "the paper" from your previous conversation.

### How is this different from other RAG systems?

Most open-source RAG systems (RAGFlow, Dify, AnythingLLM, etc.) use one retrieval strategy for every question. PaperPrism adapts.

**Query-adaptive retrieval** — every question gets classified, and the classification drives the entire retrieval strategy:

| Question Type | Example | Chunks Searched | Top-k | Per Paper |
|---|---|---|:---:|:---:|
| **Factual** | "What is the IC50 of compound X?" | fine, table, caption | 50 | max 3 |
| **Methods** | "How are stapled peptides synthesized?" | section, fine (Methods/Experimental only) | 50 | max 5 |
| **Summary** | "Summarize this paper" | abstract, section, full | 20 | max 10 |
| **Comparative** | "How does SRS compare to fluorescence?" | abstract, section | 100 | max 2 |
| **Limitations** | "What are the limitations of this approach?" | section (Discussion/Conclusion only) | 50 | max 4 |

No other open-source RAG system dynamically adjusts chunk types, top-k, section filters, and per-paper limits based on what you're asking.

**Citation verification** — every `[Source N]` citation is semantically verified against the actual source passage. Citations are color-coded by confidence: green (verified), amber (partially supported), red (weak support).

### How was it benchmarked?

Tested against direct Claude (same Opus 4.5 model, no RAG) on 15 queries with blind, randomized evaluation. A separate Claude Sonnet judge scored both answers without knowing which system produced which answer. Reference passages from the papers were provided for fact-checking.

| | PaperPrism | Direct Claude |
|---|:---:|:---:|
| **Accuracy** | **3.7** / 5 | 3.0 / 5 |
| **Completeness** | **4.0** / 5 | 3.7 / 5 |
| **Grounding** | **4.2** / 5 | 1.9 / 5 |
| **Wins** | **13** | 1 |

The grounding gap (4.2 vs 1.9) is structural — PaperPrism's answers are traceable to specific paper passages; Claude's are not.

## License

[BSL 1.1](LICENSE) — source available, free for non-commercial and academic use. Each release converts to Apache 2.0 after 4 years. Commercial use requires a license — contact yang.mengy@northeastern.edu.
