# ARC: Research Paper RAG System — Full Pipeline Architecture

> A deep-dive technical document covering every stage of the ARC pipeline, the design rationale behind each decision, and how the components work together end-to-end.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Infrastructure & Deployment](#2-infrastructure--deployment)
3. [Ingestion Pipeline: PDF to Vectors](#3-ingestion-pipeline-pdf-to-vectors)
   - 3.1 [PDF Extraction (MinerU)](#31-pdf-extraction-mineru)
   - 3.2 [Section Detection](#32-section-detection)
   - 3.3 [Multi-Type Chunking](#33-multi-type-chunking)
   - 3.4 [Embedding Generation (Voyage AI)](#34-embedding-generation-voyage-ai)
   - 3.5 [Sparse Vector Generation (BM25)](#35-sparse-vector-generation-bm25)
   - 3.6 [Vector Storage (Qdrant)](#36-vector-storage-qdrant)
   - 3.7 [Batch Indexing with Checkpointing](#37-batch-indexing-with-checkpointing)
4. [Query Pipeline: Question to Answer](#4-query-pipeline-question-to-answer)
   - 4.0 [Conversation Reference Resolution](#40-conversation-reference-resolution)
   - 4.1 [Entity Extraction](#41-entity-extraction)
   - 4.2 [Query Classification (Dual-Strategy)](#42-query-classification-dual-strategy)
   - 4.3 [Query Expansion (LLM-Based)](#43-query-expansion-llm-based)
   - 4.4 [Query Decomposition](#44-query-decomposition)
   - 4.5 [Embedding (with optional HyDE)](#45-embedding-with-optional-hyde)
   - 4.6 [Hybrid Retrieval (Dense + Sparse)](#46-hybrid-retrieval-dense--sparse)
   - 4.7 [Entity Boosting](#47-entity-boosting)
   - 4.8 [Reranking (Cohere)](#48-reranking-cohere)
   - 4.9 [Retrieval Quality Evaluation](#49-retrieval-quality-evaluation)
   - 4.10 [Conditional Re-Retrieval](#410-conditional-re-retrieval)
   - 4.11 [Parent Chunk Expansion](#411-parent-chunk-expansion)
   - 4.12 [Answer Generation (Claude)](#412-answer-generation-claude)
   - 4.13 [Citation Verification](#413-citation-verification)
   - 4.14 [Web Search (Optional)](#414-web-search-optional)
   - 4.15 [Conversation Memory Update](#415-conversation-memory-update)
5. [Data Model & Storage](#5-data-model--storage)
6. [Caching Strategy](#6-caching-strategy)
7. [API Layer & Streaming](#7-api-layer--streaming)
8. [Authentication & User Preferences](#8-authentication--user-preferences)
9. [Data Cleaning & PDF Classification](#9-data-cleaning--pdf-classification)
10. [Design Rationale: Why This Architecture](#10-design-rationale-why-this-architecture)

---

## 1. System Overview

ARC is a Retrieval-Augmented Generation (RAG) system purpose-built for querying scientific research papers. It is designed for a single researcher (or small team) who wants to have an intelligent conversation with their personal paper library — asking factual questions, comparing methods across papers, positioning novelty claims, and getting cited answers grounded in their actual uploaded literature.

### High-Level Data Flow

```
                     ┌──────────────────────────────────────────────┐
                     │              INGESTION PIPELINE               │
                     │                                              │
   PDF Files ──────▶ │  MinerU ──▶ Section Detection ──▶ Chunker   │
                     │     │           (6 chunk types)              │
                     │     ▼                                        │
                     │  Tables, Captions, Figures                   │
                     │     │                                        │
                     │     ▼                                        │
                     │  Voyage AI Embeddings (1024-dim)             │
                     │  + BM25 Sparse Vectors                      │
                     │     │                                        │
                     │     ▼                                        │
                     │  Qdrant Vector Database                     │
                     └──────────────────────────────────────────────┘

                     ┌──────────────────────────────────────────────┐
                     │               QUERY PIPELINE                  │
                     │                                              │
   User Query ─────▶ │  Classify (top-2) ──▶ LLM Expand             │
                     │     │                                        │
                     │     ▼                                        │
                     │  Decompose? ──▶ Embed ──▶ Search each       │
                     │     │           (per sub-query if complex)   │
                     │     ▼                                        │
                     │  Merge + Dedup (merged chunk types)          │
                     │     │                                        │
                     │     ▼                                        │
                     │  Rerank (Cohere)                             │
                     │     │                                        │
                     │     ▼                                        │
                     │  Quality Eval (Haiku: gaps?)                 │
                     │     │                                        │
                     │     ├─ OK ──▶ Parent Expansion               │
                     │     └─ Gaps ──▶ Re-Retrieve ──▶ Re-Rerank   │
                     │                      │                       │
                     │                      ▼                       │
                     │  Claude Opus (Answer Generation)             │
                     │     │                                        │
                     │     ▼                                        │
                     │  Citation Verification (per-usage) ──▶ Resp  │
                     └──────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Why This Choice |
|---|---|---|
| **API Framework** | FastAPI (Python) | Async-native, built-in OpenAPI docs, excellent for streaming SSE |
| **PDF Extraction** | MinerU (PDF-Extract-Kit) | Layout-aware extraction with table/formula/OCR support — far superior to naive text extraction |
| **Embeddings** | Voyage AI `voyage-3-large` (1024-dim) | State-of-the-art retrieval embeddings; outperforms OpenAI/Cohere on MTEB benchmarks for scientific text |
| **Vector Database** | Qdrant (self-hosted, Docker) | Native hybrid search with Prefetch+RRF, payload indices, sparse vector support — all in one system |
| **Reranker** | Cohere `rerank-v3.5` | Cross-encoder reranking catches semantic matches that bi-encoder embedding misses |
| **LLM (Main)** | Claude Opus 4.5 | Best reasoning for synthesizing multi-source scientific answers with proper citations |
| **LLM (Fast)** | Claude Haiku 4.5 | Cost-efficient for HyDE generation, query expansion, entity extraction, retrieval quality evaluation, citation verification |
| **LLM (Classifier)** | Claude Sonnet 4.5 | Good reasoning for query classification without needing full Opus |
| **Sparse Search** | Custom BM25 | Lexical matching for exact scientific terms (chemical names, gene symbols) that semantic search misses |
| **Database** | SQLite (async) | Zero-config, sufficient for single-user/small-team; stores conversations, preferences, upload tasks |
| **Auth** | JWT + bcrypt | Standard stateless auth; single-user mode with sensible defaults |

---

## 2. Infrastructure & Deployment

### Architecture

```
┌─────────────────────────────────────────────┐
│                Mac Mini (Local)               │
│                                               │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │ FastAPI   │  │ Qdrant   │  │  SQLite DB │ │
│  │ :8000     │  │ :6333    │  │  app.db    │ │
│  └─────┬────┘  └──────────┘  └────────────┘ │
│        │                                      │
│        │  Cloudflare Tunnel                   │
│        ▼                                      │
│  ┌──────────┐                                │
│  │ Internet │ ◀───── Frontend (Vercel)       │
│  └──────────┘                                │
└─────────────────────────────────────────────┘

External APIs:
  ├── Anthropic (Claude) — Answer generation, classification, HyDE
  ├── Voyage AI — Embedding generation
  └── Cohere — Reranking
```

### Why Self-Hosted on Mac Mini

- **Data sovereignty**: Research papers stay on local hardware; no sensitive data leaves the machine except for embedding/LLM API calls (which send text chunks, not full PDFs).
- **Cost**: Qdrant is free when self-hosted. Cloud-hosted vector DBs charge per-vector pricing that scales poorly with large paper libraries.
- **Latency**: Local vector search is sub-millisecond. No network round-trip for retrieval.
- **Storage**: External 2TB drive holds the Qdrant data (~227 GB for the full collection); internal SSD holds PDFs and code.

### Server Management

The system includes shell scripts for operational reliability:

- **`restart_server.sh`**: Full restart sequence — kills backend, restarts Docker/Qdrant, waits for health check, starts FastAPI, restarts Cloudflare Tunnel.
- **`check_status.sh`**: Diagnostic tool that verifies Docker, Qdrant, backend, API, and tunnel status.
- **`com.arc.autostart.plist`**: macOS LaunchAgent for auto-start after reboot/OS update.

---

## 3. Ingestion Pipeline: PDF to Vectors

This pipeline transforms raw PDF files into searchable, semantically-indexed chunks stored in Qdrant. Every design decision is optimized for scientific paper retrieval.

### 3.1 PDF Extraction (MinerU)

**File**: `preprocessing/pdf_processor.py` — `MinerUExtractor` class

**What it does**: Converts PDF bytes into structured content (text, tables, figures, captions, metadata).

**Why MinerU over simpler extractors** (PyMuPDF, pdfplumber, etc.):

Scientific papers are visually complex — two-column layouts, embedded tables with chemical data, mathematical formulas, figure captions intermixed with body text. Naive text extractors produce garbled output because they read linearly without understanding layout.

MinerU uses a multi-model pipeline:
1. **DocLayout-YOLO**: Detects page layout regions (text blocks, tables, figures, headers, footers)
2. **StructEqTable**: Extracts structured table data (preserving rows/columns)
3. **UniMERNet**: Recognizes mathematical and chemical formulas
4. **OCR**: Handles scanned documents or image-heavy pages

**Extraction flow** (MinerU v3 API):
```
PDF bytes
  ├── doc_analyze_streaming() ──▶ Layout detection + region classification
  │     (callback-based: on_doc_ready delivers middle_json)
  └── union_make() ──▶ Markdown + Content list
       ├── Text blocks (with section structure preserved)
       ├── Tables (LaTeX/markdown format)
       ├── Figures (with bounding boxes)
       └── Captions (regex-extracted from markdown)
```

**Timeout & Fallback**: MinerU is GPU-intensive. If extraction exceeds 900 seconds, the system falls back to `pypdfium2` for basic text extraction. This ensures no single problematic PDF blocks the entire pipeline.

**Memory Safety**: Extraction runs in a `ThreadPoolExecutor` with daemon threads. On timeout, the future is cancelled and `gc.collect()` is called to reclaim model state. macOS `spawn` multiprocessing is forced to avoid fork() issues.

**Metadata Enrichment**: After extraction, the processor attempts:
1. PDF metadata (title, authors from document properties via `pypdfium2`)
2. DOI extraction (from first 3 pages of text, with regex patterns for various DOI formats)
3. CrossRef API lookup (if DOI found — fetches authoritative title, authors, year, journal)

### 3.2 Section Detection

**File**: `preprocessing/section_detector.py` — `SectionDetector` class

**What it does**: Identifies logical sections (Introduction, Methods, Results, Discussion, etc.) from extracted text.

**Why this matters**: Different query types need different sections. A "methods" query should search Methods sections; a "novelty" query should search Discussion and Introduction. Without section detection, you're searching everything and diluting relevance.

**Detection approach**: Line-by-line regex matching against known section header patterns:

```python
SECTION_PATTERNS = [
    # Combined sections (common in chemistry)
    ("results and discussion", "results_discussion"),
    ("materials and methods", "methods"),
    # Standard sections
    ("introduction", "introduction"),
    ("methods", "methods"),
    ("results", "results"),
    ("discussion", "discussion"),
    ("conclusion", "conclusion"),
    # Chemistry-specific
    ("synthesis", "synthesis"),
    ("characterization", "characterization"),
    ("biological evaluation", "bioactivity"),
    ("SAR", "sar"),
    # ...
]
```

**Subsection handling**: Numbered subsections (e.g., "2.3 Cell Culture") are detected and inherit their parent section's normalized name. The raw subsection header text is preserved for context.

**Abstract extraction**: Special regex handling for the abstract, which uses multiple patterns to handle variations:
- "Abstract" followed by Introduction/Keywords
- "Abstract:" followed by text until double newline
- Searches within first 10,000 characters only

### 3.3 Multi-Type Chunking

**File**: `preprocessing/chunker.py` — `PaperChunker` class

This is the core design innovation of the ingestion pipeline. Instead of creating one type of chunk (like most RAG systems), ARC creates **6 distinct chunk types**, each optimized for different retrieval needs.

#### The 6 Chunk Types

| Type | Purpose | Token Limit | When Used in Retrieval |
|---|---|---|---|
| **ABSTRACT** | Full paper abstract as one chunk | 300 tokens (≈1200 chars) | Summary queries, comparative queries (need overview of many papers) |
| **SECTION** | Logical sections (Introduction, Methods, Results, etc.) | 2000 tokens (3000 for high-value sections like Discussion/Results) | Methods queries (filter to methods sections), framing queries |
| **FINE** | Semantic paragraph-level chunks with sentence boundaries | 500 tokens, 128 token overlap | Factual queries (specific facts, values, mechanisms) |
| **CAPTION** | Figure and table captions with `[Figure/Table Caption]` prefix | Variable | Factual queries about specific figures or data |
| **TABLE** | Extracted table content with `[Table Content]` prefix | Variable | Factual queries about IC50 values, activity data, etc. |
| **FULL** | Mean-pooled embedding of all chunks (paper-level representation) | N/A (derived) | Summary queries needing whole-paper similarity |

#### Why 6 Chunk Types Instead of 1

The fundamental problem with single-chunk-type RAG: **the optimal chunk size depends on the question being asked**.

- A factual query ("What is the IC50 of compound X?") needs a small, precise chunk — a 2000-token section would dilute the answer with irrelevant context.
- A summary query ("Summarize this paper's contribution") needs broad context — a 500-token chunk can't capture the full narrative.
- A comparative query ("How does SRS compare to fluorescence microscopy?") needs abstracts from many papers, not deep dives into one.
- A methods query ("How are stapled peptides synthesized?") needs the Methods section specifically, not the Introduction or Discussion.

Multi-type chunking lets the query classifier select the right chunk types per query, rather than forcing one chunk size to serve all purposes.

#### Chunking Implementation Details

**Sentence-aware splitting**: Fine chunks never split mid-sentence. The chunker uses regex to detect sentence boundaries (`(?<=[.!?])\s+(?=[A-Z])`) and accumulates sentences until the token target is reached.

**Paragraph-aware processing**: Before sentence splitting, text is divided into paragraphs (double newlines or indentation patterns). The chunker tries to keep paragraphs intact. Only when a single paragraph exceeds the target does it fall down to sentence-level splitting.

**Contextual headers**: Every fine chunk is prefixed with its section context: `[Methods] ...chunk text...`. This means that even when a fine chunk is retrieved in isolation, the LLM knows which section it came from — critical for interpreting methods details vs. discussion claims.

**Parent-child relationships**: Fine chunks store a `parent_chunk_id` linking back to their section chunk. During retrieval, if a fine chunk is selected, its parent section can be fetched for additional context (see [4.9 Parent Chunk Expansion](#49-parent-chunk-expansion)).

**High-value section boost**: Discussion and Results sections get an extra 1000 tokens (3000 total) because they contain the most unique insights and interpretations.

**Fallback**: If the section detector finds no structure (e.g., a poorly formatted PDF), the entire text is chunked using the fine-chunk strategy with a generic `[Content]` prefix.

**Tokenizer**: `tiktoken` with `cl100k_base` encoding (same tokenizer as Claude and GPT-4) ensures accurate token counting.

### 3.4 Embedding Generation (Voyage AI)

**File**: `retrieval/embedder.py` — `VoyageEmbedder` class

**Model**: `voyage-3-large` producing **1024-dimensional** dense vectors.

**Why Voyage AI**: At the time of building, `voyage-3-large` ranked highest on the MTEB retrieval benchmarks for scientific/technical text. It supports separate `input_type` for documents vs. queries, which improves asymmetric retrieval (short query → long document).

**Batch processing**: Texts are embedded in batches of 128 per API call, with rate limiting (2000 RPM, 3M TPM). Exponential backoff on rate limit errors (1s, 2s, 4s, 8s, 16s).

**Mean-pooled full-paper embedding**: For the FULL chunk type, all section embeddings of a paper are averaged (mean-pooled) and L2-normalized. This creates a single 1024-dim vector representing the entire paper's semantic content — useful for paper-level similarity search without the information loss of truncation.

```python
# Mean pooling with L2 normalization
embeddings_array = np.array(embeddings)
mean_embedding = np.mean(embeddings_array, axis=0)
norm = np.linalg.norm(mean_embedding)
mean_embedding = mean_embedding / norm  # Unit vector
```

### 3.5 Sparse Vector Generation (BM25)

**File**: `retrieval/bm25.py` — `BM25Vectorizer` class

**What it does**: Generates sparse BM25 vectors alongside the dense Voyage embeddings, enabling hybrid search.

**Why BM25 matters for scientific text**: Dense embeddings excel at semantic similarity ("similar meaning, different words") but can miss **exact lexical matches** that are critical in science:
- Chemical names: "SRS microscopy" ≠ "stimulated Raman scattering" in embedding space, but a user searching for "SRS" needs exact lexical matching.
- Gene symbols: "BRCA1", "TP53" — these are identifiers, not semantic concepts.
- Acronyms: "HPLC", "LC-MS", "NMR" — embeddings may confuse similar-sounding techniques.

**BM25 scoring formula**:
```
score(term, doc) = IDF(term) × (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × (|doc| / avgdl)))
```
Where `k1 = 1.5`, `b = 0.75`, and IDF is computed from the indexed corpus.

**Tokenization**: Extracts alphanumeric tokens, preserves scientific notation (e.g., "IC50", "1.5M"), filters scientific stopwords (common words like "study", "method", "result" that appear in every paper).

**IDF persistence**: IDF values are cached to `data/bm25_idf_cache.json` so they persist across server restarts. Incremental updates add new documents without recomputing from scratch.

**Hash-based vocabulary**: Terms are mapped to indices via `hash(term) % 50000` for a fixed-size sparse vector space. This avoids maintaining an explicit vocabulary while keeping collision rates low.

### 3.6 Vector Storage (Qdrant)

**File**: `retrieval/qdrant_store.py` — `QdrantStore` class

**Collection design**: All 6 chunk types live in a **single Qdrant collection** (`research_papers`), differentiated by the `chunk_type` payload field. This design (vs. separate collections per type) enables:
- Single-query retrieval across multiple chunk types with `MatchAny` filters
- Unified payload indices
- Simpler collection management

**Vector configuration**:
- **Dense**: 1024-dim, cosine distance (Voyage AI vectors are normalized)
- **Sparse**: Named sparse vector `"bm25"` (on-disk index for memory efficiency)

**Payload indices** (created on collection initialization for efficient filtering):
```
_chunk_id  → KEYWORD index
chunk_type → KEYWORD index
paper_id   → KEYWORD index
section_name → KEYWORD index
```

**Deterministic point IDs**: Chunk IDs are converted to UUIDs using `uuid5(NAMESPACE_UUID, chunk_id)` — deterministic and collision-resistant. This means re-indexing the same paper produces the same point IDs, enabling idempotent upserts.

**Hybrid search implementation**: Uses Qdrant's native `Prefetch + RRF` (Reciprocal Rank Fusion):
```python
prefetch=[
    Prefetch(query=dense_embedding, using="", limit=100),    # Dense
    Prefetch(query=sparse_vector, using="bm25", limit=100),  # Sparse
],
query=FusionQuery(fusion=Fusion.RRF)  # Reciprocal Rank Fusion
```

This is more principled than client-side score mixing because Qdrant handles normalization and fusion internally.

### 3.7 Batch Indexing with Checkpointing

**File**: `index_papers.py`

**Pipeline for each PDF**:
```
1. Scan PDF directory for .pdf files
2. Check checkpoint — skip already-indexed papers
3. For each unprocessed PDF:
   a. MinerU extraction (with 900s timeout)
   b. PaperChunker creates 6 chunk types
   c. VoyageEmbedder generates dense embeddings (batched)
   d. BM25Vectorizer generates sparse vectors
   e. QdrantStore upserts all points
   f. Save checkpoint (paper_id, file_hash)
4. Update BM25 IDF cache incrementally
5. Save IDF cache to disk
```

**Checkpointing**: After each successful paper, a checkpoint is written to `data/indexing_checkpoint.json`. If the process is interrupted (crash, rate limit, timeout), re-running `index_papers.py` resumes from the last checkpoint.

**Duplicate detection**: Uses MD5 file hashes to detect duplicate PDFs even with different filenames.

**Rate limit handling**: If a Voyage API rate limit is detected, the indexer pauses with exponential backoff rather than failing. Critical API errors (auth, payment) halt the process entirely.

---

## 4. Query Pipeline: Question to Answer

The query pipeline is the runtime core of PaperPrism. It transforms a user's natural language question into a cited, source-grounded answer. The pipeline uses dual-strategy classification with an LLM quality feedback loop — a conditional re-retrieval step that detects semantic gaps before answer generation.

**File**: `retrieval/query_engine.py` — `QueryEngine.query()` method

### 4.0 Conversation Reference Resolution

**File**: `retrieval/conversation_memory.py` — `ConversationMemory.resolve_references()`

Before any processing, the system resolves pronouns and references from multi-turn conversation:

- `"it"` / `"this"` / `"that"` → last mentioned subject from assistant's response
- `"the paper"` / `"this paper"` → last discussed paper title
- `"the method"` → last mentioned technique (detected via patterns like "using X chromatography")
- `"they"` / `"these"` → last mentioned plural subject

Very short queries (≤3 words) get prefixed with context from the previous question: `"Regarding 'How are stapled peptides synthesized...': What about the yield?"`.

**Why**: Without reference resolution, a follow-up like "What about its IC50?" would search for "its IC50" — meaningless to the retrieval system. After resolution, it becomes "What about [compound name]'s IC50?" — a precise, retrievable query.

### 4.1 Entity Extraction

**File**: `retrieval/entity_extractor.py` — `LLMEntityExtractor` class

Extracts domain-specific entities from the query for later use in boosting and filtering:

| Category | Examples | Detection Method |
|---|---|---|
| Chemicals | Drug codes (AB-12345), compound names | Regex + LLM |
| Proteins/Genes | ERα, BRCA1, TP53, LL-37 | Regex + LLM |
| Methods | HPLC, SRS, Western blot, flow cytometry | Regex + LLM |
| Organisms | E. coli, HeLa, HEK293, mouse | Regex + LLM |
| Metrics | IC50, EC50, nM, μM, pH values | Regex + LLM |

**Dual approach**: LLM extraction (Haiku) for accuracy, with rule-based regex fallback if the API call fails. The LLM understands context — "SRS" in "SRS microscopy" is a method, but "SRS" in "SRS gene" is a gene.

**Why extract entities**: These are used in entity boosting (Step 4.6) to upweight search results that contain the same entities mentioned in the query. This is especially important for scientific queries where the exact entity name is the most critical retrieval signal.

### 4.2 Query Classification (Dual-Strategy)

**File**: `retrieval/query_classifier.py` — `QueryClassifier` class

The classifier determines the **type** of question being asked, which controls the entire retrieval strategy.

#### 8 Query Types

| Type | Description | Example | Retrieval Strategy |
|---|---|---|---|
| **FACTUAL** | Specific facts, values, mechanisms | "What is the IC50 of compound X?" | fine, table, caption chunks; top-k=50 |
| **METHODS** | Technical protocols, procedures | "How are stapled peptides synthesized?" | section, fine chunks; filter to methods/experimental sections; top-k=50 |
| **SUMMARY** | Summarize a paper or findings | "Summarize this paper's key findings" | abstract, section, full chunks; top-k=20 |
| **COMPARATIVE** | Compare approaches across papers | "How does SRS compare to fluorescence?" | abstract, section chunks; top-k=100 (need many papers) |
| **NOVELTY** | Assess prior art, gaps, what's new | "Has SRS been applied to biofilm penetration?" | abstract, section; filter to intro/discussion; top-k=50 |
| **LIMITATIONS** | Constraints, caveats, weaknesses | "What are the limitations of this approach?" | section, abstract; filter to discussion/conclusion; top-k=50 |
| **FRAMING** | How to position/justify research | "How do I frame Raman imaging as enabling?" | abstract, section chunks; top-k=30 |
| **GENERAL** | Doesn't fit above categories | "Tell me about this topic" | all chunk types; top-k=50 |

#### Why Classification Matters

Each query type has a fundamentally different retrieval need:

- **FACTUAL** queries need small, precise chunks (fine/table/caption) because the answer is a specific value buried in text. Searching section-level chunks would drown the signal in noise.
- **SUMMARY** queries need abstract and full-paper embeddings because they require holistic understanding, not granular details.
- **COMPARATIVE** queries need high top-k (100) across many papers because they synthesize information from multiple sources, each contributing just one data point.
- **METHODS** queries benefit from section filtering — there's no point searching the Introduction when the user wants synthesis protocols from the Methods section.

Without classification, every query uses the same retrieval strategy, which means either precision suffers (too much irrelevant context) or recall suffers (not enough diverse sources).

#### Dual-Strategy Classification

Instead of betting everything on a single classification, the system uses **`classify_multi()`** to get the top-2 most relevant query types. This hedges against misclassification:

1. **Primary type** drives: section filter, per-paper limits, answer generation system prompt
2. **Merged chunk types** from both top-2 types are used for search — casting a wider retrieval net
3. **Max top-k** from both strategies ensures enough candidates for reranking
4. **Section filters** are unioned — if primary suggests "methods" and secondary suggests "discussion", both are searched

**LLM classification** (Sonnet): A structured prompt asks Claude to return the top-3 types ranked by confidence. The prompt includes domain-neutral disambiguation rules and few-shot examples.

**Confidence threshold**: If primary type confidence < 0.6, the system falls back to GENERAL strategy.

**Heuristic fallback**: If LLM classification fails or is disabled, a keyword-based heuristic classifier uses ~200 signal terms for zero-cost classification.

**Why dual-strategy**: A query like "How were these compounds synthesized and what were the results?" is both METHODS and FACTUAL. Single classification forces a choice; dual-strategy searches both chunk type sets and lets the reranker sort out what's most relevant.

### 4.3 Query Expansion (LLM-Based)

**File**: `retrieval/query_expander.py` — `QueryExpander` class

Generates domain-appropriate synonyms at query time using Claude Haiku. Works across all scientific domains without hardcoded dictionaries.

**Example**:
```
Input:  "CRISPR off-target effects"
Output: "CRISPR off-target effects (related terms: Cas9 specificity, guide RNA mismatch, genome editing fidelity, unintended mutations)"
```

**How it works**: A Haiku call generates 3-5 alternative terms/phrasings that papers might use for the same concepts. The prompt asks for abbreviation expansions, synonyms, and related technical terminology.

**Why LLM-based**: The system is domain-agnostic — users upload papers on any topic. A static synonym dictionary only works for one domain. The LLM adapts to whatever terminology appears in the query, whether it's genomics, materials science, or economics.

**Fallback**: On LLM failure, the query is used as-is. No expansion is better than bad expansion.

**Cost**: ~200ms latency, one Haiku call per query. Negligible cost relative to the downstream Opus generation call.

### 4.4 Query Decomposition

**File**: `retrieval/query_engine.py` — `_decompose_query()`

Complex queries that ask about multiple distinct aspects are decomposed into 2-3 focused sub-queries, each searched independently.

**Example**:
```
Input:  "How does BERT compare to GPT in pre-training approach and downstream performance?"
Output: ["What are the differences between BERT and GPT pre-training approaches?",
         "How do BERT and GPT compare in downstream task performance?"]
```

**How it works**: Haiku analyzes the query and returns a JSON response indicating whether decomposition is needed. Only triggers for genuinely multi-aspect queries — comparisons, conjunctions, cause-and-effect chains. Simple queries like "What is the IC50?" pass through unchanged.

**Multi-query retrieval**: When decomposed, the system:
1. Searches the original expanded query (primary results)
2. Embeds and searches each sub-query independently
3. Merges all results, deduplicating by chunk ID
4. The merged set goes to reranking — the reranker sorts out relevance across all sub-query contributions

**Why this matters**: A comparative query like "How does BERT compare to GPT?" might retrieve chunks heavily about one model but miss the other. By searching "BERT pre-training" and "GPT pre-training" separately, both sides of the comparison get adequate retrieval coverage.

**Provenance tracking**: Each result from a sub-query is tagged with `_sub_query` for debugging which sub-query contributed which chunks.

### 4.5 Embedding (with optional HyDE)

**File**: `retrieval/hyde.py` — `HyDE` and `HyDEEmbedder` classes

#### Standard Embedding

The expanded query is embedded using `VoyageEmbedder.embed_query()` with `input_type="query"` (Voyage AI optimizes differently for queries vs. documents).

#### HyDE (Hypothetical Document Embeddings)

When enabled, instead of embedding the raw query, the system:

1. **Generates a hypothetical answer** using Claude Haiku — a paragraph written in academic style that would plausibly appear in a paper answering the query
2. **Combines** the original query + hypothetical answer
3. **Embeds the combined text**

```
Query: "What is the mechanism of LL-37 against biofilms?"

Hypothetical (generated by Haiku):
"LL-37, a 37-residue human cathelicidin peptide, disrupts biofilm integrity 
through multiple mechanisms. At sub-MIC concentrations, LL-37 interferes with 
quorum sensing by downregulating Las and Rhl systems in Pseudomonas aeruginosa, 
reducing biofilm biomass by 40-60%..."

Embedding input: "Query: What is the mechanism... \n\nRelevant excerpt: LL-37, a 37-residue..."
```

**Why HyDE works**: A short query like "LL-37 mechanism against biofilms" occupies a very different region of embedding space than the actual paper text that answers it. The hypothetical answer is semantically closer to real paper text, bridging the query-document gap.

**Why it's optional**: HyDE adds ~500ms (Haiku generation) + API cost. For simple factual queries, direct embedding works fine. HyDE shines on complex, abstract queries where the gap between query and document text is large.

**Query-type-specific prompts**: HyDE generates different hypothetical styles based on query type:
- METHODS → generates a Methods section excerpt
- FACTUAL → generates a technical paragraph with specific values
- DISCUSSION → generates a Discussion section excerpt

### 4.6 Hybrid Retrieval (Dense + Sparse)

**File**: `retrieval/qdrant_store.py` — `QdrantStore.hybrid_search()`

Two parallel retrieval paths, fused via Reciprocal Rank Fusion:

```
                    ┌─────────────────────────┐
                    │     Expanded Query       │
                    └──────────┬──────────────┘
                               │
               ┌───────────────┼───────────────┐
               ▼                               ▼
    ┌────────────────────┐          ┌────────────────────┐
    │  Dense Search       │          │  Sparse Search     │
    │  (Voyage embedding) │          │  (BM25 vectors)    │
    │  Cosine similarity  │          │  Lexical matching  │
    │  Prefetch: 100      │          │  Prefetch: 100     │
    └────────┬───────────┘          └────────┬───────────┘
             │                               │
             └──────────┬───────────────────┘
                        ▼
              ┌──────────────────┐
              │  RRF Fusion      │
              │  (Qdrant native) │
              │  Top-50 results  │
              └──────────────────┘
```

**Reciprocal Rank Fusion (RRF)**: Instead of weighted score averaging (which requires normalization across incompatible score spaces), RRF combines by rank position:

```
RRF_score(doc) = Σ 1 / (k + rank_in_list)
```

This is robust because it doesn't depend on absolute score values — only relative ordering matters.

**Filtering**: Results are filtered by chunk type and optionally by section name, based on the classification strategy. For a METHODS query:
```python
chunk_types=["section", "fine"]
section_filter=["methods", "experimental", "synthesis"]
```

**Fallback**: If hybrid search fails (e.g., sparse vectors not yet indexed), the system automatically falls back to dense-only search.

### 4.7 Entity Boosting

After retrieval, results are boosted based on entity overlap with the query:

```python
for result in results:
    entity_score = entity_extractor.score_chunk_relevance(query, result['text'])
    result['score'] *= (1 + 0.1 * entity_score)
```

**How scoring works**: Entities are extracted from both the query and each chunk. The overlap ratio (matched entities / query entities) becomes a multiplicative boost factor (up to 10% boost).

**Why a multiplicative boost, not a filter**: Filtering would exclude potentially relevant chunks that discuss the entity without mentioning it by name (e.g., a paragraph about "the peptide's activity" without naming LL-37). Boosting gently promotes entity-containing chunks without excluding others.

### 4.8 Reranking (Cohere)

**File**: `retrieval/reranker.py` — `CohereReranker` class

**Model**: `rerank-v3.5` — a cross-encoder that scores each (query, document) pair jointly.

**Why reranking is essential**: Bi-encoder embedding search (Step 4.6) is fast but approximate. It encodes query and document independently and compares their vectors. A cross-encoder processes query and document **together**, enabling attention across both texts — catching nuanced relevance that independent encoding misses.

**Two-stage retrieval** (cast wide, then filter tight):
```
Step 4.6: Retrieve top-50 candidates (fast, bi-encoder)
Step 4.8: Rerank to top-15 (slow, cross-encoder)
```

This architecture is standard in production search systems. The first stage uses cheap vector similarity to narrow 50,000+ chunks to 50 candidates. The second stage uses expensive cross-encoder attention to find the truly best 15.

**Per-paper deduplication with soft caps**: `rerank_with_metadata()` enforces a `max_per_paper` limit (varies by query type: 2-25 depending on whether single-paper or cross-corpus). Caps are **soft** — chunks with rerank_score > 0.5 are allowed through up to 2x the quota, preventing high-quality chunks from being discarded purely for diversity. This prevents one paper from dominating while still allowing deep coverage when one paper is genuinely the best source.

**Query-type-specific limits**:
- FACTUAL: 3 per paper (want diversity)
- METHODS: 5 per paper (details span chunks)
- SUMMARY: 10 per paper (deep single-paper analysis)
- COMPARATIVE: 2 per paper (need breadth across many papers)

**Adaptive behavior when specific papers are selected**:
- 1 paper selected → max_per_paper=25, rerank_top_n doubled (deep dive)
- 2-3 papers → max_per_paper=15
- 4+ papers → max_per_paper=8

### 4.9 Retrieval Quality Evaluation

**File**: `retrieval/query_engine.py` — `_evaluate_retrieval_quality()`

After reranking, an LLM (Haiku) evaluates whether the retrieved chunks actually cover the query — catching semantic gaps that reranker scores can't detect.

**How it works**: The top-5 reranked chunks are sent to Haiku with the query. The model rates coverage on a 1-5 scale:
- **5**: Fully covered — ready to generate
- **4**: Mostly covered — minor gaps
- **3**: Partially covered — important aspects missing
- **2**: Poorly covered — most info missing
- **1**: Not covered — chunks are irrelevant

If confidence ≤ 3, the evaluator also returns:
- **What's missing**: Natural language description of gaps
- **Search terms**: 2-3 specific terms for targeted re-retrieval

**Why this matters**: Reranker scores measure relevance of individual chunks to the query. But high-scoring chunks can still miss key aspects. A query about "synthesis protocol and yield" might retrieve great chunks about the synthesis but nothing about yields. The LLM catches this gap.

**Cost**: One Haiku call per query (~200ms). Runs on every query — quality over speed.

### 4.10 Conditional Re-Retrieval

**File**: `retrieval/query_engine.py`

When the quality evaluator detects gaps, the system performs a single targeted re-retrieval:

```
Quality Eval → confidence ≤ 3 → Re-Retrieval
  1. Build broadened query: original + LLM-suggested search terms
  2. Re-embed the broadened query
  3. Search ALL chunk types (no section filter) with 1.5× top-k
  4. Re-rerank results
  5. Compare: use re-retrieval only if new max_score > original
```

**Why targeted, not just broader**: The quality evaluator returns specific search terms based on what's missing. Searching for "yield, purification efficiency" is far more effective than blindly broadening.

**Hard limit**: Maximum 1 retry. No recursive re-retrieval loops. Most queries pass quality eval and skip this step entirely.

**Frontend**: The "Quality Check" and "Re-Retrieval" pipeline steps show in the UI. Quality Check displays the confidence score (e.g., "4/5"). Re-Retrieval shows "improved" or "skipped".

### 4.11 Parent Chunk Expansion

**File**: `retrieval/query_engine.py` — `_expand_fine_chunks()`

When a fine chunk is retrieved, it may lack sufficient context. This step fetches the parent section chunk and attaches the first 500 characters as `parent_context`.

**Batch retrieval**: All parent chunk IDs are collected and fetched in a single batch call (`get_chunks_by_ids()`), rather than one-by-one — minimizing Qdrant round-trips.

**Why**: A fine chunk might say "The IC50 was 2.3 nM" without mentioning which compound or assay. The parent section chunk provides that context, enabling Claude to generate a properly contextualized answer.

### 4.12 Answer Generation (Claude)

**File**: `retrieval/query_engine.py` — `_generate_answer()`

**Model**: Claude Opus 4.5 for maximum reasoning quality.

**Query-type-specific system prompts**: Each of the 8 query types has two prompt variants:

- **Concise mode**: Short, focused answers (max 16,384 tokens)
- **Detailed mode**: Comprehensive, structured responses (max 32,768 tokens)

Example prompt for FACTUAL (concise):
> "You are a research assistant answering factual questions about scientific literature. Based on the retrieved sources, provide a direct, accurate answer. Include specific values, definitions, or mechanisms when available. Cite sources using [Source N] format."

Example prompt for NOVELTY (detailed):
> "Provide a comprehensive assessment covering: Prior Art Summary, Gap Analysis, Potential Novel Contributions, Strength of Novelty Claims, Differentiation Strategy, Risk Assessment, Supporting Evidence..."

**Source formatting**: Retrieved chunks are formatted as numbered sources:
```
[Source 1] (Paper: "Title...", Section: methods, Type: fine)
Text of the chunk...

[Source 2] (Paper: "Title...", Section: results, Type: table)
Text of the chunk...
```

**Conversation history**: Up to 2000 tokens of previous conversation turns are included in the messages array, giving Claude multi-turn awareness.

**General Knowledge mode**: When enabled, an addendum instructs Claude to first answer from sources with citations, then optionally add a "## Additional Context (General Knowledge)" section clearly marked as coming from training knowledge rather than uploaded papers.

**PDF Upload mode**: When enabled, full PDF documents (base64-encoded) are sent to Claude alongside the RAG chunks. Claude can then cross-reference the pre-identified relevant chunks with the complete paper context.

**Custom system prompts**: Users can customize the system prompt per query type and response mode via their preferences, overriding the defaults.

**Streaming**: When a `stream_callback` is provided (SSE endpoint), Claude's response is streamed token-by-token via the Anthropic streaming API. Each chunk is emitted as an SSE event to the frontend in real-time.

**Retry logic**: Rate limit errors (429) and server errors (5xx) trigger exponential backoff retry (up to 3 attempts, 1s → 2s → 4s delay).

### 4.13 Citation Verification

**File**: `retrieval/citation_verifier.py` — `CitationVerifier` class

After answer generation, the system verifies that each `[Source N]` citation actually supports the specific claim it's attached to.

**Per-usage verification**: When the same source is cited in multiple paragraphs making different claims, each usage is verified independently. The claim extractor splits on paragraph boundaries first, then sentences — respecting markdown structure instead of naively splitting on periods. Each (source_id, claim) pair gets its own verification with a contextual explanation.

**LLM-based verification (Haiku)**: Each (claim, source_text) pair is sent to Haiku, which returns a JSON response:
```json
{"verdict": "SUPPORTED", "confidence": 0.92, "explanation": "The source directly states the IC50 value of 2.3 nM in Table 2..."}
```

**Basic fallback (keyword overlap)**: If the LLM call fails, falls back to keyword analysis that reports specific matched/missing terms (e.g., "Partial match — found: synthesis, peptide; missing: yield, purification").

**Streaming verification**: During streaming responses, citations are verified in real-time. The `StreamingCitationVerifier` deduplicates by `(source_id, claim)` tuples — so the same source cited in two different paragraphs gets two separate verifications.

**Frontend display**: Citation badges are color-coded (green ≥70%, amber 30-70%, red <30%). Badges are clickable — a modal shows the full claim, detailed explanation, and confidence bar. Hover tooltip shows a brief status hint.

**Why verify citations**: LLMs hallucinate citations. Claude might attribute a claim to [Source 3] when the information actually comes from [Source 7] or from its own training data. Per-usage verification catches these misattributions at each point where a source is cited.

### 4.14 Web Search (Optional)

When both `enable_web_search` and `enable_general_knowledge` are enabled, after the RAG answer is generated, a separate Claude call is made with web search tool access to fetch publicly available information.

**Separation design**: Web search results are streamed as a distinct section after the RAG answer, clearly delineated. This prevents web results from contaminating the source-cited RAG answer.

### 4.15 Conversation Memory Update

After answer generation:
1. The user query is added to conversation history
2. The assistant answer and its sources are added
3. Paper context (paper_id → title mapping) is updated from the top 5 sources
4. History is trimmed to the last 10 turns (20 messages), keeping the first 2 messages for context

---

## 5. Data Model & Storage

### SQLite Database (Async via aiosqlite)

**File**: `database/models.py`

```
┌──────────┐     ┌──────────────┐     ┌──────────┐
│  User    │────▶│ Conversation │────▶│ Message  │
│          │     │              │     │          │
│ id       │     │ id           │     │ id       │
│ username │     │ user_id (FK) │     │ conv_id  │
│ pass_hash│     │ title        │     │ role     │
└──────┬───┘     │ created_at   │     │ content  │
       │         │ updated_at   │     │ metadata │
       │         └──────────────┘     └──────────┘
       │
       ├────▶ ┌──────────────┐
       │      │ UserMemory   │  (user context facts)
       │      │ category:    │  background / preference / interest
       │      │ content      │
       │      └──────────────┘
       │
       ├────▶ ┌──────────────────┐
       │      │ UserPreferences  │  (query settings per user)
       │      │ query_type       │
       │      │ top_k            │
       │      │ temperature      │
       │      │ response_mode    │
       │      │ enable_hyde      │
       │      │ enable_expansion │
       │      │ enable_citation  │
       │      │ enable_general_  │
       │      │   knowledge      │
       │      │ enable_web_search│
       │      │ enable_pdf_upload│
       │      │ system_prompts   │  (JSON - custom prompts)
       │      └──────────────────┘
       │
       └────▶ ┌──────────────┐
              │ UploadTask   │  (batch paper processing)
              │ batch_id     │
              │ file_name    │
              │ status       │  pending → uploading → processing →
              │ priority     │  extracting → embedding → indexing →
              │ progress     │  complete | error
              │ error_message│
              └──────────────┘
```

### Qdrant Collection Schema

```
Collection: research_papers

Point:
  id: UUID5(chunk_id)    ← deterministic
  vector: float[1024]     ← Voyage AI dense embedding
  sparse["bm25"]:         ← BM25 sparse vector
    indices: int[]
    values: float[]
  payload:
    _chunk_id: string
    chunk_type: "abstract" | "section" | "fine" | "caption" | "table" | "full"
    paper_id: string
    text: string
    title: string
    authors: string[]
    year: int?
    doi: string?
    section_name: string
    subsection_name: string?
    parent_chunk_id: string?  (fine chunks only)
    file_name: string
    project_tag: string?
    research_area: string?
    token_count: int

Payload Indices:
  _chunk_id    → KEYWORD
  chunk_type   → KEYWORD
  paper_id     → KEYWORD
  section_name → KEYWORD
```

---

## 6. Caching Strategy

**File**: `retrieval/cache.py` — `RAGCache` class

Three cache layers, all LRU with 1-hour TTL:

| Cache | Key | Value | Purpose |
|---|---|---|---|
| **Embedding cache** | query text hash | 1024-dim vector | Avoid re-embedding repeated queries |
| **Search results cache** | query + chunk_types + section_filter | Qdrant results | Avoid re-searching for same query |
| **HyDE cache** | query text hash | Hypothetical document | Avoid re-generating HyDE for same query |

**Automatic invalidation**: When new papers are indexed, `QdrantStore` tracks recently upserted paper IDs. On the next query, the cache checks for new papers and invalidates all entries if the index has changed. This prevents stale results.

**Thread safety**: The cache uses `threading.RLock` for thread-safe access from concurrent API requests.

**Cache statistics**: Hit/miss counts are tracked and exposed via the `/stats` endpoint.

---

## 7. API Layer & Streaming

**File**: `api/main.py`

### Key Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `POST /query` | Non-streaming | Full RAG query, returns complete JSON response |
| `POST /query/stream` | SSE streaming | Real-time streaming with step-by-step progress |
| `GET /health` | Health check | Checks all dependencies (Qdrant, API keys, DB) |
| `GET /health/quick` | Quick check | Lightweight status (no dependency checks) |
| `POST /conversation/clear` | Action | Reset conversation memory |
| `GET /stats` | Metrics | Pipeline statistics, cache info, analytics |
| `GET /papers` | List | Paginated paper listing with metadata |
| `POST /papers/upload` | Upload | Upload and index new PDFs |
| `DELETE /papers/{id}` | Delete | Remove paper (file + Qdrant cleanup) |
| `POST /auth/login` | Auth | JWT token generation |
| `POST /auth/refresh` | Auth | Token refresh |

### SSE Streaming Protocol

The `/query/stream` endpoint emits Server-Sent Events with this sequence:

```
event: entities         → {found: [...entities]}
event: classification   → {type, confidence, reasoning, chunk_types, secondary_types}
event: expansion        → {added_terms, expanded}
event: decomposition    → {sub_queries: [...]} | {skipped: true}
event: hyde             → {used: bool}
event: retrieval        → {count, cache_hit, decomposed, sub_queries}
event: reranking        → {output_count, success}
event: quality_eval     → {confidence: 1-5, missing?, search_terms?}
event: re_retrieval     → {status, reason?, improved?, search_terms?} | {skipped: true}
event: generation       → {status: "starting", source_count}
event: answer_chunk     → {chunk: "...token..."}  (repeated for each token)
event: answer_complete  → {answer: "...full text..."}
event: citation_verified → {citation_id, claim, confidence, is_valid, explanation}
event: web_search       → {status: "starting" | "complete"}
event: web_search_chunk  → {chunk: "...token..."}
event: done             → {}
```

This allows the frontend to show:
1. Pipeline progress (which step is active)
2. Classification result (what type of query was detected)
3. Streaming answer text (token by token)
4. Citation verification badges (as they're verified)
5. Web search results (after RAG answer completes)

### Middleware

- **Rate limiting**: Sliding window, 60 requests/minute per IP
- **Request tracing**: Every request gets a unique correlation ID for log tracing
- **Graceful shutdown**: Tracks in-flight requests; `/health` returns 503 during shutdown
- **CORS**: Configurable origins for frontend access

---

## 8. Authentication & User Preferences

### Authentication Flow

```
POST /auth/login { username, password }
  → Verify bcrypt hash
  → Generate JWT (24h expiry)
  → Return { token, user_id }

All subsequent requests:
  Authorization: Bearer <jwt_token>
  → Decode & verify JWT
  → Attach user to request context
```

### User Preferences

Each user has a `UserPreferences` record controlling query behavior:

| Setting | Type | Default | Effect |
|---|---|---|---|
| `query_type` | string? | null (auto) | Override query classification |
| `top_k` | int? | null (auto) | Override final result count |
| `temperature` | float? | null (0.3) | LLM temperature |
| `response_mode` | string | "detailed" | "concise" or "detailed" |
| `enable_hyde` | bool | false | HyDE for query embedding |
| `enable_expansion` | bool | true | Domain synonym expansion |
| `enable_citation_check` | bool | false | Citation verification |
| `enable_general_knowledge` | bool | true | Allow LLM to supplement with training knowledge |
| `enable_web_search` | bool | false | Web search after RAG answer |
| `enable_pdf_upload` | bool | false | Send full PDFs to Claude |
| `system_prompts` | JSON? | null | Custom system prompts per query type |

---

## 9. Data Cleaning & PDF Classification

**File**: `data_cleaning/classifier.py` and `data_cleaning/filters/`

Before papers enter the indexing pipeline, a classification system filters out non-paper PDFs that might be in a research folder:

### Filter Chain

```
PDF → FilenameFilter → ContentFilter → MetadataFilter → LLMFilter → Result
```

1. **FilenameFilter**: Rejects based on filename patterns (receipts, invoices, homework)
2. **ContentFilter**: Checks minimum text length, presence of abstract
3. **MetadataFilter**: Page count (too few = likely not a paper), language detection
4. **LLMFilter**: Claude-based content classification for edge cases

### Classification Results

- **PAPER**: Valid research paper → proceed to indexing
- **REJECTED**: Not a paper (receipt, homework, lecture notes, presentation, etc.)
- **UNCERTAIN**: Needs manual review via the `review_tool.py` interface

---

## 10. Design Rationale: Why This Architecture

### Why RAG Instead of Fine-Tuning

Fine-tuning a model on your papers would embed the knowledge in the model's weights — but you lose:
- **Citation**: Can't point to specific source passages
- **Updatability**: Adding a new paper requires re-training
- **Transparency**: Can't see what the model is basing its answer on
- **Accuracy**: Fine-tuned models still hallucinate, but now you can't verify against sources

RAG keeps the knowledge in a searchable, citable database. The LLM's job is synthesis and reasoning, not memorization.

### Why Hybrid Search (Dense + Sparse)

Neither dense nor sparse search alone is sufficient for scientific text:

- **Dense only** (embeddings): Good at "this means the same thing" but bad at exact term matching. A query for "BRCA1" might retrieve chunks about cancer genetics that don't mention BRCA1.
- **Sparse only** (BM25): Good at exact matching but bad at semantic understanding. A query about "drug resistance mechanisms" wouldn't find chunks about "antimicrobial tolerance pathways".

Hybrid search combines both, using RRF to merge rankings without requiring comparable score scales.

### Why Cohere Reranking After Retrieval

Bi-encoder retrieval (Voyage AI) scores each document independently of the query. It's fast (single embedding comparison) but misses query-document interactions.

Cross-encoder reranking (Cohere) processes (query, document) as a pair with full attention, catching:
- Negation: "proteins that do NOT bind to..." (bi-encoder might match documents about binding)
- Specificity: "IC50 of compound X" vs. a chunk that mentions IC50 and compound X separately but not together
- Relevance subtleties: A chunk that is topically related but doesn't actually answer the question

The two-stage architecture (fast bi-encoder → slow cross-encoder) is the standard approach in production search because it balances speed and quality.

### Why Multiple Claude Models

| Model | Used For | Why |
|---|---|---|
| **Opus** | Answer generation | Best reasoning for multi-source synthesis and accurate citation |
| **Sonnet** | Query classification | Good reasoning for classification but doesn't need Opus-level synthesis |
| **Haiku** | HyDE, query expansion, entity extraction, retrieval quality evaluation, citation verification | Fast and cheap for targeted tasks that don't need deep reasoning |

Using Opus for everything would be prohibitively expensive and slow. Using Haiku for answer generation would sacrifice quality on complex multi-source synthesis. The tiered approach matches model capability to task complexity.

### Why 6 Chunk Types Instead of Just "Small Chunks"

The naive RAG approach: split everything into 500-token chunks and retrieve the top-k.

The problem: For a summary query, 500-token chunks provide fragments, not narratives. For a comparative query across 20 papers, 500-token chunks from 3-4 papers dominate the results. For a methods query, chunks from the Introduction dilute the actual protocol details.

Multi-type chunking + query classification solves this by matching retrieval granularity to query intent. It's more complex to build, but it's the difference between a RAG system that sort-of works and one that consistently gives useful, well-grounded answers.

### Why Self-Hosted Qdrant

Cloud vector databases (Pinecone, Weaviate Cloud) charge per-vector pricing. With 6 chunk types per paper and potentially thousands of papers, costs scale fast. Self-hosted Qdrant on the Mac Mini is:
- **Free**: No per-vector charges
- **Fast**: Sub-millisecond local search
- **Feature-rich**: Native hybrid search, sparse vectors, payload indices — all used in this architecture
- **Controllable**: Data stays on local hardware

The tradeoff is operational complexity (Docker, backups, restarts), which the shell scripts and LaunchAgent handle.

---

*This document reflects PaperPrism as built. The system is designed to be domain-agnostic — no hardcoded domain knowledge. LLM-based expansion, dual-strategy classification, and quality-gated re-retrieval adapt to whatever scientific domain the user's papers cover. Every design decision was made to solve a specific problem encountered during development — from MinerU replacing simpler extractors that garbled two-column papers, to the quality feedback loop replacing single-pass retrieval that couldn't recover from bad first-pass results.*
