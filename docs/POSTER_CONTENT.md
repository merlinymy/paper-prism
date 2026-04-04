# Paper Prism Research Poster — Content & Layout Guide

> Copy text directly. Let the diagrams do the talking.

---

## Poster Specs

- **Size**: 48" wide x 36" tall (landscape)
- **Tool**: Canva, Google Slides (48x36in), or PowerPoint
- **Font sizes**: Title 72-96pt, Headers 36-44pt, Body 24-28pt, Captions 18-20pt

---

## Layout (3-Column, Top Banner)

```
┌──────────────────────────────────────────────────────────────────┐
│                         TITLE BANNER                              │
├──────────────────────┬──────────────────────┬─────────────────────┤
│                      │                      │                     │
│   THE PROBLEM        │                      │   BENCHMARK         │
│                      │   HOW IT WORKS       │   RESULTS           │
│                      │   (4-block pipeline) │                     │
├──────────────────────┤                      ├─────────────────────┤
│                      │                      │                     │
│   TECH STACK         │                      │   SCREENSHOT        │
│                      │                      │                     │
└──────────────────────┴──────────────────────┴─────────────────────┘
```

---

## Section-by-Section Content

### TITLE BANNER

**Title** (72-96pt, bold):
```
Paper Prism: A Query-Adaptive RAG Pipeline for Scientific Literature
```

**Subtitle** (36pt):
```
Different questions need different retrieval — adaptive chunking,
query-driven search, quality feedback loop, per-citation verification
```

**Author line** (28pt):
```
[Your Name] — [Your Program] — Oakland University
```

---

### SECTION 1: The Problem (Left Column, Top)

**Header**: The Problem

```
Researchers need answers grounded in THEIR papers.

Current RAG systems use one-size-fits-all retrieval:

  • "How many attention heads?" needs a precise
    paragraph — not a 2000-token section

  • "Compare DPR vs ColBERT" needs abstracts from
    many papers — not deep chunks from one

  • One retrieval strategy cannot serve both
```

---

### SECTION 2: How It Works (Center Column)

Visual centerpiece — full center column.

**Header**: How Paper Prism Works

Four color-coded blocks. Each block title includes a **contrast line** in smaller/italic text showing what typical systems do vs. what Paper Prism does. This makes differentiation visible without a separate comparison section.

```
┌─────────────────────────────────────────────────────────────┐
│  ❶  ONE PAPER → SIX REPRESENTATIONS  (purple)              │
│  Typical RAG: 1 chunk type (fixed-size splits)              │
│                                                             │
│   ┌─────┐     ┌──────────┐                                 │
│   │     │ ──▶ │ ABSTRACT │  broad queries                  │
│   │     │ ──▶ │ SECTION  │  methods, results, discussion   │
│   │ PDF │ ──▶ │ FINE     │  precise paragraphs (500 tok)   │
│   │     │ ──▶ │ TABLE    │  extracted data tables           │
│   │     │ ──▶ │ CAPTION  │  figure & table captions         │
│   │     │ ──▶ │ FULL     │  whole-paper embedding           │
│   └─────┘     └──────────┘                                  │
│                                                             │
│   MinerU extraction • Voyage AI embeddings • Qdrant storage │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  ❷  QUERY TYPE → RETRIEVAL STRATEGY  (blue)                │
│  Typical RAG: same search for every question                │
│                                                             │
│  ┌──────────────────┬──────────────┬───────┬────────────┐  │
│  │ Question         │ Chunks       │ Top-k │ Sections   │  │
│  ├──────────────────┼──────────────┼───────┼────────────┤  │
│  │ "How many heads  │ fine, table, │  50   │ all        │  │
│  │  in Transformer?"│ caption      │       │            │  │
│  ├──────────────────┼──────────────┼───────┼────────────┤  │
│  │ "How does BERT's │ section,     │  50   │ Methods/   │  │
│  │  MLM work?"      │ fine         │       │ Experiment │  │
│  ├──────────────────┼──────────────┼───────┼────────────┤  │
│  │ "Summarize the   │ abstract,    │  20   │ all        │  │
│  │  LoRA paper"     │ section, full│       │            │  │
│  ├──────────────────┼──────────────┼───────┼────────────┤  │
│  │ "Compare DPR     │ abstract,    │ 100   │ all        │  │
│  │  vs ColBERT"     │ section      │       │            │  │
│  └──────────────────┴──────────────┴───────┴────────────┘  │
│                                                             │
│   LLM classifies top-3 types • merges top-2 strategies     │
└─────────────────────────────────────────────────────────────┘
                            │
              Hybrid search (dense + sparse) → Cohere reranking
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  ❸  QUALITY FEEDBACK LOOP  (green)                          │
│  Typical RAG: single-pass retrieval, no quality check       │
│                                                             │
│        ┌────────────────┐                                   │
│        │ LLM evaluates: │                                   │
│        │ "Do these       │──── 4-5 / 5 ────▶ Generate       │
│        │  chunks cover   │                                  │
│        │  the question?" │                                  │
│        └────────┬───────┘                                   │
│            1-3 / 5                                          │
│           (gaps found)                                      │
│                 ▼                                           │
│        Re-retrieve with  ──── Re-rerank ──▶ Generate        │
│        LLM-suggested terms                                  │
│                                                             │
│   Example: "LoRA rank and GPU memory" → good rank chunks    │
│   but nothing about memory → re-search → both covered       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  ❹  PER-CITATION VERIFICATION  (orange)                     │
│  Typical RAG: shows sources but never checks claims         │
│                                                             │
│  "LoRA reduces params by 10,000x [Source 2] and matches     │
│   full fine-tuning on GPT-3 [Source 2]"                     │
│                                                             │
│   [Source 2] + "10,000x"                                    │
│     → ✅ SUPPORTED (95%) — matches abstract                 │
│                                                             │
│   [Source 2] + "matches full fine-tuning"                   │
│     → ⚠️ PARTIAL (48%) — "comparable", not exact match      │
│                                                             │
│   Same source, two claims → two separate verifications      │
└─────────────────────────────────────────────────────────────┘
```

**Small text below**:
```
15 stages total incl. entity extraction, query expansion,
decomposition, HyDE, entity boosting, conversation memory.
```

---

### SECTION 3: Benchmark Results (Right Column)

**Header**: Evaluation

**TOP: PaperPrism vs. Direct Claude**

```
15 queries • blind A/B • reference-grounded

  Accuracy       ██████████████████   3.67
                 ████████████████     3.33

  Completeness   ████████████████████ 4.0
                 ██████████████████   3.67

  Grounding      ██████████████████   3.8
                 ████████████         2.53

  ■ PaperPrism   ■ Direct Claude

  Won 11 of 15 (2 ties, 2 Claude wins)
```

**BOTTOM: 100-Query Benchmark**

```
GPT-5.4-mini judge • cross-family eval

  Context Recall    ████████████████████  89%
  Claim Accuracy    ████████████████████  89%
  Completeness      █████████████████     68%
  Faithfulness      ████████████████      67%
  Citation Accuracy ████████████████      65%
  Refusal Accuracy  ██████████████        60%

  Factual (30)      95% claims, 98% complete
  Robustness (10)  100% claims
  Cross-paper (25)  69% claims — hardest
  Refusal (10)      6/10 correctly refused
```

**Methodology** (small text):
```
Claude Sonnet generates, OpenAI gpt-5.4-mini evaluates.
100 queries: 30 factual, 25 cross-paper, 15 exploratory,
10 adversarial, 10 refusal, 10 robustness.
```

---

### SECTION 4: Tech Stack (Left Column, Bottom)

**Header**: Tech Stack

```
 PDF Extraction        Embeddings         Vector Search
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   MinerU     │   │  Voyage AI   │   │   Qdrant     │
│  Layout-aware│──▶│ voyage-3-large──▶│  Hybrid:     │
│  OCR, tables │   │  1024-dim    │   │  Dense + BM25│
└──────────────┘   └──────────────┘   └──────────────┘

   Reranking         LLM Generation       Evaluation
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Cohere     │   │  Claude 4.5  │   │   OpenAI     │
│ rerank-v3.5  │──▶│ Opus (answer)│   │ gpt-5.4-mini │
│ Cross-encoder│   │ Haiku (fast) │   │ Cross-family │
└──────────────┘   │Sonnet (class)│   │  evaluation  │
                   └──────────────┘   └──────────────┘

Self-hosted Mac Mini • Docker • FastAPI • SQLite
```

---

### SECTION 5: Screenshot (Right Column, Bottom)

One screenshot of the UI showing a completed response with citation verification badges (green/amber/red). Crop tight to the answer + badges + source cards.

Use AI/ML demo corpus. Dark mode. 1920px+ width.

---

## Color Scheme

- **Background**: White or light gray (#f9fafb)
- **Block ❶ (Chunking)**: Purple (#8b5cf6)
- **Block ❷ (Classification)**: Blue (#3b82f6)
- **Block ❸ (Quality loop)**: Green (#22c55e)
- **Block ❹ (Verification)**: Orange (#f59e0b)
- **Contrast lines**: Gray italic — visually recedes but readable
- **Text**: Dark gray (#1f2937)
