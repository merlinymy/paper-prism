# ARC Research Poster — Complete Content & Layout Guide

> Everything you need to create your poster. Copy the text directly, follow the layout, take the screenshots described.

---

## Poster Specs

- **Size**: 48" wide x 36" tall (landscape) — standard academic poster, most Makerspaces support this
- **Tool**: Use **Canva** (free, has poster templates), Google Slides (set custom size to 48x36 inches), or PowerPoint
- **Font sizes**: Title = 72-96pt, Section headers = 36-44pt, Body text = 24-28pt, Captions = 18-20pt
- **Rule of thumb**: If you can't read it from 4 feet away, the text is too small

---

## Layout (3-Column, Top Banner)

```
┌──────────────────────────────────────────────────────────────────┐
│                         TITLE BANNER                              │
│                     Name / Affiliation                             │
├──────────────────────┬──────────────────────┬─────────────────────┤
│                      │                      │                     │
│   THE PROBLEM        │   THE PIPELINE       │   BENCHMARK:        │
│   + APPROACH         │   (big diagram)      │   ARC vs. CLAUDE    │
│                      │                      │   (bar chart)       │
│                      │                      │                     │
│                      │                      │                     │
├──────────────────────┤                      ├─────────────────────┤
│                      │                      │                     │
│   MULTI-TYPE         │                      │   WHAT MAKES IT     │
│   CHUNKING           │                      │   DIFFERENT         │
│   (diagram)          │                      │                     │
│                      │                      │                     │
├──────────────────────┼──────────────────────┼─────────────────────┤
│                      │                      │                     │
│   QUERY-ADAPTIVE     │   TECH STACK         │   TRY IT LIVE       │
│   RETRIEVAL          │                      │   + screenshots     │
│   (table)            │                      │   (arrow to laptop) │
│                      │                      │                     │
└──────────────────────┴──────────────────────┴─────────────────────┘
```

---

## Section-by-Section Content

### TITLE BANNER

**Title** (72-96pt, bold):
```
ARC: A Query-Adaptive Retrieval Pipeline for Scientific Literature
```

**Subtitle** (36pt):
```
A quality-adaptive RAG system with retrieval feedback loop — wins 13/15 vs. direct Claude on blind evaluation
```

**Author line** (28pt):
```
[Your Name] — [Your Program] — Oakland University
```

---

### SECTION 1: The Problem (Left Column, Top)

**Header**: The Problem

**Body text** (keep it short — 4-5 bullet points max):

```
Researchers need answers grounded in THEIR papers — not generic LLM responses.

Current tools fall short:

  • ChatGPT/Claude hallucinate citations and have no access 
    to your private paper library

  • Existing open-source RAG systems use largely uniform 
    retrieval — they don't adapt chunk types, result counts,
    section filters, or diversity limits per query type

  • A question about a specific IC50 value needs precise, 
    small chunks from a data table

  • A question comparing methods across 20 papers needs broad 
    retrieval from many abstracts

  • One retrieval strategy cannot serve both
```

**Callout box** (highlighted background):
```
Key Insight: Different questions need fundamentally 
different retrieval strategies. An adaptive pipeline
with quality feedback loop wins 13 of 15 queries
against raw Claude on blind, reference-grounded evaluation.
```

---

### SECTION 2: The Pipeline (Center Column — THE BIG DIAGRAM)

This is the visual centerpiece. Make it take up most of the center column.

**Header**: The Adaptive Pipeline

**Diagram content** — draw this as a vertical flowchart with 3 color-coded phases. Each box has the stage name and a one-line description:

```
─── QUERY PROCESSING (blue) ───────────────────────

  ① Reference Resolution
     "it" → "the compound from Table 2"

  ② Entity Extraction
     Found: BRCA1 (protein), HPLC (method), IC50 (metric)

  ③ Dual-Strategy Classification ← KEY STAGE
     Top-2: METHODS (94%), FACTUAL (72%)
     Merges chunk types from both strategies

  ④ LLM Query Expansion
     Haiku generates domain-appropriate synonyms
     Added: "solid-phase synthesis", "Fmoc chemistry"

  ⑤ Query Decomposition (conditional)
     "Compare BERT and GPT pre-training and performance"
     → ["BERT vs GPT pre-training?", "BERT vs GPT performance?"]

─── RETRIEVAL (green) ─────────────────────────────

  ⑥ Embedding (+ optional HyDE)
     Generate hypothetical Methods excerpt, embed it

  ⑦ Hybrid Search (per sub-query if decomposed)
     Dense (Voyage AI) + Sparse (BM25) → RRF fusion
     Merge + dedup results across sub-queries

  ⑧ Entity Boosting
     Upweight chunks containing query entities

  ⑨ Reranking
     Cohere cross-encoder: 50 → 15 results
     Soft per-paper caps (high scores bypass limit)

  ⑩ Quality Evaluation ← UNIQUE
     Haiku rates coverage 1-5, identifies gaps

  ⑪ Conditional Re-Retrieval
     If gaps found → targeted search with
     LLM-suggested terms → re-rerank

  ⑫ Parent Chunk Expansion
     Fine chunks fetch parent section for context

─── GENERATION (orange) ───────────────────────────

  ⑬ Answer Generation
     Claude Opus with query-type-specific prompt

  ⑭ Citation Verification ← UNIQUE
     Per-usage: each citation checked independently
     Clickable modal with claim + confidence + explanation

  ⑮ Conversation Memory
     Track context for follow-up questions
```

**At the bottom of the diagram, add**:
```
Total latency: 5-12 seconds per query (fast path: ~6s, re-retrieval path: ~10s)
```

---

### SECTION 3: What Makes It Different (Right Column, Middle)

**Header**: What Makes ARC Different

**Compared to 7 open-source RAG products + 1 commercial benchmark** (small text: RAGFlow, Dify, AnythingLLM, PrivateGPT, Quivr, Onyx, Kotaemon + Perplexity)

**Format as a checklist with ARC vs. Others**:

```
✓ 6 domain-specific chunk types per paper
  Others: Generic or template-based splitting (1-3 strategies)
  → Tables, captions, and sections are separately searchable

✓ Query classification drives retrieval parameters
  Others: May route queries, but don't adapt chunk types,
  top-k, section filters, or per-paper limits per query
  → "What is the IC50?" and "Compare these methods"
    trigger completely different search strategies

✓ Citation verification (per-usage)
  Others: Show source chunks but never verify claim accuracy
  → Each usage of [Source N] is independently checked
    against the source text — clickable modal with details

✓ Query decomposition for complex questions
  Others: Single query, single search
  → "Compare BERT and GPT" decomposes into sub-queries,
    each searched independently, results merged + reranked

✓ Retrieval quality feedback loop
  Others: Single-pass retrieval with no quality check
  → LLM evaluates retrieval coverage, triggers targeted
    re-retrieval with specific search terms when gaps found

✓ Cross-family evaluation benchmark
  Others: Self-evaluate (same model judges its own output)
  → 100-query suite with claim-level ground truth
    Claude generates, GPT-5.4-mini evaluates
    RAGAS-style metrics: faithfulness, recall, precision

✓ 15-stage adaptive pipeline
  Others: Typically 3-6 stages
  → Each stage addresses a specific failure mode
    in simpler systems
```

---

### SECTION 4: Multi-Type Chunking (Left Column, Bottom)

**Header**: One Paper, Six Representations

**Diagram** — show a single PDF icon on the left, with 6 arrows pointing to 6 labeled boxes:

```
                    ┌─────────────────┐
                 ┌─▶│ ABSTRACT        │  Paper overview (300 tokens)
                 │  └─────────────────┘
                 │  ┌─────────────────┐
                 ├─▶│ SECTION         │  Methods, Results, Discussion (2000-3000 tokens)
                 │  └─────────────────┘
   ┌─────┐      │  ┌─────────────────┐
   │ PDF │──────┼─▶│ FINE            │  Precise paragraphs (500 tokens, 128 overlap)
   └─────┘      │  └─────────────────┘
                 │  ┌─────────────────┐
                 ├─▶│ TABLE           │  Extracted data tables
                 │  └─────────────────┘
                 │  ┌─────────────────┐
                 ├─▶│ CAPTION         │  Figure & table captions
                 │  └─────────────────┘
                 │  ┌─────────────────┐
                 └─▶│ FULL            │  Mean-pooled paper embedding
                    └─────────────────┘
```

**Caption below diagram**:
```
Different query types search different chunk types.
A factual query searches FINE + TABLE + CAPTION.
A summary query searches ABSTRACT + SECTION + FULL.
A methods query searches SECTION + FINE (filtered to Methods/Experimental).
```

---

### SECTION 5: Query-Adaptive Retrieval (Left Column, Bottom)

**Header**: Same System, Different Strategy Per Question

**Table** (make it a clean, colored table — this is the "aha" moment):

```
┌─────────────────────────────┬──────────────┬────────────────────┬───────┬───────────┐
│ Example Query               │ Detected     │ Chunks Searched    │ Top-k │ Max/Paper │
│                             │ Type         │                    │       │           │
├─────────────────────────────┼──────────────┼────────────────────┼───────┼───────────┤
│ "How many attention heads   │ FACTUAL      │ fine, table,       │  50   │     3     │
│  does the Transformer use?" │              │ caption            │       │           │
├─────────────────────────────┼──────────────┼────────────────────┼───────┼───────────┤
│ "How does BERT's masked     │ METHODS      │ section, fine      │  50   │     5     │
│  LM pre-training work?"     │              │ (Methods/Exp)      │       │           │
├─────────────────────────────┼──────────────┼────────────────────┼───────┼───────────┤
│ "Summarize this paper"      │ SUMMARY      │ abstract, section, │  20   │    10     │
│                             │              │ full               │       │           │
├─────────────────────────────┼──────────────┼────────────────────┼───────┼───────────┤
│ "How does DPR compare to    │ COMPARATIVE  │ abstract, section  │ 100   │     2     │
│  ColBERT for retrieval?"    │              │                    │       │           │
├─────────────────────────────┼──────────────┼────────────────────┼───────┼───────────┤
│ "What are the limitations   │ LIMITATIONS  │ section, abstract  │  50   │     4     │
│  of few-shot learning?"     │              │ (Discussion/Concl) │       │           │
└─────────────────────────────┴──────────────┴────────────────────┴───────┴───────────┘
```

**Caption**:
```
Dual-strategy classification (top-2 types) determines retrieval.
Each type triggers different chunk types, result counts, section
filters, and diversity limits. Merged strategies cast a wider net.
```

---

### SECTION 6: Benchmark Results (Right Column, Top)

**Header**: Evaluation Results

**Two subsections side by side:**

**LEFT: PaperPrism vs. Direct Claude** (blind A/B comparison)

**Subheader**: 15 queries • blind judge (randomized A/B) • reference passages for fact-checking

```
  Accuracy       ███████████████████████████████████████            3.7
                 ██████████████████████████████                     3.3

  Completeness   ████████████████████████████████████████           4.0
                 █████████████████████████████████████              3.7

  Grounding      ██████████████████████████████████████             3.8
                 █████████████████████████                          2.5

                 ■ PaperPrism (RAG)   ■ Direct Claude (no RAG)
```

**Big callout**: `PaperPrism won 11 of 15 queries.`

**RIGHT: RAGAS-Style Metrics** (100-query comprehensive benchmark)

**Subheader**: 100 queries across 6 categories • GPT-5.4-mini judge (cross-family) • claim-level ground truth

```
  Context Recall       ████████████████████████████████████████    89%
  Claim Accuracy       ███████████████████████████████████████     87%
  Citation Accuracy    ██████████████████████████████████          72%
  Faithfulness         █████████████████████████████████           69%
  Completeness         ████████████████████████████████            67%
  Refusal Accuracy     █████████████████████████                   50%
```

**Breakdown by query type** (smaller table):
```
  Factual (30 queries)        97% claim accuracy, 97% completeness
  Robustness (10 queries)    100% claim accuracy (typos, broken English)
  Cross-paper (25 queries)    60% claim accuracy (multi-paper synthesis)
  Adversarial (10 queries)    95% claim accuracy, 25% completeness
  Refusal (10 queries)         5/10 correctly refused
```

**Caption**:
```
Left: Blind A/B — PaperPrism's grounded answers beat raw Claude,
especially on evidence traceability (3.8 vs 2.5).
Right: 100-query benchmark with claim-level ground truth.
Factual queries are the strongest (97% accuracy). Cross-paper
synthesis is the hardest open challenge (60%).
```

**Methodology note** (small text):
```
A/B comparison: blind judge, randomized per query. Comprehensive
benchmark: 100 queries (30 factual, 25 cross-paper, 15 exploratory,
10 adversarial, 10 refusal, 10 robustness). Generation by Claude
Sonnet, evaluation by OpenAI gpt-5.4-mini — cross-family to
eliminate self-evaluation bias.
```

---

### SECTION 7: Tech Stack (Center Column, Bottom)

**Header**: Technology Stack

**Simple horizontal layout** with icons/logos:

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
```

**One-liner below**:
```
Self-hosted on Mac Mini • Qdrant in Docker • FastAPI backend
Cloudflare Tunnel for remote access • SQLite for conversations
```

---

### SECTION 8: Try It Live (Right Column, Bottom)

**Header**: Try It Live →

**One screenshot of the UI** (see Screenshot Guide) — showing a completed response with citation badges and the pipeline progress steps. Crop to fit.

**Below the screenshot:**

```
  ┌─────────────────────────────────────┐
  │                                     │
  │   20 AI/ML papers loaded.           │
  │   Ask it anything.                  │
  │   Watch the pipeline adapt.         │
  │                                     │
  │             ────────────▶           │
  │             (laptop here)           │
  └─────────────────────────────────────┘
```

**Small text at bottom**:
```
In daily use by a research scientist for biology
and chemistry literature. Wins 13/15 queries against
raw Claude on blind, reference-grounded evaluation.
```

---

## Screenshots to Take

Take these from your running frontend. Use the AI/ML demo corpus for screenshots if possible, or your existing bio/chem corpus works too (the screenshots are about showing the UI, not the content).

### Screenshot A: Pipeline Progress
1. Start a query with all features enabled
2. While the pipeline is running (or just after), capture the pipeline step grid showing:
   - Completed steps (green checkmarks)
   - The active step (blue spinner)
   - Step details (entity count, query type, etc.)
3. Crop to just the pipeline progress area

### Screenshot B: Citation Verification
1. Use a completed response that has [Source N] citations
2. Make sure citation check was enabled — badges should show green/amber/red
3. Hover over one citation badge to show the tooltip with confidence %
4. Capture the response area including at least 2-3 citation badges and 1-2 source cards below
5. If you can't capture the hover tooltip, that's fine — the color-coded badges alone tell the story

### Screenshot C (optional): Paper Library
1. Show the library page with several papers listed
2. Shows metadata (title, authors, year, chunk count)
3. Shows the semantic search bar
4. Good for demonstrating "this is a real, complete product"

### Screenshot D (optional): Full Chat Interface
1. Show a multi-turn conversation with 2-3 queries
2. Shows the sidebar with conversation list
3. Shows the query input with advanced options expanded
4. Demonstrates the full product experience

**Screenshot tips**:
- Use dark mode — it photographs and prints better on posters
- Crop tightly to the relevant UI area
- Resolution: at least 1920px wide for clarity at poster print size

---

## Color Scheme for the Poster

Keep it clean and consistent with your app's color scheme:

- **Background**: White or very light gray (#f9fafb)
- **Section headers**: Dark blue (#1e3a5f) or your app's primary blue
- **Pipeline diagram — Query Processing stages**: Blue (#3b82f6)
- **Pipeline diagram — Retrieval stages**: Green (#22c55e)
- **Pipeline diagram — Generation stages**: Orange/Amber (#f59e0b)
- **Accent/highlights**: Your app's primary blue
- **Text**: Dark gray (#1f2937), not pure black

---

## Common Poster Mistakes to Avoid

1. **Too much text** — If a section has more than 6-7 lines, cut it. The poster supplements your verbal explanation; it doesn't replace it.
2. **Tiny fonts** — Nothing below 20pt. If it doesn't fit, cut words, not font size.
3. **No visual hierarchy** — The pipeline diagram and the query-adaptive table should be the two biggest visual elements. Someone walking by should see those first.
4. **Walls of bullets** — Use diagrams, tables, and screenshots instead of bullet lists wherever possible.
5. **Too many colors** — Stick to 3-4 colors max plus grays.
6. **No "try it" call-to-action** — The laptop demo is your strongest selling point. Make the poster actively direct people to it.
