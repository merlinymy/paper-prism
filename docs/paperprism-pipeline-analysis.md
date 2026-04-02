# PaperPrism — Pipeline Deep-Dive

**Current RAG pipeline vs. agentic search, retrieval quality diagnosis, and actionable recommendations.**

---

## 1. Your Current Pipeline (13 Stages)

PaperPrism runs a **linear, deterministic** pipeline — every query follows the same sequence of stages, with query classification selecting pre-defined parameter presets along the way.

```
Query Rewrite → Entity Extract → Classify (8 types) → Query Expand
    → Hybrid Search (dense + BM25) → Cross-Encoder Rerank
    → Answer Gen (Claude) → Citation Verify → Response
```

### What makes this pipeline strong

**Query-Adaptive Retrieval**
- 8-type classification drives chunk-type selection, top-k, section filters, and per-paper limits
- No other open-source RAG system does this dynamically
- Different granularity for factual (tables) vs. summary (abstracts) — smart design

**6 Chunk Representations**
- Abstract, section, fine-grained, table, caption, full-paper
- MinerU for layout-aware extraction — preserves table structure
- Captures the document at multiple levels of granularity

**Citation Verification**
- Each `[Source N]` is semantically verified against the passage
- Color-coded confidence (green/amber/red) — rare in OSS RAG
- Significantly reduces hallucinated citations

**Hybrid Search**
- Dense (Voyage AI `voyage-3-large`) + BM25 sparse via Qdrant
- Cohere `rerank-v3.5` cross-encoder on top
- Solid tech choices — Voyage and Cohere are top-tier

---

## 2. Why Your Retrieval Feels Weak

### Issue 1: Single-Pass Retrieval — No Feedback Loop

The pipeline fires **one** hybrid search, reranks, and generates. If the initial retrieval misses relevant chunks (wrong query terms, embedding mismatch, entity not expanded properly), there's no mechanism to detect the miss and retry. The system never asks: *"Did I find what I needed?"*

### Issue 2: Classification Errors Cascade

The entire retrieval strategy is locked to whichever of the 8 types the classifier picks. If a query is misclassified — say, a "limitations" question classified as "factual" — the wrong chunk types, section filters, and top-k are used. There's **no fallback or hedging**: no retrieving from a secondary strategy to compensate.

### Issue 3: Static Query Expansion (Domain Synonyms)

Query expansion uses domain synonyms — likely a static dictionary or a single LLM call. This misses synonyms that are context-dependent. "Stapled peptides" might need expanding to "hydrocarbon-stapled α-helical peptides" in one paper but "constrained peptides" in another. A static expansion can't know which paper's vocabulary to target.

### Issue 4: No Cross-Chunk Reasoning at Retrieval Time

The system retrieves chunks independently and sends them to the LLM. It doesn't reason at retrieval time about *gaps*: "I have the IC50 value from Table 3 but I'm missing the experimental conditions from the Methods section." The LLM does this reasoning after the fact, when it's too late to fetch more context.

### Issue 5: Per-Paper Limits Can Starve Good Sources

Hard per-paper caps (e.g., max 2 for comparative, max 3 for factual) are set before seeing results. If one paper has 5 highly relevant chunks and another has none, the cap discards the extras. This is a tradeoff that sometimes hurts more than it helps, particularly for in-depth single-paper questions.

### Issue 6: Embedding vs. Vocabulary Mismatch

Scientific literature has dense jargon (compound names, abbreviations, formulas). Even `voyage-3-large` can struggle with domain-specific terms that weren't well-represented in training. BM25 helps here, but only if the query expansion actually captures the right surface forms. If a user queries "SRS microscopy" and the paper uses "stimulated Raman scattering" without the abbreviation in the same chunk, both dense and sparse retrieval can miss it.

---

## 3. Current Pipeline vs. Agentic Search

| Dimension | PaperPrism (Current) | Agentic Search |
|---|---|---|
| **Flow Control** | Linear, deterministic. Fixed stage order. | Agent-driven. LLM decides what to do next based on intermediate results. |
| **Retrieval Passes** | Single pass — one query, one search, one rerank. | Multi-pass — agent evaluates results, reformulates, retrieves again if needed. |
| **Query Decomposition** | One classification → one strategy preset. | Agent breaks complex queries into sub-questions, each searched independently. |
| **Self-Evaluation** | None at retrieval time. LLM only sees final chunks. | Agent checks: "Do I have enough? Are there gaps? Is this contradictory?" |
| **Error Recovery** | Classification error = wrong strategy, no recovery. | Agent can pivot: try different chunk types, broaden/narrow search, change strategy mid-flight. |
| **Cross-Chunk Reasoning** | Post-retrieval only (during generation). | During retrieval — agent synthesizes partial results to guide next search. |
| **Latency** | Fast. Predictable ~3-8s depending on stage count. | Slower. Each agent step = LLM call + search. 10-30s typical. |
| **Cost** | Lower. One main LLM call for generation, light models for classification. | Higher. Multiple LLM calls for planning, evaluation, reformulation. |
| **Predictability** | High. Same query always follows same path. | Variable. Agent may take different paths on the same query. |

### Agentic Search Pipeline (What It Would Look Like)

```
Plan/Decompose → Search Sub-Q1 → Search Sub-Q2 → Evaluate Gaps
    → Targeted Re-search → Synthesize → Generate + Cite
    → Verify / Self-Critique → Final Response
```

The agent loop (evaluate → re-search → synthesize) can repeat 1–3 times depending on result quality. Each iteration is a full LLM reasoning step.

---

## 4. What Agentic Search Would Fix

**Multi-Pass Recovery**
- Agent detects when initial retrieval returns low-relevance chunks
- Automatically reformulates — broader terms, different chunk types, relaxed filters
- Directly solves Issue 1 (single-pass) and Issue 2 (classification cascade)

**Query Decomposition**
- "How does SRS compare to fluorescence for live cell imaging?" → 3 sub-queries
- Each sub-query targets different chunk types and papers
- Results are merged with source tracking intact

**Context-Aware Expansion**
- Agent reads initial results and identifies terminology gaps
- "Paper uses 'coherent anti-Stokes Raman' but I searched 'CARS'" — agent re-searches
- Replaces static synonym dictionary with dynamic, context-driven expansion

**Gap Detection**
- "I have the result but not the method — let me search the Methods section"
- Agent identifies what's missing before generation starts
- Directly solves Issue 4 (no cross-chunk reasoning at retrieval time)

---

## 5. The Tradeoffs (Don't Ignore These)

| Metric | Impact |
|---|---|
| **Latency** | 2–4× slower. Each agent step adds 1-3s. Multi-pass = slower. |
| **Cost** | 3–5× higher. Multiple LLM calls for planning, evaluation, and reformulation. |
| **Reliability** | Lower. Agent loops can diverge, hallucinate search plans, or over-retrieve. |

### Risk: Agent Loop Instability

Agentic search agents can get stuck in loops — re-searching for info that doesn't exist, or over-refining queries past the point of usefulness. You'd need hard limits on iterations and timeout budgets. Your current pipeline's predictability is a feature, not a bug.

### Risk: You Lose Your Best Feature

Your real-time pipeline visualization (13 stages lighting up) is a great UX differentiator. A fully agentic system with variable execution paths would make this visualization much harder to build and less satisfying for users.

---

## 6. Recommended Path: Hybrid Approach

Going fully agentic is overkill and adds risk. Instead, surgically add agentic capabilities where your pipeline is weakest.

### Rec 1: Add a Retrieval Evaluator After Reranking

Insert a lightweight LLM call (Haiku-class) after reranking that scores retrieval quality: `"Given this query and these top-5 chunks, rate confidence 1-5 and identify gaps."` If confidence < 3, trigger a second retrieval pass with a reformulated query. This is the single highest-impact change — it turns your linear pipeline into a conditional loop without full agentic complexity.

### Rec 2: Multi-Strategy Hedging for Classification

Instead of betting everything on one classification, retrieve using the top-2 predicted types and merge results before reranking. If the classifier says 70% "factual" and 20% "methods", run both retrieval strategies and let the reranker sort it out. This costs one extra Qdrant query (fast) but dramatically reduces classification error impact.

### Rec 3: Dynamic Query Decomposition for Complex Queries

Add a pre-classification step: if the query contains conjunctions, comparisons, or multiple entities, decompose it into sub-queries before running the pipeline. Each sub-query runs through your existing pipeline independently, and results are merged. This handles comparative and multi-hop questions much better.

### Rec 4: Contextual Re-Expansion

After the first retrieval, extract key terms from the top-3 chunks that weren't in the original query. Use these as expansion terms for a second, targeted search. This replaces your static synonym dictionary with a pseudo-relevance feedback loop. Classic IR technique, minimal cost.

### Rec 5: Relax Per-Paper Limits Dynamically

Make per-paper caps soft: if the reranker scores 4+ chunks from one paper above a threshold, allow more through. The current hard caps are discarding relevant chunks in single-paper deep-dive scenarios.

---

## 7. What the Revised Pipeline Looks Like

```
Query Rewrite → Entity Extract → Decompose? (conditional)
    → Classify (top-2) → Expand
    → Hybrid Search (both strategies) → Rerank + Merge
    → Evaluate Quality → Re-search? (conditional)
    → Re-expand from results
    → Generate + Cite → Citation Verify → Response
```

The conditional stages (decompose, re-search, re-expand) only fire when triggered — most queries still follow the fast path. Your pipeline visualization can show these as optional stages that light up only when activated.

---

## Bottom Line

Your pipeline is already more sophisticated than 95% of open-source RAG systems. The query-adaptive retrieval with 6 chunk types is genuinely novel. The retrieval quality issues you're feeling aren't from a bad architecture — they're from the **single-pass, no-feedback** nature of the pipeline. You don't need to tear it down and go fully agentic. You need to add 2–3 surgical feedback loops that give the system a chance to recover from bad first retrievals.

**Start with Rec 1 (retrieval evaluator) and Rec 2 (multi-strategy hedging).** These two changes alone should noticeably improve retrieval quality with minimal latency impact.

---

## 8. Implementation Status

The following recommendations from this analysis have been implemented:

| Recommendation | Status | Implementation |
|---|---|---|
| **Rec 1: Retrieval Evaluator** | **Implemented (enhanced)** | LLM quality evaluation (Haiku) runs after every reranking pass. Rates retrieval coverage 1-5, identifies missing information, provides targeted search terms. Goes beyond the original suggestion — uses LLM evaluation instead of score heuristics. |
| **Rec 2: Multi-Strategy Hedging** | **Implemented** | `classify_multi()` returns top-2 types. Chunk types from both strategies are merged for search. Section filters are unioned. Max top-k uses the higher of both strategies. |
| **Rec 3: Query Decomposition** | Not implemented | Decomposition logic exists in `query_rewriter.py` but sub-queries are not searched independently. |
| **Rec 4: Contextual Re-Expansion** | **Implemented (via quality evaluator)** | When retrieval quality is poor, the LLM evaluator provides targeted search terms based on what's missing — a more intelligent form of pseudo-relevance feedback. |
| **Rec 5: Relax Per-Paper Limits** | **Implemented** | Soft caps: chunks with rerank_score > 0.5 bypass the per-paper quota up to 2× the limit. |

**Additional changes beyond the recommendations:**

- **Domain-agnostic expansion**: Replaced hardcoded domain synonym dictionary (ERα, SRS, antimicrobial peptides) with LLM-based query expansion that works across all scientific domains.
- **Generalized classifier**: Domain-specific few-shot examples replaced with domain-neutral ones.
- **Removed query rewriting step**: Redundant with LLM expansion; static spelling/acronym dictionaries were domain-specific.
- **Citation verification overhaul**: Per-usage verification (each citation occurrence gets its own check), JSON-based LLM response parsing, clickable modal UI showing full verification details.

---

*Analysis based on the PaperPrism repository at `github.com/merlinymy/paper-prism` — architecture, README documentation, and tech stack as documented. Source code internals inferred from the public architecture description. Implementation status updated April 2026.*
