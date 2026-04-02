# Changelog

## [1.1.0] — 2026-04-01

### Adaptive Retrieval Pipeline
- **Dual-strategy classification**: Queries classified into top-2 types; chunk types merged for broader search coverage
- **LLM query expansion**: Haiku generates domain-appropriate synonyms at query time (replaces hardcoded dictionary)
- **Retrieval quality evaluation**: Haiku rates retrieval coverage 1-5 and identifies gaps after reranking
- **Conditional re-retrieval**: Targeted re-search with LLM-suggested terms when quality is insufficient
- **Soft per-paper caps**: High-scoring chunks (>0.5) bypass quota up to 2x limit

### Citation Verification
- **Per-usage verification**: Each citation occurrence verified independently with its own explanation
- **JSON-based LLM responses**: Fixes confidence parsing (was falling back to 50% keyword overlap)
- **Clickable citation modal**: Click badges to see full claim, explanation, and confidence bar

### Domain-Agnostic Design
- Deleted hardcoded domain synonyms (`domain_synonyms.py`)
- Generalized classifier few-shot examples
- Removed query rewriting step (redundant with LLM expansion)

### Infrastructure Fixes
- Dockerfile: `libgl1-mesa-glx` → `libgl1` (Debian Trixie)
- Qdrant healthcheck: bash TCP (no curl in image)
- Pinned `bcrypt<4.1.0` for passlib compatibility
- MinerU v3 API (`doc_analyze_streaming`)
- Auto-create Qdrant collection on startup

### Frontend
- Pipeline visualization: added Quality Check and Re-Retrieval steps
- Classification shows secondary types
- Expansion shows LLM-generated terms
- Response metadata shows expansion terms and re-retrieval status

## [1.0.0] — 2026-03-15

Initial release.
