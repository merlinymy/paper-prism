"""Test queries for the AI/ML demo paper corpus (20 papers).

Covers all 8 query types with ground-truth expectations based on
the actual content of the indexed papers.
"""

from evaluation.test_queries import TestQuery, QueryType

DEMO_QUERIES = [
    # === FACTUAL (6 queries) ===
    TestQuery(
        query_id="demo_factual_001",
        query="How many attention heads does the base Transformer model use?",
        query_type=QueryType.FACTUAL,
        expected_topics=["attention", "head", "multi-head", "base model"],
        expected_entities=["Transformer"],
        expected_chunk_types=["fine", "table"],
    ),
    TestQuery(
        query_id="demo_factual_002",
        query="What is the perplexity score reported for GPT-3 on Penn Treebank?",
        query_type=QueryType.FACTUAL,
        expected_topics=["perplexity", "language model", "benchmark"],
        expected_entities=["GPT-3"],
        expected_chunk_types=["fine", "table"],
    ),
    TestQuery(
        query_id="demo_factual_003",
        query="What embedding dimension does the Dense Passage Retrieval model use?",
        query_type=QueryType.FACTUAL,
        expected_topics=["embedding", "dimension", "vector", "representation"],
        expected_entities=["DPR"],
        expected_chunk_types=["fine", "table"],
    ),
    TestQuery(
        query_id="demo_factual_004",
        query="How many parameters does LLaMA 65B have and what is its training data size?",
        query_type=QueryType.FACTUAL,
        expected_topics=["parameter", "billion", "training", "token"],
        expected_entities=["LLaMA"],
        expected_chunk_types=["fine", "table"],
    ),
    TestQuery(
        query_id="demo_factual_005",
        query="What is the rank used in LoRA for GPT-3 adaptation?",
        query_type=QueryType.FACTUAL,
        expected_topics=["rank", "low-rank", "adaptation", "parameter"],
        expected_entities=["LoRA"],
        expected_chunk_types=["fine", "table"],
    ),
    TestQuery(
        query_id="demo_factual_006",
        query="What sliding window size does Mistral 7B use for attention?",
        query_type=QueryType.FACTUAL,
        expected_topics=["sliding window", "attention", "context"],
        expected_entities=["Mistral"],
        expected_chunk_types=["fine", "section"],
    ),

    # === METHODS (5 queries) ===
    TestQuery(
        query_id="demo_methods_001",
        query="How is RLHF implemented in InstructGPT? What is the training procedure?",
        query_type=QueryType.METHODS,
        expected_topics=["reinforcement learning", "human feedback", "reward model", "PPO"],
        expected_entities=["RLHF", "InstructGPT"],
        expected_chunk_types=["section", "fine"],
    ),
    TestQuery(
        query_id="demo_methods_002",
        query="How does BERT's masked language modeling pre-training work?",
        query_type=QueryType.METHODS,
        expected_topics=["masked", "pre-training", "token", "bidirectional"],
        expected_entities=["BERT", "MLM"],
        expected_chunk_types=["section", "fine"],
    ),
    TestQuery(
        query_id="demo_methods_003",
        query="How are passages retrieved and scored in the Dense Passage Retrieval system?",
        query_type=QueryType.METHODS,
        expected_topics=["retrieval", "encoder", "inner product", "FAISS"],
        expected_entities=["DPR"],
        expected_chunk_types=["section", "fine"],
    ),
    TestQuery(
        query_id="demo_methods_004",
        query="How does Self-RAG decide when to retrieve and how to critique its own generations?",
        query_type=QueryType.METHODS,
        expected_topics=["reflection", "token", "retrieve", "critique", "generation"],
        expected_entities=["Self-RAG"],
        expected_chunk_types=["section", "fine"],
    ),
    TestQuery(
        query_id="demo_methods_005",
        query="What training procedure does LoRA use to adapt large language models?",
        query_type=QueryType.METHODS,
        expected_topics=["low-rank", "matrix", "frozen", "adapter"],
        expected_entities=["LoRA"],
        expected_chunk_types=["section", "fine"],
    ),

    # === SUMMARY (3 queries) ===
    TestQuery(
        query_id="demo_summary_001",
        query="Summarize the key contributions of the Attention Is All You Need paper.",
        query_type=QueryType.SUMMARY,
        expected_topics=["self-attention", "encoder", "decoder", "parallelization"],
        expected_entities=["Transformer"],
        expected_chunk_types=["abstract", "section"],
    ),
    TestQuery(
        query_id="demo_summary_002",
        query="What are the main findings of the Lost in the Middle paper?",
        query_type=QueryType.SUMMARY,
        expected_topics=["position", "middle", "context", "performance", "degradation"],
        expected_entities=[],
        expected_chunk_types=["abstract", "section"],
    ),
    TestQuery(
        query_id="demo_summary_003",
        query="Summarize the RAPTOR paper's approach to retrieval.",
        query_type=QueryType.SUMMARY,
        expected_topics=["tree", "recursive", "abstractive", "clustering", "summarization"],
        expected_entities=["RAPTOR"],
        expected_chunk_types=["abstract", "section"],
    ),

    # === COMPARATIVE (4 queries) ===
    TestQuery(
        query_id="demo_comparative_001",
        query="How does BERT's pre-training approach compare to GPT's?",
        query_type=QueryType.COMPARATIVE,
        expected_topics=["bidirectional", "autoregressive", "masked", "left-to-right"],
        expected_entities=["BERT", "GPT"],
        expected_chunk_types=["abstract", "section"],
    ),
    TestQuery(
        query_id="demo_comparative_002",
        query="Compare Dense Passage Retrieval with ColBERT's approach to passage search.",
        query_type=QueryType.COMPARATIVE,
        expected_topics=["dense", "late interaction", "token-level", "single vector"],
        expected_entities=["DPR", "ColBERT"],
        expected_chunk_types=["abstract", "section"],
    ),
    TestQuery(
        query_id="demo_comparative_003",
        query="How does RAG compare to Self-RAG in terms of retrieval strategy?",
        query_type=QueryType.COMPARATIVE,
        expected_topics=["retrieval", "generation", "adaptive", "always retrieve"],
        expected_entities=["RAG", "Self-RAG"],
        expected_chunk_types=["abstract", "section"],
    ),
    TestQuery(
        query_id="demo_comparative_004",
        query="Compare chain-of-thought prompting with tree of thoughts for reasoning.",
        query_type=QueryType.COMPARATIVE,
        expected_topics=["reasoning", "step", "branch", "exploration", "search"],
        expected_entities=["chain-of-thought", "tree of thoughts"],
        expected_chunk_types=["abstract", "section"],
    ),

    # === NOVELTY (3 queries) ===
    TestQuery(
        query_id="demo_novelty_001",
        query="What was novel about the Transformer architecture when it was introduced?",
        query_type=QueryType.NOVELTY,
        expected_topics=["novel", "self-attention", "recurrence", "parallel"],
        expected_entities=["Transformer"],
        expected_chunk_types=["abstract", "section"],
    ),
    TestQuery(
        query_id="demo_novelty_002",
        query="What gap did HyDE address in zero-shot dense retrieval?",
        query_type=QueryType.NOVELTY,
        expected_topics=["zero-shot", "hypothetical", "relevance", "gap"],
        expected_entities=["HyDE"],
        expected_chunk_types=["abstract", "section"],
    ),
    TestQuery(
        query_id="demo_novelty_003",
        query="Has Toolformer's approach to teaching LLMs to use tools been done before?",
        query_type=QueryType.NOVELTY,
        expected_topics=["tool", "API", "self-supervised", "prior work"],
        expected_entities=["Toolformer"],
        expected_chunk_types=["abstract", "section"],
    ),

    # === LIMITATIONS (3 queries) ===
    TestQuery(
        query_id="demo_limitations_001",
        query="What are the limitations of few-shot learning with GPT-3?",
        query_type=QueryType.LIMITATIONS,
        expected_topics=["limitation", "bias", "cost", "hallucination", "prompt"],
        expected_entities=["GPT-3"],
        expected_chunk_types=["section"],
    ),
    TestQuery(
        query_id="demo_limitations_002",
        query="What are the known weaknesses of RLHF as used in InstructGPT?",
        query_type=QueryType.LIMITATIONS,
        expected_topics=["limitation", "reward", "alignment", "bias", "human"],
        expected_entities=["RLHF", "InstructGPT"],
        expected_chunk_types=["section"],
    ),
    TestQuery(
        query_id="demo_limitations_003",
        query="What caveats does the RAG paper mention about retrieval-augmented generation?",
        query_type=QueryType.LIMITATIONS,
        expected_topics=["limitation", "retrieval", "quality", "noise"],
        expected_entities=["RAG"],
        expected_chunk_types=["section"],
    ),

    # === FRAMING (3 queries) ===
    TestQuery(
        query_id="demo_framing_001",
        query="How do I argue that retrieval-augmented generation is better than pure parametric models?",
        query_type=QueryType.FRAMING,
        expected_topics=["knowledge", "factual", "grounding", "hallucination", "updatable"],
        expected_entities=["RAG"],
        expected_chunk_types=["abstract", "section"],
        requires_human_eval=True,
    ),
    TestQuery(
        query_id="demo_framing_002",
        query="How should I position Constitutional AI as an improvement over standard RLHF?",
        query_type=QueryType.FRAMING,
        expected_topics=["constitutional", "principle", "harmless", "scalable"],
        expected_entities=["Constitutional AI", "RLHF"],
        expected_chunk_types=["abstract", "section"],
        requires_human_eval=True,
    ),
    TestQuery(
        query_id="demo_framing_003",
        query="How do I justify using LoRA instead of full fine-tuning for adapting large models?",
        query_type=QueryType.FRAMING,
        expected_topics=["efficient", "parameter", "cost", "comparable", "performance"],
        expected_entities=["LoRA"],
        expected_chunk_types=["abstract", "section"],
        requires_human_eval=True,
    ),
]
