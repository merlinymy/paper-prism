"""100 benchmark queries for the AI/ML demo corpus (20 papers).

Written from the perspective of a postdoc researcher studying LLMs,
retrieval-augmented generation, and efficient adaptation methods.

Distribution:
  - 30 single-paper factual (specific numbers, methods, results)
  - 25 cross-paper synthesis (multi-paper reasoning, trends)
  - 15 vague/exploratory (real user behavior)
  - 10 adversarial/boundary (partial corpus coverage, reasoning beyond stated)
  - 10 refusal (unanswerable from corpus)
  - 10 robustness (typos, paraphrases, non-native phrasing)
"""

from evaluation.comprehensive_benchmark import BenchmarkQuery, GroundTruthClaim

# Paper ID reference:
# c8b3e6e172a7 = Attention Is All You Need (Vaswani 2017)
# 55a4e64633de = LoRA (Hu 2021)
# a63eb71679c1 = DPR (Karpukhin 2020)
# 29e8a4397a48 = BERT (Devlin 2019)
# cae01624e02c = Self-RAG (Asai 2023)
# 0004c00e9958 = GPT-3 (Brown 2020)
# cb02ca4b7682 = LLaMA (Touvron 2023)
# eef16825bcb4 = Mistral 7B (Jiang 2023)
# 7d45caedf31f = InstructGPT (Ouyang 2022)
# f1470ed318c7 = Lost in the Middle (Liu 2023)
# da5b38b62bfd = RAPTOR (Sarthi 2024)
# bef7f1f5382e = HyDE (Gao 2022)
# 4ddeacf6ee26 = Chain of Thought (Wei 2022)
# 2dcba435c0eb = RAG (Lewis 2020)
# 90f334a9266b = ColBERT (Khattab 2020)
# 061b473a798e = Constitutional AI (Bai 2022)
# 0d32b6118fcc = Toolformer (Schick 2023)
# 802732405ac0 = Tree of Thoughts (Yao 2023)
# bc2b47623304 = GPT-2 (Radford 2019)
# 01aa469d9460 = RAG Survey (Gao 2024)


BENCHMARK_QUERIES_V2 = [

    # =========================================================================
    # SINGLE-PAPER FACTUAL (30 queries)
    # =========================================================================

    # --- Transformer ---
    BenchmarkQuery(
        query_id="sf_001",
        query="How many encoder and decoder layers does the base Transformer model have?",
        category="factual",
        ground_truth_answer="The base Transformer has 6 encoder layers and 6 decoder layers.",
        ground_truth_claims=[
            GroundTruthClaim("Base Transformer has 6 encoder layers", "c8b3e6e172a7", "c8b3e6e172a7_abstract", "numerical"),
            GroundTruthClaim("Base Transformer has 6 decoder layers", "c8b3e6e172a7", "c8b3e6e172a7_abstract", "numerical"),
        ],
        relevant_paper_ids=["c8b3e6e172a7"],
        relevant_chunk_ids=["c8b3e6e172a7_abstract"],
    ),
    BenchmarkQuery(
        query_id="sf_002",
        query="What hardware was used to train the Transformer and how long did training take?",
        category="factual",
        ground_truth_answer="The base Transformer was trained on 8 NVIDIA P100 GPUs, with each training step taking about 0.4 seconds. The base models were trained for 100,000 steps (about 12 hours).",
        ground_truth_claims=[
            GroundTruthClaim("Trained on 8 NVIDIA P100 GPUs", "c8b3e6e172a7", "c8b3e6e172a7_fine_29", "numerical"),
            GroundTruthClaim("Each training step took about 0.4 seconds", "c8b3e6e172a7", "c8b3e6e172a7_fine_29", "numerical"),
        ],
        relevant_paper_ids=["c8b3e6e172a7"],
        relevant_chunk_ids=["c8b3e6e172a7_fine_29"],
    ),
    BenchmarkQuery(
        query_id="sf_003",
        query="What BLEU score did the Transformer achieve on WMT 2014 English-to-German translation?",
        category="factual",
        ground_truth_answer="The big Transformer model achieved a BLEU score of 28.4 on WMT 2014 English-to-German, improving over the existing best results by more than 2 BLEU.",
        ground_truth_claims=[
            GroundTruthClaim("Transformer achieved 28.4 BLEU on EN-DE WMT 2014", "c8b3e6e172a7", "c8b3e6e172a7_abstract", "numerical"),
        ],
        relevant_paper_ids=["c8b3e6e172a7"],
        relevant_chunk_ids=["c8b3e6e172a7_abstract"],
    ),

    # --- LoRA ---
    BenchmarkQuery(
        query_id="sf_004",
        query="By how much does LoRA reduce the number of trainable parameters compared to full fine-tuning on GPT-3?",
        category="factual",
        ground_truth_answer="LoRA reduces trainable parameters by 10,000x and GPU memory by 3x compared to full fine-tuning of GPT-3 175B.",
        ground_truth_claims=[
            GroundTruthClaim("LoRA reduces trainable parameters by 10,000x", "55a4e64633de", "55a4e64633de_abstract", "numerical"),
            GroundTruthClaim("LoRA reduces GPU memory by 3x", "55a4e64633de", "55a4e64633de_abstract", "numerical"),
        ],
        relevant_paper_ids=["55a4e64633de"],
        relevant_chunk_ids=["55a4e64633de_abstract"],
    ),
    BenchmarkQuery(
        query_id="sf_005",
        query="What does LoRA stand for and what is the core idea?",
        category="factual",
        ground_truth_answer="LoRA stands for Low-Rank Adaptation. The core idea is to freeze pre-trained model weights and inject trainable rank decomposition matrices into each layer, so only the low-rank matrices are trained.",
        ground_truth_claims=[
            GroundTruthClaim("LoRA stands for Low-Rank Adaptation", "55a4e64633de", "55a4e64633de_abstract", "factual"),
            GroundTruthClaim("Freezes pre-trained weights and injects low-rank matrices", "55a4e64633de", "55a4e64633de_abstract", "methodological"),
        ],
        relevant_paper_ids=["55a4e64633de"],
        relevant_chunk_ids=["55a4e64633de_abstract"],
    ),

    # --- BERT ---
    BenchmarkQuery(
        query_id="sf_006",
        query="What are the two pre-training tasks used by BERT?",
        category="factual",
        ground_truth_answer="BERT uses two pre-training tasks: (1) Masked Language Modeling (MLM), where random tokens are masked and predicted, and (2) Next Sentence Prediction (NSP), where the model predicts whether two sentences are consecutive.",
        ground_truth_claims=[
            GroundTruthClaim("BERT uses Masked Language Modeling (MLM)", "29e8a4397a48", "29e8a4397a48_abstract", "factual"),
            GroundTruthClaim("BERT uses Next Sentence Prediction (NSP)", "29e8a4397a48", "29e8a4397a48_abstract", "factual"),
        ],
        relevant_paper_ids=["29e8a4397a48"],
        relevant_chunk_ids=["29e8a4397a48_abstract"],
    ),
    BenchmarkQuery(
        query_id="sf_007",
        query="How many parameters does BERT-Large have?",
        category="factual",
        ground_truth_answer="BERT-Large has 340 million parameters (24 layers, 1024 hidden size, 16 attention heads).",
        ground_truth_claims=[
            GroundTruthClaim("BERT-Large has 340M parameters", "29e8a4397a48", "29e8a4397a48_abstract", "numerical"),
        ],
        relevant_paper_ids=["29e8a4397a48"],
        relevant_chunk_ids=["29e8a4397a48_abstract"],
    ),

    # --- GPT-3 ---
    BenchmarkQuery(
        query_id="sf_008",
        query="How many parameters does GPT-3 have and what is the context window size?",
        category="factual",
        ground_truth_answer="GPT-3 has 175 billion parameters and uses a context window of 2048 tokens.",
        ground_truth_claims=[
            GroundTruthClaim("GPT-3 has 175 billion parameters", "0004c00e9958", "0004c00e9958_abstract", "numerical"),
        ],
        relevant_paper_ids=["0004c00e9958"],
        relevant_chunk_ids=["0004c00e9958_abstract"],
    ),
    BenchmarkQuery(
        query_id="sf_009",
        query="What are the three evaluation settings used in the GPT-3 paper?",
        category="factual",
        ground_truth_answer="The GPT-3 paper evaluates in three settings: zero-shot (task description only), one-shot (one example), and few-shot (a few examples) — all without gradient updates.",
        ground_truth_claims=[
            GroundTruthClaim("GPT-3 uses zero-shot, one-shot, and few-shot settings", "0004c00e9958", "0004c00e9958_abstract", "factual"),
            GroundTruthClaim("All settings are without gradient updates", "0004c00e9958", "0004c00e9958_abstract", "factual"),
        ],
        relevant_paper_ids=["0004c00e9958"],
        relevant_chunk_ids=["0004c00e9958_abstract"],
    ),

    # --- LLaMA ---
    BenchmarkQuery(
        query_id="sf_010",
        query="What training data did LLaMA use? Was any of it proprietary?",
        category="factual",
        ground_truth_answer="LLaMA was trained exclusively on publicly available datasets, without using any proprietary data. The training data includes CommonCrawl, C4, GitHub, Wikipedia, books, ArXiv, and StackExchange.",
        ground_truth_claims=[
            GroundTruthClaim("LLaMA uses only publicly available data", "cb02ca4b7682", "cb02ca4b7682_abstract", "factual"),
        ],
        relevant_paper_ids=["cb02ca4b7682"],
        relevant_chunk_ids=["cb02ca4b7682_abstract"],
    ),
    BenchmarkQuery(
        query_id="sf_011",
        query="How does LLaMA-13B compare to GPT-3 175B on benchmarks?",
        category="factual",
        ground_truth_answer="LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, despite being more than 10x smaller.",
        ground_truth_claims=[
            GroundTruthClaim("LLaMA-13B outperforms GPT-3 175B on most benchmarks", "cb02ca4b7682", "cb02ca4b7682_abstract", "factual"),
        ],
        relevant_paper_ids=["cb02ca4b7682"],
        relevant_chunk_ids=["cb02ca4b7682_abstract"],
    ),

    # --- Mistral ---
    BenchmarkQuery(
        query_id="sf_012",
        query="What attention mechanisms does Mistral 7B use?",
        category="factual",
        ground_truth_answer="Mistral 7B uses grouped-query attention (GQA) for faster inference and sliding window attention (SWA) to handle longer sequences at reduced cost.",
        ground_truth_claims=[
            GroundTruthClaim("Mistral uses grouped-query attention (GQA)", "eef16825bcb4", "eef16825bcb4_abstract", "factual"),
            GroundTruthClaim("Mistral uses sliding window attention (SWA)", "eef16825bcb4", "eef16825bcb4_abstract", "factual"),
        ],
        relevant_paper_ids=["eef16825bcb4"],
        relevant_chunk_ids=["eef16825bcb4_abstract"],
    ),

    # --- InstructGPT ---
    BenchmarkQuery(
        query_id="sf_013",
        query="How many human labelers were used for InstructGPT's training data?",
        category="factual",
        ground_truth_answer="InstructGPT used a team of about 40 contractors to provide human feedback for training.",
        ground_truth_claims=[
            GroundTruthClaim("InstructGPT used about 40 human labelers", "7d45caedf31f", "7d45caedf31f_abstract", "numerical"),
        ],
        relevant_paper_ids=["7d45caedf31f"],
        relevant_chunk_ids=["7d45caedf31f_abstract"],
    ),
    BenchmarkQuery(
        query_id="sf_014",
        query="What model size was used as the base for InstructGPT and how does it compare to the larger GPT-3?",
        category="factual",
        ground_truth_answer="InstructGPT's 1.3B parameter model (with RLHF) is preferred over the 175B GPT-3 outputs despite being 100x smaller.",
        ground_truth_claims=[
            GroundTruthClaim("InstructGPT 1.3B with RLHF preferred over 175B GPT-3", "7d45caedf31f", "7d45caedf31f_abstract", "factual"),
        ],
        relevant_paper_ids=["7d45caedf31f"],
        relevant_chunk_ids=["7d45caedf31f_abstract"],
    ),

    # --- Self-RAG ---
    BenchmarkQuery(
        query_id="sf_015",
        query="What are the four types of reflection tokens in Self-RAG?",
        category="factual",
        ground_truth_answer="Self-RAG uses four reflection tokens: Retrieve (whether to retrieve), IsRel (relevance of retrieved passage), IsSup (whether passage supports generation), and IsUse (overall utility of response).",
        ground_truth_claims=[
            GroundTruthClaim("Self-RAG uses Retrieve, IsRel, IsSup, and IsUse tokens", "cae01624e02c", "cae01624e02c_fine_20", "factual"),
        ],
        relevant_paper_ids=["cae01624e02c"],
        relevant_chunk_ids=["cae01624e02c_fine_20"],
    ),

    # --- DPR ---
    BenchmarkQuery(
        query_id="sf_016",
        query="How does DPR encode passages and queries?",
        category="factual",
        ground_truth_answer="DPR uses two independent BERT encoders — one for passages and one for queries. Each produces a dense vector, and relevance is computed via dot product similarity.",
        ground_truth_claims=[
            GroundTruthClaim("DPR uses separate BERT encoders for passages and queries", "a63eb71679c1", "a63eb71679c1_abstract", "methodological"),
            GroundTruthClaim("Relevance is computed via dot product", "a63eb71679c1", "a63eb71679c1_abstract", "methodological"),
        ],
        relevant_paper_ids=["a63eb71679c1"],
        relevant_chunk_ids=["a63eb71679c1_abstract"],
    ),
    BenchmarkQuery(
        query_id="sf_017",
        query="On which datasets does DPR outperform BM25?",
        category="factual",
        ground_truth_answer="DPR outperforms BM25 on Natural Questions, TriviaQA, WebQuestions, and CuratedTREC. SQuAD is the exception where BM25 performs comparably.",
        ground_truth_claims=[
            GroundTruthClaim("DPR outperforms BM25 on most QA datasets", "a63eb71679c1", "a63eb71679c1_fine_29", "factual"),
            GroundTruthClaim("SQuAD is an exception where BM25 is competitive", "a63eb71679c1", "a63eb71679c1_fine_29", "factual"),
        ],
        relevant_paper_ids=["a63eb71679c1"],
        relevant_chunk_ids=["a63eb71679c1_fine_29"],
    ),

    # --- ColBERT ---
    BenchmarkQuery(
        query_id="sf_018",
        query="What is ColBERT's late interaction mechanism?",
        category="factual",
        ground_truth_answer="ColBERT encodes queries and documents into sets of token-level embeddings independently, then computes relevance via a cheap but expressive interaction step using MaxSim — the maximum similarity between each query token and all document tokens.",
        ground_truth_claims=[
            GroundTruthClaim("ColBERT uses token-level embeddings with late interaction", "90f334a9266b", "90f334a9266b_abstract", "methodological"),
        ],
        relevant_paper_ids=["90f334a9266b"],
        relevant_chunk_ids=["90f334a9266b_abstract"],
    ),

    # --- Lost in the Middle ---
    BenchmarkQuery(
        query_id="sf_019",
        query="What is the main finding of the 'Lost in the Middle' paper?",
        category="factual",
        ground_truth_answer="Language models perform best when relevant information is at the beginning or end of the input context, but performance degrades significantly when it's in the middle. This U-shaped performance curve holds across multiple models.",
        ground_truth_claims=[
            GroundTruthClaim("Performance degrades when relevant info is in the middle of context", "f1470ed318c7", "f1470ed318c7_abstract", "factual"),
            GroundTruthClaim("Performance follows a U-shaped curve based on position", "f1470ed318c7", "f1470ed318c7_abstract", "factual"),
        ],
        relevant_paper_ids=["f1470ed318c7"],
        relevant_chunk_ids=["f1470ed318c7_abstract"],
    ),

    # --- Chain of Thought ---
    BenchmarkQuery(
        query_id="sf_020",
        query="At what model scale does chain-of-thought prompting become effective?",
        category="factual",
        ground_truth_answer="Chain-of-thought prompting is an emergent ability of sufficiently large language models — it provides little benefit for small models but significantly improves reasoning in models with ~100B+ parameters.",
        ground_truth_claims=[
            GroundTruthClaim("Chain-of-thought is emergent in large models", "4ddeacf6ee26", "4ddeacf6ee26_abstract", "factual"),
        ],
        relevant_paper_ids=["4ddeacf6ee26"],
        relevant_chunk_ids=["4ddeacf6ee26_abstract"],
    ),

    # --- RAG ---
    BenchmarkQuery(
        query_id="sf_021",
        query="What are the two RAG variants described in the original RAG paper?",
        category="factual",
        ground_truth_answer="The RAG paper introduces RAG-Sequence (uses the same retrieved document for the entire sequence) and RAG-Token (can use different documents for different tokens).",
        ground_truth_claims=[
            GroundTruthClaim("RAG has two variants: RAG-Sequence and RAG-Token", "2dcba435c0eb", "2dcba435c0eb_abstract", "factual"),
        ],
        relevant_paper_ids=["2dcba435c0eb"],
        relevant_chunk_ids=["2dcba435c0eb_abstract"],
    ),

    # --- HyDE ---
    BenchmarkQuery(
        query_id="sf_022",
        query="How does HyDE generate hypothetical documents?",
        category="factual",
        ground_truth_answer="HyDE uses an instruction-following language model (like InstructGPT) to generate a hypothetical document given a query. This document is then encoded with a contrastive encoder to retrieve real relevant documents.",
        ground_truth_claims=[
            GroundTruthClaim("HyDE generates hypothetical documents using an instruction-following LM", "bef7f1f5382e", "bef7f1f5382e_abstract", "methodological"),
            GroundTruthClaim("The hypothetical document is encoded to retrieve real documents", "bef7f1f5382e", "bef7f1f5382e_abstract", "methodological"),
        ],
        relevant_paper_ids=["bef7f1f5382e"],
        relevant_chunk_ids=["bef7f1f5382e_abstract"],
    ),

    # --- RAPTOR ---
    BenchmarkQuery(
        query_id="sf_023",
        query="How does RAPTOR construct its tree structure?",
        category="factual",
        ground_truth_answer="RAPTOR recursively clusters text chunks using embedding similarity, then summarizes each cluster. These summaries become nodes in a tree, and the process repeats to form higher-level abstractions.",
        ground_truth_claims=[
            GroundTruthClaim("RAPTOR clusters text chunks by embedding similarity", "da5b38b62bfd", "da5b38b62bfd_abstract", "methodological"),
            GroundTruthClaim("Clusters are summarized recursively to build a tree", "da5b38b62bfd", "da5b38b62bfd_abstract", "methodological"),
        ],
        relevant_paper_ids=["da5b38b62bfd"],
        relevant_chunk_ids=["da5b38b62bfd_abstract"],
    ),

    # --- Constitutional AI ---
    BenchmarkQuery(
        query_id="sf_024",
        query="What is the RLAIF component of Constitutional AI?",
        category="factual",
        ground_truth_answer="RLAIF (Reinforcement Learning from AI Feedback) replaces human feedback with AI-generated feedback. The AI uses a set of principles (a constitution) to evaluate and improve responses, training a harmless assistant without human labels for harmful outputs.",
        ground_truth_claims=[
            GroundTruthClaim("Constitutional AI uses AI feedback instead of human labels", "061b473a798e", "061b473a798e_abstract", "methodological"),
            GroundTruthClaim("A constitution (set of principles) guides the AI feedback", "061b473a798e", "061b473a798e_abstract", "factual"),
        ],
        relevant_paper_ids=["061b473a798e"],
        relevant_chunk_ids=["061b473a798e_abstract"],
    ),

    # --- Toolformer ---
    BenchmarkQuery(
        query_id="sf_025",
        query="What external tools does Toolformer learn to use?",
        category="factual",
        ground_truth_answer="Toolformer learns to use a calculator, a question answering system, a search engine, a translation system, and a calendar API.",
        ground_truth_claims=[
            GroundTruthClaim("Toolformer learns to use calculator, QA, search, translation, and calendar APIs", "0d32b6118fcc", "0d32b6118fcc_abstract", "factual"),
        ],
        relevant_paper_ids=["0d32b6118fcc"],
        relevant_chunk_ids=["0d32b6118fcc_abstract"],
    ),

    # --- Tree of Thoughts ---
    BenchmarkQuery(
        query_id="sf_026",
        query="What search algorithms does Tree of Thoughts use?",
        category="factual",
        ground_truth_answer="Tree of Thoughts uses breadth-first search (BFS) and depth-first search (DFS) to explore the space of reasoning paths, with the LLM providing heuristic evaluations at each step.",
        ground_truth_claims=[
            GroundTruthClaim("Tree of Thoughts uses BFS and DFS", "802732405ac0", "802732405ac0_abstract", "methodological"),
        ],
        relevant_paper_ids=["802732405ac0"],
        relevant_chunk_ids=["802732405ac0_abstract"],
    ),

    # --- GPT-2 ---
    BenchmarkQuery(
        query_id="sf_027",
        query="What dataset was GPT-2 trained on?",
        category="factual",
        ground_truth_answer="GPT-2 was trained on WebText, a dataset of millions of webpages scraped from outbound Reddit links with at least 3 karma.",
        ground_truth_claims=[
            GroundTruthClaim("GPT-2 was trained on WebText", "bc2b47623304", "bc2b47623304_abstract", "factual"),
        ],
        relevant_paper_ids=["bc2b47623304"],
        relevant_chunk_ids=["bc2b47623304_abstract"],
    ),
    BenchmarkQuery(
        query_id="sf_028",
        query="What zero-shot tasks could GPT-2 perform without any fine-tuning?",
        category="factual",
        ground_truth_answer="GPT-2 demonstrated zero-shot performance on reading comprehension, translation, summarization, and question answering, reaching state-of-the-art on several benchmarks without task-specific training.",
        ground_truth_claims=[
            GroundTruthClaim("GPT-2 performs zero-shot on reading comprehension, translation, summarization, and QA", "bc2b47623304", "bc2b47623304_abstract", "factual"),
        ],
        relevant_paper_ids=["bc2b47623304"],
        relevant_chunk_ids=["bc2b47623304_abstract"],
    ),

    # --- Misc single-paper ---
    BenchmarkQuery(
        query_id="sf_029",
        query="What is the key difference between RAG-Sequence and RAG-Token?",
        category="factual",
        ground_truth_answer="In RAG-Sequence, the same retrieved document is used to generate the entire output sequence. In RAG-Token, different documents can be used to generate each output token, allowing the model to draw from multiple sources within a single answer.",
        ground_truth_claims=[
            GroundTruthClaim("RAG-Sequence uses one document for the whole sequence", "2dcba435c0eb", "2dcba435c0eb_abstract", "factual"),
            GroundTruthClaim("RAG-Token can use different documents per token", "2dcba435c0eb", "2dcba435c0eb_abstract", "factual"),
        ],
        relevant_paper_ids=["2dcba435c0eb"],
        relevant_chunk_ids=["2dcba435c0eb_abstract"],
    ),
    BenchmarkQuery(
        query_id="sf_030",
        query="How does Toolformer decide when to insert API calls?",
        category="factual",
        ground_truth_answer="Toolformer uses a self-supervised approach: it samples potential API calls, executes them, and keeps only those that reduce perplexity on the next tokens. This creates training data without human annotation.",
        ground_truth_claims=[
            GroundTruthClaim("Toolformer self-supervisedly decides API calls by perplexity reduction", "0d32b6118fcc", "0d32b6118fcc_abstract", "methodological"),
        ],
        relevant_paper_ids=["0d32b6118fcc"],
        relevant_chunk_ids=["0d32b6118fcc_abstract"],
    ),

    # =========================================================================
    # CROSS-PAPER SYNTHESIS (25 queries)
    # =========================================================================

    BenchmarkQuery(
        query_id="cs_001",
        query="What are the different retrieval strategies proposed across these papers and how do they differ?",
        category="cross_paper",
        ground_truth_answer="Multiple retrieval strategies: DPR uses dense single-vector encoding with dot product; ColBERT uses late interaction with token-level matching; RAG integrates retrieval into generation; Self-RAG adds selective retrieval with reflection tokens; RAPTOR uses recursive tree-structured summarization; HyDE generates hypothetical documents for zero-shot retrieval.",
        ground_truth_claims=[
            GroundTruthClaim("DPR uses dense single-vector encoding", "a63eb71679c1", "a63eb71679c1_abstract", "factual"),
            GroundTruthClaim("ColBERT uses late interaction", "90f334a9266b", "90f334a9266b_abstract", "factual"),
            GroundTruthClaim("Self-RAG uses selective retrieval with reflection", "cae01624e02c", "cae01624e02c_abstract", "factual"),
            GroundTruthClaim("RAPTOR uses recursive tree summarization", "da5b38b62bfd", "da5b38b62bfd_abstract", "factual"),
            GroundTruthClaim("HyDE uses hypothetical document generation", "bef7f1f5382e", "bef7f1f5382e_abstract", "factual"),
        ],
        relevant_paper_ids=["a63eb71679c1", "90f334a9266b", "2dcba435c0eb", "cae01624e02c", "da5b38b62bfd", "bef7f1f5382e"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_002",
        query="How has the approach to alignment evolved from InstructGPT to Constitutional AI?",
        category="cross_paper",
        ground_truth_answer="InstructGPT uses RLHF with human labelers providing demonstrations and comparisons. Constitutional AI replaces human feedback with AI feedback guided by a set of principles (constitution), reducing dependence on human annotation while maintaining alignment quality.",
        ground_truth_claims=[
            GroundTruthClaim("InstructGPT relies on human labelers for RLHF", "7d45caedf31f", "7d45caedf31f_abstract", "factual"),
            GroundTruthClaim("Constitutional AI replaces human feedback with AI feedback", "061b473a798e", "061b473a798e_abstract", "factual"),
        ],
        relevant_paper_ids=["7d45caedf31f", "061b473a798e"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_003",
        query="What is the trend in model sizes across the language models in this corpus?",
        category="cross_paper",
        ground_truth_answer="Model sizes grew from GPT-2 (1.5B, 2019) to GPT-3 (175B, 2020) to LLaMA (7-65B, 2023), but then Mistral 7B (2023) showed a smaller model could outperform much larger ones, and LoRA showed efficient adaptation eliminates the need for full-scale models.",
        ground_truth_claims=[
            GroundTruthClaim("GPT-2 has 1.5B parameters", "bc2b47623304", "bc2b47623304_abstract", "numerical"),
            GroundTruthClaim("GPT-3 has 175B parameters", "0004c00e9958", "0004c00e9958_abstract", "numerical"),
            GroundTruthClaim("LLaMA ranges from 7B to 65B", "cb02ca4b7682", "cb02ca4b7682_abstract", "numerical"),
            GroundTruthClaim("Mistral 7B outperforms larger models", "eef16825bcb4", "eef16825bcb4_abstract", "factual"),
        ],
        relevant_paper_ids=["bc2b47623304", "0004c00e9958", "cb02ca4b7682", "eef16825bcb4", "55a4e64633de"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_004",
        query="Compare DPR, ColBERT, and HyDE — what tradeoffs does each make?",
        category="cross_paper",
        ground_truth_answer="DPR is simple and fast (single vector per passage) but loses fine-grained matching. ColBERT retains token-level interaction for better quality but requires more storage. HyDE requires no training data but adds latency from hypothetical document generation.",
        ground_truth_claims=[
            GroundTruthClaim("DPR uses single vectors, fast but less granular", "a63eb71679c1", "a63eb71679c1_abstract", "factual"),
            GroundTruthClaim("ColBERT uses token-level interaction, better quality but more storage", "90f334a9266b", "90f334a9266b_abstract", "factual"),
            GroundTruthClaim("HyDE requires no training data but adds generation latency", "bef7f1f5382e", "bef7f1f5382e_abstract", "factual"),
        ],
        relevant_paper_ids=["a63eb71679c1", "90f334a9266b", "bef7f1f5382e"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_005",
        query="Which papers in this corpus address the problem of hallucination and what solutions do they propose?",
        category="cross_paper",
        ground_truth_answer="RAG reduces hallucination by grounding in retrieved passages. Self-RAG adds reflection tokens for self-critique. InstructGPT/RLHF aligns output with human preferences. Constitutional AI uses principles for self-correction. The Lost in the Middle paper shows context positioning affects accuracy.",
        ground_truth_claims=[
            GroundTruthClaim("RAG grounds in retrieved passages", "2dcba435c0eb", "2dcba435c0eb_abstract", "factual"),
            GroundTruthClaim("Self-RAG uses reflection tokens", "cae01624e02c", "cae01624e02c_abstract", "factual"),
            GroundTruthClaim("RLHF aligns with human preferences", "7d45caedf31f", "7d45caedf31f_abstract", "factual"),
            GroundTruthClaim("Constitutional AI uses principles for self-correction", "061b473a798e", "061b473a798e_abstract", "factual"),
        ],
        relevant_paper_ids=["2dcba435c0eb", "cae01624e02c", "7d45caedf31f", "061b473a798e", "f1470ed318c7"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_006",
        query="How do the prompting strategies in Chain-of-Thought and Tree of Thoughts differ in their approach to reasoning?",
        category="cross_paper",
        ground_truth_answer="Chain-of-thought uses a single linear chain of reasoning steps via few-shot examples. Tree of Thoughts generalizes this by exploring multiple reasoning paths using search (BFS/DFS) and allowing backtracking when a path fails.",
        ground_truth_claims=[
            GroundTruthClaim("CoT uses linear reasoning chains via few-shot", "4ddeacf6ee26", "4ddeacf6ee26_abstract", "factual"),
            GroundTruthClaim("ToT explores multiple paths with search and backtracking", "802732405ac0", "802732405ac0_abstract", "factual"),
        ],
        relevant_paper_ids=["4ddeacf6ee26", "802732405ac0"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_007",
        query="What papers discuss parameter-efficient methods and what approaches do they take?",
        category="cross_paper",
        ground_truth_answer="LoRA injects trainable low-rank matrices while freezing pre-trained weights. Toolformer teaches models to use external tools instead of encoding all knowledge. The concept of adapters and efficient fine-tuning is also discussed in the context of BERT fine-tuning.",
        ground_truth_claims=[
            GroundTruthClaim("LoRA uses trainable low-rank matrices", "55a4e64633de", "55a4e64633de_abstract", "factual"),
            GroundTruthClaim("Toolformer uses external tools for efficiency", "0d32b6118fcc", "0d32b6118fcc_abstract", "factual"),
        ],
        relevant_paper_ids=["55a4e64633de", "0d32b6118fcc"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_008",
        query="Which models in this corpus use the Transformer architecture and how do they modify it?",
        category="cross_paper",
        ground_truth_answer="BERT uses only the encoder with bidirectional attention. GPT-2/GPT-3 use only the decoder with causal attention. LLaMA adds RoPE embeddings, SwiGLU activation, and pre-normalization. Mistral adds grouped-query attention and sliding window attention.",
        ground_truth_claims=[
            GroundTruthClaim("BERT uses Transformer encoder with bidirectional attention", "29e8a4397a48", "29e8a4397a48_abstract", "factual"),
            GroundTruthClaim("GPT models use Transformer decoder with causal attention", "0004c00e9958", "0004c00e9958_abstract", "factual"),
            GroundTruthClaim("LLaMA adds RoPE and SwiGLU", "cb02ca4b7682", "cb02ca4b7682_abstract", "factual"),
            GroundTruthClaim("Mistral adds GQA and sliding window attention", "eef16825bcb4", "eef16825bcb4_abstract", "factual"),
        ],
        relevant_paper_ids=["c8b3e6e172a7", "29e8a4397a48", "0004c00e9958", "cb02ca4b7682", "eef16825bcb4"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_009",
        query="How does retrieval-augmented generation evolve from RAG to Self-RAG to RAPTOR?",
        category="cross_paper",
        ground_truth_answer="RAG (2020) always retrieves and marginalizes over passages. Self-RAG (2023) adds selective retrieval with reflection tokens — the model decides when to retrieve and critiques its own outputs. RAPTOR (2024) pre-processes the corpus into a tree of summaries for multi-granularity retrieval.",
        ground_truth_claims=[
            GroundTruthClaim("RAG always retrieves and marginalizes", "2dcba435c0eb", "2dcba435c0eb_abstract", "factual"),
            GroundTruthClaim("Self-RAG adds selective retrieval with reflection", "cae01624e02c", "cae01624e02c_abstract", "factual"),
            GroundTruthClaim("RAPTOR builds a tree of summaries", "da5b38b62bfd", "da5b38b62bfd_abstract", "factual"),
        ],
        relevant_paper_ids=["2dcba435c0eb", "cae01624e02c", "da5b38b62bfd"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_010",
        query="What evidence across these papers supports the scaling hypothesis — that larger models are better?",
        category="cross_paper",
        ground_truth_answer="GPT-3 shows few-shot abilities emerge with scale. Chain-of-thought only works at ~100B+ parameters. However, LLaMA-13B outperforms GPT-3 175B with better data, and Mistral 7B beats larger models — suggesting training data quality and architecture matter more than raw size.",
        ground_truth_claims=[
            GroundTruthClaim("GPT-3 shows emergent few-shot abilities at scale", "0004c00e9958", "0004c00e9958_abstract", "factual"),
            GroundTruthClaim("Chain-of-thought requires large models (~100B+)", "4ddeacf6ee26", "4ddeacf6ee26_abstract", "factual"),
            GroundTruthClaim("LLaMA-13B outperforms GPT-3 175B", "cb02ca4b7682", "cb02ca4b7682_abstract", "factual"),
            GroundTruthClaim("Mistral 7B beats larger open models", "eef16825bcb4", "eef16825bcb4_abstract", "factual"),
        ],
        relevant_paper_ids=["0004c00e9958", "4ddeacf6ee26", "cb02ca4b7682", "eef16825bcb4"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_011",
        query="What are the different ways these papers evaluate their models?",
        category="cross_paper",
        ground_truth_answer="BLEU scores for translation (Transformer), perplexity for language modeling (GPT-2/3), F1/exact match for QA (DPR, BERT), human preference ratings (InstructGPT), top-k accuracy for retrieval (DPR, ColBERT), and task-specific benchmarks like GLUE/SuperGLUE (BERT).",
        ground_truth_claims=[
            GroundTruthClaim("Transformer uses BLEU for translation", "c8b3e6e172a7", "c8b3e6e172a7_abstract", "factual"),
            GroundTruthClaim("InstructGPT uses human preference ratings", "7d45caedf31f", "7d45caedf31f_abstract", "factual"),
            GroundTruthClaim("BERT evaluates on GLUE benchmarks", "29e8a4397a48", "29e8a4397a48_abstract", "factual"),
        ],
        relevant_paper_ids=["c8b3e6e172a7", "0004c00e9958", "29e8a4397a48", "7d45caedf31f", "a63eb71679c1"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_012",
        query="If I'm building a QA system, which papers should I read and what would each contribute?",
        category="cross_paper",
        ground_truth_answer="DPR for dense passage retrieval. ColBERT for efficient token-level matching. RAG for integrating retrieval with generation. Self-RAG for selective retrieval with quality control. RAPTOR for hierarchical document representation. HyDE for zero-shot retrieval. Lost in the Middle for understanding context window limitations.",
        ground_truth_claims=[
            GroundTruthClaim("DPR provides dense passage retrieval", "a63eb71679c1", "a63eb71679c1_abstract", "factual"),
            GroundTruthClaim("RAG integrates retrieval with generation", "2dcba435c0eb", "2dcba435c0eb_abstract", "factual"),
            GroundTruthClaim("Self-RAG adds quality control to retrieval", "cae01624e02c", "cae01624e02c_abstract", "factual"),
        ],
        relevant_paper_ids=["a63eb71679c1", "90f334a9266b", "2dcba435c0eb", "cae01624e02c", "da5b38b62bfd", "bef7f1f5382e", "f1470ed318c7"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_013",
        query="What do GPT-2 and GPT-3 papers tell us about the relationship between pre-training data and capabilities?",
        category="cross_paper",
        ground_truth_answer="GPT-2 showed WebText-trained models can do zero-shot tasks. GPT-3 scaled this up showing few-shot learning emerges with 175B parameters trained on a larger web corpus. Both demonstrate that pre-training on diverse internet text enables task generalization without fine-tuning.",
        ground_truth_claims=[
            GroundTruthClaim("GPT-2 showed zero-shot capabilities from WebText", "bc2b47623304", "bc2b47623304_abstract", "factual"),
            GroundTruthClaim("GPT-3 showed few-shot learning at 175B scale", "0004c00e9958", "0004c00e9958_abstract", "factual"),
        ],
        relevant_paper_ids=["bc2b47623304", "0004c00e9958"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_014",
        query="How do different papers handle the quality-efficiency tradeoff in retrieval?",
        category="cross_paper",
        ground_truth_answer="DPR is fast (single vector) but misses nuanced matching. ColBERT balances quality and efficiency with pre-computed token embeddings and late interaction. RAPTOR trades indexing cost for retrieval quality with hierarchical summaries. HyDE trades inference latency for zero-shot capability.",
        ground_truth_claims=[
            GroundTruthClaim("DPR uses single vectors for speed", "a63eb71679c1", "a63eb71679c1_abstract", "factual"),
            GroundTruthClaim("ColBERT uses late interaction for balance", "90f334a9266b", "90f334a9266b_abstract", "factual"),
            GroundTruthClaim("RAPTOR trades indexing cost for retrieval quality", "da5b38b62bfd", "da5b38b62bfd_abstract", "factual"),
        ],
        relevant_paper_ids=["a63eb71679c1", "90f334a9266b", "da5b38b62bfd", "bef7f1f5382e"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_015",
        query="What papers discuss attention mechanisms and what variants do they introduce?",
        category="cross_paper",
        ground_truth_answer="The Transformer introduces multi-head self-attention. BERT uses bidirectional attention (encoder only). GPT uses causal/masked attention (decoder only). Mistral introduces sliding window attention and grouped-query attention. ColBERT uses attention for late interaction scoring.",
        ground_truth_claims=[
            GroundTruthClaim("Transformer introduces multi-head self-attention", "c8b3e6e172a7", "c8b3e6e172a7_abstract", "factual"),
            GroundTruthClaim("Mistral introduces sliding window and grouped-query attention", "eef16825bcb4", "eef16825bcb4_abstract", "factual"),
        ],
        relevant_paper_ids=["c8b3e6e172a7", "29e8a4397a48", "0004c00e9958", "eef16825bcb4", "90f334a9266b"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_016",
        query="What open-source models are introduced in these papers and what makes them significant?",
        category="cross_paper",
        ground_truth_answer="LLaMA showed public-data-only models can match proprietary ones. Mistral 7B demonstrated a small open model can beat much larger ones. BERT made bidirectional pre-training available. GPT-2 was initially withheld then released, sparking discussion about open release.",
        ground_truth_claims=[
            GroundTruthClaim("LLaMA uses only public data", "cb02ca4b7682", "cb02ca4b7682_abstract", "factual"),
            GroundTruthClaim("Mistral 7B is an efficient open model", "eef16825bcb4", "eef16825bcb4_abstract", "factual"),
        ],
        relevant_paper_ids=["cb02ca4b7682", "eef16825bcb4", "29e8a4397a48", "bc2b47623304"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_017",
        query="How has the concept of 'in-context learning' developed across GPT-2, GPT-3, and chain-of-thought?",
        category="cross_paper",
        ground_truth_answer="GPT-2 showed models can learn tasks from context without fine-tuning. GPT-3 formalized this as zero/one/few-shot in-context learning at scale. Chain-of-thought extended it by showing that including reasoning steps in the examples dramatically improves complex reasoning.",
        ground_truth_claims=[
            GroundTruthClaim("GPT-2 demonstrated in-context learning without fine-tuning", "bc2b47623304", "bc2b47623304_abstract", "factual"),
            GroundTruthClaim("GPT-3 formalized zero/one/few-shot learning", "0004c00e9958", "0004c00e9958_abstract", "factual"),
            GroundTruthClaim("Chain-of-thought adds reasoning steps to in-context examples", "4ddeacf6ee26", "4ddeacf6ee26_abstract", "factual"),
        ],
        relevant_paper_ids=["bc2b47623304", "0004c00e9958", "4ddeacf6ee26"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_018",
        query="Across these papers, what are the main approaches to reducing the cost of deploying large models?",
        category="cross_paper",
        ground_truth_answer="LoRA reduces fine-tuning cost via low-rank adaptation. Mistral uses GQA for faster inference. Knowledge distillation (mentioned in several papers). Toolformer offloads computation to external tools. Smaller models trained better (LLaMA philosophy).",
        ground_truth_claims=[
            GroundTruthClaim("LoRA reduces fine-tuning cost", "55a4e64633de", "55a4e64633de_abstract", "factual"),
            GroundTruthClaim("Mistral uses GQA for efficiency", "eef16825bcb4", "eef16825bcb4_abstract", "factual"),
            GroundTruthClaim("Toolformer offloads to external tools", "0d32b6118fcc", "0d32b6118fcc_abstract", "factual"),
        ],
        relevant_paper_ids=["55a4e64633de", "eef16825bcb4", "0d32b6118fcc", "cb02ca4b7682"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_019",
        query="What are the implications of the 'Lost in the Middle' findings for RAG system design?",
        category="cross_paper",
        ground_truth_answer="The Lost in the Middle paper shows LLMs attend to the beginning and end of context, ignoring the middle. For RAG systems (DPR, RAG, Self-RAG), this means relevant passages should be placed at the beginning or end of the context, not buried in the middle. This affects how reranked passages should be ordered before generation.",
        ground_truth_claims=[
            GroundTruthClaim("LLMs attend to beginning and end of context, not middle", "f1470ed318c7", "f1470ed318c7_abstract", "factual"),
            GroundTruthClaim("RAG systems should position relevant passages carefully", "f1470ed318c7", "f1470ed318c7_abstract", "relational"),
        ],
        relevant_paper_ids=["f1470ed318c7", "2dcba435c0eb", "cae01624e02c"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_020",
        query="Compare BERT's and GPT's pre-training objectives. What are the strengths and weaknesses of each approach?",
        category="cross_paper",
        ground_truth_answer="BERT uses masked language modeling (bidirectional, good for understanding tasks like NER and QA) but can't generate text. GPT uses autoregressive/causal LM (good for generation) but only sees left context. BERT excels on NLU benchmarks; GPT excels on generation and in-context learning.",
        ground_truth_claims=[
            GroundTruthClaim("BERT uses bidirectional masked LM", "29e8a4397a48", "29e8a4397a48_abstract", "factual"),
            GroundTruthClaim("GPT uses autoregressive causal LM", "0004c00e9958", "0004c00e9958_abstract", "factual"),
        ],
        relevant_paper_ids=["29e8a4397a48", "0004c00e9958", "bc2b47623304"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_021",
        query="What role does human feedback play across the alignment papers?",
        category="cross_paper",
        ground_truth_answer="InstructGPT uses direct human feedback (demonstrations + comparisons) via RLHF. Constitutional AI reduces human involvement by using AI feedback guided by principles. Both aim to align model behavior with human values but differ in the degree of human involvement.",
        ground_truth_claims=[
            GroundTruthClaim("InstructGPT uses direct human demonstrations and comparisons", "7d45caedf31f", "7d45caedf31f_abstract", "factual"),
            GroundTruthClaim("Constitutional AI reduces human involvement with AI feedback", "061b473a798e", "061b473a798e_abstract", "factual"),
        ],
        relevant_paper_ids=["7d45caedf31f", "061b473a798e"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_022",
        query="What do these papers collectively say about the importance of training data quality vs quantity?",
        category="cross_paper",
        ground_truth_answer="LLaMA shows high-quality public data alone can match proprietary-data models. GPT-2's WebText was curated via Reddit karma filtering. GPT-3 used scale (300B tokens). The tension between quality and quantity runs through the corpus: LLaMA-13B beats GPT-3 175B with better data, not more parameters.",
        ground_truth_claims=[
            GroundTruthClaim("LLaMA matches proprietary models with public data", "cb02ca4b7682", "cb02ca4b7682_abstract", "factual"),
            GroundTruthClaim("GPT-2 curated WebText via Reddit karma", "bc2b47623304", "bc2b47623304_abstract", "factual"),
        ],
        relevant_paper_ids=["cb02ca4b7682", "bc2b47623304", "0004c00e9958"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_023",
        query="How do the encoder-only, decoder-only, and encoder-decoder architectures compare across these papers?",
        category="cross_paper",
        ground_truth_answer="BERT is encoder-only (bidirectional, good for understanding). GPT-2/3/LLaMA/Mistral are decoder-only (autoregressive, good for generation). The original Transformer and RAG use encoder-decoder. DPR and ColBERT use encoder-only for retrieval.",
        ground_truth_claims=[
            GroundTruthClaim("BERT is encoder-only", "29e8a4397a48", "29e8a4397a48_abstract", "factual"),
            GroundTruthClaim("GPT models are decoder-only", "0004c00e9958", "0004c00e9958_abstract", "factual"),
            GroundTruthClaim("Original Transformer is encoder-decoder", "c8b3e6e172a7", "c8b3e6e172a7_abstract", "factual"),
        ],
        relevant_paper_ids=["29e8a4397a48", "0004c00e9958", "c8b3e6e172a7", "cb02ca4b7682", "eef16825bcb4"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_024",
        query="What papers address the challenge of LLMs making things up, and do they solve it?",
        category="cross_paper",
        ground_truth_answer="RAG reduces hallucination by grounding in retrieved documents. Self-RAG adds self-evaluation to detect unsupported claims. InstructGPT improves truthfulness through human feedback. Constitutional AI uses principles. None fully solve hallucination — it remains an open problem.",
        ground_truth_claims=[
            GroundTruthClaim("RAG grounds generations in retrieved documents", "2dcba435c0eb", "2dcba435c0eb_abstract", "factual"),
            GroundTruthClaim("Self-RAG self-evaluates for unsupported claims", "cae01624e02c", "cae01624e02c_abstract", "factual"),
            GroundTruthClaim("InstructGPT improves truthfulness via RLHF", "7d45caedf31f", "7d45caedf31f_abstract", "factual"),
        ],
        relevant_paper_ids=["2dcba435c0eb", "cae01624e02c", "7d45caedf31f", "061b473a798e"],
        relevant_chunk_ids=[],
    ),
    BenchmarkQuery(
        query_id="cs_025",
        query="Create a timeline of the key papers in this corpus and their main contributions.",
        category="cross_paper",
        ground_truth_answer="2017: Transformer (attention mechanism). 2018: BERT (bidirectional pre-training). 2019: GPT-2 (zero-shot multitask). 2020: GPT-3 (few-shot learning), DPR (dense retrieval), RAG (retrieval-augmented generation), ColBERT (late interaction). 2021: LoRA (efficient adaptation). 2022: InstructGPT (RLHF), Constitutional AI, Chain-of-Thought, HyDE. 2023: LLaMA (open models), Mistral, Self-RAG, Tree of Thoughts, Toolformer, Lost in the Middle. 2024: RAPTOR.",
        ground_truth_claims=[
            GroundTruthClaim("Transformer was published in 2017", "c8b3e6e172a7", "c8b3e6e172a7_abstract", "factual"),
            GroundTruthClaim("BERT was published in 2018/2019", "29e8a4397a48", "29e8a4397a48_abstract", "factual"),
            GroundTruthClaim("GPT-3 was published in 2020", "0004c00e9958", "0004c00e9958_abstract", "factual"),
        ],
        relevant_paper_ids=["c8b3e6e172a7", "29e8a4397a48", "0004c00e9958", "cb02ca4b7682", "55a4e64633de"],
        relevant_chunk_ids=[],
    ),

    # =========================================================================
    # VAGUE / EXPLORATORY (15 queries)
    # =========================================================================

    BenchmarkQuery(
        query_id="ve_001",
        query="What's interesting about the attention mechanisms in these papers?",
        category="exploratory",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["c8b3e6e172a7", "eef16825bcb4", "29e8a4397a48", "90f334a9266b"],
        relevant_chunk_ids=[],
        notes="Open-ended. Should mention self-attention, GQA, sliding window, late interaction.",
    ),
    BenchmarkQuery(
        query_id="ve_002",
        query="Give me an overview of the training approaches used across these papers.",
        category="exploratory",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["29e8a4397a48", "0004c00e9958", "7d45caedf31f", "55a4e64633de", "061b473a798e"],
        relevant_chunk_ids=[],
        notes="Should cover pre-training, fine-tuning, RLHF, LoRA, constitutional AI.",
    ),
    BenchmarkQuery(
        query_id="ve_003",
        query="What should I read first if I care about efficiency?",
        category="exploratory",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["55a4e64633de", "eef16825bcb4", "90f334a9266b"],
        relevant_chunk_ids=[],
        notes="Should recommend LoRA, Mistral, possibly ColBERT. Reasoning should be clear.",
    ),
    BenchmarkQuery(
        query_id="ve_004",
        query="I'm writing a literature review on retrieval-augmented generation. What are the key themes?",
        category="exploratory",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["2dcba435c0eb", "cae01624e02c", "a63eb71679c1", "90f334a9266b", "da5b38b62bfd", "bef7f1f5382e", "01aa469d9460"],
        relevant_chunk_ids=[],
        notes="Should identify themes: retrieval methods, generation integration, quality control, evaluation.",
    ),
    BenchmarkQuery(
        query_id="ve_005",
        query="What are the most cited ideas from this set of papers?",
        category="exploratory",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["c8b3e6e172a7", "29e8a4397a48", "0004c00e9958"],
        relevant_chunk_ids=[],
        notes="Should mention Transformer/self-attention, BERT pre-training, GPT-3 few-shot.",
    ),
    BenchmarkQuery(
        query_id="ve_006",
        query="What problems remain unsolved based on these papers?",
        category="exploratory",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["f1470ed318c7", "cae01624e02c", "7d45caedf31f"],
        relevant_chunk_ids=[],
        notes="Should mention hallucination, lost-in-middle, alignment challenges, retrieval quality.",
    ),
    BenchmarkQuery(
        query_id="ve_007",
        query="Explain the RAG landscape to me like I'm new to it.",
        category="exploratory",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["2dcba435c0eb", "a63eb71679c1", "cae01624e02c", "da5b38b62bfd", "bef7f1f5382e"],
        relevant_chunk_ids=[],
        notes="Should give a clear overview from basic RAG to advanced approaches.",
    ),
    BenchmarkQuery(
        query_id="ve_008",
        query="What's the deal with RLHF? Is it worth the complexity?",
        category="exploratory",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["7d45caedf31f", "061b473a798e"],
        relevant_chunk_ids=[],
        notes="Should discuss InstructGPT's RLHF approach, Constitutional AI's alternative, and tradeoffs.",
    ),
    BenchmarkQuery(
        query_id="ve_009",
        query="How do these papers relate to each other? What's the big picture?",
        category="exploratory",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["c8b3e6e172a7", "29e8a4397a48", "0004c00e9958", "2dcba435c0eb"],
        relevant_chunk_ids=[],
        notes="Should map out the research landscape: foundations → scaling → retrieval → alignment.",
    ),
    BenchmarkQuery(
        query_id="ve_010",
        query="If I could only read 5 of these 20 papers, which ones and why?",
        category="exploratory",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["c8b3e6e172a7", "29e8a4397a48", "0004c00e9958", "2dcba435c0eb", "55a4e64633de"],
        relevant_chunk_ids=[],
        notes="Should pick foundational papers with clear reasoning. Multiple valid answers.",
    ),
    BenchmarkQuery(
        query_id="ve_011",
        query="What are the practical implications of these papers for deploying LLMs?",
        category="exploratory",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["55a4e64633de", "eef16825bcb4", "7d45caedf31f", "0d32b6118fcc"],
        relevant_chunk_ids=[],
        notes="Should discuss LoRA for adaptation, Mistral for efficient deployment, RLHF for alignment.",
    ),
    BenchmarkQuery(
        query_id="ve_012",
        query="Walk me through how you'd build a RAG system using ideas from these papers.",
        category="exploratory",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["a63eb71679c1", "90f334a9266b", "2dcba435c0eb", "cae01624e02c", "bef7f1f5382e", "f1470ed318c7"],
        relevant_chunk_ids=[],
        notes="Should describe retrieval (DPR/ColBERT), generation (RAG), quality (Self-RAG), positioning (Lost in Middle).",
    ),
    BenchmarkQuery(
        query_id="ve_013",
        query="What makes a good language model based on what these papers show?",
        category="exploratory",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["cb02ca4b7682", "eef16825bcb4", "0004c00e9958", "7d45caedf31f"],
        relevant_chunk_ids=[],
        notes="Should discuss scale, data quality, architecture choices, alignment.",
    ),
    BenchmarkQuery(
        query_id="ve_014",
        query="I need to present a 5-minute talk on these papers. What are the three biggest takeaways?",
        category="exploratory",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["c8b3e6e172a7", "0004c00e9958", "2dcba435c0eb"],
        relevant_chunk_ids=[],
        notes="Should synthesize across the corpus. Multiple valid answers.",
    ),
    BenchmarkQuery(
        query_id="ve_015",
        query="What are the ethical concerns raised across these papers?",
        category="exploratory",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["0004c00e9958", "7d45caedf31f", "061b473a798e", "bc2b47623304"],
        relevant_chunk_ids=[],
        notes="Should mention bias, misuse, hallucination, alignment. GPT-3 and InstructGPT discuss these.",
    ),

    # =========================================================================
    # ADVERSARIAL / BOUNDARY (10 queries)
    # =========================================================================

    BenchmarkQuery(
        query_id="adv_001",
        query="How does Flash Attention compare to the attention mechanism in the Transformer paper?",
        category="adversarial",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["c8b3e6e172a7"],
        relevant_chunk_ids=[],
        notes="Flash Attention is NOT in the corpus. Should describe Transformer attention and note Flash Attention is not covered.",
    ),
    BenchmarkQuery(
        query_id="adv_002",
        query="What optimizer does LLaMA use and how does it compare to the Adam optimizer used in the original Transformer?",
        category="adversarial",
        ground_truth_answer="",
        ground_truth_claims=[
            GroundTruthClaim("LLaMA uses AdamW optimizer", "cb02ca4b7682", "cb02ca4b7682_fine_25", "factual"),
            GroundTruthClaim("Original Transformer uses Adam with custom learning rate schedule", "c8b3e6e172a7", "c8b3e6e172a7_fine_29", "factual"),
        ],
        relevant_paper_ids=["cb02ca4b7682", "c8b3e6e172a7"],
        relevant_chunk_ids=[],
        notes="Both papers have optimizer details but they're buried in implementation sections. Tests deep retrieval.",
    ),
    BenchmarkQuery(
        query_id="adv_003",
        query="Does BERT use dropout and at what rate?",
        category="adversarial",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["29e8a4397a48"],
        relevant_chunk_ids=[],
        notes="This detail may or may not be extractable from the BERT paper chunks. Tests specificity of retrieval.",
    ),
    BenchmarkQuery(
        query_id="adv_004",
        query="How would you combine LoRA with Self-RAG for an efficient retrieval-augmented system?",
        category="adversarial",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["55a4e64633de", "cae01624e02c"],
        relevant_chunk_ids=[],
        notes="This combination is not discussed in any paper — requires reasoning beyond what's stated.",
    ),
    BenchmarkQuery(
        query_id="adv_005",
        query="What is the perplexity of Mistral 7B on WikiText-103?",
        category="adversarial",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["eef16825bcb4"],
        relevant_chunk_ids=[],
        notes="Mistral paper may not report WikiText-103 perplexity specifically. Tests partial information handling.",
    ),
    BenchmarkQuery(
        query_id="adv_006",
        query="Is Constitutional AI better than RLHF?",
        category="adversarial",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["061b473a798e", "7d45caedf31f"],
        relevant_chunk_ids=[],
        notes="Leading question — should present nuanced comparison, not a simple yes/no.",
    ),
    BenchmarkQuery(
        query_id="adv_007",
        query="Can GPT-3 do chain-of-thought reasoning?",
        category="adversarial",
        ground_truth_answer="",
        ground_truth_claims=[
            GroundTruthClaim("Chain-of-thought works with sufficiently large models", "4ddeacf6ee26", "4ddeacf6ee26_abstract", "factual"),
            GroundTruthClaim("GPT-3 175B is among the models tested", "4ddeacf6ee26", "4ddeacf6ee26_abstract", "factual"),
        ],
        relevant_paper_ids=["4ddeacf6ee26", "0004c00e9958"],
        relevant_chunk_ids=[],
        notes="Requires connecting the CoT paper's findings about model scale with GPT-3's size.",
    ),
    BenchmarkQuery(
        query_id="adv_008",
        query="What would happen if you applied RAPTOR's tree structure to ColBERT's index?",
        category="adversarial",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["da5b38b62bfd", "90f334a9266b"],
        relevant_chunk_ids=[],
        notes="Hypothetical combination. Should describe both approaches and reason about potential combination.",
    ),
    BenchmarkQuery(
        query_id="adv_009",
        query="Does the RAG paper use DPR for its retriever?",
        category="adversarial",
        ground_truth_answer="Yes, the original RAG paper uses DPR (Dense Passage Retrieval) as its retrieval component.",
        ground_truth_claims=[
            GroundTruthClaim("RAG uses DPR for retrieval", "2dcba435c0eb", "2dcba435c0eb_abstract", "factual"),
        ],
        relevant_paper_ids=["2dcba435c0eb", "a63eb71679c1"],
        relevant_chunk_ids=[],
        notes="Tests cross-paper connection. RAG explicitly builds on DPR.",
    ),
    BenchmarkQuery(
        query_id="adv_010",
        query="Which paper has the highest reported benchmark score and what was it?",
        category="adversarial",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=[],
        relevant_chunk_ids=[],
        notes="Ill-defined question — benchmark scores aren't comparable across different tasks. Should note this.",
    ),

    # =========================================================================
    # REFUSAL (10 queries)
    # =========================================================================

    BenchmarkQuery(
        query_id="ref_001",
        query="What is the training cost in dollars for GPT-4?",
        category="refusal",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=[],
        relevant_chunk_ids=[],
        unanswerable=True,
        notes="GPT-4 is not in the corpus.",
    ),
    BenchmarkQuery(
        query_id="ref_002",
        query="What is AlphaFold's protein structure prediction accuracy?",
        category="refusal",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=[],
        relevant_chunk_ids=[],
        unanswerable=True,
        notes="Completely different domain — not in corpus.",
    ),
    BenchmarkQuery(
        query_id="ref_003",
        query="How many GPUs did Llama 2 use for training?",
        category="refusal",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=[],
        relevant_chunk_ids=[],
        unanswerable=True,
        notes="Llama 2 is NOT in the corpus (only LLaMA 1). Near-miss — tests precision.",
    ),
    BenchmarkQuery(
        query_id="ref_004",
        query="What is the reward model architecture used by Claude?",
        category="refusal",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=[],
        relevant_chunk_ids=[],
        unanswerable=True,
        notes="Claude specifics not in corpus. Constitutional AI paper exists but doesn't describe Claude.",
    ),
    BenchmarkQuery(
        query_id="ref_005",
        query="What ROUGE scores does PEGASUS achieve on summarization?",
        category="refusal",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=[],
        relevant_chunk_ids=[],
        unanswerable=True,
        notes="PEGASUS is not in the corpus.",
    ),
    BenchmarkQuery(
        query_id="ref_006",
        query="How does Gemini's multimodal architecture work?",
        category="refusal",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=[],
        relevant_chunk_ids=[],
        unanswerable=True,
        notes="Gemini not in corpus.",
    ),
    BenchmarkQuery(
        query_id="ref_007",
        query="What is the exact learning rate schedule for Mixtral 8x7B?",
        category="refusal",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=[],
        relevant_chunk_ids=[],
        unanswerable=True,
        notes="Mixtral not in corpus. Mistral 7B is — tests near-miss precision.",
    ),
    BenchmarkQuery(
        query_id="ref_008",
        query="How does RETRO compare to RAG for retrieval-augmented language modeling?",
        category="refusal",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=[],
        relevant_chunk_ids=[],
        unanswerable=True,
        notes="RETRO not in corpus. RAG is. Should describe RAG but note RETRO is not available.",
    ),
    BenchmarkQuery(
        query_id="ref_009",
        query="What is the FLOPs count for training BERT-Large?",
        category="refusal",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=[],
        relevant_chunk_ids=[],
        unanswerable=True,
        notes="BERT paper likely doesn't report FLOPs. Should refuse or note this isn't in the paper.",
    ),
    BenchmarkQuery(
        query_id="ref_010",
        query="Compare Mamba's state-space architecture with the Transformer.",
        category="refusal",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=[],
        relevant_chunk_ids=[],
        unanswerable=True,
        notes="Mamba not in corpus. Transformer is. Should note limitation.",
    ),

    # =========================================================================
    # ROBUSTNESS (10 queries)
    # =========================================================================

    BenchmarkQuery(
        query_id="rob_001",
        query="How does the bidirectional encoder representations from transformers model handle pretraining?",
        category="robustness",
        ground_truth_answer="BERT uses masked language modeling where 15% of tokens are masked and predicted using bidirectional context.",
        ground_truth_claims=[
            GroundTruthClaim("BERT uses masked language modeling", "29e8a4397a48", "29e8a4397a48_abstract", "factual"),
        ],
        relevant_paper_ids=["29e8a4397a48"],
        relevant_chunk_ids=[],
        notes="Full name instead of acronym — tests expansion/matching.",
    ),
    BenchmarkQuery(
        query_id="rob_002",
        query="What is the low-rank adaption method for large langauge models?",
        category="robustness",
        ground_truth_answer="LoRA freezes pre-trained weights and injects trainable low-rank decomposition matrices.",
        ground_truth_claims=[
            GroundTruthClaim("LoRA uses low-rank matrices while freezing weights", "55a4e64633de", "55a4e64633de_abstract", "factual"),
        ],
        relevant_paper_ids=["55a4e64633de"],
        relevant_chunk_ids=[],
        notes="Two typos: 'adaption' and 'langauge' — tests robustness.",
    ),
    BenchmarkQuery(
        query_id="rob_003",
        query="Tell me about the paper where they make language model smaller but still good?",
        category="robustness",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=["55a4e64633de", "eef16825bcb4", "cb02ca4b7682"],
        relevant_chunk_ids=[],
        notes="Vague, informal English. Could be LoRA, Mistral, or LLaMA. Tests interpretation.",
    ),
    BenchmarkQuery(
        query_id="rob_004",
        query="which paper talk about the tree search for thinking in LLM?",
        category="robustness",
        ground_truth_answer="Tree of Thoughts uses BFS/DFS to explore multiple reasoning paths.",
        ground_truth_claims=[
            GroundTruthClaim("Tree of Thoughts uses search for reasoning", "802732405ac0", "802732405ac0_abstract", "factual"),
        ],
        relevant_paper_ids=["802732405ac0"],
        relevant_chunk_ids=[],
        notes="Non-native English phrasing. Tests query understanding.",
    ),
    BenchmarkQuery(
        query_id="rob_005",
        query="what is DPR and how does it works?",
        category="robustness",
        ground_truth_answer="DPR (Dense Passage Retrieval) uses two BERT encoders to encode passages and queries into dense vectors. Relevance is computed via dot product.",
        ground_truth_claims=[
            GroundTruthClaim("DPR uses dual BERT encoders", "a63eb71679c1", "a63eb71679c1_abstract", "factual"),
        ],
        relevant_paper_ids=["a63eb71679c1"],
        relevant_chunk_ids=[],
        notes="Grammar error ('works' instead of 'work'). Lowercase. Tests robustness.",
    ),
    BenchmarkQuery(
        query_id="rob_006",
        query="Papers that discuss making AI safer and more aligned?",
        category="robustness",
        ground_truth_answer="InstructGPT uses RLHF for alignment. Constitutional AI uses AI-written principles. Both address making LLMs safer and more helpful.",
        ground_truth_claims=[
            GroundTruthClaim("InstructGPT uses RLHF", "7d45caedf31f", "7d45caedf31f_abstract", "factual"),
            GroundTruthClaim("Constitutional AI uses principles", "061b473a798e", "061b473a798e_abstract", "factual"),
        ],
        relevant_paper_ids=["7d45caedf31f", "061b473a798e"],
        relevant_chunk_ids=[],
        notes="Fragment query, not a complete question. Tests interpretation.",
    ),
    BenchmarkQuery(
        query_id="rob_007",
        query="Explain the hypothetical document embedding approach to information retrieval",
        category="robustness",
        ground_truth_answer="HyDE generates a hypothetical answer document using an LLM, then encodes it to retrieve real relevant documents — bridging the query-document gap without training data.",
        ground_truth_claims=[
            GroundTruthClaim("HyDE generates hypothetical documents for retrieval", "bef7f1f5382e", "bef7f1f5382e_abstract", "factual"),
        ],
        relevant_paper_ids=["bef7f1f5382e"],
        relevant_chunk_ids=[],
        notes="Uses formal description instead of acronym HyDE. Tests matching.",
    ),
    BenchmarkQuery(
        query_id="rob_008",
        query="transformr attention paper how many head",
        category="robustness",
        ground_truth_answer="The base Transformer uses 8 attention heads.",
        ground_truth_claims=[
            GroundTruthClaim("Base Transformer has 8 attention heads", "c8b3e6e172a7", "c8b3e6e172a7_abstract", "numerical"),
        ],
        relevant_paper_ids=["c8b3e6e172a7"],
        relevant_chunk_ids=[],
        notes="Very broken query: typo, missing words, fragments. Tests extreme robustness.",
    ),
    BenchmarkQuery(
        query_id="rob_009",
        query="What's the RLHF thing and why should I care?",
        category="robustness",
        ground_truth_answer="RLHF (Reinforcement Learning from Human Feedback) is a method for aligning language models with human preferences, used in InstructGPT. It makes models more helpful, harmless, and honest.",
        ground_truth_claims=[
            GroundTruthClaim("RLHF aligns models with human preferences", "7d45caedf31f", "7d45caedf31f_abstract", "factual"),
        ],
        relevant_paper_ids=["7d45caedf31f"],
        relevant_chunk_ids=[],
        notes="Informal/colloquial phrasing. Tests understanding of casual queries.",
    ),
    BenchmarkQuery(
        query_id="rob_010",
        query="passage retrieval using dense vectors without sparse methods like tfidf or bm25",
        category="robustness",
        ground_truth_answer="DPR shows dense retrieval alone can outperform BM25 on multiple QA datasets, using learned BERT encoders instead of sparse bag-of-words methods.",
        ground_truth_claims=[
            GroundTruthClaim("DPR outperforms BM25 with dense vectors alone", "a63eb71679c1", "a63eb71679c1_abstract", "factual"),
        ],
        relevant_paper_ids=["a63eb71679c1"],
        relevant_chunk_ids=[],
        notes="Descriptive phrase, not a question. No question mark. Tests intent understanding.",
    ),
]
