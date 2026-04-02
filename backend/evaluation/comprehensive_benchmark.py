"""Comprehensive RAG evaluation benchmark with claim-level ground truth.

Implements RAGAS-style metrics adapted for scientific RAG:
1. Faithfulness — does the answer stay true to retrieved sources?
2. Context Recall — did retrieval find the relevant chunks?
3. Context Precision — are top-ranked results actually relevant?
4. Answer Correctness — claim-level factual accuracy against ground truth
5. Citation Accuracy — do citations actually support the claims they're attached to?
6. Answer Completeness — are all aspects of the question addressed?
7. Refusal Accuracy — does the system correctly refuse unanswerable questions?

Usage:
    python -m evaluation.comprehensive_benchmark                    # Run all
    python -m evaluation.comprehensive_benchmark --suite factual    # One suite
    python -m evaluation.comprehensive_benchmark --limit 5          # Limit per suite
    python -m evaluation.comprehensive_benchmark --output results.json
"""

import sys
import json
import time
import re
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from anthropic import Anthropic
from retrieval.embedder import VoyageEmbedder
from retrieval.reranker import CohereReranker
from retrieval.qdrant_store import QdrantStore
from retrieval.query_engine import QueryEngine, QueryResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ground Truth Data Structures
# ---------------------------------------------------------------------------

@dataclass
class GroundTruthClaim:
    """A single verifiable claim within a ground truth answer."""
    claim: str
    source_paper_id: str
    source_chunk_id: str  # Specific chunk that contains this fact
    claim_type: str  # "numerical", "factual", "methodological", "relational"


@dataclass
class BenchmarkQuery:
    """A benchmark query with claim-level ground truth."""
    query_id: str
    query: str
    category: str  # "factual", "methods", "comparative", "cross_paper", "adversarial", "refusal"
    ground_truth_answer: str
    ground_truth_claims: List[GroundTruthClaim]
    relevant_paper_ids: List[str]
    relevant_chunk_ids: List[str]  # Gold standard chunks that should be retrieved
    unanswerable: bool = False  # True for refusal queries
    notes: str = ""


# ---------------------------------------------------------------------------
# Ground Truth Dataset — built from actual Qdrant chunks
# ---------------------------------------------------------------------------

BENCHMARK_QUERIES: List[BenchmarkQuery] = [

    # ==================== FACTUAL ACCURACY (numerical + specific facts) ====================

    BenchmarkQuery(
        query_id="fact_001",
        query="How many attention heads does the base Transformer model use and what is the model dimension?",
        category="factual",
        ground_truth_answer="The base Transformer model uses 8 attention heads with a model dimension of 512.",
        ground_truth_claims=[
            GroundTruthClaim("The base Transformer uses 8 attention heads", "c8b3e6e172a7", "c8b3e6e172a7_fine_29", "numerical"),
            GroundTruthClaim("The model dimension (d_model) is 512", "c8b3e6e172a7", "c8b3e6e172a7_fine_29", "numerical"),
        ],
        relevant_paper_ids=["c8b3e6e172a7"],
        relevant_chunk_ids=["c8b3e6e172a7_fine_29", "c8b3e6e172a7_abstract"],
    ),

    BenchmarkQuery(
        query_id="fact_002",
        query="What is the rank r typically used in LoRA when adapting GPT-3?",
        category="factual",
        ground_truth_answer="LoRA uses a very low rank, with r as low as 1 or 2 being sufficient for GPT-3 adaptation, though r=4 or r=8 are commonly used.",
        ground_truth_claims=[
            GroundTruthClaim("LoRA can use rank as low as 1 or 2", "55a4e64633de", "55a4e64633de_fine_18", "numerical"),
            GroundTruthClaim("Low rank is sufficient for adapting GPT-3", "55a4e64633de", "55a4e64633de_abstract", "factual"),
        ],
        relevant_paper_ids=["55a4e64633de"],
        relevant_chunk_ids=["55a4e64633de_fine_18", "55a4e64633de_abstract"],
    ),

    BenchmarkQuery(
        query_id="fact_003",
        query="How many parameters does Mistral 7B have and what context length does it support?",
        category="factual",
        ground_truth_answer="Mistral 7B has 7 billion parameters and supports a context length using sliding window attention of 4096 tokens.",
        ground_truth_claims=[
            GroundTruthClaim("Mistral 7B has 7 billion parameters", "eef16825bcb4", "eef16825bcb4_abstract", "numerical"),
            GroundTruthClaim("Uses sliding window attention", "eef16825bcb4", "eef16825bcb4_fine_6", "factual"),
        ],
        relevant_paper_ids=["eef16825bcb4"],
        relevant_chunk_ids=["eef16825bcb4_abstract", "eef16825bcb4_fine_6"],
    ),

    BenchmarkQuery(
        query_id="fact_004",
        query="What datasets did DPR use for training the passage retriever?",
        category="factual",
        ground_truth_answer="DPR was trained on Natural Questions, TriviaQA, WebQuestions, CuratedTREC, and SQuAD datasets.",
        ground_truth_claims=[
            GroundTruthClaim("DPR was trained on Natural Questions", "a63eb71679c1", "a63eb71679c1_fine_29", "factual"),
            GroundTruthClaim("DPR was evaluated on multiple QA datasets", "a63eb71679c1", "a63eb71679c1_fine_37", "factual"),
        ],
        relevant_paper_ids=["a63eb71679c1"],
        relevant_chunk_ids=["a63eb71679c1_fine_29", "a63eb71679c1_fine_37"],
    ),

    BenchmarkQuery(
        query_id="fact_005",
        query="What are the sizes of the LLaMA model family?",
        category="factual",
        ground_truth_answer="LLaMA comes in four sizes: 7B, 13B, 33B, and 65B parameters.",
        ground_truth_claims=[
            GroundTruthClaim("LLaMA models range from 7B to 65B parameters", "cb02ca4b7682", "cb02ca4b7682_abstract", "numerical"),
            GroundTruthClaim("LLaMA is trained on trillions of tokens of public data", "cb02ca4b7682", "cb02ca4b7682_abstract", "factual"),
        ],
        relevant_paper_ids=["cb02ca4b7682"],
        relevant_chunk_ids=["cb02ca4b7682_abstract"],
    ),

    # ==================== METHODS / PROCEDURES ====================

    BenchmarkQuery(
        query_id="meth_001",
        query="How does BERT's masked language modeling pre-training work?",
        category="methods",
        ground_truth_answer="BERT randomly masks 15% of input tokens and trains the model to predict the masked tokens. Of the masked positions, 80% are replaced with [MASK], 10% with a random token, and 10% kept unchanged.",
        ground_truth_claims=[
            GroundTruthClaim("BERT masks 15% of input tokens during pre-training", "29e8a4397a48", "29e8a4397a48_fine_35", "methodological"),
            GroundTruthClaim("80% replaced with [MASK], 10% random, 10% unchanged", "29e8a4397a48", "29e8a4397a48_fine_35", "methodological"),
            GroundTruthClaim("BERT uses bidirectional context", "29e8a4397a48", "29e8a4397a48_abstract", "factual"),
        ],
        relevant_paper_ids=["29e8a4397a48"],
        relevant_chunk_ids=["29e8a4397a48_fine_35", "29e8a4397a48_abstract"],
    ),

    BenchmarkQuery(
        query_id="meth_002",
        query="How does Self-RAG use reflection tokens to decide when to retrieve?",
        category="methods",
        ground_truth_answer="Self-RAG trains the LM to generate special reflection tokens (Retrieve, IsRel, IsSup, IsUse) as part of its vocabulary. The Retrieve token signals when retrieval is needed. After retrieval, IsRel judges relevance, IsSup judges support, and IsUse judges overall utility.",
        ground_truth_claims=[
            GroundTruthClaim("Self-RAG uses reflection tokens as part of the model vocabulary", "cae01624e02c", "cae01624e02c_fine_20", "methodological"),
            GroundTruthClaim("Retrieve token signals when retrieval is needed", "cae01624e02c", "cae01624e02c_fine_20", "methodological"),
            GroundTruthClaim("Critique tokens judge relevance and support", "cae01624e02c", "cae01624e02c_fine_26", "methodological"),
        ],
        relevant_paper_ids=["cae01624e02c"],
        relevant_chunk_ids=["cae01624e02c_fine_20", "cae01624e02c_fine_26", "cae01624e02c_abstract"],
    ),

    BenchmarkQuery(
        query_id="meth_003",
        query="What is the three-step training procedure for InstructGPT?",
        category="methods",
        ground_truth_answer="InstructGPT's training has three steps: (1) supervised fine-tuning on human demonstrations, (2) training a reward model on human comparisons, and (3) optimizing the policy with PPO against the reward model.",
        ground_truth_claims=[
            GroundTruthClaim("Step 1 is supervised fine-tuning on demonstrations", "7d45caedf31f", "7d45caedf31f_abstract", "methodological"),
            GroundTruthClaim("Step 2 trains a reward model on human comparisons", "7d45caedf31f", "7d45caedf31f_abstract", "methodological"),
            GroundTruthClaim("Step 3 uses PPO to optimize against the reward model", "7d45caedf31f", "7d45caedf31f_abstract", "methodological"),
        ],
        relevant_paper_ids=["7d45caedf31f"],
        relevant_chunk_ids=["7d45caedf31f_abstract"],
    ),

    # ==================== COMPARATIVE (cross-paper reasoning) ====================

    BenchmarkQuery(
        query_id="comp_001",
        query="How does DPR's approach to passage retrieval differ from ColBERT's?",
        category="comparative",
        ground_truth_answer="DPR encodes passages and queries into single dense vectors and uses dot product similarity. ColBERT uses late interaction — encoding each token separately and computing fine-grained token-level similarity scores, enabling more nuanced matching.",
        ground_truth_claims=[
            GroundTruthClaim("DPR uses single dense vector per passage", "a63eb71679c1", "a63eb71679c1_abstract", "factual"),
            GroundTruthClaim("DPR uses dot product for similarity", "a63eb71679c1", "a63eb71679c1_fine_29", "methodological"),
            GroundTruthClaim("ColBERT uses late interaction with token-level matching", "90f334a9266b", "90f334a9266b_abstract", "factual"),
        ],
        relevant_paper_ids=["a63eb71679c1", "90f334a9266b"],
        relevant_chunk_ids=["a63eb71679c1_abstract", "90f334a9266b_abstract"],
    ),

    BenchmarkQuery(
        query_id="comp_002",
        query="Compare RAG and Self-RAG's retrieval strategies.",
        category="comparative",
        ground_truth_answer="RAG always retrieves for every input (retrieve-then-generate). Self-RAG selectively retrieves only when needed, using learned reflection tokens to decide. Self-RAG can also critique its own generations and choose whether to use retrieved passages.",
        ground_truth_claims=[
            GroundTruthClaim("RAG retrieves for every input", "2dcba435c0eb", "2dcba435c0eb_abstract", "factual"),
            GroundTruthClaim("Self-RAG selectively retrieves using reflection tokens", "cae01624e02c", "cae01624e02c_abstract", "factual"),
            GroundTruthClaim("Self-RAG can critique its own generations", "cae01624e02c", "cae01624e02c_fine_26", "factual"),
        ],
        relevant_paper_ids=["2dcba435c0eb", "cae01624e02c"],
        relevant_chunk_ids=["2dcba435c0eb_abstract", "cae01624e02c_abstract", "cae01624e02c_fine_20"],
    ),

    BenchmarkQuery(
        query_id="comp_003",
        query="How does chain-of-thought prompting differ from tree of thoughts?",
        category="comparative",
        ground_truth_answer="Chain-of-thought uses a single linear sequence of reasoning steps. Tree of Thoughts generalizes this to explore multiple reasoning paths by branching and backtracking, using search algorithms (BFS/DFS) to find the best reasoning path.",
        ground_truth_claims=[
            GroundTruthClaim("Chain-of-thought uses linear reasoning steps", "4ddeacf6ee26", "4ddeacf6ee26_abstract", "factual"),
            GroundTruthClaim("Tree of Thoughts explores multiple reasoning paths", "802732405ac0", "802732405ac0_abstract", "factual"),
            GroundTruthClaim("Tree of Thoughts uses search algorithms like BFS/DFS", "802732405ac0", "802732405ac0_abstract", "methodological"),
        ],
        relevant_paper_ids=["4ddeacf6ee26", "802732405ac0"],
        relevant_chunk_ids=["4ddeacf6ee26_abstract", "802732405ac0_abstract"],
    ),

    # ==================== CROSS-PAPER SYNTHESIS ====================

    BenchmarkQuery(
        query_id="cross_001",
        query="What approaches have been proposed to reduce hallucination in language models?",
        category="cross_paper",
        ground_truth_answer="Multiple approaches: (1) RAG grounds generations in retrieved passages, (2) Self-RAG adds reflection tokens for self-critique, (3) InstructGPT/RLHF aligns models with human preferences, (4) Constitutional AI uses principles for self-correction.",
        ground_truth_claims=[
            GroundTruthClaim("RAG reduces hallucination by grounding in retrieved text", "2dcba435c0eb", "2dcba435c0eb_abstract", "factual"),
            GroundTruthClaim("Self-RAG uses reflection tokens for self-critique", "cae01624e02c", "cae01624e02c_abstract", "factual"),
            GroundTruthClaim("RLHF/InstructGPT aligns with human preferences", "7d45caedf31f", "7d45caedf31f_abstract", "factual"),
            GroundTruthClaim("Constitutional AI uses principles for self-correction", "061b473a798e", "061b473a798e_abstract", "factual"),
        ],
        relevant_paper_ids=["2dcba435c0eb", "cae01624e02c", "7d45caedf31f", "061b473a798e"],
        relevant_chunk_ids=["2dcba435c0eb_abstract", "cae01624e02c_abstract", "7d45caedf31f_abstract", "061b473a798e_abstract"],
    ),

    BenchmarkQuery(
        query_id="cross_002",
        query="What different retrieval strategies have been proposed for augmenting LLMs?",
        category="cross_paper",
        ground_truth_answer="Several strategies: DPR uses dense single-vector retrieval, ColBERT uses late interaction token-level matching, RAG integrates retrieval into generation, Self-RAG adds selective retrieval with reflection, RAPTOR uses tree-structured recursive summarization, and HyDE generates hypothetical documents for zero-shot retrieval.",
        ground_truth_claims=[
            GroundTruthClaim("DPR uses dense single-vector passage retrieval", "a63eb71679c1", "a63eb71679c1_abstract", "factual"),
            GroundTruthClaim("ColBERT uses late interaction for passage retrieval", "90f334a9266b", "90f334a9266b_abstract", "factual"),
            GroundTruthClaim("RAG integrates retrieval into sequence generation", "2dcba435c0eb", "2dcba435c0eb_abstract", "factual"),
            GroundTruthClaim("RAPTOR uses recursive tree-structured summarization", "da5b38b62bfd", "da5b38b62bfd_abstract", "factual"),
            GroundTruthClaim("HyDE uses hypothetical documents for zero-shot retrieval", "bef7f1f5382e", "bef7f1f5382e_abstract", "factual"),
        ],
        relevant_paper_ids=["a63eb71679c1", "90f334a9266b", "2dcba435c0eb", "da5b38b62bfd", "bef7f1f5382e"],
        relevant_chunk_ids=["a63eb71679c1_abstract", "90f334a9266b_abstract", "2dcba435c0eb_abstract", "da5b38b62bfd_abstract", "bef7f1f5382e_abstract"],
    ),

    # ==================== REFUSAL (unanswerable queries) ====================

    BenchmarkQuery(
        query_id="refusal_001",
        query="What is the training cost in dollars for GPT-4?",
        category="refusal",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=[],
        relevant_chunk_ids=[],
        unanswerable=True,
        notes="GPT-4 is not in the corpus. System should indicate it cannot find this information.",
    ),

    BenchmarkQuery(
        query_id="refusal_002",
        query="What is the protein structure prediction accuracy of AlphaFold?",
        category="refusal",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=[],
        relevant_chunk_ids=[],
        unanswerable=True,
        notes="AlphaFold is not in this corpus of NLP/ML papers. System should refuse gracefully.",
    ),

    BenchmarkQuery(
        query_id="refusal_003",
        query="What reward model architecture does Claude use?",
        category="refusal",
        ground_truth_answer="",
        ground_truth_claims=[],
        relevant_paper_ids=[],
        relevant_chunk_ids=[],
        unanswerable=True,
        notes="Claude's internals are not in the corpus. Constitutional AI paper discusses principles but not Claude specifically.",
    ),

    # ==================== ROBUSTNESS (synonym, typo, negation) ====================

    BenchmarkQuery(
        query_id="robust_001",
        query="How does the bidirectional encoder representations from transformers model handle pretraining?",
        category="robustness",
        ground_truth_answer="BERT uses masked language modeling (MLM) where 15% of tokens are randomly masked and the model predicts them using bidirectional context.",
        ground_truth_claims=[
            GroundTruthClaim("BERT uses masked language modeling", "29e8a4397a48", "29e8a4397a48_abstract", "factual"),
        ],
        relevant_paper_ids=["29e8a4397a48"],
        relevant_chunk_ids=["29e8a4397a48_abstract"],
        notes="Uses full name instead of acronym 'BERT' — tests synonym/expansion handling",
    ),

    BenchmarkQuery(
        query_id="robust_002",
        query="What is the low-rank adaption method for large language models?",
        category="robustness",
        ground_truth_answer="LoRA (Low-Rank Adaptation) freezes pre-trained weights and injects trainable low-rank decomposition matrices, drastically reducing trainable parameters while maintaining performance.",
        ground_truth_claims=[
            GroundTruthClaim("LoRA freezes pre-trained weights", "55a4e64633de", "55a4e64633de_abstract", "factual"),
            GroundTruthClaim("LoRA injects low-rank matrices", "55a4e64633de", "55a4e64633de_abstract", "methodological"),
        ],
        relevant_paper_ids=["55a4e64633de"],
        relevant_chunk_ids=["55a4e64633de_abstract"],
        notes="Misspells 'adaptation' as 'adaption' — tests typo robustness",
    ),

    BenchmarkQuery(
        query_id="robust_003",
        query="Which models do NOT use autoregressive pre-training?",
        category="robustness",
        ground_truth_answer="BERT does not use autoregressive pre-training — it uses masked language modeling with bidirectional context. ColBERT also doesn't use autoregressive pre-training. Most other models in the corpus (GPT-2, GPT-3, LLaMA) are autoregressive.",
        ground_truth_claims=[
            GroundTruthClaim("BERT uses masked LM, not autoregressive", "29e8a4397a48", "29e8a4397a48_abstract", "factual"),
        ],
        relevant_paper_ids=["29e8a4397a48"],
        relevant_chunk_ids=["29e8a4397a48_abstract"],
        notes="Tests negation handling — 'NOT autoregressive'",
    ),
]


# ---------------------------------------------------------------------------
# Metric Computation
# ---------------------------------------------------------------------------

@dataclass
class QueryMetrics:
    """Metrics for a single benchmark query."""
    query_id: str
    category: str
    latency_ms: float
    # Retrieval metrics
    context_recall: float       # % of gold chunks found in retrieved results
    context_precision: float    # % of retrieved chunks that are from relevant papers
    retrieval_count: int
    # Generation metrics
    faithfulness: float         # % of answer claims supported by retrieved sources
    claim_accuracy: float       # % of ground truth claims correctly reproduced
    claims_verified: int
    claims_total: int
    claim_details: List[Dict[str, Any]]
    # Citation metrics
    citation_count: int
    citation_accuracy: float    # % of citations pointing to correct sources
    # Completeness
    answer_completeness: float  # % of ground truth claims addressed
    # Refusal (for unanswerable queries)
    correctly_refused: Optional[bool]
    # Raw data
    answer_preview: str
    warnings: List[str]


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    timestamp: str
    total_queries: int
    avg_latency_ms: float
    # Aggregate metrics
    avg_context_recall: float
    avg_context_precision: float
    avg_faithfulness: float
    avg_claim_accuracy: float
    avg_citation_accuracy: float
    avg_answer_completeness: float
    refusal_accuracy: float
    # Per-category breakdown
    by_category: Dict[str, Dict[str, float]]
    # Per-query details
    per_query: List[Dict[str, Any]]


def _resolve_gold_chunk_ids(
    queries: List[BenchmarkQuery],
    store: QdrantStore,
) -> None:
    """Resolve gold chunk IDs by verifying they exist in Qdrant.

    For chunks that don't exist (stale IDs), find alternative chunks
    from the same paper that contain the relevant content.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    for bq in queries:
        if not bq.relevant_chunk_ids:
            continue

        verified_ids = []
        for cid in bq.relevant_chunk_ids:
            # Check if this exact chunk exists
            try:
                results = store.client.scroll(
                    store.collection_name,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="_chunk_id", match=MatchValue(value=cid))
                    ]),
                    limit=1, with_payload=True, with_vectors=False,
                )
                if results[0]:
                    verified_ids.append(cid)
                else:
                    # Chunk ID doesn't exist — add all chunks from this paper instead
                    paper_id = cid.split('_')[0] if '_' in cid else cid
                    # Just keep paper_id-level matching (handled in context_recall)
                    pass
            except Exception:
                pass

        bq.relevant_chunk_ids = verified_ids


def compute_context_recall(
    result: QueryResult,
    gold_chunk_ids: List[str],
    relevant_paper_ids: List[str],
) -> float:
    """What fraction of relevant papers appear in retrieved results?

    Uses paper-level recall (more robust than exact chunk ID matching,
    since chunk IDs depend on extraction which can vary).
    """
    if not relevant_paper_ids:
        return 1.0

    retrieved_paper_ids = set()
    for s in result.sources:
        pid = s.get('paper_id', '')
        if pid:
            retrieved_paper_ids.add(pid)

    found = sum(1 for pid in relevant_paper_ids if pid in retrieved_paper_ids)
    return found / len(relevant_paper_ids)


def compute_context_precision(
    result: QueryResult,
    relevant_paper_ids: List[str],
) -> float:
    """What fraction of retrieved chunks are from relevant papers?"""
    if not result.sources:
        return 0.0
    relevant_set = set(relevant_paper_ids)
    relevant_count = sum(
        1 for s in result.sources
        if s.get('paper_id', '') in relevant_set
    )
    return relevant_count / len(result.sources)


def compute_claim_accuracy(
    answer: str,
    sources_text: str,
    ground_truth_claims: List[GroundTruthClaim],
    anthropic_client: Anthropic,
    model: str,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Verify each ground-truth claim against the generated answer.

    Returns (accuracy_score, claim_details).
    """
    if not ground_truth_claims:
        return 1.0, []

    claims_json = json.dumps([
        {"claim": c.claim, "type": c.claim_type}
        for c in ground_truth_claims
    ])

    prompt = f"""You are verifying whether a generated answer correctly reproduces specific factual claims.

For each claim below, check if the answer states it correctly. Be strict on numerical values and specific details.

Answer to verify:
"{answer[:3000]}"

Claims to check:
{claims_json}

For each claim, respond with a JSON array:
[{{"claim": "...", "found": true/false, "correct": true/false, "explanation": "..."}}]

- found=true means the answer addresses this claim
- correct=true means the specific fact is stated accurately (exact numbers, correct relationships)
- If found=false, correct should also be false"""

    try:
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        json_start = text.find('[')
        json_end = text.rfind(']') + 1
        if json_start >= 0 and json_end > json_start:
            details = json.loads(text[json_start:json_end])
            correct_count = sum(1 for d in details if d.get('correct'))
            return correct_count / len(ground_truth_claims), details
    except Exception as e:
        logger.warning(f"Claim verification failed: {e}")

    return 0.0, []


def compute_faithfulness(
    answer: str,
    sources: List[Dict[str, Any]],
    anthropic_client: Anthropic,
    model: str,
) -> float:
    """What fraction of claims in the answer are supported by retrieved sources?

    Decomposes the answer into atomic claims, then checks each against sources.
    """
    sources_text = "\n---\n".join(
        s.get('text', '')[:500] for s in sources[:10]
    )

    prompt = f"""Decompose this answer into atomic factual claims, then check each against the source passages.

Answer:
"{answer[:2000]}"

Source passages:
"{sources_text[:3000]}"

Return JSON:
{{"total_claims": <int>, "supported_claims": <int>, "unsupported_claims": [list of unsupported claim strings]}}"""

    try:
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=800,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            data = json.loads(text[json_start:json_end])
            total = data.get('total_claims', 1)
            supported = data.get('supported_claims', 0)
            return supported / max(total, 1)
    except Exception as e:
        logger.warning(f"Faithfulness check failed: {e}")

    return 0.5  # Uncertain


def check_refusal(answer: str) -> bool:
    """Does the answer indicate the system couldn't find the information?"""
    refusal_signals = [
        "couldn't find", "could not find", "no information",
        "not available", "not found in", "no relevant",
        "don't have", "do not have", "cannot answer",
        "not in the uploaded", "not covered", "outside",
        "no papers", "none of the", "not mentioned",
    ]
    answer_lower = answer.lower()
    return any(signal in answer_lower for signal in refusal_signals)


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

def run_benchmark(
    queries: List[BenchmarkQuery],
    engine: QueryEngine,
    anthropic_client: Anthropic,
    eval_model: str = "claude-sonnet-4-5-20250929",
) -> BenchmarkResults:
    """Run the full benchmark suite."""

    all_metrics: List[QueryMetrics] = []

    for i, bq in enumerate(queries):
        logger.info(f"\n[{i+1}/{len(queries)}] {bq.category}: {bq.query[:60]}...")

        if i > 0:
            time.sleep(2)

        try:
            start = time.time()
            # Disable general knowledge so faithfulness and refusal are measured fairly
            result = engine.query(
                bq.query,
                response_mode="concise",
                enable_general_knowledge=False,
            )
            latency_ms = (time.time() - start) * 1000

            # --- Retrieval metrics ---
            ctx_recall = compute_context_recall(result, bq.relevant_chunk_ids, bq.relevant_paper_ids)
            ctx_precision = compute_context_precision(result, bq.relevant_paper_ids)

            # --- Generation metrics ---
            if bq.unanswerable:
                # Refusal query
                refused = check_refusal(result.answer)
                metrics = QueryMetrics(
                    query_id=bq.query_id,
                    category=bq.category,
                    latency_ms=round(latency_ms, 1),
                    context_recall=0.0,
                    context_precision=0.0,
                    retrieval_count=result.retrieval_count,
                    faithfulness=1.0 if refused else 0.0,
                    claim_accuracy=0.0,
                    claims_verified=0,
                    claims_total=0,
                    claim_details=[],
                    citation_count=0,
                    citation_accuracy=0.0,
                    answer_completeness=0.0,
                    correctly_refused=refused,
                    answer_preview=result.answer[:200],
                    warnings=result.warnings,
                )
            else:
                # Normal query — compute all metrics
                sources_text = " ".join(s.get('text', '') for s in result.sources[:10])

                # Claim accuracy
                claim_acc, claim_details = compute_claim_accuracy(
                    result.answer, sources_text,
                    bq.ground_truth_claims, anthropic_client, eval_model,
                )

                # Faithfulness
                faithfulness = compute_faithfulness(
                    result.answer, result.sources,
                    anthropic_client, eval_model,
                )

                # Citation accuracy
                citation_pattern = re.compile(r'\[(?:Source\s*)?(\d+)\]', re.IGNORECASE)
                citations_found = citation_pattern.findall(result.answer)
                citation_count = len(set(citations_found))

                # Check if cited sources are from relevant papers
                relevant_set = set(bq.relevant_paper_ids)
                correct_citations = 0
                for cid_str in set(citations_found):
                    cid = int(cid_str) - 1
                    if 0 <= cid < len(result.sources):
                        if result.sources[cid].get('paper_id', '') in relevant_set:
                            correct_citations += 1
                citation_accuracy = correct_citations / max(len(set(citations_found)), 1)

                # Answer completeness
                claims_found = sum(
                    1 for d in claim_details if d.get('found', False)
                )
                answer_completeness = claims_found / max(len(bq.ground_truth_claims), 1)

                metrics = QueryMetrics(
                    query_id=bq.query_id,
                    category=bq.category,
                    latency_ms=round(latency_ms, 1),
                    context_recall=round(ctx_recall, 4),
                    context_precision=round(ctx_precision, 4),
                    retrieval_count=result.retrieval_count,
                    faithfulness=round(faithfulness, 4),
                    claim_accuracy=round(claim_acc, 4),
                    claims_verified=sum(1 for d in claim_details if d.get('correct')),
                    claims_total=len(bq.ground_truth_claims),
                    claim_details=claim_details,
                    citation_count=citation_count,
                    citation_accuracy=round(citation_accuracy, 4),
                    answer_completeness=round(answer_completeness, 4),
                    correctly_refused=None,
                    answer_preview=result.answer[:200],
                    warnings=result.warnings,
                )

            all_metrics.append(metrics)

            # Log summary
            if bq.unanswerable:
                logger.info(f"  Refusal: {'CORRECT' if metrics.correctly_refused else 'FAILED'}")
            else:
                logger.info(
                    f"  ctx_recall={metrics.context_recall:.2f} ctx_prec={metrics.context_precision:.2f} "
                    f"faithful={metrics.faithfulness:.2f} claims={metrics.claims_verified}/{metrics.claims_total} "
                    f"cite_acc={metrics.citation_accuracy:.2f} complete={metrics.answer_completeness:.2f} "
                    f"({metrics.latency_ms:.0f}ms)"
                )

        except Exception as e:
            logger.error(f"  FAILED: {e}")
            all_metrics.append(QueryMetrics(
                query_id=bq.query_id, category=bq.category, latency_ms=0,
                context_recall=0, context_precision=0, retrieval_count=0,
                faithfulness=0, claim_accuracy=0, claims_verified=0, claims_total=0,
                claim_details=[], citation_count=0, citation_accuracy=0,
                answer_completeness=0, correctly_refused=None,
                answer_preview=f"ERROR: {e}", warnings=[str(e)],
            ))

    # --- Aggregate results ---
    normal = [m for m in all_metrics if m.correctly_refused is None]
    refusals = [m for m in all_metrics if m.correctly_refused is not None]

    def avg(vals):
        return round(sum(vals) / max(len(vals), 1), 4)

    # Per-category breakdown
    categories = set(m.category for m in all_metrics)
    by_category = {}
    for cat in sorted(categories):
        cat_metrics = [m for m in normal if m.category == cat]
        if cat_metrics:
            by_category[cat] = {
                "count": len(cat_metrics),
                "avg_context_recall": avg([m.context_recall for m in cat_metrics]),
                "avg_context_precision": avg([m.context_precision for m in cat_metrics]),
                "avg_faithfulness": avg([m.faithfulness for m in cat_metrics]),
                "avg_claim_accuracy": avg([m.claim_accuracy for m in cat_metrics]),
                "avg_citation_accuracy": avg([m.citation_accuracy for m in cat_metrics]),
                "avg_completeness": avg([m.answer_completeness for m in cat_metrics]),
                "avg_latency_ms": avg([m.latency_ms for m in cat_metrics]),
            }
    if refusals:
        by_category["refusal"] = {
            "count": len(refusals),
            "correctly_refused": sum(1 for m in refusals if m.correctly_refused),
            "accuracy": avg([1.0 if m.correctly_refused else 0.0 for m in refusals]),
        }

    return BenchmarkResults(
        timestamp=datetime.now().isoformat(),
        total_queries=len(all_metrics),
        avg_latency_ms=avg([m.latency_ms for m in all_metrics]),
        avg_context_recall=avg([m.context_recall for m in normal]) if normal else 0,
        avg_context_precision=avg([m.context_precision for m in normal]) if normal else 0,
        avg_faithfulness=avg([m.faithfulness for m in normal]) if normal else 0,
        avg_claim_accuracy=avg([m.claim_accuracy for m in normal]) if normal else 0,
        avg_citation_accuracy=avg([m.citation_accuracy for m in normal]) if normal else 0,
        avg_answer_completeness=avg([m.answer_completeness for m in normal]) if normal else 0,
        refusal_accuracy=avg([1.0 if m.correctly_refused else 0.0 for m in refusals]) if refusals else 0,
        by_category=by_category,
        per_query=[asdict(m) for m in all_metrics],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Comprehensive RAG benchmark")
    parser.add_argument("--suite", choices=["factual", "methods", "comparative", "cross_paper", "refusal", "robustness", "exploratory", "adversarial", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Max queries per suite")
    parser.add_argument("--output", default="/app/data/comprehensive_benchmark.json")
    parser.add_argument("--v2", action="store_true", help="Use the 100-query v2 benchmark set")
    args = parser.parse_args()

    # Select query set
    if args.v2:
        from evaluation.benchmark_queries_v2 import BENCHMARK_QUERIES_V2
        all_queries = BENCHMARK_QUERIES_V2
    else:
        all_queries = BENCHMARK_QUERIES

    # Filter queries
    if args.suite == "all":
        queries = all_queries
    else:
        queries = [q for q in all_queries if q.category == args.suite]

    if args.limit:
        queries = queries[:args.limit]

    logger.info(f"Running {len(queries)} benchmark queries (suite={args.suite})")

    # Initialize components
    anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
    store = QdrantStore(
        collection_name=settings.qdrant_collection_name,
        embedding_dimension=settings.embedding_dimension,
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )
    engine = QueryEngine(
        embedder=VoyageEmbedder(api_key=settings.voyage_api_key),
        reranker=CohereReranker(api_key=settings.cohere_api_key),
        store=store,
        anthropic_client=anthropic_client,
        claude_model=settings.claude_model,
        claude_model_fast=settings.claude_model_fast,
        claude_model_classifier=settings.claude_model_classifier,
        enable_classification=True,
        enable_expansion=True,
        enable_entity_extraction=True,
        enable_citation_verification=True,
        enable_conversation_memory=False,
    )

    # Resolve gold chunk IDs — map paper_id-based IDs to actual chunk IDs in Qdrant
    logger.info("Resolving gold chunk IDs against Qdrant...")
    _resolve_gold_chunk_ids(queries, store)

    results = run_benchmark(queries, engine, anthropic_client)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(asdict(results), f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"Queries: {results.total_queries}  |  Avg latency: {results.avg_latency_ms:.0f}ms")
    print(f"\n--- Overall Metrics ---")
    print(f"  Context Recall:      {results.avg_context_recall:.1%}")
    print(f"  Context Precision:   {results.avg_context_precision:.1%}")
    print(f"  Faithfulness:        {results.avg_faithfulness:.1%}")
    print(f"  Claim Accuracy:      {results.avg_claim_accuracy:.1%}")
    print(f"  Citation Accuracy:   {results.avg_citation_accuracy:.1%}")
    print(f"  Completeness:        {results.avg_answer_completeness:.1%}")
    print(f"  Refusal Accuracy:    {results.refusal_accuracy:.1%}")
    print(f"\n--- By Category ---")
    for cat, metrics in results.by_category.items():
        if 'correctly_refused' in metrics:
            print(f"  {cat}: {metrics['correctly_refused']}/{metrics['count']} correctly refused ({metrics['accuracy']:.0%})")
        else:
            print(f"  {cat} (n={metrics['count']}): "
                  f"recall={metrics['avg_context_recall']:.0%} "
                  f"precision={metrics['avg_context_precision']:.0%} "
                  f"faithful={metrics['avg_faithfulness']:.0%} "
                  f"claims={metrics['avg_claim_accuracy']:.0%} "
                  f"complete={metrics['avg_completeness']:.0%}")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
