"""Benchmark suite for Paper Prism RAG pipeline.

Three benchmark types:
1. Ablation Study — disable features one at a time, measure degradation
2. Paper Prism vs. Direct Claude — compare RAG answers to raw LLM answers
3. Classification Accuracy — does the classifier detect query types correctly

Usage:
    python -m evaluation.benchmark                      # Run all benchmarks
    python -m evaluation.benchmark --type ablation      # Ablation only
    python -m evaluation.benchmark --type comparison    # Paper Prism vs Claude only
    python -m evaluation.benchmark --type classification # Classification only
    python -m evaluation.benchmark --limit 10           # Limit queries per benchmark
    python -m evaluation.benchmark --output results.json # Custom output path
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from anthropic import Anthropic
from retrieval.embedder import VoyageEmbedder
from retrieval.reranker import CohereReranker
from retrieval.qdrant_store import QdrantStore
from retrieval.query_engine import QueryEngine
from evaluation.test_queries import load_test_queries, TestQuery, QueryType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    """Result from one ablation configuration."""
    config_name: str
    description: str
    queries_run: int
    avg_latency_ms: float
    topic_coverage: float        # % of expected topics found in answer + sources
    entity_coverage: float       # % of expected entities found
    citation_rate: float         # % of answers containing [Source N] citations
    classification_accuracy: float  # % of queries classified correctly
    avg_retrieval_count: float
    avg_reranked_count: float
    per_query: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Result from Paper Prism vs. Direct Claude comparison."""
    query_id: str
    query: str
    query_type: str
    arc_answer: str
    claude_answer: str
    arc_sources_count: int
    arc_has_citations: bool
    arc_latency_ms: float
    claude_latency_ms: float
    # LLM judge scores (1-5)
    judge_arc_accuracy: int
    judge_arc_completeness: int
    judge_arc_grounding: int
    judge_claude_accuracy: int
    judge_claude_completeness: int
    judge_claude_grounding: int
    judge_explanation: str


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are an expert evaluator comparing two answers to a research question.

You are given the question, two candidate answers, and reference passages from the actual papers. Use the reference passages to verify factual claims in each answer.

**Question:** {query}

**Reference passages from the papers** (use these to check factual accuracy):
{reference_passages}

**Answer A:**
{answer_a}

**Answer B:**
{answer_b}

Rate EACH answer on these criteria (1-5 scale):

1. **Accuracy** — Are the factual claims correct when checked against the reference passages? Are specific values, names, and details accurate?
   1=Mostly wrong or fabricated, 2=Several factual errors, 3=Mostly correct, 4=Accurate, 5=Highly accurate with verifiable specific details

2. **Completeness** — Does it fully address the question? Does it cover the key aspects?
   1=Barely addresses it, 2=Partial, 3=Adequate, 4=Thorough, 5=Comprehensive

3. **Grounding** — Is the answer traceable to specific evidence rather than vague or unsupported claims?
   1=No evidence/vague, 2=Generic claims, 3=Some specifics, 4=Well-evidenced, 5=Every claim supported with specific evidence

Respond in this EXACT format (no other text):
A_ACCURACY: <1-5>
A_COMPLETENESS: <1-5>
A_GROUNDING: <1-5>
B_ACCURACY: <1-5>
B_COMPLETENESS: <1-5>
B_GROUNDING: <1-5>
EXPLANATION: <2-3 sentences comparing the two answers>"""


def judge_answers(
    anthropic_client: Anthropic,
    query: str,
    arc_answer: str,
    claude_answer: str,
    reference_passages: str = "",
    model: str = "claude-sonnet-4-5-20250929",
) -> Dict[str, Any]:
    """Use a separate LLM call to judge two answers.

    Fully blind: A/B assignment is randomized per query so the judge
    cannot learn which system is which across queries.
    Reference passages are provided so the judge can fact-check claims.
    """
    import random

    # Randomize which answer is A and which is B
    arc_is_a = random.choice([True, False])

    if arc_is_a:
        answer_a = arc_answer[:3000]
        answer_b = claude_answer[:3000]
    else:
        answer_a = claude_answer[:3000]
        answer_b = arc_answer[:3000]

    # Provide reference passages (truncated) or a note if none available
    if not reference_passages:
        reference_passages = "(No reference passages available — judge based on your knowledge)"

    prompt = JUDGE_PROMPT.format(
        query=query,
        reference_passages=reference_passages[:4000],
        answer_a=answer_a,
        answer_b=answer_b,
    )

    try:
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=300,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        raw_scores = _parse_judge_response(text)

        # Map back from randomized A/B to arc/claude
        if arc_is_a:
            return raw_scores  # A=arc, B=claude — already correct
        else:
            # Swap: A was claude, B was arc
            return {
                "arc_accuracy": raw_scores["claude_accuracy"],
                "arc_completeness": raw_scores["claude_completeness"],
                "arc_grounding": raw_scores["claude_grounding"],
                "claude_accuracy": raw_scores["arc_accuracy"],
                "claude_completeness": raw_scores["arc_completeness"],
                "claude_grounding": raw_scores["arc_grounding"],
                "explanation": raw_scores["explanation"],
            }
    except Exception as e:
        logger.warning(f"Judge failed: {e}")
        return {
            "arc_accuracy": 3, "arc_completeness": 3, "arc_grounding": 3,
            "claude_accuracy": 3, "claude_completeness": 3, "claude_grounding": 3,
            "explanation": f"Judge error: {e}",
        }


def _parse_judge_response(text: str) -> Dict[str, Any]:
    """Parse the judge's structured response."""
    result = {
        "arc_accuracy": 3, "arc_completeness": 3, "arc_grounding": 3,
        "claude_accuracy": 3, "claude_completeness": 3, "claude_grounding": 3,
        "explanation": "",
    }
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("A_ACCURACY:"):
            result["arc_accuracy"] = int(line.split(":")[1].strip())
        elif line.startswith("A_COMPLETENESS:"):
            result["arc_completeness"] = int(line.split(":")[1].strip())
        elif line.startswith("A_GROUNDING:"):
            result["arc_grounding"] = int(line.split(":")[1].strip())
        elif line.startswith("B_ACCURACY:"):
            result["claude_accuracy"] = int(line.split(":")[1].strip())
        elif line.startswith("B_COMPLETENESS:"):
            result["claude_completeness"] = int(line.split(":")[1].strip())
        elif line.startswith("B_GROUNDING:"):
            result["claude_grounding"] = int(line.split(":")[1].strip())
        elif line.startswith("EXPLANATION:"):
            result["explanation"] = line.split(":", 1)[1].strip()
    return result


# ---------------------------------------------------------------------------
# Benchmark 1: Ablation Study
# ---------------------------------------------------------------------------

def run_ablation_study(
    queries: List[TestQuery],
    anthropic_client: Anthropic,
    embedder: VoyageEmbedder,
    reranker: CohereReranker,
    store: QdrantStore,
) -> List[AblationResult]:
    """Run the same queries under different pipeline configurations."""

    configs = [
        {
            "name": "full_pipeline",
            "description": "All features enabled (baseline)",
            "kwargs": {
                "enable_classification": True,
                "enable_expansion": True,
                "enable_hyde": False,  # Keep HyDE off by default as in production
                "enable_entity_extraction": True,
                "enable_citation_verification": True,
                "enable_conversation_memory": True,
                "enable_hybrid_search": True,
            },
        },
        {
            "name": "no_classification",
            "description": "Query classification disabled — all queries use GENERAL strategy",
            "kwargs": {
                "enable_classification": False,
                "enable_expansion": True,
                "enable_hyde": False,
                "enable_entity_extraction": True,
                "enable_citation_verification": True,
                "enable_conversation_memory": True,
                "enable_hybrid_search": True,
            },
        },
        {
            "name": "no_expansion",
            "description": "Query expansion disabled — no domain synonym addition",
            "kwargs": {
                "enable_classification": True,
                "enable_expansion": False,
                "enable_hyde": False,
                "enable_entity_extraction": True,
                "enable_citation_verification": True,
                "enable_conversation_memory": True,
                "enable_hybrid_search": True,
            },
        },
        {
            "name": "no_hybrid_search",
            "description": "Dense-only search — BM25 sparse vectors disabled",
            "kwargs": {
                "enable_classification": True,
                "enable_expansion": True,
                "enable_hyde": False,
                "enable_entity_extraction": True,
                "enable_citation_verification": True,
                "enable_conversation_memory": True,
                "enable_hybrid_search": False,
            },
        },
        {
            "name": "no_entity_extraction",
            "description": "Entity extraction disabled — no entity boosting of results",
            "kwargs": {
                "enable_classification": True,
                "enable_expansion": True,
                "enable_hyde": False,
                "enable_entity_extraction": False,
                "enable_citation_verification": True,
                "enable_conversation_memory": True,
                "enable_hybrid_search": True,
            },
        },
        {
            "name": "minimal_pipeline",
            "description": "Minimal — embed, search, generate (no classification, expansion, entities, hybrid)",
            "kwargs": {
                "enable_classification": False,
                "enable_expansion": False,
                "enable_hyde": False,
                "enable_entity_extraction": False,
                "enable_citation_verification": False,
                "enable_conversation_memory": False,
                "enable_hybrid_search": False,
            },
        },
    ]

    results = []

    for config in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"ABLATION: {config['name']} — {config['description']}")
        logger.info(f"{'='*60}")

        # Create a fresh QueryEngine with this configuration
        engine = QueryEngine(
            embedder=embedder,
            reranker=reranker,
            store=store,
            anthropic_client=anthropic_client,
            claude_model=settings.claude_model,
            claude_model_fast=settings.claude_model_fast,
            claude_model_classifier=settings.claude_model_classifier,
            **config["kwargs"],
        )

        per_query_results = []
        latencies = []
        topic_hits = 0
        topic_total = 0
        entity_hits = 0
        entity_total = 0
        citation_count = 0
        classification_correct = 0

        for i, tq in enumerate(queries):
            logger.info(f"  [{i+1}/{len(queries)}] {tq.query_id}: {tq.query[:60]}...")

            # Rate limit: pause between queries to avoid API throttling
            if i > 0:
                time.sleep(2)

            try:
                start = time.time()
                result = engine.query(tq.query, response_mode="concise")
                latency_ms = (time.time() - start) * 1000

                # Timeout guard: skip if a single query took > 120s (likely hung)
                if latency_ms > 120000:
                    logger.warning(f"  TIMEOUT: {latency_ms:.0f}ms — skipping")
                    per_query_results.append({"query_id": tq.query_id, "error": "timeout"})
                    continue

                latencies.append(latency_ms)

                # Check topic coverage
                all_text = (result.answer + " " +
                            " ".join(s.get("text", "") for s in result.sources)).lower()
                found_topics = [t for t in tq.expected_topics if t.lower() in all_text]
                topic_hits += len(found_topics)
                topic_total += len(tq.expected_topics)

                # Check entity coverage
                found_entities = [e for e in tq.expected_entities if e.lower() in all_text]
                entity_hits += len(found_entities)
                entity_total += len(tq.expected_entities)

                # Check citations
                has_citations = "[Source" in result.answer or "[source" in result.answer
                if has_citations:
                    citation_count += 1

                # Check classification
                if result.query_type.value == tq.query_type.value:
                    classification_correct += 1

                per_query_results.append({
                    "query_id": tq.query_id,
                    "query_type_expected": tq.query_type.value,
                    "query_type_detected": result.query_type.value,
                    "classification_correct": result.query_type.value == tq.query_type.value,
                    "latency_ms": round(latency_ms, 1),
                    "retrieval_count": result.retrieval_count,
                    "reranked_count": result.reranked_count,
                    "topic_coverage": len(found_topics) / max(len(tq.expected_topics), 1),
                    "entity_coverage": len(found_entities) / max(len(tq.expected_entities), 1),
                    "has_citations": has_citations,
                    "answer_length": len(result.answer),
                })

                logger.info(f"    OK ({latency_ms:.0f}ms) type={result.query_type.value} topics={len(found_topics)}/{len(tq.expected_topics)}")

            except Exception as e:
                logger.error(f"  FAILED: {e}")
                per_query_results.append({
                    "query_id": tq.query_id,
                    "error": str(e),
                })

        n = len(queries)
        results.append(AblationResult(
            config_name=config["name"],
            description=config["description"],
            queries_run=n,
            avg_latency_ms=round(sum(latencies) / max(len(latencies), 1), 1),
            topic_coverage=round(topic_hits / max(topic_total, 1), 4),
            entity_coverage=round(entity_hits / max(entity_total, 1), 4),
            citation_rate=round(citation_count / max(n, 1), 4),
            classification_accuracy=round(classification_correct / max(n, 1), 4),
            avg_retrieval_count=round(
                sum(r.get("retrieval_count", 0) for r in per_query_results if "error" not in r) / max(n, 1), 1
            ),
            avg_reranked_count=round(
                sum(r.get("reranked_count", 0) for r in per_query_results if "error" not in r) / max(n, 1), 1
            ),
            per_query=per_query_results,
        ))

        logger.info(f"  Result: topic={results[-1].topic_coverage:.1%}, "
                     f"entity={results[-1].entity_coverage:.1%}, "
                     f"citations={results[-1].citation_rate:.1%}, "
                     f"classification={results[-1].classification_accuracy:.1%}")

        # Pause between ablation configs to avoid rate limits
        logger.info("  Pausing 10s before next config...")
        time.sleep(10)

    return results


# ---------------------------------------------------------------------------
# Benchmark 2: Paper Prism vs. Direct Claude
# ---------------------------------------------------------------------------

def run_comparison(
    queries: List[TestQuery],
    query_engine: QueryEngine,
    anthropic_client: Anthropic,
) -> List[ComparisonResult]:
    """Compare Paper Prism pipeline answers to direct Claude answers."""

    results = []

    for i, tq in enumerate(queries):
        logger.info(f"\n[{i+1}/{len(queries)}] Comparing: {tq.query[:60]}...")

        # Rate limit: pause between queries
        if i > 0:
            time.sleep(3)

        # --- Paper Prism answer ---
        try:
            start = time.time()
            arc_result = query_engine.query(tq.query, response_mode="concise")
            arc_latency = (time.time() - start) * 1000
            arc_answer = arc_result.answer
            arc_sources = len(arc_result.sources)
            arc_has_citations = "[Source" in arc_answer
        except Exception as e:
            logger.error(f"  Paper Prism failed: {e}")
            arc_answer = f"[ERROR: {e}]"
            arc_latency = 0
            arc_sources = 0
            arc_has_citations = False

        # --- Direct Claude answer (no RAG, no sources) ---
        try:
            start = time.time()
            claude_response = anthropic_client.messages.create(
                model=settings.claude_model,
                max_tokens=2048,
                temperature=0.3,
                system="You are a research assistant. Answer the following question about scientific research. Be specific and accurate.",
                messages=[{"role": "user", "content": tq.query}],
            )
            claude_latency = (time.time() - start) * 1000
            claude_answer = claude_response.content[0].text
        except Exception as e:
            logger.error(f"  Claude direct failed: {e}")
            claude_answer = f"[ERROR: {e}]"
            claude_latency = 0

        # --- Build reference passages from Paper Prism's retrieved sources for fact-checking ---
        reference_passages = ""
        try:
            if arc_sources > 0:
                passages = []
                for j, src in enumerate(arc_result.sources[:8]):  # Top 8 sources
                    title = src.get("title", "Unknown")
                    text = src.get("text", "")[:400]
                    passages.append(f"[Passage {j+1}] ({title}): {text}")
                reference_passages = "\n\n".join(passages)
        except Exception:
            reference_passages = ""

        # --- LLM Judge (blind, randomized A/B, with reference passages) ---
        logger.info("  Judging answers (blind)...")
        judge_scores = judge_answers(
            anthropic_client, tq.query, arc_answer, claude_answer,
            reference_passages=reference_passages,
        )

        results.append(ComparisonResult(
            query_id=tq.query_id,
            query=tq.query,
            query_type=tq.query_type.value,
            arc_answer=arc_answer,
            claude_answer=claude_answer,
            arc_sources_count=arc_sources,
            arc_has_citations=arc_has_citations,
            arc_latency_ms=round(arc_latency, 1),
            claude_latency_ms=round(claude_latency, 1),
            judge_arc_accuracy=judge_scores["arc_accuracy"],
            judge_arc_completeness=judge_scores["arc_completeness"],
            judge_arc_grounding=judge_scores["arc_grounding"],
            judge_claude_accuracy=judge_scores["claude_accuracy"],
            judge_claude_completeness=judge_scores["claude_completeness"],
            judge_claude_grounding=judge_scores["claude_grounding"],
            judge_explanation=judge_scores["explanation"],
        ))

        logger.info(f"  Paper Prism: acc={judge_scores['arc_accuracy']} comp={judge_scores['arc_completeness']} ground={judge_scores['arc_grounding']}")
        logger.info(f"  Claude: acc={judge_scores['claude_accuracy']} comp={judge_scores['claude_completeness']} ground={judge_scores['claude_grounding']}")

    return results


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(
    ablation_results: Optional[List[AblationResult]],
    comparison_results: Optional[List[ComparisonResult]],
    output_path: Path,
):
    """Generate a combined benchmark report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "benchmark_version": "1.0",
    }

    # --- Ablation summary ---
    if ablation_results:
        report["ablation"] = {
            "configs": [],
            "summary_table": [],
        }
        for ar in ablation_results:
            report["ablation"]["configs"].append(asdict(ar))
            report["ablation"]["summary_table"].append({
                "config": ar.config_name,
                "topic_coverage": f"{ar.topic_coverage:.1%}",
                "entity_coverage": f"{ar.entity_coverage:.1%}",
                "citation_rate": f"{ar.citation_rate:.1%}",
                "classification_accuracy": f"{ar.classification_accuracy:.1%}",
                "avg_latency_ms": f"{ar.avg_latency_ms:.0f}",
            })

        # Compute deltas from baseline
        baseline = ablation_results[0]  # full_pipeline
        report["ablation"]["deltas_from_baseline"] = []
        for ar in ablation_results[1:]:
            report["ablation"]["deltas_from_baseline"].append({
                "config": ar.config_name,
                "topic_coverage_delta": f"{(ar.topic_coverage - baseline.topic_coverage):+.1%}",
                "entity_coverage_delta": f"{(ar.entity_coverage - baseline.entity_coverage):+.1%}",
                "citation_rate_delta": f"{(ar.citation_rate - baseline.citation_rate):+.1%}",
                "classification_accuracy_delta": f"{(ar.classification_accuracy - baseline.classification_accuracy):+.1%}",
                "latency_delta_ms": f"{(ar.avg_latency_ms - baseline.avg_latency_ms):+.0f}",
            })

    # --- Comparison summary ---
    if comparison_results:
        n = len(comparison_results)
        arc_acc = sum(r.judge_arc_accuracy for r in comparison_results) / n
        arc_comp = sum(r.judge_arc_completeness for r in comparison_results) / n
        arc_ground = sum(r.judge_arc_grounding for r in comparison_results) / n
        claude_acc = sum(r.judge_claude_accuracy for r in comparison_results) / n
        claude_comp = sum(r.judge_claude_completeness for r in comparison_results) / n
        claude_ground = sum(r.judge_claude_grounding for r in comparison_results) / n

        arc_wins = sum(
            1 for r in comparison_results
            if (r.judge_arc_accuracy + r.judge_arc_completeness + r.judge_arc_grounding) >
               (r.judge_claude_accuracy + r.judge_claude_completeness + r.judge_claude_grounding)
        )
        claude_wins = sum(
            1 for r in comparison_results
            if (r.judge_claude_accuracy + r.judge_claude_completeness + r.judge_claude_grounding) >
               (r.judge_arc_accuracy + r.judge_arc_completeness + r.judge_arc_grounding)
        )
        ties = n - arc_wins - claude_wins

        report["comparison"] = {
            "total_queries": n,
            "arc_wins": arc_wins,
            "claude_wins": claude_wins,
            "ties": ties,
            "arc_avg_scores": {
                "accuracy": round(arc_acc, 2),
                "completeness": round(arc_comp, 2),
                "grounding": round(arc_ground, 2),
                "overall": round((arc_acc + arc_comp + arc_ground) / 3, 2),
            },
            "claude_avg_scores": {
                "accuracy": round(claude_acc, 2),
                "completeness": round(claude_comp, 2),
                "grounding": round(claude_ground, 2),
                "overall": round((claude_acc + claude_comp + claude_ground) / 3, 2),
            },
            "per_query": [asdict(r) for r in comparison_results],
        }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport saved to {output_path}")

    # Print summary to terminal
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    if ablation_results:
        print("\n--- ABLATION STUDY ---")
        print(f"{'Config':<25} {'Topics':>8} {'Entities':>10} {'Citations':>10} {'ClassAcc':>10} {'Latency':>10}")
        print("-" * 73)
        for ar in ablation_results:
            print(f"{ar.config_name:<25} {ar.topic_coverage:>7.1%} {ar.entity_coverage:>9.1%} "
                  f"{ar.citation_rate:>9.1%} {ar.classification_accuracy:>9.1%} {ar.avg_latency_ms:>8.0f}ms")

        print("\n  Deltas from full pipeline:")
        baseline = ablation_results[0]
        for ar in ablation_results[1:]:
            topic_d = ar.topic_coverage - baseline.topic_coverage
            entity_d = ar.entity_coverage - baseline.entity_coverage
            print(f"    {ar.config_name:<25} topics: {topic_d:+.1%}  entities: {entity_d:+.1%}")

    if comparison_results:
        print("\n--- PAPER PRISM vs. DIRECT CLAUDE ---")
        print(f"  Paper Prism wins: {arc_wins}/{n}  |  Claude wins: {claude_wins}/{n}  |  Ties: {ties}/{n}")
        print(f"  Paper Prism avg scores — Accuracy: {arc_acc:.1f}/5  Completeness: {arc_comp:.1f}/5  Grounding: {arc_ground:.1f}/5")
        print(f"  Claude avg scores — Accuracy: {claude_acc:.1f}/5  Completeness: {claude_comp:.1f}/5  Grounding: {claude_ground:.1f}/5")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Paper Prism benchmarks")
    parser.add_argument("--type", choices=["ablation", "comparison", "classification", "all"],
                        default="all", help="Benchmark type to run")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max queries per benchmark (default: all 50)")
    parser.add_argument("--output", type=str, default="data/benchmark_results.json",
                        help="Output path for results JSON")
    args = parser.parse_args()

    # Initialize shared components
    logger.info("Initializing components...")
    anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
    embedder = VoyageEmbedder(api_key=settings.voyage_api_key)
    reranker = CohereReranker(api_key=settings.cohere_api_key)
    store = QdrantStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection_name,
        enable_hybrid=True,
    )

    # Load test queries — prefer demo queries for the AI/ML corpus
    try:
        from evaluation.demo_test_queries import DEMO_QUERIES
        queries = DEMO_QUERIES
        logger.info(f"Using demo AI/ML test queries ({len(queries)} queries)")
    except ImportError:
        queries = load_test_queries()
        logger.info(f"Using default test queries ({len(queries)} queries)")

    if args.limit:
        queries = queries[:args.limit]
    logger.info(f"Running with {len(queries)} queries")

    ablation_results = None
    comparison_results = None

    # --- Run ablation ---
    if args.type in ("ablation", "all"):
        logger.info("\n\n========== ABLATION STUDY ==========")
        ablation_results = run_ablation_study(
            queries, anthropic_client, embedder, reranker, store,
        )

    # --- Run comparison ---
    if args.type in ("comparison", "all"):
        logger.info("\n\n========== PAPER PRISM vs. DIRECT CLAUDE ==========")
        # Use a subset for comparison (it's expensive — 3 API calls per query)
        comparison_queries = queries[:min(len(queries), args.limit or 15)]

        # Build full-featured engine for comparison
        engine = QueryEngine(
            embedder=embedder,
            reranker=reranker,
            store=store,
            anthropic_client=anthropic_client,
            claude_model=settings.claude_model,
            claude_model_fast=settings.claude_model_fast,
            claude_model_classifier=settings.claude_model_classifier,
            enable_classification=True,
            enable_expansion=True,
            enable_entity_extraction=True,
            enable_citation_verification=True,
            enable_conversation_memory=False,  # Disable for fair per-query comparison
            enable_hybrid_search=True,
        )
        comparison_results = run_comparison(
            comparison_queries, engine, anthropic_client,
        )

    # --- Classification-only (fast, cheap) ---
    if args.type == "classification":
        logger.info("\n\n========== CLASSIFICATION ACCURACY ==========")
        # Just run the ablation with full_pipeline config — classification_accuracy is included
        ablation_results = run_ablation_study(
            queries, anthropic_client, embedder, reranker, store,
        )
        # Only keep the full_pipeline result
        ablation_results = [ablation_results[0]]

    # Generate report
    generate_report(
        ablation_results=ablation_results,
        comparison_results=comparison_results,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
