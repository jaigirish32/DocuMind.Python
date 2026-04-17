"""
DocuMind/eval/recall_eval.py

Layer 1 — Retrieval Evaluation
Measures Recall@5 on a gold set of questions with known-correct pages.

Usage:
    python -m DocuMind.eval.recall_eval

What it does:
    1. Loads gold_set.json
    2. For each question, runs hybrid_search with top_k=5
    3. Checks if any returned chunk's pageNumber is in ground_truth_pages
    4. Reports Recall@5 overall and per category
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from collections import defaultdict

from DocuMind.azure.embedding_client import EmbeddingClient
from DocuMind.search.azure_search_store import AzureSearchStore
from DocuMind.core.logging.logger import get_logger

logger = get_logger(__name__)

GOLD_SET_PATH = Path(__file__).parent / "gold_set.json"
TOP_K         = 5
USER_ID       = "1"   # the user who uploaded the docs — matches your test setup


async def evaluate_question(
    question_data: dict,
    embedder:      EmbeddingClient,
    store:         AzureSearchStore,
) -> dict:
    """
    Run one gold-set question through retrieval and return the evaluation result.
    """
    question           = question_data["question"]
    document_id        = question_data["document_id"]
    ground_truth_pages = set(question_data["ground_truth_pages"])
    answerable         = question_data["answerable"]

    # Unanswerable questions can't have retrieval Recall — skip retrieval check
    if not answerable:
        return {
            "id":         question_data["id"],
            "question":   question,
            "category":   question_data["category"],
            "skipped":    True,
            "reason":     "unanswerable — retrieval recall not applicable",
        }

    # Embed the question
    embeddings = await embedder.create_embeddings([question])
    if not embeddings:
        return {
            "id":       question_data["id"],
            "question": question,
            "category": question_data["category"],
            "error":    "embedding failed",
        }

    # Run hybrid search
    chunks = await store.hybrid_search(
        query       = question,
        embedding   = embeddings[0],
        top_k       = TOP_K,
        document_id = document_id,
        user_id     = USER_ID,
    )

    # Extract unique pages from returned chunks (order preserved)
    retrieved_pages = []
    for c in chunks:
        p = c.get("page_number")
        if p is not None and p not in retrieved_pages:
            retrieved_pages.append(p)

    # Check hit/miss
    hit = any(p in ground_truth_pages for p in retrieved_pages)

    return {
        "id":              question_data["id"],
        "question":        question,
        "category":        question_data["category"],
        "ground_truth":    sorted(ground_truth_pages),
        "retrieved_pages": retrieved_pages,
        "num_chunks":      len(chunks),
        "hit":             hit,
    }


def print_report(results: list[dict]) -> None:
    """Pretty-print the evaluation report."""
    print("\n" + "=" * 80)
    print(" LAYER 1 — RETRIEVAL EVALUATION REPORT (Recall@{})".format(TOP_K))
    print("=" * 80 + "\n")

    # Separate evaluated vs skipped
    evaluated = [r for r in results if not r.get("skipped") and not r.get("error")]
    skipped   = [r for r in results if r.get("skipped")]
    errored   = [r for r in results if r.get("error")]

    # Per-question detail
    print(" Per-question results:")
    print(" " + "-" * 78)
    for r in results:
        if r.get("error"):
            status = "ERROR"
        elif r.get("skipped"):
            status = "SKIP "
        elif r.get("hit"):
            status = "HIT  "
        else:
            status = "MISS "

        print(f"  [{status}] {r['id']:4} | {r['category']:15} | {r['question'][:50]}")
        if not r.get("skipped") and not r.get("error"):
            print(f"          ground truth: {r['ground_truth']}, retrieved: {r['retrieved_pages']}, chunks: {r['num_chunks']}")

    # Overall Recall@5
    print("\n Overall:")
    print(" " + "-" * 78)
    if evaluated:
        hits  = sum(1 for r in evaluated if r["hit"])
        total = len(evaluated)
        recall = hits / total
        print(f"  Recall@{TOP_K}: {hits}/{total} = {recall:.2%}")
    else:
        print(f"  No answerable questions evaluated.")

    # Per-category breakdown
    by_category = defaultdict(list)
    for r in evaluated:
        by_category[r["category"]].append(r)

    if by_category:
        print("\n By category:")
        print(" " + "-" * 78)
        for cat, rs in by_category.items():
            hits  = sum(1 for r in rs if r["hit"])
            total = len(rs)
            print(f"  {cat:15}: {hits}/{total} = {hits / total:.2%}")

    # Skipped / errored
    if skipped:
        print(f"\n Skipped (unanswerable, not counted in recall): {len(skipped)}")
    if errored:
        print(f"\n Errored: {len(errored)}")

    print("\n" + "=" * 80)


async def main() -> None:
    # Load gold set
    with open(GOLD_SET_PATH, encoding="utf-8") as f:
        gold_set = json.load(f)

    logger.info("Gold set loaded", count=len(gold_set))

    # Initialize clients
    embedder = EmbeddingClient()
    store    = AzureSearchStore()

    # Run evaluation
    async with embedder:
        results = []
        for q in gold_set:
            logger.info("Evaluating", id=q["id"], question=q["question"][:50])
            result = await evaluate_question(q, embedder, store)
            results.append(result)

    # Print report
    print_report(results)


if __name__ == "__main__":
    asyncio.run(main())