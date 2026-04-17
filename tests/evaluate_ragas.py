# tests/evaluate_ragas.py
from __future__ import annotations

import asyncio
import json
import sys
import os
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from DocuMind.core.settings import get_settings
from DocuMind.azure.chat_client import ChatClient
from DocuMind.azure.embedding_client import EmbeddingClient
from DocuMind.search.azure_search_store import AzureSearchStore
from DocuMind.agents.documind_agent import DocuMindAgent
from tests.ragas_dataset import DOCUMENT_TEST_MAP

COLLECTED_FILE = "ragas_collected.json"
RESULTS_FILE   = "ragas_results.json"


# ── Step 1 — Collect (run once, saves to JSON) ────────────────────────────────

async def collect_outputs() -> list[dict]:
    """
    Run DocuMind on every test question.
    Saves results to ragas_collected.json so you never need to re-run.
    """
    settings = get_settings()
    embedder = EmbeddingClient()
    chat     = ChatClient()
    store    = AzureSearchStore()
    agent    = DocuMindAgent(
        embedder = embedder,
        chat     = chat,
        store    = store,
    )

    all_results = []

    for document_id, test_cases in DOCUMENT_TEST_MAP.items():
        print(f"\nDocument: {document_id}")
        print("-" * 60)

        for i, test in enumerate(test_cases):
            question     = test["question"]
            ground_truth = test["ground_truth"]
            print(f"  [{i+1}/{len(test_cases)}] {question}")

            try:
                # Ask DocuMind
                result = await agent.ask_structured(
                    question     = question,
                    document_id  = document_id,
                    document_ids = None,
                    history      = [],
                )

                # Get chunks for RAGAS contexts
                embeddings = await embedder.create_embeddings([question])
                chunks     = await store.hybrid_search(
                    query        = question,
                    embedding    = embeddings[0],
                    top_k        = 5,
                    document_id  = document_id,
                )
                contexts = [
                    c["content"] for c in chunks
                    if c.get("content", "").strip()
                ] or ["No context retrieved"]

                print(f"     Answer:   {result.answer[:70]}...")
                print(f"     Pages:    {result.source_pages}")
                print(f"     Contexts: {len(contexts)} chunks")

                all_results.append({
                    "question":     question,
                    "answer":       result.answer,
                    "contexts":     contexts,
                    "ground_truth": ground_truth,
                    "document_id":  document_id,
                    "source_pages": result.source_pages,
                })

            except Exception as e:
                print(f"     ERROR: {e}")
                all_results.append({
                    "question":     question,
                    "answer":       f"ERROR: {str(e)}",
                    "contexts":     ["Error retrieving context"],
                    "ground_truth": ground_truth,
                    "document_id":  document_id,
                    "source_pages": [],
                })

    # Save to file — never need to re-collect
    with open(COLLECTED_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved {len(all_results)} results to {COLLECTED_FILE} ✅")
    print("Now run: python -m tests.evaluate_ragas evaluate")
    return all_results


# ── Step 2 — Evaluate (loads from JSON, runs RAGAS) ───────────────────────────

def run_ragas(results: list[dict]) -> None:
    """
    Run RAGAS on collected results.
    Loads from ragas_collected.json — no DocuMind calls needed.
    """
    settings = get_settings()

    print(f"\nEvaluating {len(results)} questions...")
    print("="*60)

    # Build dataset
    dataset = Dataset.from_dict({
        "question":     [r["question"]     for r in results],
        "answer":       [r["answer"]       for r in results],
        "contexts":     [r["contexts"]     for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    # Use modern RAGAS API — no LangChain
    try:
        from openai import AzureOpenAI
        from ragas.llms import llm_factory
        from ragas.embeddings import embedding_factory

        azure_client = AzureOpenAI(
            azure_endpoint = settings.azure_openai_endpoint,
            api_key        = settings.azure_openai_key,
            api_version    = "2024-02-01",
        )

        ragas_llm = llm_factory(
            model  = settings.azure_openai_chat_deployment,
            client = azure_client,
        )
        ragas_embeddings = embedding_factory(
            provider = "openai",
            model    = settings.azure_openai_embedding_deployment,
            client   = azure_client,
        )

        scores = evaluate(
            dataset    = dataset,
            metrics    = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm        = ragas_llm,
            embeddings = ragas_embeddings,
        )

    except (ImportError, Exception) as e:
        # Fallback to LangChain if modern API fails
        print(f"Modern API failed ({e}), falling back to LangChain...")
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        azure_llm = AzureChatOpenAI(
            azure_endpoint   = settings.azure_openai_endpoint,
            api_key          = settings.azure_openai_key,
            azure_deployment = settings.azure_openai_chat_deployment,
            api_version      = "2024-02-01",
            temperature      = 0,
        )
        azure_embeddings = AzureOpenAIEmbeddings(
            azure_endpoint   = settings.azure_openai_endpoint,
            api_key          = settings.azure_openai_key,
            azure_deployment = settings.azure_openai_embedding_deployment,
            api_version      = "2024-02-01",
        )
        scores = evaluate(
            dataset    = dataset,
            metrics    = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm        = LangchainLLMWrapper(azure_llm),
            embeddings = LangchainEmbeddingsWrapper(azure_embeddings),
        )

    # Print results
    df = scores.to_pandas()

    print("\n" + "="*60)
    print("RESULTS PER QUESTION")
    print("="*60)
    for _, row in df.iterrows():
        doc = next(
            (r["document_id"].split("_")[0] for r in results
             if r["question"] == row["question"]), ""
        )
        print(f"\nQ: {row['question'][:55]}")
        print(f"   Faithfulness:      {row.get('faithfulness', 'N/A'):.3f}")
        print(f"   Answer relevancy:  {row.get('answer_relevancy', 'N/A'):.3f}")
        print(f"   Context precision: {row.get('context_precision', 'N/A'):.3f}")
        print(f"   Context recall:    {row.get('context_recall', 'N/A'):.3f}")

    print("\n" + "="*60)
    print("MEAN SCORES — BASELINE")
    print("="*60)
    metrics = ['faithfulness', 'answer_relevancy',
               'context_precision', 'context_recall']
    mean_scores = {}
    for metric in metrics:
        if metric in df.columns:
            score = df[metric].mean()
            mean_scores[metric] = round(score, 3)
            bar = "█" * int(score * 20)
            print(f"  {metric:<22} {score:.3f}  {bar}")

    # Save results
    output = {
        "mean_scores":  mean_scores,
        "per_question": df.to_dict(orient="records"),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved to {RESULTS_FILE} ✅")
    print("\nShare these scores — we'll identify what to fix next!")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "both"

    if command == "collect":
        # Only collect — no RAGAS calls
        print("STEP 1 — Collecting DocuMind outputs...")
        await collect_outputs()

    elif command == "evaluate":
        # Only evaluate — load from saved file
        print("STEP 2 — Running RAGAS evaluation...")
        if not os.path.exists(COLLECTED_FILE):
            print(f"ERROR: {COLLECTED_FILE} not found.")
            print("Run collection first: python -m tests.evaluate_ragas collect")
            return
        with open(COLLECTED_FILE) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results from {COLLECTED_FILE}")
        run_ragas(results)

    else:
        # Run both — default
        print("STEP 1 — Collecting DocuMind outputs...")
        results = await collect_outputs()
        print("\nSTEP 2 — Running RAGAS evaluation...")
        run_ragas(results)


if __name__ == "__main__":
    asyncio.run(main())