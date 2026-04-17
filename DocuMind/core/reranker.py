from __future__ import annotations

import math
from DocuMind.core.logging.logger import get_logger
from langsmith import traceable
logger = get_logger(__name__)


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    """Euclidean norm of a vector."""
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    
    norm_a = _norm(a)
    norm_b = _norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return _dot(a, b) / (norm_a * norm_b)

@traceable(name="mmr_rerank")
def mmr_rerank(
    query_embedding:  list[float],
    chunks:           list[dict],
    chunk_embeddings: list[list[float]],
    top_k:            int   = 10,
    diversity:        float = 0.5,
) -> list[dict]:
    """
    Maximal Marginal Relevance re-ranking.

    diversity controls the balance:
      0.0 = pure relevance (same as no re-ranking)
      1.0 = pure diversity (ignores relevance score)
      0.5 = balanced (recommended default)

    Returns top_k chunks ordered by MMR score.
    """
    if not chunks:
        return []

    if len(chunks) != len(chunk_embeddings):
        logger.warning(
            "Chunk count does not match embedding count — skipping MMR",
            chunks=len(chunks),
            embeddings=len(chunk_embeddings),
        )
        return chunks[:top_k]

    # Relevance score of each chunk against the query
    relevance_scores = [
        cosine_similarity(query_embedding, emb)
        for emb in chunk_embeddings
    ]

    selected_indices   = []
    remaining_indices  = list(range(len(chunks)))

    while len(selected_indices) < top_k and remaining_indices:

        best_index = None
        best_score = float("-inf")

        for idx in remaining_indices:

            relevance = relevance_scores[idx]

            # Similarity to already-selected chunks
            # We take the MAX similarity to any selected chunk
            # (penalise if similar to ANY already picked chunk)
            if selected_indices:
                max_sim = max(
                    cosine_similarity(chunk_embeddings[idx], chunk_embeddings[s])
                    for s in selected_indices
                )
            else:
                max_sim = 0.0

            # MMR score formula
            mmr_score = (
                diversity       * relevance
                - (1 - diversity) * max_sim
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_index = idx

        if best_index is None:
            break

        selected_indices.append(best_index)
        remaining_indices.remove(best_index)

    result = [chunks[i] for i in selected_indices]

    logger.info(
        "MMR re-ranking complete",
        input_chunks=len(chunks),
        output_chunks=len(result),
        diversity=diversity,
    )
    return result