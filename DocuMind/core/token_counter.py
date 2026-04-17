from __future__ import annotations

import tiktoken
from DocuMind.core.logging.logger import get_logger

logger = get_logger(__name__)

# GPT-4o and GPT-4o-mini both use cl100k_base encoding
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Leave room for system prompt, question, and response
# Total GPT-4o window = 128k, but we keep context lean
MAX_CONTEXT_TOKENS = 6000
MAX_QUESTION_TOKENS = 500

def count_tokens(text: str) -> int:
    """Return the number of tokens in a string."""
    return len(_ENCODING.encode(text))

def validate_question(question: str) -> None:
    """
    Raise ValueError if the question exceeds the allowed token limit.
    Call this before sending anything to the LLM.
    """
    n = count_tokens(question)
    if n > MAX_QUESTION_TOKENS:
        raise ValueError(
            f"Question is too long ({n} tokens). "
            f"Maximum allowed is {MAX_QUESTION_TOKENS} tokens."
        )
    
def truncate_chunks_to_limit(
    chunks: list[dict],
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> list[dict]:
    """
    Walk through retrieved chunks in order and keep adding them
    until the token budget is exhausted. Returns the subset that fits.

    This means the highest-ranked chunks (reranked first) are always
    kept — lower-ranked ones get dropped when budget runs out.
    """
    kept   = []
    total  = 0

    for chunk in chunks:
        text   = chunk.get("content", "") or chunk.get("text", "")
        tokens = count_tokens(text)

        if total + tokens > max_tokens:
            logger.warning(
                "Context token limit reached — dropping remaining chunks",
                kept=len(kept),
                dropped=len(chunks) - len(kept),
                tokens_used=total,
                limit=max_tokens,
            )
            break

        kept.append(chunk)
        total += tokens

    logger.info("Context budget", chunks_kept=len(kept), tokens_used=total)
    return kept