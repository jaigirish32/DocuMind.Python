from __future__ import annotations

import re
from DocuMind.core.logging.logger import get_logger
from DocuMind.documents.models.document_chunk import DocumentChunk
from DocuMind.documents.raw.raw_document import RawDocument

logger = get_logger(__name__)

# ── Default constants ─────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE   = 1500
DEFAULT_OVERLAP_SIZE = 300
DEFAULT_MIN_BLOCK    = 50

# Default boilerplate patterns — applied to ALL documents
DEFAULT_BOILERPLATE = [
    "confidential - for internal use only",
    "unauthorized distribution is prohibited",
    "this document contains proprietary information",
]


class ChunkBuilder:
    def __init__(
        self,
        chunk_size:           int       = DEFAULT_CHUNK_SIZE,
        overlap_size:         int       = DEFAULT_OVERLAP_SIZE,
        min_block_size:       int       = DEFAULT_MIN_BLOCK,
        boilerplate_patterns: list[str] = None,
    ) -> None:
        self._chunk_size    = chunk_size
        self._overlap_size  = overlap_size
        self._min_block     = min_block_size

        # Merge default + custom boilerplate patterns
        self._boilerplate = DEFAULT_BOILERPLATE.copy()
        if boilerplate_patterns:
            self._boilerplate.extend(boilerplate_patterns)

    def build_chunks(
        self,
        document:    RawDocument,
        document_id: str,
    ) -> list[DocumentChunk]:
        """Main entry point."""
        blocks = self._extract_blocks(document)
        chunks = self._merge_into_chunks(blocks, document, document_id)

        logger.info(
            "Chunks built",
            chunks=len(chunks),
            document_id=document_id,
        )
        return chunks

    # ── Private ───────────────────────────────────────────────────────────────

    def _extract_blocks(self, document: RawDocument) -> list[dict]:
        """Extract, clean and deduplicate text blocks."""
        blocks          = []
        seen            = set()
        current_section = ""

        for page in document.pages:
            for raw_block in page.blocks:

                text = raw_block.text.strip()

                # Clean block text — remove boilerplate lines
                text = self._clean_block_text(raw_block.text)

                if not text.strip():
                    continue

                if len(text.strip()) < self._min_block:
                    continue

                # Skip if entire block is boilerplate
                if self._is_boilerplate(text):
                    continue

                if _is_section_heading(text):
                    current_section = text.strip()

                embedding_text = (
                    f"[Section: {current_section}]\n{text}"
                    if current_section else text
                )

                dedup_key = f"{page.page_number}|{text.lower()}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                blocks.append({
                    "text":           text,
                    "embedding_text": embedding_text,
                    "page_number":    page.page_number,
                })

        return blocks

    def _merge_into_chunks(
        self,
        blocks:      list[dict],
        document:    RawDocument,
        document_id: str,
    ) -> list[DocumentChunk]:
        """Merge blocks into size-bounded chunks with overlap."""
        chunks: list[DocumentChunk] = []

        current_text       = ""
        current_embed_text = ""
        current_page       = -1
        chunk_index        = 0

        def flush():
            nonlocal current_text, current_embed_text, chunk_index

            if not current_text.strip():
                return

            chunk = DocumentChunk(
                chunk_id       = f"{document_id}_p{current_page}_{chunk_index}",
                document_id    = document_id,
                document_name  = document.source_path.split("\\")[-1],
                page_number    = current_page,
                text           = current_text.strip(),
                normalized_text= current_text.strip().lower(),
                embedding_text = current_embed_text.strip(),
                source         = document.source_path,
            )
            chunks.append(chunk)
            chunk_index += 1

            if len(current_text) > self._overlap_size:
                overlap_start = len(current_text) - self._overlap_size
                while overlap_start < len(current_text) and \
                      current_text[overlap_start] not in (" ", "\n"):
                    overlap_start += 1
                current_text       = current_text[overlap_start:].strip()
                current_embed_text = current_embed_text[overlap_start:].strip()
            else:
                current_text       = ""
                current_embed_text = ""

        for block in blocks:

            if current_page != -1 and block["page_number"] != current_page:
                flush()
                current_text       = ""
                current_embed_text = ""
                current_page       = block["page_number"]

            if not current_text:
                current_page = block["page_number"]

            if current_text and \
               len(current_text) + len(block["text"]) >= self._chunk_size:
                flush()

            current_text       += block["text"] + "\n"
            current_embed_text += block["embedding_text"] + "\n"

        flush()
        return chunks

    def _is_boilerplate(self, text: str) -> bool:
        lower = text.lower()
        if re.match(r"^\d+/\d+$", text.strip()):
            return True
        if re.match(r"^page\s+\d+\s+of\s+\d+$", lower.strip()):
            return True
        return any(pattern in lower for pattern in self._boilerplate)
    
    def _clean_block_text(self, text: str) -> str:
        clean_lines = []

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Skip line if it matches any boilerplate pattern
            if self._is_boilerplate_line(line):
                continue
        clean_lines.append(line)
        return "\n".join(clean_lines)
    
    def _is_boilerplate_line(self, line: str) -> bool:
        lower = line.lower()   # ← lowercase the line first
        # Skip page number patterns
        if re.match(r"^\d+/\d+$", line.strip()):
            return True
        if re.match(r"^page\s+\d+\s+of\s+\d+$", lower.strip()):
            return True

        # Compare lowercase line against lowercase patterns
        return any(pattern.lower() in lower for pattern in self._boilerplate)


# ── Module level helper ───────────────────────────────────────────────────────

def _is_section_heading(text: str) -> bool:
    """Detect section headings."""
    text = text.strip()

    if len(text) < 5 or len(text) > 120:
        return False

    # Reject form checkboxes
    if text[:2] in ("o ", "x "):
        return False

    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False

    upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)

    if upper_ratio > 0.7:
        return True

    words = text.split()
    if 2 <= len(words) <= 6:
        upper_starts = sum(1 for w in words if w and w[0].isupper())
        if upper_starts / len(words) > 0.6:
            return True

    return False