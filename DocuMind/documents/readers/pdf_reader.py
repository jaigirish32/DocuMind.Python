from __future__ import annotations

import pdfplumber
from pathlib import Path

from DocuMind.core.logging.logger import get_logger
from DocuMind.core.errors.exceptions import DocumentParseError
from DocuMind.documents.raw.raw_document import (
    RawDocument, RawPage, RawBlock, RawLine, RawWord, BoundingBox
)

logger = get_logger(__name__)


class PdfReader:
    """
    Reads PDF files using pdfplumber.
    Extracts both text blocks and tables per page.

    Pipeline:
        PDF → extract_words()   → text paragraphs and headings
            → extract_tables()  → structured table rows
        Both combined into RawDocument
    """

    def read(self, path: Path) -> RawDocument:
        logger.info("Opening PDF", path=str(path))

        try:
            with pdfplumber.open(path) as pdf:
                return self._parse(pdf, path)
        except FileNotFoundError:
            raise DocumentParseError(str(path), "File not found")
        except Exception as e:
            raise DocumentParseError(str(path), str(e))

    # ── Private ───────────────────────────────────────────────────────────────

    def _parse(self, pdf: pdfplumber.PDF, path: Path) -> RawDocument:
        doc = RawDocument(source_path=str(path))

        for page in pdf.pages:
            raw_page = self._parse_page(page)
            doc.pages.append(raw_page)

        logger.info("PDF parsed", pages=len(doc.pages), path=str(path))
        return doc

    def _parse_page(self, page: pdfplumber.page.Page) -> RawPage:
        """
        Parse a single page — extracts both text blocks and tables.
        """
        raw_page = RawPage(page_number=page.page_number)

        # Step 1 — find table bounding boxes so we can exclude
        # those regions from text extraction
        table_bboxes = [t.bbox for t in page.find_tables()]

        # Step 2 — extract text blocks (excluding table areas)
        text_blocks = self._extract_text_blocks(page, table_bboxes)
        raw_page.blocks.extend(text_blocks)

        # Step 3 — extract tables as structured blocks
        table_blocks = self._extract_table_blocks(page)
        raw_page.blocks.extend(table_blocks)

        return raw_page

    def _extract_text_blocks(
        self,
        page:        pdfplumber.page.Page,
        table_bboxes: list[tuple],
    ) -> list[RawBlock]:
        """
        Extract text words from page — skipping table regions.
        """
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True,
            extra_attrs=["fontname", "size"],
        )

        if not words:
            return []

        # Filter out words that fall inside table bounding boxes
        if table_bboxes:
            words = [
                w for w in words
                if not self._is_in_table(w, table_bboxes)
            ]

        if not words:
            return []

        lines  = self._group_into_lines(words)
        blocks = self._group_into_blocks(lines)

        result = []
        for block_lines in blocks:
            raw_block = RawBlock()

            for line_words in block_lines:
                raw_line = RawLine()

                for w in line_words:
                    raw_word = RawWord(
                        text      = w["text"],
                        bounds    = BoundingBox(
                            x0     = w["x0"],
                            top    = w["top"],
                            x1     = w["x1"],
                            bottom = w["bottom"],
                        ),
                        font_name = w.get("fontname", ""),
                        font_size = w.get("size", 0.0),
                        is_bold   = "bold" in w.get("fontname", "").lower(),
                        is_italic = "italic" in w.get("fontname", "").lower()
                                    or "oblique" in w.get("fontname", "").lower(),
                    )
                    raw_line.words.append(raw_word)

                if raw_line.words:
                    raw_block.lines.append(raw_line)

            if raw_block.lines:
                result.append(raw_block)

        return result

    def _extract_table_blocks(
        self,
        page: pdfplumber.page.Page,
    ) -> list[RawBlock]:
        """
        Extract tables as structured RawBlocks.

        Each table row becomes a RawLine.
        Format: "Header1: Value1 | Header2: Value2"

        Example:
            "Total revenues: $96,773M | Automotive: $82,418M"

        This keeps label and value together — solving the
        main problem with the C++ Poppler approach.
        """
        tables = page.extract_tables()
        result = []

        for table in tables:
            if not table or len(table) < 2:
                continue

            # First row is headers
            headers = [
                cell.strip() if cell else ""
                for cell in table[0]
            ]

            raw_block = RawBlock()

            # Process each data row
            for row in table[1:]:
                if not row or all(cell is None or cell.strip() == "" for cell in row):
                    continue

                # Build "Header: Value | Header: Value" format
                parts = []
                for i, cell in enumerate(row):
                    if cell is None:
                        continue
                    cell = cell.strip()
                    if not cell:
                        continue

                    header = headers[i] if i < len(headers) else ""
                    if header:
                        parts.append(f"{header}: {cell}")
                    else:
                        parts.append(cell)

                if not parts:
                    continue

                row_text = " | ".join(parts)

                # Build RawLine from row text
                raw_line = RawLine()
                raw_word = RawWord(text=row_text)
                raw_line.words.append(raw_word)
                raw_block.lines.append(raw_line)

            if raw_block.lines:
                result.append(raw_block)

        return result

    def _is_in_table(
        self,
        word:         dict,
        table_bboxes: list[tuple],
    ) -> bool:
        """
        Check if a word falls inside any table bounding box.
        Prevents duplicate content — word already captured in table.
        """
        wx0, wtop, wx1, wbottom = (
            word["x0"], word["top"], word["x1"], word["bottom"]
        )
        for x0, top, x1, bottom in table_bboxes:
            if wx0 >= x0 and wtop >= top and wx1 <= x1 and wbottom <= bottom:
                return True
        return False

    def _group_into_lines(self, words: list[dict]) -> list[list[dict]]:
        """Group words into lines by y-position."""
        if not words:
            return []

        lines: list[list[dict]] = []
        current_line: list[dict] = [words[0]]
        current_top = words[0]["top"]

        for word in words[1:]:
            if abs(word["top"] - current_top) <= 3:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
                current_top  = word["top"]

        if current_line:
            lines.append(current_line)

        return lines

    def _group_into_blocks(
        self,
        lines: list[list[dict]],
    ) -> list[list[list[dict]]]:
        """Group lines into blocks by vertical gap."""
        if not lines:
            return []

        blocks: list[list[list[dict]]] = []
        current_block: list[list[dict]] = [lines[0]]

        for i in range(1, len(lines)):
            prev_bottom = max(w["bottom"] for w in lines[i - 1])
            curr_top    = min(w["top"]    for w in lines[i])
            gap         = curr_top - prev_bottom

            if gap > 10:
                blocks.append(current_block)
                current_block = [lines[i]]
            else:
                current_block.append(lines[i])

        if current_block:
            blocks.append(current_block)

        return blocks