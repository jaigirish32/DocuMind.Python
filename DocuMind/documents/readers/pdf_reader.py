from __future__ import annotations
import pdfplumber
from pathlib import Path
import re
from typing import List, Optional
from collections import defaultdict
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

        if len(blocks) == 1 and len(lines) > 5:
            blocks = [[line] for line in lines]

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

    def _extract_table_blocks(self, page) -> List[RawBlock]:
        tables = page.extract_tables()
        result = []

        for table in tables:
            # if self.is_bad_table(table):
            #     logger.info("Bad table detected → skipping")
            #     continue

            # --------------------------------------------------
            # 1. Find the row that contains the years
            # --------------------------------------------------
            year_row_idx = None
            best_year_count = 0

            for idx, row in enumerate(table):
                # Count cells that look like a 4-digit year (1900-2099)
                # Also allow "2023 (1)" or "FY2023" if needed
                year_count = 0
                for cell in row:
                    if cell is None:
                        continue
                    cell_str = str(cell).strip()
                    # Simple pattern: 4 digits, optionally preceded by "FY" or followed by "(...)"
                    # Adjust if your PDF uses different formatting
                    if re.match(r'^(?:FY)?(19|20)\d{2}(?:\s*\(.*\))?$', cell_str):
                        year_count += 1
                if year_count > best_year_count:
                    best_year_count = year_count
                    year_row_idx = idx

            # If no row with years found, fallback to first non‑empty row
            if year_row_idx is None:
                # Find the first row that has at least 2 non‑empty cells
                for idx, row in enumerate(table):
                    non_empty = [c for c in row if c is not None and str(c).strip()]
                    if len(non_empty) >= 2:
                        year_row_idx = idx
                        break
                # If still none, fallback to first row
                if year_row_idx is None:
                    year_row_idx = 0

            # --------------------------------------------------
            # 2. Build column → year mapping
            # --------------------------------------------------
            header_row = table[year_row_idx]
            # Data rows are all rows except the header row
            data_rows = [row for i, row in enumerate(table) if i != year_row_idx]

            col_to_year = {}
            for col, cell in enumerate(header_row):
                if cell is None:
                    continue
                cell_str = str(cell).strip()
                # Extract the year part (e.g., from "2023 (1)" or "FY2023")
                match = re.search(r'(19|20)\d{2}', cell_str)
                if match:
                    col_to_year[col] = match.group(0)   # store just the year digits
                else:
                    # If this cell is not a year, keep track for debugging
                    pass

            # If we still have no years, fallback to using column indices
            if not col_to_year:
                # Use the first few columns as "Year 1", "Year 2", etc.
                # This should not happen for the Tesla 10-K, but just in case.
                for col in range(1, len(header_row)):
                    col_to_year[col] = f"Year {col}"

            # --------------------------------------------------
            # 3. Generate text lines
            # --------------------------------------------------
            raw_block = RawBlock()
            for row in data_rows:
                if not row or all(cell is None or str(cell).strip() == "" for cell in row):
                    continue

                # The first cell is the label (e.g., "Total revenues")
                label = str(row[0]).strip() if row[0] else ""

                if not label:
                    continue

                # For each column that maps to a year, create a line
                for col, year in col_to_year.items():
                    if col < len(row):
                        value_cell = row[col]
                        if value_cell is not None and str(value_cell).strip():
                            value = str(value_cell).strip()
                            # Clean up common formatting: remove $ and commas
                            value = value.replace('$', '').replace(',', '')
                            text = f"{label} for {year} is {value}"
                            raw_line = RawLine()
                            raw_line.words.append(RawWord(text=text))
                            raw_block.lines.append(raw_line)

            if raw_block.lines:
                result.append(raw_block)

        return result

    def is_bad_table(self,table):
        if not table or len(table) < 2:
            return True

        # header must contain years
        headers = table[0]

        year_count = sum(
            1 for h in headers
            if h and any(y in h for y in ["2023", "2022", "2021"])
        )

        # if no proper year columns → bad table
        if year_count == 0:
            return True

        # too many empty cells
        empty = sum(1 for row in table for c in row if not c)
        total = sum(len(row) for row in table)

        return (empty / total) > 0.4

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
        """Group words into lines by y-position (more strict)."""
        if not words:
            return []

        lines = []
        current_line = [words[0]]
        current_top = words[0]["top"]

        for word in words[1:]:
            # 🔥 stricter threshold
            if abs(word["top"] - current_top) <= 1:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
                current_top = word["top"]

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
    