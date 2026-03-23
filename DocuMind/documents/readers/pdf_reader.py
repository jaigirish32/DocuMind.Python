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
    def read(self, path: Path) -> RawDocument:
       
        logger.info("Opening PDF", path=str(path))

        try:
            with pdfplumber.open(path) as pdf:
                return self._parse(pdf, path)

        except FileNotFoundError:
            raise DocumentParseError(str(path), "File not found")
        except Exception as e:
            raise DocumentParseError(str(path), str(e))
        
    def _parse(self, pdf: pdfplumber.PDF, path: Path) -> RawDocument:
        """Build RawDocument from all pages."""
        doc = RawDocument(source_path=str(path))

        for page in pdf.pages:
            raw_page = self._parse_page(page)
            doc.pages.append(raw_page)

        logger.info("PDF parsed", pages=len(doc.pages), path=str(path))
        return doc
    
    def _parse_page(self, page: pdfplumber.page.Page) -> RawPage:
        """Parse a single page into RawPage."""
        raw_page = RawPage(page_number=page.page_number)

        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True,
            extra_attrs=["fontname", "size"],
        )

        if not words:
            return raw_page

        lines  = self._group_into_lines(words)
        blocks = self._group_into_blocks(lines)

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
                raw_page.blocks.append(raw_block)

        return raw_page
    
    def _group_into_lines(self, words: list[dict]) -> list[list[dict]]:
        
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
        self, lines: list[list[dict]]
    ) -> list[list[list[dict]]]:
        
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