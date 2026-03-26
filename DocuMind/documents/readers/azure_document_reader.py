from __future__ import annotations

from pathlib import Path

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.core.credentials import AzureKeyCredential

from DocuMind.core.settings import get_settings
from DocuMind.core.logging.logger import get_logger
from DocuMind.core.errors.exceptions import DocumentParseError
from DocuMind.documents.raw.raw_document import (
    RawDocument, RawPage, RawBlock, RawLine, RawWord, BoundingBox
)

logger = get_logger(__name__)


class AzureDocumentReader:
    """
    Reads PDF files using Azure Document Intelligence (prebuilt-layout).
    Handles text, complex tables, and scanned/OCR pages.
    Returns the same RawDocument structure as PdfReader.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = DocumentIntelligenceClient(
            endpoint   = settings.azure_document_intelligence_endpoint,
            credential = AzureKeyCredential(settings.azure_document_intelligence_key),
        )

    def read(self, path: Path) -> RawDocument:
        logger.info("AzureDocumentReader reading", path=str(path))

        try:
            with open(path, "rb") as f:
                poller = self._client.begin_analyze_document(
                    "prebuilt-layout",
                    body=f,
                )
            result: AnalyzeResult = poller.result()
        except FileNotFoundError:
            raise DocumentParseError(str(path), "File not found")
        except Exception as e:
            raise DocumentParseError(str(path), str(e))

        doc = RawDocument(source_path=str(path))

        # Build one RawPage per page
        page_map: dict[int, RawPage] = {}
        for page in result.pages or []:
            raw_page = RawPage(page_number=page.page_number)
            page_map[page.page_number] = raw_page
            doc.pages.append(raw_page)

        # Paragraphs
        for para in result.paragraphs or []:
            text = para.content.strip()
            if not text:
                continue
            page_num = self._page_of(para.bounding_regions)
            raw_page = page_map.get(page_num)
            if raw_page is None:
                continue
            block = RawBlock()
            line  = RawLine()
            line.words.append(RawWord(text=text))
            block.lines.append(line)
            raw_page.blocks.append(block)

        # Tables
        for table in result.tables or []:
            page_num = self._page_of(table.bounding_regions)
            raw_page = page_map.get(page_num)
            if raw_page is None:
                continue
            block = self._table_to_block(table)
            if block.lines:
                raw_page.blocks.append(block)

        logger.info("AzureDocumentReader done", pages=len(doc.pages), path=str(path))
        return doc

    # ── Table conversion ──────────────────────────────────────────────────────

    def _table_to_block(self, table) -> RawBlock:
        block = RawBlock()

        grid: dict[tuple[int, int], str] = {}
        max_row = 0
        max_col = 0

        for cell in table.cells or []:
            grid[(cell.row_index, cell.column_index)] = (cell.content or "").strip()
            if cell.row_index > max_row:
                max_row = cell.row_index
            if cell.column_index > max_col:
                max_col = cell.column_index

        if max_row == 0 and max_col == 0:
            return block

        headers = [grid.get((0, col), "") for col in range(max_col + 1)]

        for row in range(1, max_row + 1):
            cells = [grid.get((row, col), "") for col in range(max_col + 1)]
            if not any(c for c in cells):
                continue

            label = cells[0] if cells else ""
            parts = []
            for col in range(1, max_col + 1):
                val    = grid.get((row, col), "")
                header = headers[col] if col < len(headers) else ""
                if val:
                    parts.append(f"{header}: {val}" if header else val)

            if parts:
                text = f"{label}: {' | '.join(parts)}" if label else " | ".join(parts)
            else:
                text = label

            if text.strip():
                line = RawLine()
                line.words.append(RawWord(text=text))
                block.lines.append(line)

        return block

    @staticmethod
    def _page_of(bounding_regions) -> int:
        if bounding_regions:
            return bounding_regions[0].page_number
        return 1