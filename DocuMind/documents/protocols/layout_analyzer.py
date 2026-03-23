from typing import Protocol
from DocuMind.documents.raw.raw_document import RawDocument
from DocuMind.documents.models.document import Document


class LayoutAnalyzer(Protocol):
    def analyze(self, raw_doc: RawDocument) -> Document:
        ...