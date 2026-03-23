from typing import Protocol
from pathlib import Path
from DocuMind.documents.raw.raw_document import RawDocument


class DocumentReader(Protocol):
    def read(self, path: Path) -> RawDocument:
        ...