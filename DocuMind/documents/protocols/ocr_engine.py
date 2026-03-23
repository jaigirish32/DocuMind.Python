from typing import Protocol
from pathlib import Path


class OcrEngine(Protocol):
    def extract_text(self, path: Path) -> str:
        ...