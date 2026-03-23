from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class BoundingBox:
    x0:     float = 0.0
    top:    float = 0.0
    x1:     float = 0.0
    bottom: float = 0.0


@dataclass
class RawWord:
    text: str = ""
    bounds: BoundingBox = field(default_factory=BoundingBox)
    font_name: str = ""
    font_size: float = 0.0
    is_bold: bool = False
    is_italic: bool = False


@dataclass
class RawLine:
    words: list[RawWord] = field(default_factory=list)
    bounds: BoundingBox = field(default_factory=BoundingBox)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
class RawBlock:
    lines: list[RawLine] = field(default_factory=list)
    bounds: BoundingBox = field(default_factory=BoundingBox)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@dataclass
class RawPage:
    page_number: int = 0
    blocks: list[RawBlock] = field(default_factory=list)


@dataclass
class RawDocument:
    source_path: str = ""
    pages: list[RawPage] = field(default_factory=list)

    def to_plain_text(self) -> str:
        """
        Returns full document as plain text with page markers.
        Python equivalent of C++ toPlainText()
        """
        parts = []
        for page in self.pages:
            parts.append(f"--- Page {page.page_number} ---")
            for block in page.blocks:
                parts.append(block.text)
        return "\n".join(parts)
