from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class ElementType(Enum):
    PARAGRAPH = "paragraph"
    IMAGE = "image"
    TABLE = "table"
    HEADER = "header"
    FOOTER = "footer"
    KEY_VALUE = "key_value"
    TEXT = "text"


@dataclass
class BoundingBox:
    x0: float = 0.0
    top: float = 0.0
    x1: float = 0.0
    bottom: float = 0.0


@dataclass
class Element:
    element_type: ElementType = ElementType.TEXT
    bounds: BoundingBox = field(default_factory=BoundingBox)
    text: str = ""


@dataclass
class Page:
    page_number: int = 0
    elements: list[Element] = field(default_factory=list)


@dataclass
class Document:
    source_path: str = ""
    pages: list[Page] = field(default_factory=list)
