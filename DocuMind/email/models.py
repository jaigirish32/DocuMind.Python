from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class EmailMessage:
    """
    Normalised email message — provider agnostic.
    """
    message_id:  str
    thread_id:   str
    subject:     str
    sender:      str
    recipients:  list[str]
    body:        str
    date:        datetime
    labels:      list[str] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)  # filenames only for now