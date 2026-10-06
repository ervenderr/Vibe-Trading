"""Sanitise untrusted external text (headlines, token symbols, risk names).

Anything fetched from the network may be attacker-controlled and is shown to an
agent or a terminal. Strip control, ANSI and invisible formatting characters,
collapse whitespace, and cap the length.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-Z\\-_]|\][^\x07\x1b]*(?:\x07|\x1b\\)?)")
_SPACE_RE = re.compile(r"\s+")
_DROP_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co"})


def sanitize_text(value: Any, max_len: int) -> str:
    """Return a single-line, printable, length-capped string ('' for None)."""
    if value is None:
        return ""
    text = _ANSI_RE.sub("", str(value))
    text = _SPACE_RE.sub(" ", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) not in _DROP_CATEGORIES)
    return text.strip()[:max_len]
