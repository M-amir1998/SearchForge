"""Conservative text normalization.

Normalization must not destroy structure that search depends on: newlines,
Markdown headings, code identifiers, punctuation inside identifiers and casing are
preserved. We only:

* apply Unicode normalization (NFC by default — NFKC is opt-in because it folds
  characters such as ``ﬁ`` or full-width digits, which can change identifiers),
* convert CRLF/CR to LF,
* drop C0/C1 control characters except ``\\n`` and ``\\t`` (and zero-width/BOM chars),
* collapse runs of horizontal whitespace and trailing spaces,
* cap consecutive blank lines.
"""

from __future__ import annotations

import re
import unicodedata

from searchforge.config.models import NormalizationConfig

#: Bump when normalization output changes; it participates in embedding cache keys.
NORMALIZATION_VERSION = "norm-v1"

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b\u200c\u200d\u2060\ufeff]")
_HSPACE = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
_TRAILING = re.compile(r"[ \t]+\n")


def normalize_text(text: str, config: NormalizationConfig | None = None) -> str:
    cfg = config or NormalizationConfig()
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    if cfg.unicode_form != "none":
        out = unicodedata.normalize(cfg.unicode_form, out)
    if cfg.strip_control_chars:
        out = _CONTROL.sub("", out)
    if cfg.collapse_whitespace:
        out = _HSPACE.sub(" ", out)
        out = _TRAILING.sub("\n", out)
        out = re.sub(r"\n{" + str(cfg.max_blank_lines + 2) + r",}", "\n" * (cfg.max_blank_lines + 1), out)
    return out.strip()


def normalize_query_text(text: str) -> str:
    """Query-side normalization: NFC, controls removed, single-spaced, trimmed."""
    out = unicodedata.normalize("NFC", text)
    out = _CONTROL.sub("", out)
    return " ".join(out.split())
