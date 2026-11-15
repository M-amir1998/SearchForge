"""Deterministic, CPU-cheap query analysis and rules-based classification.

Features: tokens, quoted phrases, identifier detection (error codes, ``snake_case``,
``camelCase``, ``Dotted.paths()``, ticket keys), numbers, question words, stopword
ratio, and ``field:value`` inline filters for *whitelisted* filterable fields.

Classification rules (first match wins; thresholds are configurable):

1. ``IDENTIFIER``        identifiers present and ≤ ``identifier_max_tokens`` tokens
2. ``FILTER_HEAVY``      ≥ 1 filter (inline or explicit) and ≤ ``filter_residual_max_tokens`` tokens
3. ``KEYWORD``           ≤ ``keyword_max_tokens`` tokens and no question word, or a short quoted query
4. ``COMPLEX_SEMANTIC``  ≥ ``complex_min_tokens`` tokens
5. ``NATURAL_LANGUAGE``  everything else
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from searchforge.domain.query import QueryAnalysis, QueryClass
from searchforge.parsing.stopwords import STOPWORDS

_QUOTED = re.compile(r'"([^"]{1,200})"')
_INLINE_FILTER = re.compile(r'\b([a-z][a-z0-9_]{0,63}):("([^"]{1,100})"|[^\s"]{1,100})')
_TOKEN = re.compile(r"[\w.\-/()]+", re.UNICODE)
_NUMBER = re.compile(r"^\d+(?:[.,]\d+)?$")
_IDENTIFIER_PATTERNS = (
    re.compile(r"^[A-Za-z]+[A-Za-z0-9]*[-_][A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*$"),  # ERR-4012, max_attempts
    re.compile(r"^[a-z]+[A-Z][A-Za-z0-9]*$"),  # camelCase
    re.compile(r"^[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*$"),  # PascalCase
    re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+(?:\(\))?$"),  # Dotted.path()
    re.compile(r"^[A-Za-z]{2,}\d+[A-Za-z0-9]*$"),  # http2, sha256
    re.compile(r"^v?\d+\.\d+(?:\.\d+)+$"),  # 1.4.2
)
QUESTION_WORDS = frozenset(
    {"how", "why", "what", "when", "where", "which", "who", "can", "should", "does", "do", "is", "are"}
)


@dataclass(frozen=True, slots=True)
class AnalyzerThresholds:
    identifier_max_tokens: int = 4
    filter_residual_max_tokens: int = 4
    keyword_max_tokens: int = 3
    complex_min_tokens: int = 9


def is_identifier(token: str) -> bool:
    t = token.strip(".,;:!?()[]{}'\"`")
    if len(t) < 2 or _NUMBER.match(t):
        return False
    if _IDENTIFIER_PATTERNS[0].match(t):
        # Hyphenated English ("rate-limit") is not an identifier; codes carry a digit,
        # an underscore or capitals ("ERR-4012", "max_attempts", "X-Request-Id").
        return any(ch.isdigit() or ch == "_" or ch.isupper() for ch in t)
    return any(p.match(t) for p in _IDENTIFIER_PATTERNS[1:])


class QueryAnalyzer:
    def __init__(
        self,
        filterable_fields: Iterable[str] = (),
        *,
        extract_inline_filters: bool = True,
        thresholds: AnalyzerThresholds | None = None,
    ) -> None:
        self.filterable = frozenset(filterable_fields)
        self.extract_inline_filters = extract_inline_filters
        self.t = thresholds or AnalyzerThresholds()

    def _extract_filters(self, text: str) -> tuple[dict[str, str], str]:
        if not self.extract_inline_filters or not self.filterable:
            return {}, text
        found: dict[str, str] = {}

        def repl(m: re.Match[str]) -> str:
            field = m.group(1)
            if field not in self.filterable:
                return m.group(0)
            found[field] = m.group(3) if m.group(3) is not None else m.group(2)
            return " "

        residual = _INLINE_FILTER.sub(repl, text)
        return found, " ".join(residual.split())

    def analyze(self, text: str, explicit_filter_count: int = 0) -> QueryAnalysis:
        extracted, residual = self._extract_filters(text)
        phrases = tuple(p.strip() for p in _QUOTED.findall(residual) if p.strip())
        raw_tokens = _TOKEN.findall(residual)
        tokens = tuple(t.strip(".,;:!?()[]{}'\"`").lower() for t in raw_tokens if t.strip(".,;:!?()[]{}'\"`"))
        identifiers = tuple(dict.fromkeys(t.strip(".,;:!?[]{}'\"`") for t in raw_tokens if is_identifier(t)))
        numbers = tuple(t for t in tokens if _NUMBER.match(t))
        has_q = bool(tokens) and (tokens[0] in QUESTION_WORDS or residual.rstrip().endswith("?"))
        stop_ratio = sum(t in STOPWORDS for t in tokens) / len(tokens) if tokens else 0.0
        n = len(tokens)
        n_filters = len(extracted) + explicit_filter_count
        if identifiers and n <= self.t.identifier_max_tokens and not has_q:
            qclass = QueryClass.IDENTIFIER
        elif n_filters >= 1 and n <= self.t.filter_residual_max_tokens:
            qclass = QueryClass.FILTER_HEAVY
        elif (n <= self.t.keyword_max_tokens and not has_q) or (phrases and n <= self.t.keyword_max_tokens + 2):
            qclass = QueryClass.KEYWORD
        elif n >= self.t.complex_min_tokens:
            qclass = QueryClass.COMPLEX_SEMANTIC
        else:
            qclass = QueryClass.NATURAL_LANGUAGE
        return QueryAnalysis(
            tokens=tokens,
            token_count=n,
            char_length=len(residual),
            quoted_phrases=phrases,
            identifiers=identifiers,
            numbers=numbers,
            has_question_word=has_q,
            stopword_ratio=stop_ratio,
            extracted_filters=extracted,
            residual_text=residual,
            query_class=qclass,
        )
