"""Deterministic query rewriting. Never silent: every rewrite yields a :class:`RewriteRecord`.

Strategies (applied in order, each recorded only if it changed something):

* ``whitespace``          collapse/trim whitespace
* ``strip_formatting``    remove Markdown emphasis/backticks/quote markers (content kept)
* ``acronym_expansion``   append configured expansions (``k8s`` → ``k8s kubernetes``)
* ``synonyms``            build *variants* by substituting configured synonyms; variants
                          are only used with ``multi_query`` (retrieved separately, then fused)

Whether rewriting helps is an empirical question: evaluate it with
``searchforge compare`` before enabling it in a profile.
"""

from __future__ import annotations

import re

from searchforge.config.models import RewriteConfig
from searchforge.domain.query import RewriteRecord

_FORMATTING = re.compile(r"(\*\*|__|`+|~~|^>\s*|^#+\s*)", re.MULTILINE)
_WORD = re.compile(r"\w[\w\-.]*", re.UNICODE)


class QueryRewriter:
    def __init__(self, config: RewriteConfig) -> None:
        self.config = config
        self._acronyms = {k.lower(): v for k, v in config.acronyms.items()}
        self._synonyms = {k.lower(): list(v) for k, v in config.synonyms.items()}

    def rewrite(self, text: str) -> RewriteRecord:
        strategies: list[str] = []
        current = " ".join(text.split())
        if current != text:
            strategies.append("whitespace")
        if self.config.enabled:
            stripped = " ".join(_FORMATTING.sub("", current).split())
            if stripped and stripped != current:
                current = stripped
                strategies.append("strip_formatting")
            words = [w.lower() for w in _WORD.findall(current)]
            expansions = [self._acronyms[w] for w in dict.fromkeys(words) if w in self._acronyms]
            expansions = [e for e in expansions if e.lower() not in current.lower()]
            if expansions:
                current = current + " " + " ".join(expansions)
                strategies.append("acronym_expansion")
        variants: list[str] = []
        if self.config.enabled and self.config.multi_query and self._synonyms:
            lowered = current.lower()
            for term, alternatives in self._synonyms.items():
                if re.search(rf"\b{re.escape(term)}\b", lowered):
                    for alt in alternatives:
                        variant = re.sub(rf"\b{re.escape(term)}\b", alt, current, flags=re.IGNORECASE)
                        if variant != current and variant not in variants:
                            variants.append(variant)
                if len(variants) >= self.config.max_variants:
                    break
            if variants:
                strategies.append("synonyms")
        return RewriteRecord(
            original=text,
            rewritten=current,
            strategies=tuple(strategies),
            variants=tuple(variants[: self.config.max_variants]),
        )
