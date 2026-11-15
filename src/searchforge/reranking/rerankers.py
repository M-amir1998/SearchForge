"""Second-stage rerankers. All implement :class:`searchforge.ports.Reranker`.

Contract: input is the top-N hydrated candidates in first-stage order; output is
the *same set* of candidates, sorted by the new score (descending), with
``signals.reranker_score``/``reranker_rank`` populated. Rerankers never add or
drop candidates (contract-tested in tests/unit).

* :class:`NoOpReranker`            identity (baseline for marginal-value measurement)
* :class:`LexicalFeatureReranker`  CPU-cheap hand-built features: query-term
  coverage, title/section coverage, exact phrase, exact identifier match, term
  proximity, plus a rank prior from the first stage. Interpretable and fast.
* :class:`CrossEncoderReranker`    optional Sentence Transformers cross-encoder
  (``searchforge[neural]``), CPU, top-N only, no download unless allowed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from searchforge.domain.result import RankingSignals, ScoredChunk
from searchforge.errors import RerankerFailure
from searchforge.parsing.stopwords import STOPWORDS
from searchforge.query.analysis import is_identifier

_WORD = re.compile(r"\w+", re.UNICODE)

DEFAULT_FEATURE_WEIGHTS: dict[str, float] = {
    "prior": 1.0,
    "coverage": 1.0,
    "title_coverage": 0.4,
    "phrase": 0.5,
    "identifier": 1.0,
    "proximity": 0.4,
}


def _stem(token: str) -> str:
    """Very light suffix stripping, sufficient for coverage features."""
    for suffix in ("ingly", "edly", "ing", "ies", "ied", "es", "ed", "ly", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)] + ("y" if suffix in ("ies", "ied") else "")
    return token


def _terms(text: str) -> list[str]:
    return [_stem(t) for t in _WORD.findall(text.lower()) if t not in STOPWORDS]


def _with_rerank(candidates: Sequence[ScoredChunk], scores: Sequence[float]) -> list[ScoredChunk]:
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
    out: list[ScoredChunk] = []
    for new_rank, i in enumerate(order, start=1):
        c = candidates[i]
        sig = replace(c.signals) if isinstance(c.signals, RankingSignals) else RankingSignals()
        sig.reranker_score = float(scores[i])
        sig.reranker_rank = new_rank
        out.append(replace(c, score=float(scores[i]), signals=sig))
    return out


class NoOpReranker:
    name = "none"

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        return list(candidates)


class LexicalFeatureReranker:
    name = "lexical_features"

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        self.weights = {**DEFAULT_FEATURE_WEIGHTS, **(weights or {})}
        unknown = set(self.weights) - set(DEFAULT_FEATURE_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown reranker features: {sorted(unknown)}")

    def features(self, query: str, candidate: ScoredChunk, position: int) -> dict[str, float]:
        q_terms = list(dict.fromkeys(_terms(query)))
        text = candidate.text
        body_tokens = _terms(text)
        body_set = set(body_tokens)
        title = " ".join(str(candidate.metadata.get(k) or "") for k in ("title", "section"))
        title_set = set(_terms(title))
        n = len(q_terms) or 1
        coverage = sum(t in body_set for t in q_terms) / n
        title_cov = sum(t in title_set for t in q_terms) / n
        lowered = " ".join(text.lower().split())
        q_norm = " ".join(_WORD.findall(query.lower()))
        phrase = 1.0 if len(q_terms) >= 2 and q_norm and q_norm in " ".join(_WORD.findall(lowered)) else 0.0
        idents = [t.strip(".,;:!?()[]{}'\"`") for t in query.split() if is_identifier(t)]
        ident = (sum(i.lower() in lowered for i in idents) / len(idents)) if idents else 0.0
        proximity = 0.0
        matched = [t for t in q_terms if t in body_set]
        if len(matched) >= 2:
            positions: dict[str, list[int]] = {}
            for i, tok in enumerate(body_tokens):
                if tok in matched:
                    positions.setdefault(tok, []).append(i)
            proximity = self._min_window(positions, len(matched))
        return {
            "prior": 1.0 / (1.0 + position),
            "coverage": coverage,
            "title_coverage": title_cov,
            "phrase": phrase,
            "identifier": ident,
            "proximity": proximity,
        }

    @staticmethod
    def _min_window(positions: Mapping[str, list[int]], needed: int) -> float:
        events = sorted((p, t) for t, ps in positions.items() for p in ps)
        counts: dict[str, int] = {}
        best = None
        left = 0
        have = 0
        for right, (pos, term) in enumerate(events):
            counts[term] = counts.get(term, 0) + 1
            if counts[term] == 1:
                have += 1
            while have == needed:
                width = pos - events[left][0] + 1
                best = width if best is None else min(best, width)
                lt = events[left][1]
                counts[lt] -= 1
                if counts[lt] == 0:
                    have -= 1
                left += 1
            _ = right
        if best is None:
            return 0.0
        return needed / best

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        scores = []
        for pos, cand in enumerate(candidates):
            feats = self.features(query, cand, pos)
            scores.append(sum(self.weights[k] * v for k, v in feats.items()))
        return _with_rerank(candidates, scores)


class CrossEncoderReranker:
    """Sentence-Transformers ``CrossEncoder`` on CPU over the top-N candidates."""

    name = "cross_encoder"

    def __init__(
        self,
        model: str,
        revision: str | None = None,
        *,
        batch_size: int = 32,
        allow_download: bool = False,
        model_obj: Any | None = None,
        max_chars: int = 2000,
    ) -> None:
        self.model_name = model
        self.batch_size = batch_size
        self.max_chars = max_chars
        if model_obj is not None:
            self._model = model_obj
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise RerankerFailure("sentence-transformers is not installed; install searchforge[neural]") from exc
        try:
            self._model = CrossEncoder(model, revision=revision, device="cpu", local_files_only=not allow_download)
        except Exception as exc:
            raise RerankerFailure(f"failed to load cross-encoder {model!r}: {exc}") from exc

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, c.text[: self.max_chars]) for c in candidates]
        try:
            raw = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        except Exception as exc:
            raise RerankerFailure(f"cross-encoder inference failed: {exc}") from exc
        return _with_rerank(candidates, [float(s) for s in raw])
