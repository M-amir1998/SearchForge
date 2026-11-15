"""Optional OpenTelemetry spans (``pip install searchforge[observability]``).

Without the extra, :func:`span` is a zero-cost no-op. Span attributes never
include raw query text — only the query hash and low-cardinality values.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

try:  # pragma: no cover - depends on optional extra
    from opentelemetry import trace as _otel_trace

    _TRACER: Any = _otel_trace.get_tracer("searchforge")
except ImportError:  # pragma: no cover
    _TRACER = None

_ALLOWED_TYPES = (str, bool, int, float)


def tracing_available() -> bool:
    return _TRACER is not None


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    if _TRACER is None:
        yield
        return
    with _TRACER.start_as_current_span(f"searchforge.{name}") as sp:
        for key, value in attributes.items():
            if isinstance(value, _ALLOWED_TYPES):
                sp.set_attribute(f"searchforge.{key}", value)
        yield
