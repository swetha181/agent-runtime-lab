"""Minimal tracing: nested spans per agent run, exported as JSON lines.

Each run gets a trace ID. Spans record name, parent, duration and attributes
(tool name, tokens, stop reason, errors), so a failed run can be replayed and
explained step by step.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

_current_span: ContextVar["Span | None"] = ContextVar("current_span", default=None)


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start: float
    duration_ms: float | None = None
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)

    def set(self, **attrs: Any) -> None:
        self.attributes.update(attrs)


class Tracer:
    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.spans: list[Span] = []

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        parent = _current_span.get()
        span = Span(
            name=name,
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:8],
            parent_id=parent.span_id if parent else None,
            start=time.perf_counter(),
            attributes=dict(attrs),
        )
        token = _current_span.set(span)
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.set(error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            span.duration_ms = round((time.perf_counter() - span.start) * 1000, 2)
            _current_span.reset(token)
            self.spans.append(span)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [asdict(s) for s in sorted(self.spans, key=lambda s: s.start)]

    def export_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for span in self.to_dicts():
                fh.write(json.dumps(span, default=str) + "\n")
