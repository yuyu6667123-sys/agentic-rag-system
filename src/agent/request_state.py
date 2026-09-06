"""Unified state for one Agent request."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from uuid import uuid4


@dataclass
class AgentRequestState:
    """Record the key decisions and outcomes of one Agent execution."""

    request_id: str
    question: str
    routing_decision: str | None = None
    tool_call_count: int = 0
    retrieval_result_count: int = 0
    best_distance: float | None = None
    retrieval_reliable: bool = False
    relevance_decision: str | None = None
    query_rewrite: str | None = None
    answer_strategy: str | None = None
    sources: list[str] = field(default_factory=list)
    total_time_ms: float | None = None
    _started_at: float = field(default_factory=time.perf_counter, repr=False)

    @classmethod
    def create(cls, question: str) -> "AgentRequestState":
        """Create an independent state object for a new request."""
        return cls(request_id=str(uuid4()), question=question)

    def finish(self) -> None:
        """Freeze the elapsed execution time when processing ends."""
        if self.total_time_ms is None:
            self.total_time_ms = round(
                (time.perf_counter() - self._started_at) * 1000,
                3,
            )

    def to_dict(self) -> dict[str, object]:
        """Return the public, JSON-serializable request state."""
        data = asdict(self)
        data.pop("_started_at", None)
        return data


__all__ = ["AgentRequestState"]
