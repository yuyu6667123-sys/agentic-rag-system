"""Run real end-to-end Agent evaluation cases.

This module deliberately uses the production Agent without monkeypatching its
model, Tool Registry, or Chroma dependencies. Each case gets a fresh Agent so
short-term Memory and request state cannot leak between cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable

from src.agent.agent import Agent
from src.tools.rag_tool import _get_vector_store


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationCase:
    """One real Agent evaluation request and its expected behavior."""

    case_id: str
    category: str
    question: str
    expected: dict[str, object]


@dataclass
class CaseResult:
    """Serializable result for one evaluation case."""

    case_id: str
    category: str
    question: str
    passed: bool
    elapsed_ms: float
    state: dict[str, object] | None = None
    answer_preview: str = ""
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "question": self.question,
            "passed": self.passed,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "state": self.state,
            "answer_preview": self.answer_preview,
            "failure_reason": self.failure_reason,
        }


@dataclass
class EvaluationReport:
    """Aggregate evaluation output."""

    results: list[CaseResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(result.passed for result in self.results)

    @property
    def failed_count(self) -> int:
        return len(self.results) - self.passed_count

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": {
                "total": len(self.results),
                "passed": self.passed_count,
                "failed": self.failed_count,
            },
            "results": [result.to_dict() for result in self.results],
            "notes": self.notes,
        }


def default_cases() -> list[EvaluationCase]:
    """Return deterministic, knowledge-base-specific real evaluation cases."""
    return [
        EvaluationCase(
            "router_plain",
            "router",
            "你好，请介绍一下你自己。",
            {"routing_decision": "llm_false", "tool_call_count": 0},
        ),
        EvaluationCase(
            "router_knowledge",
            "router",
            "什么是 RAG？",
            {"routing_prefix": "llm_true", "tool_call_count_min": 1},
        ),
        EvaluationCase(
            "retrieval_high",
            "retrieval",
            "请根据知识库说明什么是 RAG，以及它为什么能减少幻觉。",
            {"tool_call_count_min": 1},
        ),
        EvaluationCase(
            "retrieval_ambiguous",
            "retrieval",
            "请根据知识库介绍一下这个系统。",
            {"tool_call_count_min": 1},
        ),
        EvaluationCase(
            "retrieval_irrelevant",
            "retrieval",
            "知识库中有没有关于量子计算和黑洞的内容？",
            {"tool_call_count_min": 1},
        ),
        EvaluationCase(
            "rewrite",
            "query_rewrite",
            "知识库中是否有关于量子计算和黑洞的内容？",
            {"tool_call_count": 2, "query_rewrite_nonempty": True},
        ),
    ]


def _short_answer(answer: str, limit: int = 240) -> str:
    compact = " ".join(answer.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def _validate_state(state: dict[str, object], expected: dict[str, object]) -> list[str]:
    failures: list[str] = []
    routing = state.get("routing_decision")
    if "routing_decision" in expected and routing != expected["routing_decision"]:
        failures.append(
            f"routing_decision 期望 {expected['routing_decision']!r}，实际 {routing!r}"
        )
    if "routing_prefix" in expected and not str(routing).startswith(
        str(expected["routing_prefix"])
    ):
        failures.append(
            f"routing_decision 应以 {expected['routing_prefix']!r} 开头，实际 {routing!r}"
        )
    tool_count = state.get("tool_call_count")
    if "tool_call_count" in expected and tool_count != expected["tool_call_count"]:
        failures.append(
            f"tool_call_count 期望 {expected['tool_call_count']}，实际 {tool_count}"
        )
    if "tool_call_count_min" in expected and (
        not isinstance(tool_count, int) or tool_count < expected["tool_call_count_min"]
    ):
        failures.append(
            f"tool_call_count 应 >= {expected['tool_call_count_min']}，实际 {tool_count}"
        )
    if "query_rewrite_nonempty" in expected and expected["query_rewrite_nonempty"]:
        rewrite = state.get("query_rewrite")
        if not isinstance(rewrite, str) or not rewrite.strip():
            failures.append("query_rewrite 应为非空字符串")
    return failures


def run_case(case: EvaluationCase) -> CaseResult:
    """Execute one case against real Ollama and Chroma."""
    started = time.perf_counter()
    agent = Agent()
    try:
        answer = agent.answer(case.question)
        state = agent.get_request_state()
        if state is None:
            raise RuntimeError("Agent 未生成 AgentRequestState")
        failures = _validate_state(state, case.expected)
        return CaseResult(
            case_id=case.case_id,
            category=case.category,
            question=case.question,
            passed=not failures,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            state=state,
            answer_preview=_short_answer(answer),
            failure_reason="；".join(failures) if failures else None,
        )
    except Exception as exc:
        return CaseResult(
            case_id=case.case_id,
            category=case.category,
            question=case.question,
            passed=False,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            state=agent.get_request_state(),
            failure_reason=f"{type(exc).__name__}: {exc}",
        )


def run_router_evaluation() -> EvaluationReport:
    return _run_selected({"router"})


def run_retrieval_evaluation() -> EvaluationReport:
    return _run_selected({"retrieval"})


def run_rewrite_evaluation() -> EvaluationReport:
    return _run_selected({"query_rewrite"})


def run_memory_evaluation() -> EvaluationReport:
    """Run a two-turn pronoun test using one real Agent and one Memory."""
    started = time.perf_counter()
    agent = Agent()
    first_question = "什么是 RAG？"
    second_question = "它有什么优点？"
    try:
        first_answer = agent.answer(first_question)
        second_answer = agent.answer(second_question)
        history = agent.get_history()
        state = agent.get_request_state()
        failures: list[str] = []
        if len(history) != 4:
            failures.append(f"两轮对话历史应有 4 条消息，实际 {len(history)}")
        if [item.get("role") for item in history] != [
            "user",
            "assistant",
            "user",
            "assistant",
        ]:
            failures.append("历史角色顺序不是 user/assistant/user/assistant")
        if not second_answer.strip():
            failures.append("第二轮回答为空")
        if state is None:
            failures.append("第二轮未生成 AgentRequestState")
        result = CaseResult(
            case_id="memory_pronoun",
            category="memory",
            question=f"第一轮：{first_question}；第二轮：{second_question}",
            passed=not failures,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            state=state,
            answer_preview=_short_answer(second_answer),
            failure_reason="；".join(failures) if failures else None,
        )
        LOGGER.info(
            "Memory case completed: first=%s second=%s history_messages=%d",
            _short_answer(first_answer, 80),
            _short_answer(second_answer, 80),
            len(history),
        )
        return EvaluationReport(results=[result])
    except Exception as exc:
        return EvaluationReport(
            results=[
                CaseResult(
                    case_id="memory_pronoun",
                    category="memory",
                    question=f"第一轮：{first_question}；第二轮：{second_question}",
                    passed=False,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    state=agent.get_request_state(),
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
            ]
        )


def _run_selected(categories: set[str]) -> EvaluationReport:
    report = EvaluationReport()
    for case in default_cases():
        if case.category not in categories:
            continue
        result = run_case(case)
        report.results.append(result)
        LOGGER.info(
            "Evaluation case=%s passed=%s routing=%s tool_calls=%s",
            result.case_id,
            result.passed,
            (result.state or {}).get("routing_decision"),
            (result.state or {}).get("tool_call_count"),
        )
    return report


def run_all_evaluations() -> EvaluationReport:
    """Run router, retrieval, rewrite, and memory evaluations."""
    report = EvaluationReport()
    for partial in (
        _run_selected({"router"}),
        _run_selected({"retrieval"}),
        _run_selected({"query_rewrite"}),
        run_memory_evaluation(),
    ):
        report.results.extend(partial.results)
        report.notes.extend(partial.notes)
    return report


def verify_dependencies() -> list[str]:
    """Check real service prerequisites without changing project state."""
    notes: list[str] = []
    try:
        store = _get_vector_store()
        collection = getattr(store, "_collection", None)
        count = collection.count() if collection is not None else None
        notes.append(f"Chroma 初始化成功，collection_count={count}")
    except Exception as exc:
        notes.append(f"Chroma 初始化失败: {type(exc).__name__}: {exc}")
    return notes


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    report = EvaluationReport(notes=verify_dependencies())
    report.results.extend(run_all_evaluations().results)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
