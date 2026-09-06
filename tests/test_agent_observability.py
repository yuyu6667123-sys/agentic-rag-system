"""Structured logging tests for Agent request observability."""

from __future__ import annotations

from contextlib import contextmanager
from io import StringIO
import logging
import unittest

from src.agent.agent import Agent
from test_agent_request_state import _StageAwareAgent, _StubAgent


@contextmanager
def _captured_agent_logs():
    logger = logging.getLogger("src.agent.agent")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield stream
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


class _ErrorAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def _call_model(self, prompt: str) -> str:
        self.calls += 1
        if self.calls == 1:
            return '{"need_knowledge_base": false}'
        raise RuntimeError("simulated model failure")


class AgentObservabilityTests(unittest.TestCase):
    def test_rag_rewrite_logs_all_stage_summaries(self) -> None:
        agent = _StageAwareAgent(sufficient=False)
        with _captured_agent_logs() as captured:
            agent.answer("什么是 RAG？")

        state = agent.get_request_state()
        logs = captured.getvalue()
        for event in (
            "request_started",
            "router_decision",
            "tool_call",
            "retrieval",
            "relevance_decision",
            "retrieval_reliability",
            "query_rewrite",
            "answer_strategy",
            "sources",
            "request_finished",
        ):
            self.assertIn(f"event={event}", logs)

        agent_lines = [line for line in logs.splitlines() if line.startswith("[Agent]")]
        self.assertTrue(agent_lines)
        self.assertTrue(
            all(f"request_id={state['request_id']}" in line for line in agent_lines)
        )
        self.assertIn('call_count=2', logs)
        self.assertIn('best_distance=0.2', logs)
        self.assertIn('query="RAG 的核心原理"', logs)
        self.assertIn('strategy="insufficient"', logs)
        self.assertIn('sources=["rag.txt"]', logs)
        self.assertIn('outcome="completed"', logs)

        self.assertNotIn("RAG combines retrieval with generation", logs)
        self.assertNotIn("知识库检索证据", logs)
        self.assertNotIn("thinking", logs.casefold())

    def test_error_request_logs_id_and_elapsed_time(self) -> None:
        agent = _ErrorAgent()
        with _captured_agent_logs() as captured:
            with self.assertRaises(RuntimeError):
                agent.answer("普通问题")

        state = agent.get_request_state()
        logs = captured.getvalue()
        self.assertIn(f"request_id={state['request_id']}", logs)
        self.assertIn('event=request_finished', logs)
        self.assertIn('outcome="error"', logs)
        self.assertIsNotNone(state["total_time_ms"])
        self.assertIn(f"total_time_ms={state['total_time_ms']}", logs)

    def test_multiple_requests_and_streaming_use_distinct_ids(self) -> None:
        agent = _StubAgent(route_to_rag=False)
        with _captured_agent_logs() as captured:
            agent.answer("first")
            first_id = agent.get_request_state()["request_id"]
            list(agent.stream_answer("second"))
            second_id = agent.get_request_state()["request_id"]

        logs = captured.getvalue()
        self.assertNotEqual(first_id, second_id)
        self.assertIn(f"request_id={first_id}", logs)
        self.assertIn(f"request_id={second_id}", logs)
        self.assertEqual(logs.count("event=request_started"), 2)
        self.assertEqual(logs.count("event=request_finished"), 2)


if __name__ == "__main__":
    unittest.main()
