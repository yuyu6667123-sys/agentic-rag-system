"""Request-state regression tests without external model or Chroma calls."""

from __future__ import annotations

import unittest

from src.agent.agent import Agent


EVIDENCE = {
    "query": "RAG",
    "results": [
        {
            "content": "RAG combines retrieval with generation.",
            "source": "rag.txt",
            "page": None,
            "distance": 0.2,
        }
    ],
}


class _Registry:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def execute(self, name: str, **tool_input: object) -> dict:
        self.queries.append(str(tool_input["query"]))
        result = {"query": self.queries[-1], "results": []}
        result["results"] = [item.copy() for item in EVIDENCE["results"]]
        return result


class _InspectingRegistry(_Registry):
    def __init__(self) -> None:
        super().__init__()
        self.agent: Agent | None = None
        self.states_during_calls: list[dict[str, object]] = []

    def execute(self, name: str, **tool_input: object) -> dict:
        assert self.agent is not None
        state = self.agent.get_request_state()
        assert state is not None
        self.states_during_calls.append(state)
        return super().execute(name, **tool_input)


class _StubAgent(Agent):
    def __init__(
        self,
        *,
        route_to_rag: bool,
        reliable: bool = True,
        sufficient: bool = True,
    ) -> None:
        self.registry = _Registry()
        super().__init__(tool_registry=self.registry)
        self.route_to_rag = route_to_rag
        self.reliable = reliable
        self.sufficient = sufficient

    def _should_use_knowledge_base(self, question: str) -> bool:
        self.last_routing_decision = "llm_true" if self.route_to_rag else "llm_false"
        return self.route_to_rag

    def evaluate_retrieval_quality(self, question: str, evidence: dict) -> bool:
        self.last_result_count = len(evidence["results"])
        self.last_best_distance = 0.2
        self.last_relevance_decision = "llm_true" if self.reliable else "llm_false"
        self.last_retrieval_reliable = self.reliable
        self.last_loop_decision = "reliable" if self.reliable else "unreliable_relevance"
        return self.reliable

    def _check_evidence_sufficiency(self, question: str, evidence: dict) -> bool:
        return self.sufficient

    def _call_model(self, prompt: str) -> str:
        if "Query 改写器" in prompt:
            return '{"query": "RAG 的核心原理"}'
        return "stub answer"

    def _stream_model(self, prompt: str):
        yield "streamed "
        yield "answer"


class _StageAwareAgent(Agent):
    """Use production stage methods while asserting state during execution."""

    def __init__(self, *, sufficient: bool) -> None:
        registry = _InspectingRegistry()
        super().__init__(tool_registry=registry)
        registry.agent = self
        self.registry = registry
        self.sufficient = sufficient

    def _call_model(self, prompt: str) -> str:
        state = self.get_request_state()
        assert state is not None
        if "路由判断器" in prompt:
            return '{"need_knowledge_base": true}'
        if "检索相关性判断器" in prompt:
            assert state["tool_call_count"] == 1
            assert state["retrieval_result_count"] == 1
            assert state["best_distance"] == 0.2
            return '{"relevant": true}'
        if "证据充分性判断器" in prompt:
            assert state["retrieval_reliable"] is True
            assert state["relevance_decision"] == "llm_true"
            return f'{{"sufficient": {str(self.sufficient).lower()}}}'
        if "Query 改写器" in prompt:
            assert state["answer_strategy"] == "insufficient"
            return '{"query": "RAG 的核心原理"}'
        assert state["sources"] == ["rag.txt"]
        return "stage-aware answer"

    def _stream_model(self, prompt: str):
        state = self.get_request_state()
        assert state is not None
        assert state["answer_strategy"] == "reliable"
        assert state["sources"] == ["rag.txt"]
        yield "streamed "
        yield "answer"


class AgentRequestStateTests(unittest.TestCase):
    def test_plain_requests_have_independent_state_and_memory(self) -> None:
        agent = _StubAgent(route_to_rag=False)
        agent.answer("你好")
        first = agent.request_state
        first_snapshot = agent.get_request_state()
        agent.answer("再介绍一下")
        second = agent.request_state
        second_snapshot = agent.get_request_state()

        self.assertIsNot(first, second)
        self.assertNotEqual(first_snapshot["request_id"], second_snapshot["request_id"])
        self.assertEqual(first_snapshot["question"], "你好")
        self.assertEqual(second_snapshot["question"], "再介绍一下")
        self.assertEqual(second_snapshot["routing_decision"], "llm_false")
        self.assertEqual(second_snapshot["tool_call_count"], 0)
        self.assertEqual(second_snapshot["answer_strategy"], "not_needed")
        self.assertEqual(len(agent.get_history()), 4)

    def test_reliable_rag_state(self) -> None:
        agent = _StubAgent(route_to_rag=True)
        answer = agent.answer("什么是 RAG？")
        state = agent.get_request_state()

        self.assertIn("参考来源", answer)
        self.assertEqual(state["routing_decision"], "llm_true")
        self.assertEqual(state["tool_call_count"], 1)
        self.assertEqual(state["retrieval_result_count"], 1)
        self.assertEqual(state["best_distance"], 0.2)
        self.assertTrue(state["retrieval_reliable"])
        self.assertEqual(state["relevance_decision"], "llm_true")
        self.assertEqual(state["answer_strategy"], "reliable")
        self.assertEqual(state["sources"], ["rag.txt"])
        self.assertGreaterEqual(state["total_time_ms"], 0)

    def test_query_rewrite_state(self) -> None:
        agent = _StubAgent(route_to_rag=True, reliable=False)
        agent.answer("知识库里的未知细节")
        state = agent.get_request_state()

        self.assertEqual(state["tool_call_count"], 2)
        self.assertEqual(state["query_rewrite"], "RAG 的核心原理")
        self.assertEqual(state["answer_strategy"], "cautious")
        self.assertEqual(len(agent.registry.queries), 2)

    def test_streaming_records_state_after_completion(self) -> None:
        agent = _StubAgent(route_to_rag=True)
        chunks = list(agent.stream_answer("什么是 RAG？"))
        state = agent.get_request_state()

        self.assertIn("参考来源", "".join(chunks))
        self.assertEqual(state["tool_call_count"], 1)
        self.assertEqual(state["sources"], ["rag.txt"])
        self.assertIsNotNone(state["total_time_ms"])
        self.assertEqual(len(agent.get_history()), 2)

    def test_rewrite_fields_are_visible_during_each_stage(self) -> None:
        agent = _StageAwareAgent(sufficient=False)
        agent.answer("什么是 RAG？")
        first_call, second_call = agent.registry.states_during_calls
        state = agent.get_request_state()

        self.assertEqual(first_call["routing_decision"], "llm_true")
        self.assertEqual(first_call["tool_call_count"], 1)
        self.assertIsNone(first_call["query_rewrite"])
        self.assertEqual(second_call["tool_call_count"], 2)
        self.assertEqual(second_call["query_rewrite"], "RAG 的核心原理")
        self.assertEqual(state["answer_strategy"], "insufficient")
        self.assertEqual(state["sources"], ["rag.txt"])
        self.assertEqual(state["routing_decision"], agent.last_routing_decision)
        self.assertEqual(state["tool_call_count"], agent.last_tool_call_count)
        self.assertEqual(state["best_distance"], agent.last_best_distance)
        self.assertEqual(
            state["retrieval_reliable"],
            agent.last_retrieval_reliable,
        )

    def test_stream_state_is_populated_before_first_chunk(self) -> None:
        agent = _StageAwareAgent(sufficient=True)
        stream = agent.stream_answer("什么是 RAG？")
        self.assertTrue(next(stream))
        during = agent.get_request_state()

        self.assertEqual(during["routing_decision"], "llm_true")
        self.assertEqual(during["tool_call_count"], 1)
        self.assertEqual(during["retrieval_result_count"], 1)
        self.assertTrue(during["retrieval_reliable"])
        self.assertEqual(during["answer_strategy"], "reliable")
        self.assertEqual(during["sources"], ["rag.txt"])
        self.assertIsNone(during["total_time_ms"])

        list(stream)
        self.assertIsNotNone(agent.get_request_state()["total_time_ms"])


if __name__ == "__main__":
    unittest.main()
