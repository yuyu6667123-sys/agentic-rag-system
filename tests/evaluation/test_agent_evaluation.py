"""Real integration tests for Router, Retrieval, Rewrite, and Memory.

Run explicitly with ``python -m unittest tests.evaluation.test_agent_evaluation``.
These tests intentionally require a running Ollama with the configured model
and a persisted Chroma database.
"""

from __future__ import annotations

import unittest

from .runner import (
    run_memory_evaluation,
    run_retrieval_evaluation,
    run_router_evaluation,
    run_rewrite_evaluation,
    verify_dependencies,
)


class RealAgentEvaluationTests(unittest.TestCase):
    """End-to-end tests; failures include the state and service error."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dependency_notes = verify_dependencies()
        if any("失败" in note for note in cls.dependency_notes):
            raise unittest.SkipTest("; ".join(cls.dependency_notes))

    @staticmethod
    def _assert_report_passed(report) -> None:
        failures = [
            f"{item.case_id}: {item.failure_reason}; state={item.state!r}"
            for item in report.results
            if not item.passed
        ]
        if failures:
            raise AssertionError("\n".join(failures))

    def test_router_cases_and_accuracy(self) -> None:
        report = run_router_evaluation()
        self._assert_report_passed(report)
        self.assertEqual(report.passed_count / len(report.results), 1.0)

    def test_retrieval_quality_cases(self) -> None:
        report = run_retrieval_evaluation()
        self._assert_report_passed(report)
        for result in report.results:
            self.assertIsNotNone(result.state)
            self.assertIn("retrieval_result_count", result.state)
            self.assertIn("best_distance", result.state)
            self.assertIn("retrieval_reliable", result.state)
            self.assertIn("answer_strategy", result.state)

    def test_query_rewrite_second_tool_call(self) -> None:
        report = run_rewrite_evaluation()
        self._assert_report_passed(report)
        state = report.results[0].state
        self.assertEqual(state["tool_call_count"], 2)
        self.assertTrue(state["query_rewrite"])

    def test_memory_pronoun_context(self) -> None:
        report = run_memory_evaluation()
        self._assert_report_passed(report)
        state = report.results[0].state
        self.assertEqual(state["tool_call_count"] in {0, 1, 2}, True)


if __name__ == "__main__":
    unittest.main()

