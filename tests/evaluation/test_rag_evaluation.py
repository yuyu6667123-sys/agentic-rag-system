"""Retrieval quality evaluation using real Chroma and Ollama."""

from __future__ import annotations

import unittest
import os

from .runner import run_retrieval_evaluation, verify_dependencies


RUN_REAL = os.getenv("RUN_REAL_EVALUATION", "0").casefold() in {"1", "true", "yes"}


class RagEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not RUN_REAL:
            raise unittest.SkipTest(
                "真实评估默认跳过；设置 RUN_REAL_EVALUATION=1 后运行"
            )
        notes = verify_dependencies()
        if any("失败" in note for note in notes):
            raise unittest.SkipTest("; ".join(notes))

    def test_retrieval_quality_fields(self) -> None:
        report = run_retrieval_evaluation()
        failures = [r for r in report.results if not r.passed]
        self.assertFalse(
            failures,
            "RAG Case 失败："
            + "；".join(f"{r.case_id}: {r.failure_reason}" for r in failures),
        )
        for result in report.results:
            self.assertIsNotNone(result.state)
            state = result.state
            self.assertIn("retrieval_result_count", state)
            self.assertIn("best_distance", state)
            self.assertIn("retrieval_reliable", state)
            self.assertIn("answer_strategy", state)


if __name__ == "__main__":
    unittest.main()
