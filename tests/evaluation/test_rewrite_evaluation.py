"""Query Rewrite evaluation using the real Agent loop."""

from __future__ import annotations

import unittest
import os

from .runner import run_rewrite_evaluation, verify_dependencies


RUN_REAL = os.getenv("RUN_REAL_EVALUATION", "0").casefold() in {"1", "true", "yes"}


class RewriteEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not RUN_REAL:
            raise unittest.SkipTest(
                "真实评估默认跳过；设置 RUN_REAL_EVALUATION=1 后运行"
            )
        notes = verify_dependencies()
        if any("失败" in note for note in notes):
            raise unittest.SkipTest("; ".join(notes))

    def test_second_retrieval_and_query_rewrite(self) -> None:
        report = run_rewrite_evaluation()
        failures = [r for r in report.results if not r.passed]
        self.assertFalse(
            failures,
            "Rewrite Case 失败："
            + "；".join(f"{r.case_id}: {r.failure_reason}" for r in failures),
        )
        state = report.results[0].state
        self.assertGreaterEqual(state["tool_call_count"], 2)
        self.assertTrue(state["query_rewrite"])


if __name__ == "__main__":
    unittest.main()
