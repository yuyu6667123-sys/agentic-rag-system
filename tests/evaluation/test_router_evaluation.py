"""Router evaluation using real Ollama and the production Agent."""

from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from .runner import run_router_evaluation, verify_dependencies


RUN_REAL = os.getenv("RUN_REAL_EVALUATION", "0").casefold() in {"1", "true", "yes"}


class RouterEvaluationTests(unittest.TestCase):
    """Run with RUN_REAL_EVALUATION=1 to execute real model cases."""

    @classmethod
    def setUpClass(cls) -> None:
        if not RUN_REAL:
            raise unittest.SkipTest(
                "真实评估默认跳过；设置 RUN_REAL_EVALUATION=1 后运行"
            )
        notes = verify_dependencies()
        if any("失败" in note for note in notes):
            raise unittest.SkipTest("; ".join(notes))

    def test_router_accuracy(self) -> None:
        catalog = json.loads(
            (Path(__file__).with_name("evaluation_cases.json")).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(catalog["router"]["plain"]), 5)
        self.assertEqual(len(catalog["router"]["knowledge_base"]), 4)

        report = run_router_evaluation()
        failures = [r for r in report.results if not r.passed]
        self.assertFalse(
            failures,
            "Router Case 失败："
            + "；".join(f"{r.case_id}: {r.failure_reason}" for r in failures),
        )
        self.assertEqual(report.passed_count, len(report.results))


if __name__ == "__main__":
    unittest.main()
