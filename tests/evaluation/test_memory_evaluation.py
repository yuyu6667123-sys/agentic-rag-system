"""Short-term Memory and multi-user isolation evaluation."""

from __future__ import annotations

import unittest
import os

from src.api import server as server_module

from .runner import run_memory_evaluation, verify_dependencies


RUN_REAL = os.getenv("RUN_REAL_EVALUATION", "0").casefold() in {"1", "true", "yes"}


class MemoryEvaluationTests(unittest.TestCase):
    def test_pronoun_context(self) -> None:
        if not RUN_REAL:
            self.skipTest("真实评估默认跳过；设置 RUN_REAL_EVALUATION=1 后运行")
        notes = verify_dependencies()
        if any("失败" in note for note in notes):
            self.skipTest("; ".join(notes))
        report = run_memory_evaluation()
        failures = [r for r in report.results if not r.passed]
        self.assertFalse(
            failures,
            "Memory Case 失败："
            + "；".join(f"{r.case_id}: {r.failure_reason}" for r in failures),
        )
        state = report.results[0].state
        self.assertIsNotNone(state)

    def test_multi_user_memory_isolation(self) -> None:
        user_a = server_module._get_or_create_agent_for_user(910001)
        user_b = server_module._get_or_create_agent_for_user(910002)
        try:
            user_a.add_message("user", "我喜欢Python")
            user_b.add_message("user", "我喜欢Java")
            self.assertEqual(user_a.get_history()[0]["content"], "我喜欢Python")
            self.assertEqual(user_b.get_history()[0]["content"], "我喜欢Java")
            self.assertNotEqual(user_a.get_history(), user_b.get_history())
        finally:
            server_module._AGENTS_BY_USER_ID.pop(910001, None)
            server_module._AGENTS_BY_USER_ID.pop(910002, None)


if __name__ == "__main__":
    unittest.main()
