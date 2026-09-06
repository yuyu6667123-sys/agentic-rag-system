"""Failure-path evaluation. These tests simulate faults without changing src."""

from __future__ import annotations

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.agent.agent import Agent, AgentError
from src.api.server import _stream_events, app, get_agent
from src.auth.router import get_auth_service, require_current_user
from src.auth.service import (
    AuthService,
    AuthSettings,
    ExpiredVerificationCodeError,
    InvalidSessionError,
    InvalidVerificationCodeError,
)


class _FailingAgent:
    def stream_answer(self, _question: str):
        raise RuntimeError("simulated Ollama unavailable")
        yield "never"


class _EmailSender:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def send_verification_code(self, email: str, code: str) -> None:
        self.messages.append((email, code))


class StabilityEvaluationTests(unittest.TestCase):
    def test_ollama_failure_becomes_sse_error_event(self) -> None:
        events = list(_stream_events("测试 Ollama 异常", _FailingAgent()))
        self.assertEqual(len(events), 1)
        self.assertIn('"type": "error"', events[0])
        self.assertIn("simulated Ollama unavailable", events[0])

    def test_stream_client_close_does_not_commit_partial_memory(self) -> None:
        agent = Agent(ollama_timeout=1)

        def partial_model(_prompt: str):
            yield "partial"
            yield " answer"

        with patch.object(agent, "_should_use_knowledge_base", return_value=False), patch.object(
            agent, "_stream_model", side_effect=partial_model
        ):
            stream = agent.stream_answer("客户端提前关闭")
            self.assertEqual(next(stream), "partial")
            stream.close()

        self.assertEqual(agent.get_history(), [])
        self.assertIsNotNone(agent.get_request_state()["total_time_ms"])

    def test_auth_wrong_expired_and_invalid_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sender = _EmailSender()
            settings = AuthSettings(
                db_path=Path(temp_dir) / "auth.db",
                code_ttl_seconds=1,
                code_cooldown_seconds=0,
                session_ttl_seconds=1,
            )
            clock_value = [time.time()]
            service = AuthService(
                settings,
                email_sender=sender,
                clock=lambda: clock_value[0],
            )
            service.request_verification_code("123456@qq.com")
            with self.assertRaises(InvalidVerificationCodeError):
                service.verify_and_login("123456@qq.com", "000000")

            issued_at = time.time()
            clock_value[0] = issued_at
            service.request_verification_code("123456@qq.com")
            clock_value[0] = issued_at + 2
            with self.assertRaises(ExpiredVerificationCodeError):
                service.verify_and_login("123456@qq.com", sender.messages[-1][1])

            with self.assertRaises(InvalidSessionError):
                service.get_user_by_session("invalid-session")

    def test_rag_resource_failure_is_explicit(self) -> None:
        agent = Agent()
        with patch.object(agent, "_should_use_knowledge_base", return_value=True), patch.object(
            agent,
            "_invoke_tool",
            side_effect=RuntimeError("simulated Chroma unavailable"),
        ):
            with self.assertRaises(AgentError) as context:
                agent.answer("知识库资源异常")
        self.assertIn("search_knowledge_base", str(context.exception))

    def test_sse_endpoint_failure_contract(self) -> None:
        app.dependency_overrides[get_agent] = lambda: _FailingAgent()
        app.dependency_overrides[require_current_user] = lambda: {
            "id": 1,
            "email": "123456@qq.com",
        }
        try:
            response = TestClient(app).get(
                "/chat/stream",
                params={"question": "异常请求"},
            )
        finally:
            app.dependency_overrides.clear()
        self.assertEqual(response.status_code, 200)
        self.assertIn('"type": "error"', response.text)


if __name__ == "__main__":
    unittest.main()
