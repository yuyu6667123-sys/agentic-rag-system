"""Interface tests for authentication, cookies, and protected chat routes."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from src.api import server as server_module
from src.api.server import app, get_agent
from src.auth.router import get_auth_service
from src.auth.service import AuthService, AuthSettings


class _EmailSender:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def send_verification_code(self, email: str, code: str) -> None:
        self.messages.append((email, code))

    @property
    def latest_code(self) -> str:
        return self.messages[-1][1]


class _FakeAgent:
    def stream_answer(self, question: str):
        yield f"回答：{question}"
        yield "\n\n参考来源：\n[1] rag.txt"


class AuthApiTests(unittest.TestCase):
    def setUp(self) -> None:
        tests_dir = Path(__file__).resolve().parent
        self.temp_dir = tempfile.TemporaryDirectory(prefix="tmp_auth_api_", dir=tests_dir)
        self.sender = _EmailSender()
        settings = AuthSettings(
            db_path=Path(self.temp_dir.name) / "auth.db",
            code_ttl_seconds=300,
            code_cooldown_seconds=60,
            session_ttl_seconds=600,
        )
        self.service = AuthService(settings, email_sender=self.sender)
        app.dependency_overrides[get_auth_service] = lambda: self.service
        app.dependency_overrides[get_agent] = lambda: _FakeAgent()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        app.dependency_overrides.clear()
        self.temp_dir.cleanup()

    def _login(self) -> None:
        requested = self.client.post(
            "/auth/code/request",
            json={"email": "123456@qq.com"},
        )
        self.assertEqual(requested.status_code, 202)
        self.assertNotIn("code", requested.json())
        verified = self.client.post(
            "/auth/code/verify",
            json={"email": "123456@qq.com", "code": self.sender.latest_code},
        )
        self.assertEqual(verified.status_code, 200)
        self.assertNotIn("session_token", verified.json())
        cookie = verified.headers["set-cookie"].casefold()
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=lax", cookie)
        self.assertIn("max-age=600", cookie)

    def test_login_cookie_me_protected_routes_and_logout(self) -> None:
        self._login()

        self.assertIn("agent_session", self.client.cookies)
        me = self.client.get("/auth/me")
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["user"]["email"], "123456@qq.com")
        chat_page = self.client.get("/")
        self.assertEqual(chat_page.status_code, 200)
        self.assertIn('id="user-email"', chat_page.text)
        self.assertIn("/auth/me", chat_page.text)
        self.assertIn("/auth/logout", chat_page.text)
        self.assertIn("/chat/state", chat_page.text)
        self.assertEqual(self.client.get("/chat/state").status_code, 200)

        stream = self.client.get(
            "/chat/stream",
            params={"question": "什么是 RAG？"},
        )
        self.assertEqual(stream.status_code, 200)
        self.assertIn('"type": "token"', stream.text)
        self.assertTrue(stream.text.rstrip().endswith('data: {"type": "done"}'))

        logout = self.client.post("/auth/logout")
        self.assertEqual(logout.status_code, 200)
        self.assertEqual(self.client.get("/auth/me").status_code, 401)
        redirected = self.client.get("/", follow_redirects=False)
        self.assertEqual(redirected.status_code, 303)
        self.assertEqual(redirected.headers["location"], "/login")
        protected_stream = self.client.get(
            "/chat/stream",
            params={"question": "退出后的问题"},
        )
        self.assertEqual(protected_stream.status_code, 401)
        self.assertEqual(self.client.get("/chat/state").status_code, 401)

    def test_unauthenticated_chat_and_page_are_rejected(self) -> None:
        page = self.client.get("/", follow_redirects=False)
        self.assertEqual(page.status_code, 303)
        self.assertEqual(page.headers["location"], "/login")
        login = self.client.get("/login")
        self.assertEqual(login.status_code, 200)
        self.assertIn("Agentic RAG", login.text)
        response = self.client.get(
            "/chat/stream",
            params={"question": "你好"},
        )
        self.assertEqual(response.status_code, 401)

    def test_invalid_email_wrong_code_and_cooldown_statuses(self) -> None:
        invalid = self.client.post(
            "/auth/code/request",
            json={"email": "user@example.com"},
        )
        self.assertEqual(invalid.status_code, 422)

        self.client.post("/auth/code/request", json={"email": "123456@qq.com"})
        cooldown = self.client.post(
            "/auth/code/request",
            json={"email": "123456@qq.com"},
        )
        self.assertEqual(cooldown.status_code, 429)
        self.assertEqual(cooldown.headers["retry-after"], "60")

        wrong_code = "000000" if self.sender.latest_code != "000000" else "000001"
        wrong = self.client.post(
            "/auth/code/verify",
            json={"email": "123456@qq.com", "code": wrong_code},
        )
        self.assertEqual(wrong.status_code, 400)
        self.assertNotIn("agent_session", self.client.cookies)

    def test_authenticated_user_skips_login_page(self) -> None:
        self._login()

        response = self.client.get("/login", follow_redirects=False)

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/")

    def test_agents_and_memory_are_isolated_by_user_id(self) -> None:
        first_user_id = 900_001
        second_user_id = 900_002
        try:
            first = server_module._get_or_create_agent_for_user(first_user_id)
            first_again = server_module._get_or_create_agent_for_user(first_user_id)
            second = server_module._get_or_create_agent_for_user(second_user_id)
            first.add_message("user", "用户一的私有上下文")

            self.assertIs(first, first_again)
            self.assertIsNot(first, second)
            self.assertEqual(len(first.get_history()), 1)
            self.assertEqual(second.get_history(), [])
        finally:
            server_module._AGENTS_BY_USER_ID.pop(first_user_id, None)
            server_module._AGENTS_BY_USER_ID.pop(second_user_id, None)


if __name__ == "__main__":
    unittest.main()
