"""Minimal contract test for the Agent SSE endpoint."""

from fastapi.testclient import TestClient

from src.api.server import app, get_agent
from src.auth.router import require_current_user


class _FakeAgent:
    def stream_answer(self, question: str):
        assert question == "什么是 RAG？"
        yield "RAG"
        yield " 是一种检索增强生成方法。"
        yield "\n\n参考来源：\n[1] ragziliao.txt"


def test_stream_chat_sends_tokens_then_done() -> None:
    app.dependency_overrides[get_agent] = lambda: _FakeAgent()
    app.dependency_overrides[require_current_user] = lambda: {
        "id": 1,
        "email": "123456@qq.com",
    }
    try:
        response = TestClient(app).get(
            "/chat/stream",
            params={"question": "什么是 RAG？"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [line for line in response.text.splitlines() if line.startswith("data: ")]
    assert events == [
        'data: {"type": "token", "content": "RAG"}',
        'data: {"type": "token", "content": " 是一种检索增强生成方法。"}',
        'data: {"type": "token", "content": "\\n\\n参考来源：\\n[1] ragziliao.txt"}',
        'data: {"type": "done"}',
    ]
