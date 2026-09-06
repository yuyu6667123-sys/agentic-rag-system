"""FastAPI endpoint that exposes the existing Agent stream as SSE."""

from __future__ import annotations

from collections.abc import Iterator
import json
import logging
from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse

from ..agent.agent import Agent
from ..tools.rag_tool import _get_vector_store
from ..auth.router import (
    get_optional_current_user,
    require_current_user,
    router as auth_router,
)


LOGGER = logging.getLogger(__name__)
_AGENT_LOCK = Lock()
_AGENT_INSTANCES_LOCK = Lock()
_AGENTS_BY_USER_ID: dict[int, Agent] = {}

app = FastAPI(title="Agent Streaming API")
app.include_router(auth_router)
CHAT_PAGE = Path(__file__).with_name("chat.html")
LOGIN_PAGE = Path(__file__).with_name("login.html")


@app.on_event("startup")
def warm_rag_resources() -> None:
    """Warm the cached Chroma store so the first user request avoids cold start."""
    try:
        store = _get_vector_store()
        collection = getattr(store, "_collection", None)
        count = collection.count() if collection is not None else None
        LOGGER.info("[API] Chroma resources warmed: collection_count=%s", count)
    except Exception as exc:
        # Keep startup available for ordinary chat and return the existing
        # explicit RAG error if a knowledge-base request is later attempted.
        LOGGER.warning("[API] Chroma warmup failed: %s", exc)


def get_agent(
    current_user: Annotated[dict[str, object], Depends(require_current_user)],
) -> Agent:
    """Reuse one in-memory Agent per authenticated user."""
    return _get_or_create_agent_for_user(int(current_user["id"]))


def _get_or_create_agent_for_user(user_id: int) -> Agent:
    """Keep each authenticated user's short-term Memory isolated."""
    with _AGENT_INSTANCES_LOCK:
        agent = _AGENTS_BY_USER_ID.get(user_id)
        if agent is None:
            agent = Agent()
            _AGENTS_BY_USER_ID[user_id] = agent
        return agent


def _sse_data(payload: dict[str, object]) -> str:
    """Encode one JSON object as a UTF-8-safe SSE data event."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_events(
    question: str,
    agent: Agent,
    *,
    emit_status: bool = False,
) -> Iterator[str]:
    """Adapt Agent text chunks to SSE and emit one terminal event."""
    try:
        # Status events are additive.  The default keeps this helper compatible
        # with existing integrations; the production endpoint opts in below.
        if emit_status and isinstance(agent, Agent):
            yield _sse_data(
                {
                    "type": "status",
                    "stage": "analysis",
                    "message": "正在分析问题",
                }
            )
            yield _sse_data(
                {
                    "type": "status",
                    "stage": "retrieval",
                    "message": "正在检索知识库（如需要）",
                }
            )
        if emit_status and isinstance(agent, Agent):
            yield _sse_data(
                {
                    "type": "status",
                    "stage": "generation",
                    "message": "正在生成回答",
                }
            )
        # Memory is ordered user/assistant state; serialize turns on one Agent.
        with _AGENT_LOCK:
            for chunk in agent.stream_answer(question):
                yield _sse_data({"type": "token", "content": chunk})
        yield _sse_data({"type": "done"})
    except GeneratorExit:
        # A disconnected client closes the generator; Agent will not commit an
        # incomplete assistant turn to Memory.
        raise
    except Exception as exc:
        LOGGER.exception("[API] Agent Streaming 失败")
        yield _sse_data({"type": "error", "message": str(exc)})


@app.get("/chat/stream", response_class=StreamingResponse)
def stream_chat(
    question: Annotated[str, Query(min_length=1)],
    agent: Annotated[Agent, Depends(get_agent)],
    _current_user: Annotated[dict[str, object], Depends(require_current_user)],
) -> StreamingResponse:
    """Stream one Agent answer as JSON-bearing Server-Sent Events."""
    normalized_question = question.strip()
    if not normalized_question:
        raise HTTPException(status_code=422, detail="question 必须是非空字符串")

    return StreamingResponse(
        _stream_events(normalized_question, agent, emit_status=True),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/chat/state")
def chat_state(
    agent: Annotated[Agent, Depends(get_agent)],
    _current_user: Annotated[dict[str, object], Depends(require_current_user)],
) -> dict[str, object] | None:
    """Return the latest public Agent state for the authenticated user."""
    # Keep the endpoint compatible with lightweight Agent substitutes used by
    # API tests and integrations that only implement streaming.
    get_state = getattr(agent, "get_request_state", None)
    return get_state() if callable(get_state) else None


@app.get("/", response_class=FileResponse)
def chat_page(
    current_user: Annotated[
        dict[str, object] | None,
        Depends(get_optional_current_user),
    ],
) -> Response:
    """Serve the minimal same-origin browser smoke-test page."""
    if current_user is None:
        return RedirectResponse(url="/login", status_code=303)
    return FileResponse(CHAT_PAGE, media_type="text/html; charset=utf-8")


@app.get("/login", response_class=FileResponse)
def login_page(
    current_user: Annotated[
        dict[str, object] | None,
        Depends(get_optional_current_user),
    ],
) -> Response:
    """Serve authentication UI or return an authenticated user to Chat."""
    if current_user is not None:
        return RedirectResponse(url="/", status_code=303)
    return FileResponse(LOGIN_PAGE, media_type="text/html; charset=utf-8")


__all__ = ["app", "get_agent"]
