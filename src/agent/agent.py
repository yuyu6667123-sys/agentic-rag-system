"""Minimal Agentic RAG router using Ollama and one existing search Tool."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from contextvars import ContextVar
import os
import json
import logging
from pathlib import Path
import requests
import time

try:
    from ..rag_qa import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL, _call_ollama
except ImportError:
    from rag_qa import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL, _call_ollama

from dotenv import load_dotenv

try:
    from ..tools.tool_registry import (
        DEFAULT_TOOL_REGISTRY,
        RAG_TOOL_NAME,
        ToolRegistry,
    )
except ImportError:
    from tools.tool_registry import DEFAULT_TOOL_REGISTRY, RAG_TOOL_NAME, ToolRegistry

try:
    from .request_state import AgentRequestState
except ImportError:
    from request_state import AgentRequestState


SRC_DIR = Path(__file__).resolve().parent.parent
LOGGER = logging.getLogger(__name__)
MAX_HISTORY_MESSAGES = 12
DEFAULT_PROMPT_HISTORY_TURNS = 2
DEFAULT_GENERATION_MAX_NEW_TOKENS = 128
_REQUEST_LOG_ID: ContextVar[str] = ContextVar("agent_request_id", default="-")


class _RequestIdFilter(logging.Filter):
    """Add the active request id to existing and structured Agent logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = str(record.msg)
        if not message.startswith("[Agent]"):
            return True
        if "request_id=" in message:
            return True
        remainder = message.removeprefix("[Agent]").lstrip()
        record.msg = f"[Agent] request_id={_REQUEST_LOG_ID.get()} {remainder}"
        return True


LOGGER.addFilter(_RequestIdFilter())


KNOWLEDGE_BASE_KEYWORDS = (
    "rag",
    "知识库",
    "文档",
    "资料",
    "数据库",
    "向量",
    "chroma",
)

ROUTER_PROMPT_TEMPLATE = (
    "你是一个路由判断器。你的任务只有一个：判断用户问题是否需要查询当前项目知识库。"
    "凡是询问 RAG、Chroma、向量检索、项目文档或知识库内容的问题，都判定为需要查询知识库；"
    "如果需要查询知识库，只返回严格 JSON：{{\"need_knowledge_base\": true}}；"
    "如果不需要，只返回严格 JSON：{{\"need_knowledge_base\": false}}。"
    "只能返回 JSON，不要解释，不要回答用户问题，不要输出 Markdown。\n\n"
    "历史对话（仅用于理解当前问题，不要复述）：\n{history}\n\n"
    "当前用户问题：{question}"
)

EVIDENCE_CHECK_PROMPT_TEMPLATE = (
    "你是一个证据充分性判断器。判断下方知识库证据是否足以回答用户问题。"
    "只允许返回严格 JSON：{{\"sufficient\": true}} 或 {{\"sufficient\": false}}。"
    "不要解释，不要回答用户问题，不要输出 Markdown。\n\n"
    "用户问题：{question}\n\n"
    "知识库证据：\n{evidence}"
)

QUERY_REWRITE_PROMPT_TEMPLATE = (
    "你是一个知识库检索 Query 改写器。根据原始用户问题和第一次检索证据，"
    "生成一个更适合知识库检索的新 Query。只允许返回严格 JSON："
    "{{\"query\": \"新的检索问题\"}}。不要解释，不要回答用户问题，不要输出 Markdown。\n\n"
    "原始用户问题：{question}\n\n"
    "第一次检索证据：\n{evidence}"
)

DEFAULT_RELEVANCE_DISTANCE_THRESHOLD = 0.8
DEFAULT_HIGH_CONFIDENCE_DISTANCE = 0.65
DEFAULT_HIGH_CONFIDENCE_MIN_RESULTS = 4

RELEVANCE_PROMPT_TEMPLATE = (
    "你是一个检索相关性判断器。判断下方知识库检索证据是否与用户问题高度相关，"
    "并且能够帮助回答该问题。只允许返回严格 JSON：{{\"relevant\": true}} "
    "或 {{\"relevant\": false}}。只能返回 JSON，不要解释，不要回答用户问题，"
    "不要输出 Markdown。\n\n用户问题：{question}\n\n检索证据：\n{evidence}"
)

class AgentError(RuntimeError):
    """Raised when the Agent cannot obtain an Ollama response."""


def get_rag_tool_interface() -> dict[str, object]:
    """Return a copy of the public metadata for the single supported Tool."""
    return DEFAULT_TOOL_REGISTRY.get(RAG_TOOL_NAME)


def format_sources(evidence_sets: list[dict] | dict) -> list[str]:
    """Return deduplicated, display-ready source labels from Tool evidence."""
    if isinstance(evidence_sets, dict):
        evidence_sets = [evidence_sets]
    formatted: list[str] = []
    seen_sources: set[str] = set()
    # Later evidence is the rewritten/second retrieval and therefore has priority.
    for evidence in reversed(evidence_sets):
        results = evidence.get("results", []) if isinstance(evidence, dict) else []
        if not isinstance(results, list):
            continue
        for item in results:
            if not isinstance(item, dict):
                continue
            source = item.get("source")
            if source is None or not str(source).strip():
                continue
            source_text = str(source).strip()
            if source_text in seen_sources:
                continue
            seen_sources.add(source_text)
            page = item.get("page")
            if page is None or page == "":
                formatted.append(source_text)
            else:
                formatted.append(f"{source_text}，第 {page} 页")
    return formatted


class Agent:
    """Minimal Agent that routes knowledge questions to the existing RAG Tool."""

    def __init__(
        self,
        ollama_url: str | None = None,
        ollama_model: str | None = None,
        ollama_timeout: float | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        load_dotenv(SRC_DIR / ".env")
        self.ollama_url = ollama_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.ollama_timeout = (
            ollama_timeout
            if ollama_timeout is not None
            else float(os.getenv("OLLAMA_TIMEOUT", "120"))
        )
        self.relevance_distance_threshold = float(
            os.getenv(
                "RELEVANCE_DISTANCE_THRESHOLD",
                str(DEFAULT_RELEVANCE_DISTANCE_THRESHOLD),
            )
        )
        self.high_confidence_distance = float(
            os.getenv(
                "HIGH_CONFIDENCE_DISTANCE",
                str(DEFAULT_HIGH_CONFIDENCE_DISTANCE),
            )
        )
        self.high_confidence_min_results = int(
            os.getenv(
                "HIGH_CONFIDENCE_MIN_RESULTS",
                str(DEFAULT_HIGH_CONFIDENCE_MIN_RESULTS),
            )
        )
        if self.high_confidence_distance < 0:
            raise ValueError("HIGH_CONFIDENCE_DISTANCE 必须大于等于 0")
        if self.high_confidence_min_results <= 0:
            raise ValueError("HIGH_CONFIDENCE_MIN_RESULTS 必须大于 0")
        self.prompt_history_turns = int(
            os.getenv(
                "PROMPT_HISTORY_TURNS",
                str(DEFAULT_PROMPT_HISTORY_TURNS),
            )
        )
        if self.prompt_history_turns <= 0:
            raise ValueError("PROMPT_HISTORY_TURNS 必须大于 0")
        generation_limit = os.getenv(
            "OLLAMA_MAX_NEW_TOKENS",
            str(DEFAULT_GENERATION_MAX_NEW_TOKENS),
        )
        self.generation_max_new_tokens = int(generation_limit)
        if self.generation_max_new_tokens <= 0:
            raise ValueError("OLLAMA_MAX_NEW_TOKENS 必须大于 0")
        self.tool_registry = tool_registry or DEFAULT_TOOL_REGISTRY

        self.last_tool_called = False
        self.last_routing_decision: str | None = None
        self.last_tool_call_count = 0
        self.last_loop_decision: str | None = None
        self.last_retrieval_reliable = False
        self.last_best_distance: float | None = None
        self.last_result_count = 0
        self.last_relevance_decision: str | None = None
        self.answer_strategy: str | None = None
        self.last_answer_strategy: str | None = None
        self._high_confidence_fast_path = False
        self.request_state: AgentRequestState | None = None
        self.max_history_messages = MAX_HISTORY_MESSAGES
        self._history: list[dict[str, str]] = []

    def _start_request(self, question: str) -> AgentRequestState:
        """Reset compatibility fields and create state for one new request."""
        self.last_tool_called = False
        self.last_routing_decision = None
        self.last_tool_call_count = 0
        self.last_loop_decision = None
        self.last_retrieval_reliable = False
        self.last_best_distance = None
        self.last_result_count = 0
        self.last_relevance_decision = None
        self.answer_strategy = None
        self.last_answer_strategy = None
        self._high_confidence_fast_path = False
        self.request_state = AgentRequestState.create(question)
        _REQUEST_LOG_ID.set(self.request_state.request_id)
        self._log_event(
            "request_started",
            question_chars=len(question),
        )
        return self.request_state

    @staticmethod
    def _log_value(value: object) -> str:
        """Render one bounded value without logging prompts or evidence bodies."""
        if isinstance(value, str) and len(value) > 200:
            value = value[:197] + "..."
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def _log_event(
        self,
        event: str,
        *,
        level: int = logging.INFO,
        **fields: object,
    ) -> None:
        """Emit one compact key/value event for the active request."""
        details = " ".join(
            f"{name}={self._log_value(value)}" for name, value in fields.items()
        )
        suffix = f" {details}" if details else ""
        LOGGER.log(level, "[Agent] event=%s%s", event, suffix)

    def _log_performance(
        self,
        stage: str,
        started_at: float,
        **fields: object,
    ) -> None:
        """Record bounded stage timing without exposing prompts or evidence."""
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
        self._log_event(
            "performance",
            stage=stage,
            time_ms=elapsed_ms,
            **fields,
        )

    @staticmethod
    def _model_stage(prompt: str) -> str:
        """Classify an internal model call for performance logs only."""
        if "路由判断器" in prompt:
            return "router"
        if "检索相关性判断器" in prompt:
            return "relevance"
        if "证据充分性判断器" in prompt:
            return "sufficiency"
        if "Query 改写器" in prompt:
            return "query_rewrite"
        return "generation"

    def _sync_request_state(self) -> None:
        """Copy existing compatibility fields into the active request state."""
        state = self.request_state
        if state is None:
            return
        state.routing_decision = self.last_routing_decision
        state.tool_call_count = self.last_tool_call_count
        state.retrieval_result_count = self.last_result_count
        state.best_distance = self.last_best_distance
        state.retrieval_reliable = self.last_retrieval_reliable
        state.relevance_decision = self.last_relevance_decision
        state.answer_strategy = self.last_answer_strategy

    def _update_request_state(self, **updates: object) -> None:
        """Write stage results to the active state as soon as they are known."""
        state = self.request_state
        if state is None:
            return
        for field_name, value in updates.items():
            if not hasattr(state, field_name) or field_name.startswith("_"):
                raise AttributeError(f"未知的 AgentRequestState 字段: {field_name}")
            setattr(state, field_name, value)

    def _record_tool_call(self, call_count: int) -> None:
        """Keep the legacy Tool fields and request state synchronized."""
        self.last_tool_called = call_count > 0
        self.last_tool_call_count = call_count
        self._update_request_state(tool_call_count=call_count)
        self._log_event(
            "tool_call",
            tool=RAG_TOOL_NAME,
            call_count=call_count,
        )

    def _record_answer_strategy(self) -> None:
        """Publish the current compatibility strategy to request state."""
        self._update_request_state(answer_strategy=self.last_answer_strategy)
        self._log_event(
            "answer_strategy",
            strategy=self.last_answer_strategy,
        )

    def _finish_request(self, outcome: str) -> None:
        """Finalize the active state whether the request succeeds or fails."""
        if self.request_state is None:
            return
        self._sync_request_state()
        self.request_state.finish()
        self._log_performance(
            "total",
            self.request_state._started_at,
            outcome=outcome,
        )
        self._log_event(
            "request_finished",
            outcome=outcome,
            routing_decision=self.request_state.routing_decision,
            tool_call_count=self.request_state.tool_call_count,
            retrieval_result_count=self.request_state.retrieval_result_count,
            best_distance=self.request_state.best_distance,
            retrieval_reliable=self.request_state.retrieval_reliable,
            relevance_decision=self.request_state.relevance_decision,
            query_rewrite=self.request_state.query_rewrite,
            answer_strategy=self.request_state.answer_strategy,
            sources_count=len(self.request_state.sources),
            total_time_ms=self.request_state.total_time_ms,
        )
        _REQUEST_LOG_ID.set("-")

    def get_request_state(self) -> dict[str, object] | None:
        """Return a snapshot of the current or most recently completed request."""
        if self.request_state is None:
            return None
        self._sync_request_state()
        return self.request_state.to_dict()

    def add_message(self, role: str, content: str) -> None:
        """Append one short-term conversation message to in-memory history."""
        if role not in {"user", "assistant"}:
            raise ValueError("role 必须是 'user' 或 'assistant'")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content 必须是非空字符串")
        expected_role = (
            "user"
            if not self._history
            else "assistant" if self._history[-1]["role"] == "user" else "user"
        )
        if role != expected_role:
            raise ValueError(
                f"消息角色顺序无效：期望 {expected_role!r}，实际为 {role!r}"
            )
        self._history.append({"role": role, "content": content.strip()})
        removed_count = 0
        # Remove only complete oldest turns.  A newly-added user message may
        # temporarily leave one pending message until its assistant reply arrives.
        while len(self._history) > self.max_history_messages:
            del self._history[:2]
            removed_count += 2
        if removed_count:
            LOGGER.info(
                "[Agent] Memory truncated: removed=%d oldest messages, history_size=%d",
                removed_count,
                len(self._history),
            )
        LOGGER.info(
            "[Agent] Memory message added: role=%s, history_size=%d",
            role,
            len(self._history),
        )

    def get_history(self) -> list[dict[str, str]]:
        """Return a shallow copy so callers cannot mutate internal history."""
        LOGGER.info("[Agent] Memory history size: %d", len(self._history))
        return [message.copy() for message in self._history]

    def clear_history(self) -> None:
        """Clear all short-term conversation messages."""
        cleared_count = len(self._history)
        self._history.clear()
        LOGGER.info("[Agent] Memory cleared: messages=%d", cleared_count)

    @staticmethod
    def _needs_knowledge_base(question: str) -> bool:
        """Apply a small deterministic routing rule for this basic Agent."""
        lowered = question.casefold()
        return any(keyword in lowered for keyword in KNOWLEDGE_BASE_KEYWORDS)

    def _history_text_for_prompt(self) -> str:
        """Use only recent complete turns in model prompts."""
        history = self.get_history()
        max_messages = self.prompt_history_turns * 2
        if len(history) > max_messages:
            history = history[-max_messages:]
            self._log_event(
                "prompt_history_truncated",
                kept_messages=len(history),
                kept_turns=self.prompt_history_turns,
            )
        return (
            json.dumps(history, ensure_ascii=False, indent=2)
            if history
            else "（无历史对话）"
        )

    @staticmethod
    def _estimate_input_tokens(text: str) -> int:
        """Return a conservative character-based input token estimate."""
        return max(1, (len(text) + 3) // 4)

    def _log_prompt_profile(
        self,
        *,
        prompt: str,
        instruction_text: str,
        history_text: str,
        evidence_text: str,
    ) -> None:
        """Log prompt dimensions without logging prompt content."""
        self._log_event(
            "prompt_profile",
            stage="generation",
            system_chars=len(instruction_text),
            history_chars=len(history_text),
            evidence_chars=len(evidence_text),
            prompt_chars=len(prompt),
            estimated_input_tokens=self._estimate_input_tokens(prompt),
        )

    def _call_generation_model(self, prompt: str) -> str:
        """Call Ollama with a bounded generation length for final answers."""
        url = f"{self.ollama_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": self.generation_max_new_tokens},
        }
        response: requests.Response | None = None
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.ollama_timeout,
            )
            response.raise_for_status()
            data = response.json()
            answer = data.get("response")
            if not answer and isinstance(data.get("message"), dict):
                answer = data["message"].get("content")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Ollama 响应中没有有效的 response 字段")
            return answer.strip()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama 生成请求失败: url={url!r}") from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("Ollama 生成响应解析失败") from exc

    def _call_model(self, prompt: str) -> str:
        started_at = time.perf_counter()
        stage = self._model_stage(prompt)
        try:
            if stage == "generation":
                return self._call_generation_model(prompt)
            return _call_ollama(
                self.ollama_url,
                self.ollama_model,
                prompt,
                self.ollama_timeout,
            )
        except Exception as exc:
            raise AgentError(
                f"Agent 调用 Ollama 失败: url={self.ollama_url!r}, "
                f"model={self.ollama_model!r}。请确认 Ollama 服务已启动，且模型已准备。"
            ) from exc
        finally:
            self._log_performance(stage, started_at)

    def _stream_model(self, prompt: str) -> Iterator[str]:
        """Yield only Ollama ``response`` chunks from an NDJSON response."""
        started_at = time.perf_counter()
        url = f"{self.ollama_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": self.generation_max_new_tokens},
        }
        done_received = False
        response_received = False
        try:
            try:
                with requests.post(
                    url,
                    json=payload,
                    stream=True,
                    timeout=(5, self.ollama_timeout),
                ) as response:
                    response.raise_for_status()
                    response.encoding = "utf-8"
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except (TypeError, json.JSONDecodeError) as exc:
                            raise AgentError("Ollama Streaming 返回了无效的 NDJSON 数据") from exc
                        if not isinstance(data, dict):
                            raise AgentError("Ollama Streaming 的每行数据必须是 JSON 对象")
                        error = data.get("error")
                        if error:
                            raise AgentError(f"Ollama Streaming 返回错误: {error}")

                        chunk = data.get("response")
                        if isinstance(chunk, str) and chunk:
                            response_received = True
                            yield chunk
                        if data.get("done") is True:
                            done_received = True
                            break
            except AgentError:
                raise
            except requests.exceptions.RequestException as exc:
                raise AgentError(
                    f"Agent Streaming 调用 Ollama 失败: url={url!r}, "
                    f"model={self.ollama_model!r}。请确认 Ollama 服务已启动，且模型已准备。"
                ) from exc

            if not done_received:
                raise AgentError("Ollama Streaming 响应在收到 done=true 前结束")
            if not response_received:
                raise AgentError("Ollama Streaming 响应中没有有效的 response 字段")
        finally:
            self._log_performance("generation", started_at)

    def _invoke_tool(self, name: str, **kwargs: object) -> dict:
        """Look up and invoke a Tool through the Registry."""
        LOGGER.info("[Agent] Tool invoke: %s", name)
        started_at = time.perf_counter()
        try:
            return self.tool_registry.execute(name, **kwargs)
        finally:
            self._log_performance("retrieval", started_at, tool=name)

    def _should_use_knowledge_base(self, question: str) -> bool:
        """Ask Qwen for a JSON routing decision, with keyword fallback."""
        history_text = self._history_text_for_prompt()
        router_prompt = ROUTER_PROMPT_TEMPLATE.format(
            question=question,
            history=history_text,
        )
        try:
            raw_decision = self._call_model(router_prompt)
            decision = json.loads(raw_decision.strip())
            if not isinstance(decision, dict):
                raise ValueError("Router JSON 必须是对象")
            need_knowledge_base = decision.get("need_knowledge_base")
            if not isinstance(need_knowledge_base, bool):
                raise ValueError("need_knowledge_base 必须是 bool")

            self.last_routing_decision = (
                "llm_true" if need_knowledge_base else "llm_false"
            )
            self._update_request_state(
                routing_decision=self.last_routing_decision,
            )
            self._log_event(
                "router_decision",
                decision=self.last_routing_decision,
            )
            return need_knowledge_base
        except Exception as exc:
            fallback = self._needs_knowledge_base(question)
            self.last_routing_decision = (
                "keyword_fallback_true" if fallback else "keyword_fallback_false"
            )
            self._update_request_state(
                routing_decision=self.last_routing_decision,
            )
            self._log_event(
                "router_decision",
                decision=self.last_routing_decision,
                fallback=True,
            )
            LOGGER.warning(
                "[Agent] LLM Router 失败（%s），使用关键词 fallback：%s",
                exc,
                fallback,
            )
            return fallback

    @staticmethod
    def _evidence_json(evidence: dict) -> str:
        return json.dumps(evidence, ensure_ascii=False, indent=2)

    def _check_evidence_sufficiency(self, question: str, evidence: dict) -> bool:
        """Ask Qwen whether the first retrieval is sufficient."""
        prompt = EVIDENCE_CHECK_PROMPT_TEMPLATE.format(
            question=question,
            evidence=self._evidence_json(evidence),
        )
        raw_decision = self._call_model(prompt)
        decision = json.loads(raw_decision.strip())
        if not isinstance(decision, dict):
            raise ValueError("Evidence 判断 JSON 必须是对象")
        sufficient = decision.get("sufficient")
        if not isinstance(sufficient, bool):
            raise ValueError("sufficient 必须是 bool")
        return sufficient

    def _rewrite_query(self, question: str, evidence: dict) -> str:
        """Ask Qwen for one replacement retrieval query."""
        prompt = QUERY_REWRITE_PROMPT_TEMPLATE.format(
            question=question,
            evidence=self._evidence_json(evidence),
        )
        raw_rewrite = self._call_model(prompt)
        rewrite = json.loads(raw_rewrite.strip())
        if not isinstance(rewrite, dict):
            raise ValueError("Query Rewrite JSON 必须是对象")
        new_query = rewrite.get("query")
        if not isinstance(new_query, str) or not new_query.strip():
            raise ValueError("Rewrite 的 query 必须是非空字符串")
        normalized_query = new_query.strip()
        if self.request_state is not None:
            self.request_state.query_rewrite = normalized_query
        self._log_event("query_rewrite", query=normalized_query)
        return normalized_query

    def evaluate_retrieval_quality(
        self,
        question: str,
        evidence: dict,
    ) -> bool:
        """Combine result count, best distance, and Qwen relevance judgment."""
        results = evidence.get("results", [])
        if not isinstance(results, list):
            results = []
        self.last_result_count = len(results)
        self.last_best_distance = None
        self.last_relevance_decision = None
        self._high_confidence_fast_path = False
        self._update_request_state(
            retrieval_result_count=self.last_result_count,
            best_distance=None,
            relevance_decision=None,
        )
        LOGGER.info("[Agent] Retrieval result count: %d", self.last_result_count)

        if not results:
            self._log_event(
                "retrieval",
                result_count=self.last_result_count,
                best_distance=None,
            )
            self.last_retrieval_reliable = False
            self._update_request_state(retrieval_reliable=False)
            self._log_event(
                "retrieval_reliability",
                reliable=False,
                reason="no_results",
            )
            self.last_loop_decision = "unreliable_no_results"
            LOGGER.info("[Agent] Retrieval reliability: False")
            LOGGER.info("[Agent] Reliability reason: no_results")
            return False

        distances: list[float] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            try:
                distance = float(item["distance"])
            except (KeyError, TypeError, ValueError):
                continue
            if distance == distance:
                distances.append(distance)

        if not distances:
            self._log_event(
                "retrieval",
                result_count=self.last_result_count,
                best_distance=None,
            )
            self.last_retrieval_reliable = False
            self._update_request_state(retrieval_reliable=False)
            self._log_event(
                "retrieval_reliability",
                reliable=False,
                reason="distance_unavailable",
            )
            self.last_loop_decision = "unreliable_distance"
            LOGGER.info("[Agent] Best distance: unavailable")
            LOGGER.info("[Agent] Retrieval reliability: False")
            LOGGER.info("[Agent] Reliability reason: distance")
            return False

        self.last_best_distance = min(distances)
        self._update_request_state(best_distance=self.last_best_distance)
        self._log_event(
            "retrieval",
            result_count=self.last_result_count,
            best_distance=self.last_best_distance,
        )
        LOGGER.info("[Agent] Best distance: %.4f", self.last_best_distance)

        if (
            self.last_result_count >= self.high_confidence_min_results
            and self.last_best_distance <= self.high_confidence_distance
            and self.last_best_distance <= self.relevance_distance_threshold
        ):
            # A complete top-k result set with a very small distance is already
            # strong evidence.  Keep the public state explicit while avoiding
            # two redundant LLM classification calls for this request.
            self._high_confidence_fast_path = True
            self.last_relevance_decision = "skipped_high_confidence"
            self.last_retrieval_reliable = True
            self._update_request_state(
                relevance_decision=self.last_relevance_decision,
                retrieval_reliable=True,
            )
            self._log_event(
                "relevance_decision",
                decision=self.last_relevance_decision,
            )
            self._log_event(
                "retrieval_reliability",
                reliable=True,
                reason="high_confidence_fast_path",
            )
            self._log_event(
                "performance",
                stage="relevance",
                time_ms=0.0,
                skipped=True,
            )
            self._log_event(
                "performance",
                stage="sufficiency",
                time_ms=0.0,
                skipped=True,
            )
            self.last_loop_decision = "high_confidence_fast_path"
            LOGGER.info(
                "[Agent] High-confidence fast path: distance=%.4f results=%d",
                self.last_best_distance,
                self.last_result_count,
            )
            return True

        relevance_prompt = RELEVANCE_PROMPT_TEMPLATE.format(
            question=question,
            evidence=self._evidence_json(evidence),
        )
        try:
            raw_relevance = self._call_model(relevance_prompt)
            parsed = json.loads(raw_relevance.strip())
            if not isinstance(parsed, dict):
                raise ValueError("Relevance JSON 必须是对象")
            relevant = parsed.get("relevant")
            if not isinstance(relevant, bool):
                raise ValueError("relevant 必须是 bool")
            self.last_relevance_decision = "llm_true" if relevant else "llm_false"
            self._update_request_state(
                relevance_decision=self.last_relevance_decision,
            )
            self._log_event(
                "relevance_decision",
                decision=self.last_relevance_decision,
            )
            LOGGER.info("[Agent] LLM relevance: %s", relevant)
        except Exception as exc:
            distance_ok = self.last_best_distance <= self.relevance_distance_threshold
            self.last_relevance_decision = "llm_parse_fallback"
            self.last_retrieval_reliable = distance_ok
            self._update_request_state(
                relevance_decision=self.last_relevance_decision,
                retrieval_reliable=distance_ok,
            )
            self._log_event(
                "relevance_decision",
                decision=self.last_relevance_decision,
                fallback=True,
            )
            self._log_event(
                "retrieval_reliability",
                reliable=distance_ok,
                reason="distance_fallback",
            )
            self.last_loop_decision = "reliability_fallback"
            LOGGER.warning(
                "[Agent] LLM relevance 判断失败（%s），按 distance fallback：%s",
                exc,
                distance_ok,
            )
            LOGGER.info("[Agent] Retrieval reliability: %s", distance_ok)
            if not distance_ok:
                LOGGER.info("[Agent] Reliability reason: distance")
            return distance_ok

        distance_ok = self.last_best_distance <= self.relevance_distance_threshold
        if not distance_ok:
            self.last_retrieval_reliable = False
            self._update_request_state(retrieval_reliable=False)
            self._log_event(
                "retrieval_reliability",
                reliable=False,
                reason="distance",
            )
            self.last_loop_decision = "unreliable_distance"
            LOGGER.info("[Agent] Retrieval reliability: False")
            LOGGER.info("[Agent] Reliability reason: distance")
            return False
        if self.last_relevance_decision == "llm_false":
            self.last_retrieval_reliable = False
            self._update_request_state(retrieval_reliable=False)
            self._log_event(
                "retrieval_reliability",
                reliable=False,
                reason="relevance",
            )
            self.last_loop_decision = "unreliable_relevance"
            LOGGER.info("[Agent] Retrieval reliability: False")
            LOGGER.info("[Agent] Reliability reason: relevance")
            return False

        self.last_retrieval_reliable = True
        self._update_request_state(retrieval_reliable=True)
        self._log_event("retrieval_reliability", reliable=True)
        self.last_loop_decision = "reliable"
        LOGGER.info("[Agent] Retrieval reliability: True")
        return True

    def _evaluate_retrieval_quality_timed(
        self,
        question: str,
        evidence: dict,
    ) -> bool:
        """Measure quality evaluation while preserving override compatibility."""
        started_at = time.perf_counter()
        try:
            return self.evaluate_retrieval_quality(question, evidence)
        finally:
            self._log_performance("reliability", started_at)

    def _select_answer_strategy(self) -> str:
        """Map retrieval quality to a conservative answer policy."""
        if self.last_result_count == 0 or self.last_best_distance is None:
            return "insufficient"
        if self.last_retrieval_reliable:
            return "reliable"
        return "cautious"

    def _answer_with_evidence(
        self,
        question: str,
        evidence_sets: list[dict],
    ) -> str:
        """Generate the final answer from one or two structured evidence sets."""
        evidence_json = json.dumps(evidence_sets, ensure_ascii=False, indent=2)
        history_text = self._history_text_for_prompt()
        sources = format_sources(evidence_sets)
        if self.request_state is not None:
            self.request_state.sources = sources.copy()
        self._log_event("sources", count=len(sources), sources=sources)
        strategy = self.answer_strategy or "insufficient"
        strategy_instruction = {
            "reliable": (
                "证据质量可靠。请优先依据 Evidence 正常回答用户问题，"
                "可以引用其中的知识库内容，但只能陈述 Evidence 明确支持的事实，"
                "不得补充或推测未被 Evidence 支持的信息。"
            ),
            "cautious": (
                "证据存在但可靠性一般。请明确说明回答基于当前检索到的资料，"
                "优先使用‘根据当前资料’、‘目前检索到的信息显示’等谨慎表达；"
                "对不确定的信息不要补充为事实，也不要超出 Evidence 推断。"
            ),
            "insufficient": (
                "当前知识库证据不足，严禁编造或猜测答案。请明确说明当前知识库证据不足；"
                "如果这是二次检索后的 Evidence，必须直接说明无法从当前知识库确定该问题的答案。"
            ),
        }[strategy]
        source_lines = (
            "\n".join(f"[{index}] {source}" for index, source in enumerate(sources, 1))
            if sources
            else "[无可靠知识库来源]"
        )
        LOGGER.info("[Agent] Sources used: %s", ", ".join(sources) if sources else "none")
        instruction_text = (
            "你是严谨的中文问答助手。规则：只依据 Evidence 回答；"
            "Evidence 不支持的内容不得当作事实；证据不足要明确说明；"
            "只输出回答正文，不输出内部 JSON 或思考过程；不要自行编造来源。\n\n"
            f"回答策略：{strategy}\n{strategy_instruction}"
        )
        prompt = (
            f"{instruction_text}\n\n"
            f"最近对话（仅用于理解）：\n{history_text}\n\n"
            f"Evidence：\n{evidence_json}\n\n"
            f"程序确认的来源（回答末尾必须原样使用）：\n{source_lines}\n\n"
            f"用户问题：{question}"
        )
        self._log_prompt_profile(
            prompt=prompt,
            instruction_text=instruction_text,
            history_text=history_text,
            evidence_text=evidence_json,
        )
        answer = self._call_model(prompt)
        # 来源必须由程序从 Tool Evidence 生成，避免模型编造或重复引用。
        if "参考来源：" in answer:
            answer = answer.split("参考来源：", 1)[0].rstrip()
        references = "\n\n参考来源：\n" + source_lines
        return answer.rstrip() + references

    def _build_stream_evidence_prompt(
        self,
        question: str,
        evidence_sets: list[dict],
    ) -> tuple[str, str]:
        """Build a final-answer prompt that keeps Sources outside the stream."""
        evidence_json = json.dumps(evidence_sets, ensure_ascii=False, indent=2)
        history_text = self._history_text_for_prompt()
        sources = format_sources(evidence_sets)
        if self.request_state is not None:
            self.request_state.sources = sources.copy()
        self._log_event("sources", count=len(sources), sources=sources)
        strategy = self.answer_strategy or "insufficient"
        strategy_instruction = {
            "reliable": "证据质量可靠。优先依据 Evidence 正常回答，只陈述证据支持的事实。",
            "cautious": "证据可靠性一般。明确说明回答基于当前资料，使用谨慎措辞，不把不确定内容当事实。",
            "insufficient": "当前知识库证据不足。严禁编造或猜测；明确说明无法从当前知识库确定答案。",
        }[strategy]
        source_lines = (
            "\n".join(f"[{index}] {source}" for index, source in enumerate(sources, 1))
            if sources
            else "[无可靠知识库来源]"
        )
        LOGGER.info("[Agent] Sources used: %s", ", ".join(sources) if sources else "none")
        instruction_text = (
            "你是严谨的中文问答助手。只依据 Evidence 回答；"
            "Evidence 不支持的内容不得当作事实；证据不足要明确说明；"
            "只输出正文，不输出参考来源、内部 JSON 或思考过程。\n\n"
            f"回答策略：{strategy}\n{strategy_instruction}"
        )
        prompt = (
            f"{instruction_text}\n\n"
            f"最近对话（仅用于理解）：\n{history_text}\n\n"
            f"Evidence：\n{evidence_json}\n\n"
            f"程序确认的来源（不要在正文中复述）：\n{source_lines}\n\n"
            f"用户问题：{question}"
        )
        self._log_prompt_profile(
            prompt=prompt,
            instruction_text=instruction_text,
            history_text=history_text,
            evidence_text=evidence_json,
        )
        return prompt, source_lines

    def _stream_answer_with_evidence(
        self,
        question: str,
        evidence_sets: list[dict],
    ) -> Generator[str, None, str]:
        """Stream answer text and trusted Sources, then return the full answer."""
        prompt, source_lines = self._build_stream_evidence_prompt(question, evidence_sets)
        body_chunks: list[str] = []
        pending = ""
        suppress_model_sources = False
        marker = "参考来源"
        for chunk in self._stream_model(prompt):
            if suppress_model_sources:
                continue
            pending += chunk
            if marker in pending:
                visible = pending.split(marker, 1)[0].rstrip()
                if visible:
                    body_chunks.append(visible)
                    yield visible
                pending = ""
                suppress_model_sources = True
                continue
            safe_length = max(0, len(pending) - len(marker) + 1)
            if safe_length:
                visible = pending[:safe_length]
                pending = pending[safe_length:]
                body_chunks.append(visible)
                yield visible
        if pending and not suppress_model_sources:
            body_chunks.append(pending)
            yield pending
        references = "\n\n参考来源：\n" + source_lines
        yield references
        return "".join(body_chunks).rstrip() + references

    def _stream_rag_and_store(
        self,
        question: str,
        evidence_sets: list[dict],
    ) -> Generator[str, None, None]:
        """Commit a RAG turn to Memory only after its stream completes."""
        full_answer = yield from self._stream_answer_with_evidence(
            question,
            evidence_sets,
        )
        self.add_message("user", question)
        self.add_message("assistant", full_answer)

    def _answer_once(self, question: str) -> str:
        """Route ``question`` through the RAG Tool when appropriate, then answer."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question 必须是非空字符串")

        normalized_question = question.strip()

        if not self._should_use_knowledge_base(normalized_question):
            self.last_loop_decision = "not_needed"
            self.last_answer_strategy = "not_needed"
            self._record_answer_strategy()
            LOGGER.info("[Agent] Answer strategy: %s", self.last_answer_strategy)
            LOGGER.info("[Agent] 直接调用 Ollama（未调用 RAG Tool）")
            history_text = self._history_text_for_prompt()
            direct_instruction = (
                "你是严谨的中文问答助手。结合最近对话理解当前问题，"
                "直接给出自然、准确的回答；不要提及内部路由、工具或历史 JSON。"
            )
            direct_prompt = (
                f"{direct_instruction}\n\n最近对话：\n{history_text}\n\n"
                f"当前用户问题：{normalized_question}"
            )
            self._log_prompt_profile(
                prompt=direct_prompt,
                instruction_text=direct_instruction,
                history_text=history_text,
                evidence_text="",
            )
            return self._call_model(direct_prompt)

        self._record_tool_call(1)
        LOGGER.info("[Agent] RAG Tool call #1")
        try:
            evidence = self._invoke_tool(
                RAG_TOOL_NAME,
                query=normalized_question,
            )
        except Exception as exc:
            LOGGER.error("[Agent] RAG Tool 调用失败: %s", exc)
            raise AgentError(
                f"Agent 调用 search_knowledge_base 失败: {exc}"
            ) from exc

        retrieval_reliable = self._evaluate_retrieval_quality_timed(
            normalized_question,
            evidence,
        )
        reliability_fallback = self.last_loop_decision == "reliability_fallback"
        self.answer_strategy = self._select_answer_strategy()
        self.last_answer_strategy = self.answer_strategy
        self._record_answer_strategy()
        LOGGER.info("[Agent] Answer strategy: %s", self.last_answer_strategy)

        if retrieval_reliable and self._high_confidence_fast_path:
            LOGGER.info("[Agent] Evidence sufficiency skipped: high-confidence fast path")
            return self._answer_with_evidence(normalized_question, [evidence])

        if not retrieval_reliable:
            try:
                rewritten_query = self._rewrite_query(normalized_question, evidence)
            except Exception as exc:
                self.last_loop_decision = "rewrite_fallback"
                LOGGER.warning(
                    "[Agent] Query Rewrite 失败（%s），不进行第二次检索",
                    exc,
                )
                return self._answer_with_evidence(normalized_question, [evidence])

            LOGGER.info("[Agent] Query rewritten")
            self._record_tool_call(2)
            LOGGER.info("[Agent] RAG Tool call #2")
            try:
                second_evidence = self._invoke_tool(
                    RAG_TOOL_NAME,
                    query=rewritten_query,
                )
            except Exception as exc:
                LOGGER.error("[Agent] RAG Tool 调用失败: %s", exc)
                raise AgentError(
                    f"Agent 第二次调用 search_knowledge_base 失败: {exc}"
                ) from exc
            return self._answer_with_evidence(
                normalized_question,
                [evidence, second_evidence],
            )

        try:
            sufficient = self._check_evidence_sufficiency(
                normalized_question,
                evidence,
            )
        except Exception as exc:
            self.last_loop_decision = "evidence_check_fallback"
            self.answer_strategy = "cautious"
            self.last_answer_strategy = self.answer_strategy
            self._record_answer_strategy()
            LOGGER.info("[Agent] Answer strategy: %s", self.last_answer_strategy)
            LOGGER.warning(
                "[Agent] Evidence 判断失败（%s），直接使用已有 Evidence",
                exc,
            )
            return self._answer_with_evidence(normalized_question, [evidence])

        LOGGER.info("[Agent] Evidence sufficiency: %s", sufficient)
        if sufficient:
            if not reliability_fallback:
                self.last_loop_decision = "sufficient"
            return self._answer_with_evidence(normalized_question, [evidence])

        self.last_loop_decision = "insufficient_rewrite"
        self.answer_strategy = "insufficient"
        self.last_answer_strategy = self.answer_strategy
        self._record_answer_strategy()
        LOGGER.info("[Agent] Answer strategy: %s", self.last_answer_strategy)
        LOGGER.info("[Agent] Evidence sufficiency: false")
        try:
            rewritten_query = self._rewrite_query(normalized_question, evidence)
        except Exception as exc:
            self.last_loop_decision = "rewrite_fallback"
            LOGGER.warning(
                "[Agent] Query Rewrite 失败（%s），不进行第二次检索",
                exc,
            )
            return self._answer_with_evidence(normalized_question, [evidence])

        LOGGER.info("[Agent] Query rewritten")
        self._record_tool_call(2)
        LOGGER.info("[Agent] RAG Tool call #2")
        try:
            second_evidence = self._invoke_tool(
                RAG_TOOL_NAME,
                query=rewritten_query,
            )
        except Exception as exc:
            LOGGER.error("[Agent] RAG Tool 调用失败: %s", exc)
            raise AgentError(
                f"Agent 第二次调用 search_knowledge_base 失败: {exc}"
            ) from exc

        return self._answer_with_evidence(
            normalized_question,
            [evidence, second_evidence],
        )

    def _stream_answer_once(self, normalized: str) -> Iterator[str]:
        """Run the existing streaming flow inside a request-state wrapper."""
        if not self._should_use_knowledge_base(normalized):
            self.last_loop_decision = "not_needed"
            self.last_answer_strategy = "not_needed"
            self._record_answer_strategy()
            LOGGER.info("[Agent] Answer strategy: %s", self.last_answer_strategy)
            history_text = self._history_text_for_prompt()
            direct_instruction = (
                "你是严谨的中文问答助手。结合最近对话理解当前问题，"
                "只输出自然、准确的回答正文，不输出参考来源、内部 JSON 或思考过程。"
            )
            prompt = (
                f"{direct_instruction}\n\n最近对话：\n{history_text}\n\n"
                f"当前用户问题：{normalized}"
            )
            self._log_prompt_profile(
                prompt=prompt,
                instruction_text=direct_instruction,
                history_text=history_text,
                evidence_text="",
            )
            body: list[str] = []
            for chunk in self._stream_model(prompt):
                body.append(chunk)
                yield chunk
            self.add_message("user", normalized)
            self.add_message("assistant", "".join(body))
            return

        self._record_tool_call(1)
        LOGGER.info("[Agent] RAG Tool call #1")
        try:
            evidence = self._invoke_tool(RAG_TOOL_NAME, query=normalized)
        except Exception as exc:
            raise AgentError(f"Agent 调用 search_knowledge_base 失败: {exc}") from exc
        reliable = self._evaluate_retrieval_quality_timed(normalized, evidence)
        fallback = self.last_loop_decision == "reliability_fallback"
        self.answer_strategy = self.last_answer_strategy = self._select_answer_strategy()
        self._record_answer_strategy()
        LOGGER.info("[Agent] Answer strategy: %s", self.last_answer_strategy)
        evidence_sets = [evidence]
        if reliable and self._high_confidence_fast_path:
            LOGGER.info("[Agent] Evidence sufficiency skipped: high-confidence fast path")
            full_answer = yield from self._stream_answer_with_evidence(
                normalized,
                evidence_sets,
            )
            self.add_message("user", normalized)
            self.add_message("assistant", full_answer)
            return

        if reliable:
            try:
                sufficient = self._check_evidence_sufficiency(normalized, evidence)
            except Exception as exc:
                self.last_loop_decision = "evidence_check_fallback"
                self.answer_strategy = self.last_answer_strategy = "cautious"
                self._record_answer_strategy()
                LOGGER.warning("[Agent] Evidence 判断失败（%s），直接使用已有 Evidence", exc)
                sufficient = True
            if sufficient:
                if not fallback:
                    self.last_loop_decision = "sufficient"
            else:
                self.last_loop_decision = "insufficient_rewrite"
                self.answer_strategy = self.last_answer_strategy = "insufficient"
                self._record_answer_strategy()
                try:
                    rewritten = self._rewrite_query(normalized, evidence)
                except Exception as exc:
                    self.last_loop_decision = "rewrite_fallback"
                    LOGGER.warning("[Agent] Query Rewrite 失败（%s），不进行第二次检索", exc)
                else:
                    LOGGER.info("[Agent] Query rewritten")
                    self._record_tool_call(2)
                    LOGGER.info("[Agent] RAG Tool call #2")
                    try:
                        second_evidence = self._invoke_tool(
                            RAG_TOOL_NAME,
                            query=rewritten,
                        )
                    except Exception as exc:
                        LOGGER.error("[Agent] RAG Tool 调用失败: %s", exc)
                        raise AgentError(
                            f"Agent 第二次调用 search_knowledge_base 失败: {exc}"
                        ) from exc
                    evidence_sets.append(second_evidence)
        else:
            try:
                rewritten = self._rewrite_query(normalized, evidence)
            except Exception as exc:
                self.last_loop_decision = "rewrite_fallback"
                LOGGER.warning("[Agent] Query Rewrite 失败（%s），不进行第二次检索", exc)
            else:
                LOGGER.info("[Agent] Query rewritten")
                self._record_tool_call(2)
                LOGGER.info("[Agent] RAG Tool call #2")
                try:
                    second_evidence = self._invoke_tool(
                        RAG_TOOL_NAME,
                        query=rewritten,
                    )
                except Exception as exc:
                    LOGGER.error("[Agent] RAG Tool 调用失败: %s", exc)
                    raise AgentError(
                        f"Agent 第二次调用 search_knowledge_base 失败: {exc}"
                    ) from exc
                evidence_sets.append(second_evidence)

        full_answer = yield from self._stream_answer_with_evidence(normalized, evidence_sets)
        self.add_message("user", normalized)
        self.add_message("assistant", full_answer)

    def stream_answer(self, question: str) -> Iterator[str]:
        """Stream one answer and record an independent request state."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question 必须是非空字符串")
        normalized = question.strip()
        self._start_request(normalized)
        outcome = "error"
        try:
            yield from self._stream_answer_once(normalized)
            outcome = "completed"
        except GeneratorExit:
            outcome = "cancelled"
            raise
        finally:
            self._finish_request(outcome)

    def answer(self, question: str) -> str:
        """Answer once, then persist the user/assistant turn in short-term memory."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question 必须是非空字符串")

        normalized_question = question.strip()
        self._start_request(normalized_question)
        outcome = "error"
        try:
            answer = self._answer_once(normalized_question)
            self.add_message("user", normalized_question)
            self.add_message("assistant", answer)
            outcome = "completed"
            return answer
        finally:
            self._finish_request(outcome)


__all__ = ["Agent", "AgentError", "AgentRequestState", "get_rag_tool_interface"]
