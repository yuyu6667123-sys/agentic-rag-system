"""面试演示入口：调用现有 Agent 能力，不复制 Agent 业务逻辑。"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

from src.agent.agent import Agent, AgentError


ROOT_DIR = Path(__file__).resolve().parent
PERFORMANCE_REPORT = ROOT_DIR / "docs" / "performance_report.md"

MENU = """
=================================
      Agentic RAG Demo System
=================================

请选择演示模式：

1. 普通 Agent 问答
2. RAG 知识库问答
3. Query Rewrite 演示
4. Memory 多轮对话
5. Agent State 追踪
6. 性能分析展示
7. 退出
"""


def _print_state(state: dict[str, Any] | None, *, compact: bool = False) -> None:
    """Print public request state returned by the existing Agent."""
    if not state:
        print("State：暂无（请求可能在模型调用前失败）")
        return
    if compact:
        for key in (
            "request_id",
            "routing_decision",
            "tool_call_count",
            "retrieval_result_count",
            "best_distance",
            "retrieval_reliable",
            "query_rewrite",
            "answer_strategy",
            "total_time_ms",
        ):
            print(f"{key}: {state.get(key)}")
        if state.get("sources"):
            print(f"sources: {', '.join(str(item) for item in state['sources'])}")
        return
    print(json.dumps(state, ensure_ascii=False, indent=2))


def _friendly_error(exc: BaseException) -> str:
    """Turn infrastructure errors into concise instructions for a demo audience."""
    message = str(exc)
    normalized = message.casefold()
    if any(marker in normalized for marker in ("11434", "ollama", "connection refused")):
        return "Ollama 服务未启动，请先运行 `ollama serve`，并确认已准备 qwen3:4b。"
    if any(marker in normalized for marker in ("chroma", "knowledge base", "search_knowledge_base")):
        return "知识库未初始化或当前不可访问，请先完成知识库 ingest。"
    if not message:
        return "演示执行失败，请检查本地服务状态。"
    return f"演示执行失败：{message}"


def _run_answer_demo(
    title: str,
    question: str,
    *,
    show_state: bool = True,
    agent: Agent | None = None,
) -> tuple[Agent, str | None, dict[str, Any] | None]:
    print(f"\n--- {title} ---")
    print(f"Question: {question}")
    current_agent = agent or Agent()
    try:
        answer = current_agent.answer(question)
    except (AgentError, OSError, TimeoutError, ValueError) as exc:
        print(f"[Demo] {_friendly_error(exc)}")
        return current_agent, None, current_agent.get_request_state()
    except Exception as exc:  # noqa: BLE001 - demo boundary must remain usable.
        print(f"[Demo] {_friendly_error(exc)}")
        return current_agent, None, current_agent.get_request_state()

    state = current_agent.get_request_state()
    if show_state:
        print("State:")
        _print_state(state, compact=True)
    print(f"Answer:\n{answer}\n")
    return current_agent, answer, state


def demo_plain_answer() -> None:
    agent, _, state = _run_answer_demo(
        "普通 Agent 问答",
        "什么是人工智能？",
    )
    if state:
        print(
            "[Demo] Router 结果："
            f"{state.get('routing_decision')}；Tool 调用次数：{state.get('tool_call_count')}"
        )
        print("[Demo] 该结果用于展示普通问题可以绕过知识库。")
    del agent


def demo_rag_answer() -> None:
    agent, _, state = _run_answer_demo(
        "RAG 知识库问答",
        "什么是 RAG？",
    )
    if state:
        print(f"Routing: {state.get('routing_decision')}")
        print(f"Tool: rag_tool（调用 {state.get('tool_call_count')} 次）")
        print(f"Retrieval: {state.get('retrieval_result_count')} documents")
        print(f"Sources: {', '.join(state.get('sources') or []) or '无'}")
        print("[Demo] 该结果用于展示知识库增强回答和来源可追溯。")
    del agent


def demo_query_rewrite() -> None:
    print("\n--- Query Rewrite 演示 ---")
    agent = Agent()
    first_question = "什么是 RAG？"
    second_question = "它有什么优势？"
    print(f"第一轮：{first_question}")
    try:
        first_answer = agent.answer(first_question)
        first_state = agent.get_request_state()
        print(f"第一轮回答：\n{first_answer}\n")
        print("第一轮 State:")
        _print_state(first_state, compact=True)

        print(f"\n第二轮：{second_question}")
        second_answer = agent.answer(second_question)
        second_state = agent.get_request_state()
    except (AgentError, OSError, TimeoutError, ValueError) as exc:
        print(f"[Demo] {_friendly_error(exc)}")
        return
    except Exception as exc:  # noqa: BLE001 - demo boundary must remain usable.
        print(f"[Demo] {_friendly_error(exc)}")
        return

    rewrite = (second_state or {}).get("query_rewrite")
    print(f"Original Query: {second_question}")
    print(f"Rewrite Query: {rewrite or '未触发（当前证据已足够）'}")
    print(f"Tool Call Count: {(second_state or {}).get('tool_call_count')}")
    print(f"Answer:\n{second_answer}\n")
    print("第二轮 State:")
    _print_state(second_state, compact=True)
    print("[Demo] 第二轮使用了第一轮 Memory；是否实际改写由当前检索证据决定。")


def demo_memory() -> None:
    print("\n--- Memory 多轮对话 ---")
    agent = Agent()
    questions = ["我的名字叫张三。", "我叫什么？"]
    try:
        for question in questions:
            print(f"user: {question}")
            answer = agent.answer(question)
            print(f"assistant: {answer}\n")
    except (AgentError, OSError, TimeoutError, ValueError) as exc:
        print(f"[Demo] {_friendly_error(exc)}")
        return
    except Exception as exc:  # noqa: BLE001 - demo boundary must remain usable.
        print(f"[Demo] {_friendly_error(exc)}")
        return

    print("Memory:")
    for message in agent.get_history():
        print(f"{message['role']}: {message['content']}")
    print("\n[Demo] Memory 仅保存在当前 Agent 实例中，并受现有长度限制。")


def demo_state() -> None:
    print("\n--- Agent State 追踪 ---")
    agent, _, state = _run_answer_demo(
        "Agent State",
        "什么是 RAG？",
        show_state=False,
    )
    print("AgentRequestState:")
    _print_state(state)
    del agent


def _report_metric(report: str, label: str) -> str:
    pattern = rf"^\|\s*{re.escape(label)}\s*\|\s*(.*?)\s*\|\s*$"
    match = re.search(pattern, report, flags=re.MULTILINE)
    return match.group(1) if match else "报告中未找到"


def demo_performance() -> None:
    print("\n--- 性能分析展示 ---")
    try:
        report = PERFORMANCE_REPORT.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[Demo] 无法读取性能报告：{exc}")
        return

    print("数据来源：docs/performance_report.md")
    for label in ("Router", "Chroma Retrieval（含本进程 Embedding 冷启动）", "Final Generation", "Total"):
        print(f"{label}: {_report_metric(report, label)}")
    print("\n已记录的 Step16 优化：")
    for item in (
        "Fast Path：高置信检索时跳过重复判断",
        "Prompt 优化：限制历史和输入长度",
        "Embedding / Chroma 预热",
        "SSE status：反馈分析、检索、生成阶段",
    ):
        print(f"- {item}")


def _handlers() -> dict[str, Callable[[], None]]:
    return {
        "1": demo_plain_answer,
        "2": demo_rag_answer,
        "3": demo_query_rewrite,
        "4": demo_memory,
        "5": demo_state,
        "6": demo_performance,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    handlers = _handlers()
    while True:
        print(MENU)
        try:
            choice = input("请输入选项（1-7）：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDemo 已退出。")
            return
        if choice == "7":
            print("Demo 已退出。")
            return
        handler = handlers.get(choice)
        if handler is None:
            print("请输入 1-7 之间的数字。")
            continue
        handler()
        input("\n按 Enter 返回菜单...")


if __name__ == "__main__":
    main()
