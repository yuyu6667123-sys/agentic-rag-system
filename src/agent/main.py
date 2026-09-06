"""Command-line smoke test for the minimal Ollama Agent."""

from __future__ import annotations

import argparse
import logging
import sys

from .agent import Agent, AgentError, get_rag_tool_interface
from ..tools.tool_registry import ToolNotFoundError, ToolRegistry


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="最基础的 Ollama Agent 测试")
    parser.add_argument(
        "question",
        nargs="?",
        default="你好，请介绍一下你自己。",
        help="发送给 Agent 的问题",
    )
    parser.add_argument(
        "--memory-test",
        action="store_true",
        help="连续回答两轮并验证短期内存的读取与清空",
    )
    parser.add_argument(
        "--tool-info",
        action="store_true",
        help="显示统一 RAG Tool 接口描述并退出",
    )
    parser.add_argument(
        "--registry-test",
        action="store_true",
        help="测试 Tool Registry 的注册、查询、执行和未知 Tool 异常",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="使用 Ollama Streaming 逐块输出最终回答",
    )
    args = parser.parse_args()

    if args.tool_info:
        print(f"RAG Tool 接口：{get_rag_tool_interface()}")
        return

    if args.registry_test:
        _run_registry_test()
        return

    if args.memory_test:
        _run_memory_test()
        return

    print(f"用户问题：{args.question}")
    print("Agent：调用 Ollama 生成回答...")
    agent = Agent()
    try:
        if args.stream:
            print("最终回答：")
            for chunk in agent.stream_answer(args.question):
                print(chunk, end="", flush=True)
            print()
        else:
            answer = agent.answer(args.question)
    except AgentError as exc:
        print(f"Agent 错误：{exc}")
        raise SystemExit(1) from exc
    print(f"routing decision：{agent.last_routing_decision}")
    print(f"是否调用 search_knowledge_base：{'是' if agent.last_tool_called else '否'}")
    if not args.stream:
        print(f"最终回答：\n{answer}")


def _run_memory_test() -> None:
    """Run a small manual smoke test for the in-memory conversation history."""
    agent = Agent()
    questions = ["什么是 RAG？", "它有什么优点？"]
    for question in questions:
        print(f"用户问题：{question}")
        try:
            answer = agent.answer(question)
        except AgentError as exc:
            print(f"Agent 错误：{exc}")
            raise SystemExit(1) from exc
        print(f"routing decision：{agent.last_routing_decision}")
        print(f"Tool 调用次数：{agent.last_tool_call_count}")
        print(f"最终回答：\n{answer}")

    history = agent.get_history()
    print(f"Memory 历史消息数：{len(history)}")
    print(f"Memory 历史：{history}")
    agent.clear_history()
    print(f"Memory 清空后消息数：{len(agent.get_history())}")

    for index in range(14):
        role = "user" if index % 2 == 0 else "assistant"
        agent.add_message(role, f"测试消息 {index + 1}")
    bounded_history = agent.get_history()
    print(f"Memory 截断后消息数：{len(bounded_history)}")
    print(f"Memory 截断后首条/末条：{bounded_history[0]} / {bounded_history[-1]}")
    print(
        "Memory 角色顺序正确："
        f"{all(message['role'] == ('user' if index % 2 == 0 else 'assistant') for index, message in enumerate(bounded_history))}"
    )


def _run_registry_test() -> None:
    """Run a small smoke test for the Tool Registry contract."""
    registry = ToolRegistry()
    registry.register(
        name="test_tool",
        description="Registry smoke-test Tool",
        input_schema={"value": "str"},
        output_schema={"value": "str"},
        handler=lambda value: {"value": value},
    )
    description = registry.get("test_tool")
    result = registry.execute("test_tool", value="ok")
    try:
        registry.execute("missing_tool")
    except ToolNotFoundError as exc:
        unknown_error = str(exc)
    else:
        unknown_error = ""
    print(f"Registry 注册/查询：{description['name'] == 'test_tool'}")
    print(f"Registry 执行：{result}")
    print(f"Registry 未知 Tool 异常：{unknown_error}")


if __name__ == "__main__":
    main()
