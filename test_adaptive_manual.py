#!/usr/bin/env python3
"""Manual test script for adaptive planning feature.

This script tests the adaptive planning functionality without needing
the full Kimi CLI environment.

Usage:
    python test_adaptive_manual.py

Or enable in Kimi CLI config:
    [loop_control]
    adaptive_planning = true
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from kosong.message import Message
from kosong.tooling.empty import EmptyToolset

from kimi_cli.soul.adaptive import (
    ExecutionController,
    ExecutionPlan,
    Planner,
    Stage,
    StageExecutor,
    StageResult,
    Transition,
)


async def mock_llm_for_planning(prompt: str, messages: list[Message]) -> Message:
    """Mock LLM that returns a valid execution plan."""
    plan = {
        "stages": [
            {
                "id": "analyze",
                "name": "分析任务",
                "goal": "理解用户需求和当前状态",
                "exit_criteria": "明确了任务范围和目标",
                "tools_hint": ["read_file", "list_dir"],
                "max_iterations": 5,
            },
            {
                "id": "plan",
                "name": "制定计划",
                "goal": "制定详细的执行步骤",
                "exit_criteria": "有了清晰的执行计划",
                "tools_hint": ["think"],
                "max_iterations": 3,
            },
            {
                "id": "execute",
                "name": "执行计划",
                "goal": "按照计划执行任务",
                "exit_criteria": "任务执行完成",
                "tools_hint": ["read_file", "write_file", "shell"],
                "max_iterations": 10,
            },
            {
                "id": "verify",
                "name": "验证结果",
                "goal": "检查结果是否正确",
                "exit_criteria": "结果验证通过",
                "tools_hint": ["read_file"],
                "max_iterations": 5,
            },
        ],
        "transitions": [
            {"from": "analyze", "to": "plan", "condition": "分析完成"},
            {"from": "plan", "to": "execute", "condition": "计划制定完成"},
            {"from": "execute", "to": "verify", "condition": "执行完成"},
            {"from": "verify", "to": "execute", "condition": "验证失败，需要重新执行"},
        ],
        "entry": "analyze",
        "exits": ["verify"],
    }
    return Message(role="assistant", content=f"```json\n{json.dumps(plan, ensure_ascii=False)}\n```")


async def mock_llm_for_execution(prompt: str, messages: list[Message]) -> Message:
    """Mock LLM that simulates stage execution."""
    # Simulate completing the stage after a few steps
    step_count = len([m for m in messages if m.role == "assistant"])

    if step_count >= 2:
        return Message(
            role="assistant",
            content="<stage_complete>阶段执行完成，达成了目标</stage_complete>")

    return Message(
        role="assistant",
        content="Action: think({'thought': '正在分析当前状态...'})")


async def mock_tool_executor(name: str, params: dict) -> MagicMock:
    """Mock tool executor."""
    result = MagicMock()
    result.tool_call_id = f"test_{name}"
    result.tool_name = name
    result.return_value = f"[Mock result of {name} with {params}]"
    return result


async def test_planner():
    """Test the Planner component."""
    print("=" * 60)
    print("测试 Planner - 生成执行计划")
    print("=" * 60)

    planner = Planner(
        llm_call=mock_llm_for_planning,
        tools=["read_file", "write_file", "shell", "think"],
        work_dir="/tmp/test",
    )

    plan = await planner.plan("帮我创建一个 Python 项目", {})

    print(f"\n生成的计划包含 {len(plan.stages)} 个阶段:")
    for stage_id, stage in plan.stages.items():
        print(f"  - {stage_id}: {stage.name}")
        print(f"    目标: {stage.goal}")
        print(f"    退出条件: {stage.exit_criteria}")
        print(f"    建议工具: {stage.tools_hint}")
        print()

    print(f"入口阶段: {plan.entry}")
    print(f"出口阶段: {plan.exits}")
    print()

    return plan


async def test_stage_executor():
    """Test the StageExecutor component."""
    print("=" * 60)
    print("测试 StageExecutor - 执行单个阶段")
    print("=" * 60)

    executor = StageExecutor(
        llm_call=mock_llm_for_execution,
        tool_executor=mock_tool_executor,
    )

    stage = Stage(
        id="test_stage",
        name="测试阶段",
        goal="完成测试任务",
        exit_criteria="测试通过",
        max_iterations=5,
        tools_hint=["think"],
    )

    # Mock context
    context = MagicMock()
    context.token_count = 100
    context.max_context_size = 100000

    result = await executor.execute(stage, context, {})

    print(f"\n阶段执行结果:")
    print(f"  状态: {result.status}")
    print(f"  摘要: {result.summary}")
    print(f"  执行步数: {len(result.messages)}")
    print()

    return result


async def test_execution_controller():
    """Test the full ExecutionController."""
    print("=" * 60)
    print("测试 ExecutionController - 完整执行流程")
    print("=" * 60)

    planner = Planner(
        llm_call=mock_llm_for_planning,
        tools=["read_file", "write_file", "shell", "think"],
        work_dir="/tmp/test",
    )

    executor = StageExecutor(
        llm_call=mock_llm_for_execution,
        tool_executor=mock_tool_executor,
    )

    controller = ExecutionController(
        planner=planner,
        executor=executor,
        max_global_iterations=20,
        max_replans=2,
    )

    # Mock context
    context = MagicMock()
    context.token_count = 100
    context.max_context_size = 100000

    print("\n开始执行任务: '帮我创建一个 Python 项目'\n")

    result = await controller.execute("帮我创建一个 Python 项目", context)

    print("\n" + "=" * 60)
    print("执行完成!")
    print("=" * 60)
    print(f"最终状态: {result.status}")
    print(f"执行摘要: {result.summary}")

    if result.artifacts:
        print(f"\n产出物:")
        for key, value in result.artifacts.items():
            print(f"  {key}: {value}")

    return result


async def test_with_kimisoul():
    """Test integration with KimiSoul (requires full environment)."""
    print("=" * 60)
    print("测试 KimiSoul 集成 (需要完整环境)")
    print("=" * 60)

    try:
        from kimi_cli.soul.kimisoul import KimiSoul
        from kimi_cli.soul.agent import Agent, Runtime

        print("\n要测试 KimiSoul 集成，需要:")
        print("1. 配置好的 LLM provider")
        print("2. 有效的 agent 配置")
        print("3. 在配置中启用 adaptive_planning = true")
        print("\n请在实际环境中测试，或运行单元测试:")
        print("  pytest tests/core/test_adaptive.py -v")

    except ImportError as e:
        print(f"无法导入 KimiSoul: {e}")
        print("请确保在正确的环境中运行")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("自适应规划 (Adaptive Planning) 测试")
    print("=" * 60 + "\n")

    async def run_tests():
        # Test 1: Planner
        await test_planner()

        # Test 2: Stage Executor
        await test_stage_executor()

        # Test 3: Full Controller
        await test_execution_controller()

        # Test 4: Integration
        await test_with_kimisoul()

    asyncio.run(run_tests())

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
    print("\n要在 Kimi CLI 中启用自适应规划:")
    print("1. 编辑 ~/.config/kimi-cli/config.toml")
    print("2. 添加: adaptive_planning = true")
    print("3. 运行 kimi 并输入任务")


if __name__ == "__main__":
    main()
