"""
Demo script showing the three execution modes of Unified Agent Architecture.

Run with: python demo_unified_modes.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from pydantic import BaseModel, Field

from kimi_cli.soul.unified import AgentMode
from kimi_cli.soul.unified.soul import UnifiedSoul
from kimi_cli.soul.unified.agent import UnifiedAgent, ExecutionMode, ExecutionConfig
from kimi_cli.soul.unified.context import UnifiedContext, MemoryFragment, FragmentType
from kimi_cli.soul.unified.tool import UnifiedTool, unified_tool, register_tool
from kimi_cli.soul.unified.graph_engine import TaskGraph, TaskNode


# =============================================================================
# Demo Tools
# =============================================================================

class CalculatorParams(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")
    operation: str = Field(description="Operation")


@register_tool
class CalculatorTool(UnifiedTool[CalculatorParams]):
    """Calculator for demo."""
    name = "calculator"
    description = "Perform calculations"
    params = CalculatorParams
    
    async def execute(self, params: CalculatorParams) -> dict:
        ops = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else "Error",
        }
        result = ops.get(params.operation, lambda x, y: "Unknown")(params.a, params.b)
        return {"result": result, "operation": params.operation}


@unified_tool(name="greet", description="Greet someone")
async def greet(name: str, greeting: str = "Hello") -> str:
    """Greeting tool."""
    return f"{greeting}, {name}!"


# =============================================================================
# Mock Runtime Setup
# =============================================================================

class MockRuntime:
    """Mock runtime for demo."""
    llm = None
    session = type('obj', (object,), {'context_file': Path('/tmp/demo.jsonl')})()
    denwa_renji = type('obj', (object,), {
        'set_n_checkpoints': lambda x: None,
        'fetch_pending_dmail': lambda: None,
    })()
    approval = type('obj', (object,), {
        'is_yolo': lambda: False,
        'share': lambda self: self,
    })()
    config = type('obj', (object,), {
        'loop_control': type('obj', (object,), {
            'max_steps_per_turn': 50,
            'max_ralph_iterations': 0,
            'max_retries_per_step': 3,
            'reserved_context_size': 1000,
        })(),
    })()
    labor_market = type('obj', (object,), {'add_fixed_subagent': lambda *args: None})()
    skills = {}


class MockToolset:
    """Mock toolset."""
    tools = [CalculatorTool(), greet]


def create_demo_agent(mode: ExecutionMode) -> UnifiedAgent:
    """Create a demo agent with specified mode."""
    return UnifiedAgent(
        name="demo_agent",
        system_prompt="You are a helpful demo agent.",
        toolset=MockToolset(),
        runtime=MockRuntime(),
        execution_config=ExecutionConfig(
            mode=mode,
            max_steps=10,
            max_iterations=3,
            parallel_tool_calls=True,
            parallel_tool_calls_limit=5,
        ),
    )


# =============================================================================
# Demo Functions
# =============================================================================

async def demo_reactive_mode():
    """Demo: Reactive Mode (Step-by-step execution)."""
    print("\n" + "="*60)
    print("MODE 1: REACTIVE (Step-by-step execution)")
    print("="*60)
    
    agent = create_demo_agent(ExecutionMode.REACTIVE)
    context = UnifiedContext(file_backend=Path("/tmp/demo_reactive.jsonl"))
    soul = UnifiedSoul(agent, context, mode=AgentMode.REACTIVE)
    
    print(f"Agent: {soul.name}")
    print(f"Mode: {soul.mode.name}")
    print(f"Tools: {[t.name for t in agent.toolset.tools]}")
    
    # Simulate simple task
    print("\nSimulating: 'What is 5 + 3?'")
    print("Execution: Single step, direct response")
    
    # Execute tool directly
    tool = CalculatorTool()
    result = await tool(CalculatorParams(a=5, b=3, operation="add"))
    print(f"Result: {result.output}")
    
    print("\n✓ Reactive mode: Best for simple, linear tasks")


async def demo_proactive_mode():
    """Demo: Proactive Mode (Graph-based parallel execution)."""
    print("\n" + "="*60)
    print("MODE 2: PROACTIVE (Graph-based parallel execution)")
    print("="*60)
    
    agent = create_demo_agent(ExecutionMode.PROACTIVE)
    context = UnifiedContext(file_backend=Path("/tmp/demo_proactive.jsonl"))
    soul = UnifiedSoul(agent, context, mode=AgentMode.PROACTIVE)
    
    print(f"Agent: {soul.name}")
    print(f"Mode: {soul.mode.name}")
    print(f"Graph Engine: {soul._graph_engine is not None}")
    
    # Create a task graph
    print("\nCreating task graph for: 'Analyze codebase'")
    graph = TaskGraph()
    
    # Independent tasks (can run in parallel)
    graph.add_node(TaskNode(
        id="find_files",
        description="Find all Python files",
        tool=None,
        args={},
        dependencies=[],
    ))
    graph.add_node(TaskNode(
        id="check_deps",
        description="Check dependencies",
        tool=None,
        args={},
        dependencies=[],
    ))
    
    # Dependent task (runs after above complete)
    graph.add_node(TaskNode(
        id="analyze",
        description="Analyze code structure",
        tool=None,
        args={},
        dependencies=["find_files", "check_deps"],
    ))
    
    print(f"Nodes: {list(graph.nodes.keys())}")
    print(f"Ready to execute: {[n.id for n in graph.get_ready_nodes()]}")
    
    # Simulate execution
    print("\nExecution flow:")
    print("  1. find_files + check_deps (PARALLEL)")
    print("  2. analyze (after both complete)")
    
    print("\n✓ Proactive mode: Best for complex, parallelizable tasks")


async def demo_adaptive_mode():
    """Demo: Adaptive Mode (Automatic mode selection)."""
    print("\n" + "="*60)
    print("MODE 3: ADAPTIVE (Automatic mode selection)")
    print("="*60)
    
    agent = create_demo_agent(ExecutionMode.ADAPTIVE)
    context = UnifiedContext(file_backend=Path("/tmp/demo_adaptive.jsonl"))
    soul = UnifiedSoul(agent, context, mode=AgentMode.ADAPTIVE)
    
    print(f"Agent: {soul.name}")
    print(f"Mode: {soul.mode.name}")
    print(f"Auto-select: Enabled")
    
    # Test task analysis
    test_tasks = [
        ("What is 2 + 2?", "simple"),
        ("Hello", "simple"),
        ("Analyze all files and refactor code", "complex"),
        ("Multiple parallel tasks to execute", "complex"),
    ]
    
    print("\nTask analysis:")
    for task, expected in test_tasks:
        # Simple heuristic
        complexity_indicators = ["complex", "multiple", "parallel", "analyze", "refactor"]
        is_complex = any(w in task.lower() for w in complexity_indicators)
        detected = "complex" if is_complex else "simple"
        match = "✓" if detected == expected else "✗"
        print(f"  {match} '{task[:40]}...' -> {detected}")
    
    print("\n✓ Adaptive mode: Best for mixed workloads")


async def demo_mode_switching():
    """Demo: Runtime mode switching."""
    print("\n" + "="*60)
    print("FEATURE: Runtime Mode Switching")
    print("="*60)
    
    agent = create_demo_agent(ExecutionMode.REACTIVE)
    context = UnifiedContext(file_backend=Path("/tmp/demo_switch.jsonl"))
    soul = UnifiedSoul(agent, context, mode=AgentMode.REACTIVE)
    
    print(f"Initial mode: {soul.mode.name}")
    
    # Switch modes
    soul.set_mode(AgentMode.PROACTIVE)
    print(f"After switch: {soul.mode.name}")
    
    soul.set_mode(AgentMode.ADAPTIVE)
    print(f"After switch: {soul.mode.name}")
    
    print("\n✓ Modes can be switched at runtime via /mode command")


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("Unified Agent Architecture - Mode Demo")
    print("="*60)
    
    # await demo_reactive_mode()
    await demo_proactive_mode()
    # await demo_adaptive_mode()
    # await demo_mode_switching()
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nSummary:")
    print("  • REACTIVE: Step-by-step, good for simple tasks")
    print("  • PROACTIVE: Graph-based parallel, good for complex tasks")
    print("  • ADAPTIVE: Auto-select, good for mixed workloads")
    print("\nUse in code:")
    print("  soul = UnifiedSoul(agent, context, mode=AgentMode.PROACTIVE)")
    print("  await soul.run('Your task here')")
    print()


if __name__ == "__main__":
    asyncio.run(main())
