"""
Simple Demo: Unified Agent Architecture

This script demonstrates the three execution modes without requiring an LLM.
"""

import asyncio
from pathlib import Path
from pydantic import BaseModel, Field

from kimi_cli.soul.unified import AgentMode
from kimi_cli.soul.unified.agent import ExecutionMode, ExecutionConfig
from kimi_cli.soul.unified.context import UnifiedContext, MemoryFragment, FragmentType
from kimi_cli.soul.unified.tool import UnifiedTool, unified_tool
from kimi_cli.soul.unified.graph_engine import TaskGraph, TaskNode


class CalculatorParams(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")
    operation: str = Field(description="Operation")


class CalculatorTool(UnifiedTool[CalculatorParams]):
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
        return {"result": result}


async def demo_reactive():
    print("\n" + "="*60)
    print("REACTIVE MODE (Step-by-step)")
    print("="*60)
    
    # Direct tool execution
    tool = CalculatorTool()
    result = await tool(CalculatorParams(a=5, b=3, operation="multiply"))
    
    print(f"Task: Calculate 5 * 3")
    print(f"Execution: Single step")
    print(f"Result: {result.output}")
    print("\n✓ Best for: Simple, linear tasks")


async def demo_proactive():
    print("\n" + "="*60)
    print("PROACTIVE MODE (Graph-based parallel)")
    print("="*60)
    
    # Create task graph
    graph = TaskGraph()
    
    # Independent tasks (can run in parallel)
    graph.add_node(TaskNode(id="task_a", description="Task A", tool=None, dependencies=[]))
    graph.add_node(TaskNode(id="task_b", description="Task B", tool=None, dependencies=[]))
    graph.add_node(TaskNode(id="task_c", description="Task C", tool=None, dependencies=[]))
    
    # Dependent task
    graph.add_node(TaskNode(
        id="final",
        description="Final task",
        tool=None,
        dependencies=["task_a", "task_b", "task_c"]
    ))
    
    print(f"Task: Complex workflow with dependencies")
    print(f"Graph: {len(graph.nodes)} nodes")
    
    # Show execution
    ready = graph.get_ready_nodes()
    print(f"Ready to execute: {[n.id for n in ready]} (parallel)")
    
    # Mark some complete
    graph.mark_complete("task_a", "done")
    graph.mark_complete("task_b", "done")
    
    ready = graph.get_ready_nodes()
    print(f"After 2 complete: {[n.id for n in ready]} (task_c still ready)")
    
    graph.mark_complete("task_c", "done")
    
    ready = graph.get_ready_nodes()
    print(f"After all ready complete: {[n.id for n in ready]} (final now ready)")
    
    print("\n✓ Best for: Complex, parallelizable tasks")


async def demo_adaptive():
    print("\n" + "="*60)
    print("ADAPTIVE MODE (Auto-select)")
    print("="*60)
    
    test_tasks = [
        ("What is 2 + 2?", "simple"),
        ("Hello", "simple"),
        ("Analyze all files and refactor", "complex"),
        ("Multiple parallel tasks", "complex"),
    ]
    
    print("Task analysis:")
    for task, expected in test_tasks:
        # Simple heuristic
        complex_words = ["complex", "multiple", "parallel", "analyze", "refactor", "all"]
        is_complex = any(w in task.lower() for w in complex_words)
        detected = "complex" if is_complex else "simple"
        mode = "PROACTIVE" if is_complex else "REACTIVE"
        match = "✓" if detected == expected else "✗"
        print(f"  {match} '{task[:30]}...' -> {detected} -> {mode}")
    
    print("\n✓ Best for: Mixed workloads")


async def demo_comparison():
    print("\n" + "="*60)
    print("MODE COMPARISON")
    print("="*60)
    
    print("\nScenario: Analyze codebase")
    print("\nREACTIVE (Step-by-step):")
    print("  1. Find files")
    print("  2. Analyze (sequential)")
    print("  3. Generate report")
    print("  Time: ~5s (sequential)")
    
    print("\nPROACTIVE (Graph-based):")
    print("  1. [Find files] [Check deps] [Count lines] (parallel)")
    print("  2. [Analyze results]")
    print("  3. [Generate report]")
    print("  Time: ~2s (parallel)")
    
    print("\nADAPTIVE (Auto-select):")
    print("  System: 'Complex task detected'")
    print("  Action: Switch to PROACTIVE")
    print("  Result: Optimal performance")


async def main():
    print("="*60)
    print("Unified Agent Architecture - Demo")
    print("="*60)
    
    await demo_reactive()
    await demo_proactive()
    await demo_adaptive()
    await demo_comparison()
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nUsage:")
    print("  soul = UnifiedSoul(agent, context, mode=AgentMode.PROACTIVE)")
    print("  await soul.run('Your task')")


if __name__ == "__main__":
    asyncio.run(main())
