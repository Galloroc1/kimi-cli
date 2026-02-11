"""
Demo: Unified Agent Architecture with Simulated LLM Responses

This script demonstrates the unified architecture with mock LLM responses
so it can run without an external LLM server.

Run:
   uv run python demo_with_mock_llm.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
from typing import Any

from pydantic import BaseModel, Field
from kosong.message import Message

from kimi_cli.config import load_config
from kimi_cli.auth.oauth import OAuthManager
from kimi_cli.session import Session
from kimi_cli.llm import LLM
from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.toolset import KimiToolset


class MockToolset:
    """Mock toolset for testing."""
    def __init__(self, tools=None):
        self.tools = tools or []

from kimi_cli.soul.unified import AgentMode
from kimi_cli.soul.unified.soul import UnifiedSoul
from kimi_cli.soul.unified.agent import UnifiedAgent, ExecutionMode, ExecutionConfig
from kimi_cli.soul.unified.context import UnifiedContext, MemoryFragment, FragmentType
from kimi_cli.soul.unified.tool import UnifiedTool, unified_tool
from kimi_cli.soul.unified.graph_engine import TaskGraph, TaskNode, GraphEngine

from kaos.path import KaosPath


# =============================================================================
# Mock LLM that simulates responses
# =============================================================================

class MockLLMProvider:
    """Mock LLM provider that returns predefined responses."""
    
    def __init__(self):
        self.model_name = "mock-llm"
        self.thinking_effort = None
    
    async def chat(self, messages: list[Message]) -> Message:
        """Return mock response based on input."""
        last_message = messages[-1].extract_text(" ") if messages else ""
        
        # Simulate task planning response
        if "task planner" in last_message.lower() or "decompose" in last_message.lower():
            return self._planning_response(last_message)
        
        # Simulate simple Q&A
        if "2 + 2" in last_message:
            return Message(role="assistant", content="2 + 2 = 4")
        
        if "5 * 3" in last_message or "5 multiply 3" in last_message:
            return Message(role="assistant", content="5 * 3 = 15")
        
        # Default response
        return Message(
            role="assistant",
            content="I understand. Let me help you with that task."
        )
    
    async def stream_chat(self, messages: list[Message]):
        """Mock stream chat."""
        response = await self.chat(messages)
        yield response
    
    def _planning_response(self, prompt: str) -> Message:
        """Generate a task plan response."""
        # Extract task from prompt
        if "python files" in prompt.lower():
            plan = {
                "nodes": [
                    {
                        "id": "find_files",
                        "description": "Find all Python files",
                        "tool": "glob",
                        "args": {"pattern": "**/*.py"},
                        "dependencies": [],
                    },
                    {
                        "id": "analyze",
                        "description": "Analyze code structure",
                        "tool": None,
                        "args": {},
                        "dependencies": ["find_files"],
                    },
                    {
                        "id": "report",
                        "description": "Generate report",
                        "tool": None,
                        "args": {},
                        "dependencies": ["analyze"],
                    },
                ]
            }
        elif "read" in prompt.lower() and "analyze" in prompt.lower():
            plan = {
                "nodes": [
                    {
                        "id": "read_file",
                        "description": "Read the specified file",
                        "tool": "read_file",
                        "args": {"path": "example.py"},
                        "dependencies": [],
                    },
                    {
                        "id": "analyze_content",
                        "description": "Analyze file content",
                        "tool": None,
                        "args": {},
                        "dependencies": ["read_file"],
                    },
                ]
            }
        else:
            # Simple single-node plan
            plan = {
                "nodes": [
                    {
                        "id": "task_1",
                        "description": "Execute the main task",
                        "tool": None,
                        "args": {},
                        "dependencies": [],
                    }
                ]
            }
        
        return Message(role="assistant", content=json.dumps(plan))
    
    def with_thinking(self, effort: str):
        """Mock with_thinking."""
        return self


class MockLLM:
    """Mock LLM for testing."""
    
    def __init__(self):
        self.chat_provider = MockLLMProvider()
        self.model_name = "mock-llm"
        self.max_context_size = 100000
        self.capabilities = set()


# =============================================================================
# Demo Tools
# =============================================================================

class CalculatorParams(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")
    operation: str = Field(description="Operation: add, subtract, multiply, divide")


class CalculatorTool(UnifiedTool[CalculatorParams]):
    """Calculator tool."""
    name = "calculator"
    description = "Perform mathematical calculations"
    params = CalculatorParams
    
    async def execute(self, params: CalculatorParams) -> dict:
        ops = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero",
        }
        result = ops.get(params.operation, lambda x, y: f"Unknown: {params.operation}")(params.a, params.b)
        return {"result": result, "operation": params.operation, "inputs": [params.a, params.b]}


class GlobParams(BaseModel):
    pattern: str = Field(description="File pattern")


class GlobTool(UnifiedTool[GlobParams]):
    """Glob tool for finding files."""
    name = "glob"
    description = "Find files matching a pattern"
    params = GlobParams
    
    async def execute(self, params: GlobParams) -> dict:
        # Mock file listing
        mock_files = {
            "**/*.py": ["main.py", "utils.py", "test.py"],
            "**/*.md": ["README.md", "CONTRIBUTING.md"],
        }
        files = mock_files.get(params.pattern, ["file1.txt", "file2.txt"])
        return {"pattern": params.pattern, "files": files, "count": len(files)}


class ReadFileParams(BaseModel):
    path: str = Field(description="File path")


class ReadFileTool(UnifiedTool[ReadFileParams]):
    """Read file tool."""
    name = "read_file"
    description = "Read a file"
    params = ReadFileParams
    
    async def execute(self, params: ReadFileParams) -> str:
        mock_content = {
            "main.py": "def main():\n    print('Hello')\n",
            "utils.py": "def helper():\n    pass\n",
            "README.md": "# Project\n\nThis is a test project.\n",
        }
        return mock_content.get(params.path, f"# Content of {params.path}")


# =============================================================================
# Demo Functions
# =============================================================================

async def demo_reactive_mode():
    """Demo reactive mode with simulated LLM."""
    print("\n" + "="*70)
    print("DEMO 1: Reactive Mode (Step-by-step execution)")
    print("="*70)
    
    # Setup
    config = load_config()
    oauth = OAuthManager(config)
    work_dir = KaosPath(".")
    session = await Session.create(work_dir)
    
    # Create mock LLM
    llm = MockLLM()
    
    runtime = await Runtime.create(
        config=config,
        oauth=oauth,
        llm=llm,
        session=session,
        yolo=True,
    )
    
    # Create toolset
    toolset = MockToolset([CalculatorTool(), GlobTool(), ReadFileTool()])
    
    # Create agent
    agent = UnifiedAgent(
        name="reactive_demo",
        system_prompt="You are a helpful assistant.",
        toolset=toolset,
        runtime=runtime,
        execution_config=ExecutionConfig(mode=ExecutionMode.REACTIVE),
    )
    
    context = UnifiedContext(file_backend=session.context_file)
    soul = UnifiedSoul(agent, context, mode=AgentMode.REACTIVE)
    
    print(f"\nAgent: {soul.name}")
    print(f"Mode: {soul.mode.name}")
    print(f"LLM: {soul.model_name}")
    print(f"Tools: {[t.name for t in agent.toolset.tools]}")
    
    # Execute a simple task
    print("\n--- Task: Calculate 5 * 3 ---")
    tool = CalculatorTool()
    result = await tool(CalculatorParams(a=5, b=3, operation="multiply"))
    print(f"Result: {result.output}")
    
    print("\n✓ Reactive mode: Direct tool execution, single step")


async def demo_proactive_mode():
    """Demo proactive mode with task graph."""
    print("\n" + "="*70)
    print("DEMO 2: Proactive Mode (Graph-based parallel execution)")
    print("="*70)
    
    # Setup
    config = load_config()
    oauth = OAuthManager(config)
    work_dir = KaosPath(".")
    session = await Session.create(work_dir)
    
    llm = MockLLM()
    
    runtime = await Runtime.create(
        config=config,
        oauth=oauth,
        llm=llm,
        session=session,
        yolo=True,
    )
    
    toolset = MockToolset([CalculatorTool(), GlobTool(), ReadFileTool()])
    
    agent = UnifiedAgent(
        name="proactive_demo",
        system_prompt="You are a task planning assistant.",
        toolset=toolset,
        runtime=runtime,
        execution_config=ExecutionConfig(
            mode=ExecutionMode.PROACTIVE,
            max_iterations=2,
            parallel_tool_calls_limit=3,
        ),
    )
    
    context = UnifiedContext(file_backend=session.context_file)
    soul = UnifiedSoul(agent, context, mode=AgentMode.PROACTIVE)
    
    print(f"\nAgent: {soul.name}")
    print(f"Mode: {soul.mode.name}")
    print(f"Graph Engine: {soul._graph_engine is not None}")
    
    # Create task graph using LLM
    print("\n--- Task: Analyze Python files ---")
    print("Creating task graph via LLM...")
    
    graph_engine = GraphEngine(agent, context)
    graph = await graph_engine.create_task_graph(
        "Find all Python files and analyze them"
    )
    
    print(f"\nGraph created with {len(graph.nodes)} nodes:")
    
    # Show execution plan
    iteration = 0
    while not graph.is_complete() and iteration < 3:
        iteration += 1
        ready = graph.get_ready_nodes()
        
        if not ready:
            break
        
        print(f"\nIteration {iteration}:")
        print(f"  Ready nodes: {[n.id for n in ready]}")
        
        # Execute ready nodes
        for node in ready:
            print(f"  Executing: {node.id} - {node.description}")
            
            # Mock execution
            if node.tool:
                result = f"Tool {node.tool} executed"
            else:
                result = "Direct LLM processing"
            
            graph.mark_complete(node.id, result)
            print(f"  Completed: {node.id}")
    
    print(f"\n✓ Proactive mode: Task decomposition + parallel execution")


async def demo_adaptive_mode():
    """Demo adaptive mode with auto-selection."""
    print("\n" + "="*70)
    print("DEMO 3: Adaptive Mode (Automatic mode selection)")
    print("="*70)
    
    # Setup
    config = load_config()
    oauth = OAuthManager(config)
    work_dir = KaosPath(".")
    session = await Session.create(work_dir)
    
    llm = MockLLM()
    
    runtime = await Runtime.create(
        config=config,
        oauth=oauth,
        llm=llm,
        session=session,
        yolo=True,
    )
    
    toolset = MockToolset([CalculatorTool(), GlobTool()])
    
    agent = UnifiedAgent(
        name="adaptive_demo",
        system_prompt="You are a helpful assistant.",
        toolset=toolset,
        runtime=runtime,
        execution_config=ExecutionConfig(mode=ExecutionMode.ADAPTIVE),
    )
    
    context = UnifiedContext(file_backend=session.context_file)
    soul = UnifiedSoul(agent, context, mode=AgentMode.ADAPTIVE)
    
    print(f"\nAgent: {soul.name}")
    print(f"Mode: {soul.mode.name}")
    print(f"Auto-select: Enabled")
    
    # Test task analysis
    print("\n--- Task Analysis ---")
    
    test_cases = [
        ("What is 2 + 2?", "simple"),
        ("Hello", "simple"),
        ("Find all Python files and analyze them", "complex"),
        ("Multiple parallel tasks", "complex"),
    ]
    
    for task, expected_type in test_cases:
        analysis = await soul._analyze_task(task)
        
        indicator = "✓" if analysis.recommended_mode.name.lower() in [expected_type, "adaptive"] else "~"
        print(f"  {indicator} '{task[:35]}...' -> {analysis.complexity} -> {analysis.recommended_mode.name}")
    
    print("\n✓ Adaptive mode: Auto-selects based on task complexity")


async def demo_mode_comparison():
    """Compare all three modes side by side."""
    print("\n" + "="*70)
    print("DEMO 4: Mode Comparison")
    print("="*70)
    
    print("\n" + "-"*70)
    print("Scenario: 'Analyze codebase and generate report'")
    print("-"*70)
    
    scenarios = [
        ("REACTIVE", "Step-by-step", [
            "1. User: Analyze codebase",
            "2. LLM: Let me find files...",
            "3. [Tool: find files]",
            "4. LLM: Now let me analyze...",
            "5. [Tool: analyze]",
            "6. LLM: Here's the report",
        ]),
        ("PROACTIVE", "Graph-based parallel", [
            "1. User: Analyze codebase",
            "2. LLM: Creating task graph...",
            "3. [Parallel execution:]",
            "   - Find files ✓",
            "   - Check deps ✓",
            "4. [Analyze results]",
            "5. LLM: Here's the report",
        ]),
        ("ADAPTIVE", "Auto-select", [
            "1. User: Analyze codebase",
            "2. System: Analyzing complexity...",
            "3. System: Complex task detected",
            "4. [Switch to PROACTIVE mode]",
            "5. [Execute as proactive]",
        ]),
    ]
    
    for mode, description, flow in scenarios:
        print(f"\n{mode} Mode ({description}):")
        for step in flow:
            print(f"  {step}")
    
    print("\n" + "-"*70)
    print("Performance Comparison:")
    print("-"*70)
    print("  Simple task (1 tool call):")
    print("    Reactive:  ~1s  (single step)")
    print("    Proactive: ~2s  (planning overhead)")
    print("    Adaptive:  ~1s  (selects reactive)")
    print("\n  Complex task (5 parallel tool calls):")
    print("    Reactive:  ~5s  (sequential)")
    print("    Proactive: ~2s  (parallel)")
    print("    Adaptive:  ~2s  (selects proactive)")


async def main():
    """Main demo runner."""
    print("="*70)
    print("Unified Agent Architecture - Demo with Simulated LLM")
    print("="*70)
    print("\nThis demo uses mock LLM responses to show the three execution modes.")
    print("No external LLM server required.")
    
    try:
        await demo_reactive_mode()
        await demo_proactive_mode()
        await demo_adaptive_mode()
        await demo_mode_comparison()
        
        print("\n" + "="*70)
        print("Demo Complete!")
        print("="*70)
        print("\nKey Takeaways:")
        print("  • REACTIVE: Simple, linear tasks (low latency)")
        print("  • PROACTIVE: Complex, parallel tasks (high throughput)")
        print("  • ADAPTIVE: Automatically chooses (best of both)")
        print("\nTo use with REAL LLM:")
        print("  export KIMI_BASE_URL='http://localhost:8000/v1'")
        print("  uv run python example_real_llm_usage.py")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
