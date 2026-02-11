"""
End-to-end tests for Unified Agent Architecture.

This module tests the complete unified mode workflow:
1. Reactive mode (step-by-step execution)
2. Proactive mode (graph-based parallel execution)
3. Adaptive mode (automatic mode selection)
"""

from __future__ import annotations

import asyncio
import json
import pytest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import BaseModel, Field
from kosong.message import Message
from kosong.tooling import ToolOk

from kimi_cli.soul.unified import AgentMode
from kimi_cli.soul.unified.soul import UnifiedSoul, TaskAnalysis
from kimi_cli.soul.unified.agent import UnifiedAgent, ExecutionMode, ExecutionConfig
from kimi_cli.soul.unified.context import UnifiedContext, MemoryFragment, FragmentType
from kimi_cli.soul.unified.tool import UnifiedTool, unified_tool, register_tool
from kimi_cli.soul.unified.graph_engine import TaskGraph, TaskNode, GraphEngine
from kimi_cli.soul.unified.loader import UnifiedAgentConfig, load_unified_agent, create_unified_soul


# =============================================================================
# Mock Fixtures
# =============================================================================

class MockLLM:
    """Mock LLM for testing."""
    
    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or []
        self.call_count = 0
        self.chat_provider = self
        self.model_name = "mock-model"
        self.max_context_size = 100000
        self.capabilities = set()
    
    async def chat(self, messages: list[Message]) -> Message:
        """Mock chat method."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return Message(role="assistant", content=response)
        return Message(role="assistant", content="Default response")


class MockRuntime:
    """Complete mock runtime for testing."""
    
    def __init__(self, llm: MockLLM | None = None):
        self.llm = llm
        self.session = MagicMock()
        self.session.context_file = Path("/tmp/test_context.jsonl")
        self.denwa_renji = MagicMock()
        self.denwa_renji.set_n_checkpoints = lambda x: None
        self.denwa_renji.fetch_pending_dmail = lambda: None
        self.approval = MagicMock()
        self.approval.is_yolo = lambda: False
        self.approval.share = lambda: self.approval
        self.config = MagicMock()
        self.config.loop_control = MagicMock()
        self.config.loop_control.max_steps_per_turn = 50
        self.config.loop_control.max_ralph_iterations = 0
        self.config.loop_control.max_retries_per_step = 3
        self.config.loop_control.reserved_context_size = 1000
        self.labor_market = MagicMock()
        self.labor_market.add_fixed_subagent = lambda *args: None
        self.skills = {}


class MockToolset:
    """Mock toolset for testing."""
    
    def __init__(self, tools: list | None = None):
        self.tools = tools or []


# =============================================================================
# Test Tools
# =============================================================================

class CalculatorParams(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")
    operation: str = Field(description="Operation: add, subtract, multiply, divide")


@register_tool
class CalculatorTool(UnifiedTool[CalculatorParams]):
    """Calculator tool for testing."""
    name = "calculator"
    description = "Perform mathematical calculations"
    params = CalculatorParams
    
    async def execute(self, params: CalculatorParams) -> dict:
        if params.operation == "add":
            result = params.a + params.b
        elif params.operation == "subtract":
            result = params.a - params.b
        elif params.operation == "multiply":
            result = params.a * params.b
        elif params.operation == "divide":
            result = params.a / params.b if params.b != 0 else "Error: Division by zero"
        else:
            result = f"Unknown operation: {params.operation}"
        
        return {"result": result, "operation": params.operation}


class FileReadParams(BaseModel):
    path: str = Field(description="File path to read")


@register_tool
class FileReadTool(UnifiedTool[FileReadParams]):
    """File reading tool for testing."""
    name = "read_file"
    description = "Read a file from the filesystem"
    params = FileReadParams
    
    async def execute(self, params: FileReadParams) -> str:
        # Mock file reading
        mock_files = {
            "file1.py": "def hello(): pass",
            "file2.py": "class World: pass",
            "file3.py": "import os",
        }
        return mock_files.get(params.path, f"# Content of {params.path}")


# =============================================================================
# E2E Tests
# =============================================================================

@pytest.fixture
def temp_context_file(tmp_path):
    """Create a temporary context file."""
    return tmp_path / "test_context.jsonl"


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    return MockLLM()


@pytest.fixture
def mock_runtime(mock_llm):
    """Create a mock runtime."""
    return MockRuntime(llm=mock_llm)


@pytest.fixture
def unified_agent(mock_runtime):
    """Create a unified agent for testing."""
    toolset = MockToolset(tools=[CalculatorTool(), FileReadTool()])
    
    return UnifiedAgent(
        name="test_unified_agent",
        system_prompt="You are a helpful test agent.",
        toolset=toolset,
        runtime=mock_runtime,
        execution_config=ExecutionConfig(
            mode=ExecutionMode.REACTIVE,
            max_steps=10,
            max_iterations=2,
            parallel_tool_calls=True,
            parallel_tool_calls_limit=3,
        ),
    )


class TestReactiveMode:
    """Test reactive mode (step-by-step execution)."""
    
    @pytest.mark.asyncio
    async def test_reactive_mode_creation(self, unified_agent, temp_context_file):
        """Test creating soul in reactive mode."""
        context = UnifiedContext(file_backend=temp_context_file)
        soul = UnifiedSoul(
            agent=unified_agent,
            context=context,
            mode=AgentMode.REACTIVE,
        )
        
        assert soul.mode == AgentMode.REACTIVE
        assert soul.name == "test_unified_agent"
        assert soul._kimi_soul is not None
    
    @pytest.mark.asyncio
    async def test_reactive_mode_simple_task(self, unified_agent, temp_context_file, mock_llm):
        """Test reactive mode with a simple task."""
        # Setup mock response
        mock_llm.responses = ["I'll help you with that calculation."]
        
        context = UnifiedContext(file_backend=temp_context_file)
        soul = UnifiedSoul(
            agent=unified_agent,
            context=context,
            mode=AgentMode.REACTIVE,
        )
        
        # The soul should be created successfully
        assert soul.mode == AgentMode.REACTIVE
        assert soul.name == "test_unified_agent"
        
        # Verify KimiSoul is initialized
        assert soul._kimi_soul is not None
    
    @pytest.mark.asyncio
    async def test_reactive_mode_tool_execution(self, unified_agent, temp_context_file):
        """Test tool execution in reactive mode."""
        context = UnifiedContext(file_backend=temp_context_file)
        
        # Execute tool directly
        tool = CalculatorTool()
        params = CalculatorParams(a=5, b=3, operation="add")
        result = await tool(params)
        
        assert isinstance(result, ToolOk)
        assert "8" in result.output
        assert "add" in result.output


class TestProactiveMode:
    """Test proactive mode (graph-based execution)."""
    
    @pytest.mark.asyncio
    async def test_proactive_mode_creation(self, unified_agent, temp_context_file):
        """Test creating soul in proactive mode."""
        context = UnifiedContext(file_backend=temp_context_file)
        soul = UnifiedSoul(
            agent=unified_agent,
            context=context,
            mode=AgentMode.PROACTIVE,
        )
        
        assert soul.mode == AgentMode.PROACTIVE
        assert soul._graph_engine is not None
    
    @pytest.mark.asyncio
    async def test_task_graph_creation(self):
        """Test creating a task graph."""
        graph = TaskGraph()
        
        # Add nodes
        graph.add_node(TaskNode(
            id="read_files",
            description="Read all Python files",
            tool="read_file",
            args={"path": "*.py"},
            dependencies=[],
        ))
        graph.add_node(TaskNode(
            id="analyze",
            description="Analyze code structure",
            tool=None,
            args={},
            dependencies=["read_files"],
        ))
        
        assert len(graph.nodes) == 2
        assert graph.get_node("read_files") is not None
        assert graph.get_node("analyze") is not None
        
        # Check dependencies
        analyze_node = graph.get_node("analyze")
        assert analyze_node.dependencies == ["read_files"]
    
    @pytest.mark.asyncio
    async def test_graph_ready_nodes(self):
        """Test getting ready nodes from graph."""
        graph = TaskGraph()
        
        # Add independent nodes
        graph.add_node(TaskNode(id="a", description="Task A", tool=None))
        graph.add_node(TaskNode(id="b", description="Task B", tool=None))
        # Add dependent node
        graph.add_node(TaskNode(id="c", description="Task C", tool=None, dependencies=["a", "b"]))
        
        # Initially a and b should be ready
        ready = graph.get_ready_nodes()
        ready_ids = {n.id for n in ready}
        assert ready_ids == {"a", "b"}
        
        # Mark a as complete
        graph.mark_complete("a", "result_a")
        
        # b should still be ready, c not yet
        ready = graph.get_ready_nodes()
        ready_ids = {n.id for n in ready}
        assert "b" in ready_ids
        assert "c" not in ready_ids
        
        # Mark b as complete
        graph.mark_complete("b", "result_b")
        
        # Now c should be ready
        ready = graph.get_ready_nodes()
        ready_ids = {n.id for n in ready}
        assert "c" in ready_ids
    
    @pytest.mark.asyncio
    async def test_graph_parallel_execution(self):
        """Test parallel execution of independent nodes."""
        graph = TaskGraph()
        
        # Add multiple independent nodes
        for i in range(5):
            graph.add_node(TaskNode(
                id=f"task_{i}",
                description=f"Task {i}",
                tool=None,
                args={},
                dependencies=[],
            ))
        
        # All should be ready initially
        ready = graph.get_ready_nodes()
        assert len(ready) == 5
        
        # Execute all in parallel
        execution_order = []
        
        async def mock_executor(node: TaskNode) -> str:
            execution_order.append(node.id)
            await asyncio.sleep(0.01)  # Simulate work
            return f"result_{node.id}"
        
        # Execute all ready nodes
        tasks = [mock_executor(node) for node in ready]
        results = await asyncio.gather(*tasks)
        
        # All should have executed
        assert len(results) == 5
        assert len(execution_order) == 5
    
    @pytest.mark.asyncio
    async def test_graph_from_dict(self):
        """Test creating graph from dictionary."""
        data = {
            "nodes": [
                {
                    "id": "node_1",
                    "description": "First task",
                    "tool": "calculator",
                    "args": {"a": 1, "b": 2, "operation": "add"},
                    "dependencies": [],
                },
                {
                    "id": "node_2",
                    "description": "Second task",
                    "tool": None,
                    "args": {},
                    "dependencies": ["node_1"],
                },
            ]
        }
        
        graph = TaskGraph.from_dict(data)
        
        assert len(graph.nodes) == 2
        assert graph.get_node("node_1").tool == "calculator"
        assert graph.get_node("node_2").dependencies == ["node_1"]


class TestAdaptiveMode:
    """Test adaptive mode (automatic mode selection)."""
    
    @pytest.mark.asyncio
    async def test_adaptive_mode_creation(self, unified_agent, temp_context_file):
        """Test creating soul in adaptive mode."""
        context = UnifiedContext(file_backend=temp_context_file)
        soul = UnifiedSoul(
            agent=unified_agent,
            context=context,
            mode=AgentMode.ADAPTIVE,
        )
        
        assert soul.mode == AgentMode.ADAPTIVE
        # Should have graph engine for adaptive mode
        assert soul._graph_engine is not None
    
    @pytest.mark.asyncio
    async def test_task_complexity_analysis_simple(self):
        """Test analyzing simple task complexity."""
        simple_tasks = [
            "What is 2 + 2?",
            "Hello",
            "Simple question",
            "Quick check",
        ]
        
        for task in simple_tasks:
            # Simple heuristic
            complexity_indicators = {
                "simple": ["simple", "quick", "basic", "single"],
                "complex": ["complex", "multiple", "parallel", "analyze all", "refactor"],
            }
            
            task_lower = task.lower()
            simple_score = sum(1 for w in complexity_indicators["simple"] if w in task_lower)
            complex_score = sum(1 for w in complexity_indicators["complex"] if w in task_lower)
            
            # Simple tasks should have low complex_score
            assert complex_score == 0, f"Task '{task}' should not have complexity indicators"
    
    @pytest.mark.asyncio
    async def test_task_complexity_analysis_complex(self):
        """Test analyzing complex task complexity."""
        complex_tasks = [
            "Analyze all Python files and refactor them",
            "Multiple parallel tasks to execute",
            "Complex multi-step workflow",
        ]
        
        for task in complex_tasks:
            complexity_indicators = {
                "simple": ["simple", "quick", "basic", "single"],
                "complex": ["complex", "multiple", "parallel", "analyze all", "refactor"],
            }
            
            task_lower = task.lower()
            complex_score = sum(1 for w in complexity_indicators["complex"] if w in task_lower)
            
            # Complex tasks should have complexity indicators
            assert complex_score > 0, f"Task '{task}' should have complexity indicators"
    
    @pytest.mark.asyncio
    async def test_mode_switching(self, unified_agent, temp_context_file):
        """Test switching between modes."""
        context = UnifiedContext(file_backend=temp_context_file)
        soul = UnifiedSoul(
            agent=unified_agent,
            context=context,
            mode=AgentMode.REACTIVE,
        )
        
        assert soul.mode == AgentMode.REACTIVE
        
        # Switch to proactive
        soul.set_mode(AgentMode.PROACTIVE)
        assert soul.mode == AgentMode.PROACTIVE
        assert soul._graph_engine is not None
        
        # Switch to adaptive
        soul.set_mode(AgentMode.ADAPTIVE)
        assert soul.mode == AgentMode.ADAPTIVE
        
        # Switch back to reactive
        soul.set_mode(AgentMode.REACTIVE)
        assert soul.mode == AgentMode.REACTIVE


class TestUnifiedToolSystem:
    """Test unified tool system."""
    
    @pytest.mark.asyncio
    async def test_tool_decorator(self):
        """Test @unified_tool decorator."""
        @unified_tool(name="multiply", description="Multiply two numbers")
        async def multiply(a: float, b: float) -> float:
            return a * b
        
        # Should create a UnifiedTool subclass
        assert issubclass(multiply, UnifiedTool)
        assert multiply.name == "multiply"
        assert multiply.description == "Multiply two numbers"
        
        # Test execution
        tool = multiply()
        params = tool.params(a=3, b=4)
        result = await tool(params)
        
        assert isinstance(result, ToolOk)
        assert "12" in result.output
    
    @pytest.mark.asyncio
    async def test_tool_result_formats(self):
        """Test different tool result formats."""
        # Test dict result
        tool = CalculatorTool()
        result = await tool(CalculatorParams(a=10, b=5, operation="multiply"))
        assert "50" in result.output
        
        # Test string result
        @unified_tool(name="greet", description="Greet someone")
        async def greet(name: str) -> str:
            return f"Hello, {name}!"
        
        tool = greet()
        result = await tool(tool.params(name="World"))
        assert result.output == "Hello, World!"
    
    def test_tool_registry(self):
        """Test tool registry."""
        from kimi_cli.soul.unified.tool import unified_tool_registry
        
        # CalculatorTool should be registered
        tool_class = unified_tool_registry.get("calculator")
        assert tool_class is not None
        assert tool_class.name == "calculator"
        
        # Create instance
        instance = unified_tool_registry.create_instance("calculator")
        assert instance is not None
        assert isinstance(instance, CalculatorTool)


class TestUnifiedContext:
    """Test unified context functionality."""
    
    @pytest.mark.asyncio
    async def test_context_fragments(self, temp_context_file):
        """Test adding and retrieving fragments."""
        context = UnifiedContext(file_backend=temp_context_file)
        
        # Add fragments
        await context.append_fragment(MemoryFragment(
            role="user",
            content="Hello",
            fragment_type=FragmentType.USER,
        ))
        await context.append_fragment(MemoryFragment(
            role="assistant",
            content="Hi there",
            fragment_type=FragmentType.ASSISTANT,
        ))
        
        assert len(context.fragments) == 2
        assert context.fragments[0].role == "user"
        assert context.fragments[1].role == "assistant"
    
    @pytest.mark.asyncio
    async def test_context_filter_by_type(self, temp_context_file):
        """Test filtering fragments by type."""
        context = UnifiedContext(file_backend=temp_context_file)
        
        # Add different fragment types
        await context.append_fragment(MemoryFragment(
            role="user", content="Q1", fragment_type=FragmentType.USER,
        ))
        await context.append_fragment(MemoryFragment(
            role="assistant", content="Tool call", fragment_type=FragmentType.TOOL_CALL,
            tool_name="test_tool",
        ))
        await context.append_fragment(MemoryFragment(
            role="assistant", content="Tool result", fragment_type=FragmentType.TOOL_RESULT,
        ))
        
        # Filter by type
        tool_results = context.get_fragments_by_type(FragmentType.TOOL_RESULT)
        assert len(tool_results) == 1
        assert tool_results[0].content == "Tool result"
    
    @pytest.mark.asyncio
    async def test_context_persistence(self, temp_context_file):
        """Test context persistence to file."""
        context = UnifiedContext(file_backend=temp_context_file)
        
        # Add data
        await context.append_fragment(MemoryFragment(
            role="user",
            content="Test message",
            fragment_type=FragmentType.USER,
        ))
        await context.update_token_count(100)
        
        # Restore in new context
        context2 = UnifiedContext(file_backend=temp_context_file)
        restored = await context2.restore()
        
        assert restored is True
        assert len(context2.fragments) == 1
        assert context2.token_count == 100


class TestExecutionConfig:
    """Test execution configuration."""
    
    def test_default_config(self):
        """Test default execution configuration."""
        config = ExecutionConfig()
        
        assert config.mode == ExecutionMode.REACTIVE
        assert config.max_steps == 50
        assert config.max_iterations == 3
        assert config.parallel_tool_calls is True
        assert config.parallel_tool_calls_limit == 5
        assert config.auto_summarize is True
    
    def test_custom_config(self):
        """Test custom execution configuration."""
        config = ExecutionConfig(
            mode=ExecutionMode.PROACTIVE,
            max_steps=100,
            max_iterations=5,
            parallel_tool_calls=False,
            parallel_tool_calls_limit=10,
            auto_summarize=False,
        )
        
        assert config.mode == ExecutionMode.PROACTIVE
        assert config.max_steps == 100
        assert config.max_iterations == 5
        assert config.parallel_tool_calls is False
        assert config.parallel_tool_calls_limit == 10
        assert config.auto_summarize is False
    
    def test_config_builder(self):
        """Test UnifiedAgentConfig builder."""
        config = (
            UnifiedAgentConfig()
            .with_mode(ExecutionMode.PROACTIVE)
            .with_max_steps(75)
            .with_max_iterations(4)
            .with_parallel_limit(8)
            .with_auto_summarize(True)
        )
        
        exec_config = config.build_execution_config()
        
        assert exec_config.mode == ExecutionMode.PROACTIVE
        assert exec_config.max_steps == 75
        assert exec_config.max_iterations == 4
        assert exec_config.parallel_tool_calls_limit == 8


class TestModeComparison:
    """Compare different execution modes."""
    
    @pytest.mark.asyncio
    async def test_reactive_vs_proactive_modes(self, unified_agent, temp_context_file):
        """Test that reactive and proactive modes create different engines."""
        context = UnifiedContext(file_backend=temp_context_file)
        
        # Reactive mode
        reactive_soul = UnifiedSoul(
            agent=unified_agent,
            context=context,
            mode=AgentMode.REACTIVE,
        )
        assert reactive_soul.mode == AgentMode.REACTIVE
        
        # Proactive mode
        context2 = UnifiedContext(file_backend=temp_context_file)
        proactive_soul = UnifiedSoul(
            agent=unified_agent,
            context=context2,
            mode=AgentMode.PROACTIVE,
        )
        assert proactive_soul.mode == AgentMode.PROACTIVE
        assert proactive_soul._graph_engine is not None
    
    def test_execution_modes_enum(self):
        """Test ExecutionMode and AgentMode compatibility."""
        # Both enums should have same values
        assert ExecutionMode.REACTIVE.name == AgentMode.REACTIVE.name
        assert ExecutionMode.PROACTIVE.name == AgentMode.PROACTIVE.name
        assert ExecutionMode.ADAPTIVE.name == AgentMode.ADAPTIVE.name


# =============================================================================
# Main Entry Point for Manual Testing
# =============================================================================

if __name__ == "__main__":
    """Run tests manually with: python test_unified_mode_e2e.py"""
    pytest.main([__file__, "-v"])
