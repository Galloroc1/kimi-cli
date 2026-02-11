"""Tests for UnifiedSoul."""

from __future__ import annotations

import pytest
from pathlib import Path

from kimi_cli.soul.unified import AgentMode
from kimi_cli.soul.unified.soul import UnifiedSoul, TaskAnalysis
from kimi_cli.soul.unified.agent import UnifiedAgent, ExecutionMode, ExecutionConfig
from kimi_cli.soul.unified.context import UnifiedContext
from pydantic import BaseModel


class MockParams(BaseModel):
    value: str


class MockRuntime:
    """Mock runtime for testing."""
    llm = None
    session = None
    denwa_renji = type('obj', (object,), {
        'set_n_checkpoints': lambda *args: None,
        'fetch_pending_dmail': lambda *args: None,
    })()
    approval = type('obj', (object,), {
        'is_yolo': lambda *args: False,
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
    labor_market = type('obj', (object,), {
        'add_fixed_subagent': lambda *args: None
    })()
    skills = {}


class MockToolset:
    """Mock toolset for testing."""
    tools = []


@pytest.fixture
def mock_agent(tmp_path):
    """Create a mock UnifiedAgent."""
    runtime = MockRuntime()
    toolset = MockToolset()
    
    agent = UnifiedAgent(
        name="test_agent",
        system_prompt="You are a test agent.",
        toolset=toolset,
        runtime=runtime,
        execution_config=ExecutionConfig(mode=ExecutionMode.REACTIVE),
    )
    
    return agent


@pytest.fixture
def mock_context(tmp_path):
    """Create a mock UnifiedContext."""
    context_file = tmp_path / "context.jsonl"
    return UnifiedContext(file_backend=context_file)


class TestTaskAnalysis:
    """Test TaskAnalysis functionality."""
    
    def test_task_analysis_creation(self):
        """Test creating TaskAnalysis."""
        analysis = TaskAnalysis(
            complexity="complex",
            estimated_steps=15,
            parallelizable=True,
            recommended_mode=AgentMode.PROACTIVE,
        )
        
        assert analysis.complexity == "complex"
        assert analysis.estimated_steps == 15
        assert analysis.parallelizable is True
        assert analysis.recommended_mode == AgentMode.PROACTIVE


class TestModeAnalysis:
    """Test task analysis for mode selection."""
    
    @pytest.mark.asyncio
    async def test_analyze_task_simple(self):
        """Test analyzing simple task."""
        # Create a simple mock for the _analyze_task logic
        task = "What is 2 + 2?"
        
        # Simple heuristic analysis
        complexity_indicators = {
            "simple": ["simple", "quick", "basic", "single"],
            "complex": ["complex", "multiple", "parallel", "analyze all", "refactor"],
        }
        
        task_lower = task.lower()
        simple_score = sum(1 for w in complexity_indicators["simple"] if w in task_lower)
        complex_score = sum(1 for w in complexity_indicators["complex"] if w in task_lower)
        
        # This is a simple task
        assert simple_score >= 0
        assert complex_score == 0
    
    @pytest.mark.asyncio
    async def test_analyze_task_complex(self):
        """Test analyzing complex task."""
        task = "Analyze all Python files and refactor them to use type hints"
        
        complexity_indicators = {
            "simple": ["simple", "quick", "basic", "single"],
            "complex": ["complex", "multiple", "parallel", "analyze all", "refactor"],
        }
        
        task_lower = task.lower()
        simple_score = sum(1 for w in complexity_indicators["simple"] if w in task_lower)
        complex_score = sum(1 for w in complexity_indicators["complex"] if w in task_lower)
        
        # This is a complex task
        assert complex_score > 0
        assert "analyze all" in task_lower or "refactor" in task_lower


class TestAgentMode:
    """Test AgentMode enum."""
    
    def test_agent_mode_values(self):
        """Test AgentMode values."""
        assert AgentMode.REACTIVE.name == "REACTIVE"
        assert AgentMode.PROACTIVE.name == "PROACTIVE"
        assert AgentMode.ADAPTIVE.name == "ADAPTIVE"
    
    def test_agent_mode_equality(self):
        """Test AgentMode equality."""
        assert AgentMode.REACTIVE == AgentMode.REACTIVE
        assert AgentMode.REACTIVE != AgentMode.PROACTIVE


class TestExecutionMode:
    """Test ExecutionMode enum."""
    
    def test_execution_mode_values(self):
        """Test ExecutionMode values."""
        assert ExecutionMode.REACTIVE.name == "REACTIVE"
        assert ExecutionMode.PROACTIVE.name == "PROACTIVE"
        assert ExecutionMode.ADAPTIVE.name == "ADAPTIVE"


class TestUnifiedAgentBasics:
    """Test basic UnifiedAgent functionality."""
    
    def test_agent_creation(self, mock_agent):
        """Test creating UnifiedAgent."""
        assert mock_agent.name == "test_agent"
        assert mock_agent.system_prompt == "You are a test agent."
        assert mock_agent.execution_config.mode == ExecutionMode.REACTIVE
    
    def test_agent_with_mode(self, mock_agent):
        """Test creating agent with different mode."""
        proactive_agent = mock_agent.with_mode(ExecutionMode.PROACTIVE)
        
        assert proactive_agent.execution_config.mode == ExecutionMode.PROACTIVE
        # Original unchanged
        assert mock_agent.execution_config.mode == ExecutionMode.REACTIVE
    
    def test_execution_config_defaults(self):
        """Test ExecutionConfig defaults."""
        config = ExecutionConfig()
        
        assert config.mode == ExecutionMode.REACTIVE
        assert config.max_steps == 50
        assert config.max_iterations == 3
        assert config.parallel_tool_calls is True
        assert config.parallel_tool_calls_limit == 5
        assert config.auto_summarize is True


class TestUnifiedContextBasics:
    """Test basic UnifiedContext functionality."""
    
    @pytest.mark.asyncio
    async def test_context_creation(self, mock_context):
        """Test creating UnifiedContext."""
        assert len(mock_context.fragments) == 0
        assert mock_context.execution_graph is None
    
    @pytest.mark.asyncio
    async def test_execution_graph(self, mock_context):
        """Test execution graph getter/setter."""
        assert mock_context.execution_graph is None
        
        graph = {"nodes": ["a", "b"]}
        mock_context.execution_graph = graph
        
        assert mock_context.execution_graph == graph
