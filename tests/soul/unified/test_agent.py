"""Tests for UnifiedAgent."""

from __future__ import annotations

import pytest
from pathlib import Path

from kimi_cli.soul.unified.agent import (
    UnifiedAgent,
    ExecutionConfig,
    ExecutionMode,
)
from kimi_cli.soul.unified.tool import UnifiedTool, register_tool
from pydantic import BaseModel


class MockParams(BaseModel):
    value: str


class MockTool(UnifiedTool[MockParams]):
    name = "mock_tool"
    description = "A mock tool"
    params = MockParams
    
    async def execute(self, params: MockParams) -> str:
        return f"Result: {params.value}"


class TestExecutionConfig:
    """Test ExecutionConfig dataclass."""
    
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


class TestUnifiedAgent:
    """Test UnifiedAgent functionality."""
    
    @pytest.fixture
    def mock_runtime(self):
        """Create a mock runtime."""
        # This is a simplified mock
        class MockRuntime:
            llm = None
            session = None
            labor_market = type('obj', (object,), {
                'add_fixed_subagent': lambda *args: None
            })()
        
        return MockRuntime()
    
    @pytest.fixture
    def mock_toolset(self):
        """Create a mock toolset."""
        class MockToolset:
            tools = []
        
        return MockToolset()
    
    def test_agent_creation(self, mock_runtime, mock_toolset):
        """Test creating a UnifiedAgent."""
        agent = UnifiedAgent(
            name="test_agent",
            system_prompt="You are a test agent.",
            toolset=mock_toolset,
            runtime=mock_runtime,
        )
        
        assert agent.name == "test_agent"
        assert agent.system_prompt == "You are a test agent."
        assert agent.execution_config.mode == ExecutionMode.REACTIVE
    
    def test_agent_with_custom_config(self, mock_runtime, mock_toolset):
        """Test creating agent with custom execution config."""
        config = ExecutionConfig(mode=ExecutionMode.PROACTIVE)
        
        agent = UnifiedAgent(
            name="test_agent",
            system_prompt="You are a test agent.",
            toolset=mock_toolset,
            runtime=mock_runtime,
            execution_config=config,
        )
        
        assert agent.execution_config.mode == ExecutionMode.PROACTIVE
    
    def test_agent_properties(self, mock_runtime, mock_toolset):
        """Test agent properties."""
        agent = UnifiedAgent(
            name="test_agent",
            system_prompt="Test",
            toolset=mock_toolset,
            runtime=mock_runtime,
        )
        
        assert agent.llm == mock_runtime.llm
        assert agent.session == mock_runtime.session
    
    def test_with_mode(self, mock_runtime, mock_toolset):
        """Test creating agent with different mode."""
        agent = UnifiedAgent(
            name="test_agent",
            system_prompt="Test",
            toolset=mock_toolset,
            runtime=mock_runtime,
            execution_config=ExecutionConfig(mode=ExecutionMode.REACTIVE),
        )
        
        # Create copy with different mode
        proactive_agent = agent.with_mode(ExecutionMode.PROACTIVE)
        
        assert proactive_agent.name == agent.name
        assert proactive_agent.execution_config.mode == ExecutionMode.PROACTIVE
        # Original should be unchanged
        assert agent.execution_config.mode == ExecutionMode.REACTIVE
    
    def test_add_unified_tool(self, mock_runtime, mock_toolset):
        """Test adding a unified tool."""
        agent = UnifiedAgent(
            name="test_agent",
            system_prompt="Test",
            toolset=mock_toolset,
            runtime=mock_runtime,
        )
        
        # Add tool
        agent.add_unified_tool(MockTool)
        
        # Should be in unified_tools registry
        assert agent.unified_tools.get("mock_tool") is MockTool
    
    def test_get_tool_from_unified(self, mock_runtime, mock_toolset):
        """Test getting tool from unified registry."""
        agent = UnifiedAgent(
            name="test_agent",
            system_prompt="Test",
            toolset=mock_toolset,
            runtime=mock_runtime,
        )
        
        agent.add_unified_tool(MockTool)
        
        tool = agent.get_tool("mock_tool")
        
        assert tool is not None
        assert isinstance(tool, MockTool)
    
    def test_get_tool_not_found(self, mock_runtime, mock_toolset):
        """Test getting non-existent tool."""
        agent = UnifiedAgent(
            name="test_agent",
            system_prompt="Test",
            toolset=mock_toolset,
            runtime=mock_runtime,
        )
        
        tool = agent.get_tool("non_existent")
        
        assert tool is None
    
    def test_to_kimi_agent(self, mock_runtime, mock_toolset):
        """Test converting to KimiAgent."""
        from kimi_cli.soul.agent import Agent as KimiAgent
        
        agent = UnifiedAgent(
            name="test_agent",
            system_prompt="Test",
            toolset=mock_toolset,
            runtime=mock_runtime,
        )
        
        kimi_agent = agent.to_kimi_agent()
        
        assert isinstance(kimi_agent, KimiAgent)
        assert kimi_agent.name == "test_agent"
        assert kimi_agent.system_prompt == "Test"


class TestExecutionMode:
    """Test ExecutionMode enum."""
    
    def test_execution_modes(self):
        """Test that all execution modes exist."""
        assert ExecutionMode.REACTIVE.name == "REACTIVE"
        assert ExecutionMode.PROACTIVE.name == "PROACTIVE"
        assert ExecutionMode.ADAPTIVE.name == "ADAPTIVE"
    
    def test_execution_mode_values(self):
        """Test that execution modes have unique values."""
        assert ExecutionMode.REACTIVE.value != ExecutionMode.PROACTIVE.value
        assert ExecutionMode.PROACTIVE.value != ExecutionMode.ADAPTIVE.value
