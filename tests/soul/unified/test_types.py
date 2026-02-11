"""Tests for unified types (AgentMode, ExecutionMode)."""

from __future__ import annotations

import pytest

from kimi_cli.soul.unified.types import AgentMode, ExecutionMode


class TestAgentMode:
    """Test AgentMode enum."""
    
    def test_agent_mode_values(self):
        """Test that AgentMode has expected values."""
        assert AgentMode.REACTIVE.name == "REACTIVE"
        assert AgentMode.PROACTIVE.name == "PROACTIVE"
        assert AgentMode.ADAPTIVE.name == "ADAPTIVE"
    
    def test_agent_mode_auto_values(self):
        """Test that AgentMode values are auto-incremented."""
        # Auto() should give unique values
        assert AgentMode.REACTIVE.value != AgentMode.PROACTIVE.value
        assert AgentMode.PROACTIVE.value != AgentMode.ADAPTIVE.value
        assert AgentMode.REACTIVE.value != AgentMode.ADAPTIVE.value


class TestExecutionMode:
    """Test ExecutionMode enum."""
    
    def test_execution_mode_values(self):
        """Test that ExecutionMode has expected values."""
        assert ExecutionMode.REACTIVE.name == "REACTIVE"
        assert ExecutionMode.PROACTIVE.name == "PROACTIVE"
        assert ExecutionMode.ADAPTIVE.name == "ADAPTIVE"
    
    def test_execution_mode_auto_values(self):
        """Test that ExecutionMode values are auto-incremented."""
        assert ExecutionMode.REACTIVE.value != ExecutionMode.PROACTIVE.value
        assert ExecutionMode.PROACTIVE.value != ExecutionMode.ADAPTIVE.value
        assert ExecutionMode.REACTIVE.value != ExecutionMode.ADAPTIVE.value
