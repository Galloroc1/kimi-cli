"""
Unified Agent Architecture for Kimi-CLI

This module provides a unified architecture that merges:
- Kimi-CLI's Soul/Agent system (reactive, step-based)
- Agent Framework's GraphAgent (proactive, DAG-based scheduling)

The unified architecture allows agents to work in both modes:
1. Reactive Mode: Traditional step-by-step tool calling (KimiSoul)
2. Proactive Mode: Graph-based task decomposition and parallel execution (GraphSoul)

Usage:
    # Reactive mode (existing KimiSoul behavior)
    soul = UnifiedSoul(agent, context=context, mode=AgentMode.REACTIVE)
    
    # Proactive mode (GraphAgent behavior)
    soul = UnifiedSoul(agent, context=context, mode=AgentMode.PROACTIVE)
    
    # Adaptive mode (switches based on task complexity)
    soul = UnifiedSoul(agent, context=context, mode=AgentMode.ADAPTIVE)
"""

from __future__ import annotations

# Import types first (no dependencies)
from .types import AgentMode, ExecutionMode

# Import main classes
from .soul import UnifiedSoul
from .agent import UnifiedAgent
from .context import UnifiedContext

__all__ = [
    "AgentMode",
    "ExecutionMode",
    "UnifiedSoul",
    "UnifiedAgent",
    "UnifiedContext",
]
