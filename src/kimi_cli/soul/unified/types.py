"""
Type definitions for Unified Agent Architecture.

This module contains enums and types to avoid circular imports.
"""

from __future__ import annotations

from enum import Enum, auto


class AgentMode(Enum):
    """Agent execution modes."""
    
    REACTIVE = auto()
    """Step-by-step reactive execution (original KimiSoul)."""
    
    PROACTIVE = auto()
    """Proactive graph-based execution with parallel scheduling."""
    
    ADAPTIVE = auto()
    """Automatically choose mode based on task complexity."""


class ExecutionMode(Enum):
    """Agent execution modes (alias for AgentMode)."""
    
    REACTIVE = auto()
    PROACTIVE = auto()
    ADAPTIVE = auto()
