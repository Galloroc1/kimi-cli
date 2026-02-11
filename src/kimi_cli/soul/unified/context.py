"""
Unified Context - Merges Kimi-CLI's Context with Agent Framework's Memory

Key design decisions:
1. Keep Kimi-CLI's file-backed persistence (checkpoint, revert)
2. Add Agent Framework's structured memory fragments (RoleFragment, ToolFragment)
3. Support both synchronous and asynchronous message access
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os
from kosong.message import Message

from kimi_cli.soul.context import Context as KimiContext
from kimi_cli.soul.message import system
from kimi_cli.utils.logging import logger
from kimi_cli.utils.path import next_available_rotation


class FragmentType(Enum):
    """Types of memory fragments."""
    USER = auto()
    ASSISTANT = auto()
    SYSTEM = auto()
    TOOL_CALL = auto()
    TOOL_RESULT = auto()
    OBSERVATION = auto()


@dataclass
class MemoryFragment:
    """
    A structured memory fragment from Agent Framework.
    
    This provides richer metadata than plain messages, enabling:
    - Better context compression
    - Tool call tracking
    - Execution graph reconstruction
    """
    role: str
    content: str
    fragment_type: FragmentType
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # For tool calls
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: Any | None = None
    tool_success: bool | None = None
    
    # For graph execution
    node_id: str | None = None
    parent_nodes: list[str] = field(default_factory=list)
    
    def to_message(self) -> Message:
        """Convert fragment to kosong Message."""
        return Message(role=self.role, content=self.content)
    
    @classmethod
    def from_message(cls, message: Message, fragment_type: FragmentType = None) -> MemoryFragment:
        """Create fragment from kosong Message."""
        return cls(
            role=message.role,
            content=message.extract_text(" "),
            fragment_type=fragment_type or FragmentType.USER,
        )


class UnifiedContext:
    """
    Unified context that combines:
    - Kimi-CLI's file-backed persistence and checkpointing
    - Agent Framework's structured memory fragments
    - Graph execution state tracking
    """
    
    def __init__(self, file_backend: Path):
        self._kimi_context = KimiContext(file_backend)
        self._fragments: list[MemoryFragment] = []
        self._execution_graph: dict[str, Any] | None = None
        
    # === Delegated properties from KimiContext ===
    
    @property
    def history(self) -> Sequence[Message]:
        """Get message history (for kosong compatibility)."""
        return self._kimi_context.history
    
    @property
    def token_count(self) -> int:
        return self._kimi_context.token_count
    
    @property
    def n_checkpoints(self) -> int:
        return self._kimi_context.n_checkpoints
    
    @property
    def file_backend(self) -> Path:
        return self._kimi_context.file_backend
    
    # === New properties for Agent Framework ===
    
    @property
    def fragments(self) -> Sequence[MemoryFragment]:
        """Get structured memory fragments."""
        return self._fragments
    
    @property
    def execution_graph(self) -> dict[str, Any] | None:
        """Get current execution graph (for proactive mode)."""
        return self._execution_graph
    
    @execution_graph.setter
    def execution_graph(self, graph: dict[str, Any] | None):
        self._execution_graph = graph
    
    # === Persistence methods (delegated to KimiContext) ===
    
    async def restore(self) -> bool:
        """Restore context from file."""
        success = await self._kimi_context.restore()
        if success:
            # Rebuild fragments from messages
            self._fragments = [
                MemoryFragment.from_message(msg)
                for msg in self._kimi_context.history
            ]
        return success
    
    async def checkpoint(self, add_user_message: bool):
        """Create a checkpoint."""
        await self._kimi_context.checkpoint(add_user_message)
    
    async def revert_to(self, checkpoint_id: int):
        """Revert to a checkpoint."""
        await self._kimi_context.revert_to(checkpoint_id)
        # Rebuild fragments
        self._fragments = [
            MemoryFragment.from_message(msg)
            for msg in self._kimi_context.history
        ]
        self._execution_graph = None
    
    async def clear(self):
        """Clear context."""
        await self._kimi_context.clear()
        self._fragments.clear()
        self._execution_graph = None
    
    # === Message management ===
    
    async def append_message(
        self, 
        message: Message | Sequence[Message],
        fragment_type: FragmentType = None
    ):
        """Append message(s) with optional fragment type."""
        await self._kimi_context.append_message(message)
        
        # Also add to fragments
        messages = [message] if isinstance(message, Message) else message
        for msg in messages:
            self._fragments.append(
                MemoryFragment.from_message(msg, fragment_type)
            )
    
    async def append_fragment(self, fragment: MemoryFragment):
        """Append a structured fragment."""
        self._fragments.append(fragment)
        # Also append as message for persistence
        await self._kimi_context.append_message(fragment.to_message())
    
    async def update_token_count(self, token_count: int):
        """Update token count."""
        await self._kimi_context.update_token_count(token_count)
    
    # === Graph execution helpers ===
    
    def get_fragments_by_type(self, fragment_type: FragmentType) -> list[MemoryFragment]:
        """Get all fragments of a specific type."""
        return [f for f in self._fragments if f.fragment_type == fragment_type]
    
    def get_tool_call_chain(self, node_id: str) -> list[MemoryFragment]:
        """Get the chain of tool calls leading to a node."""
        chain = []
        current = next(
            (f for f in self._fragments if f.node_id == node_id),
            None
        )
        while current:
            chain.append(current)
            if current.parent_nodes:
                parent_id = current.parent_nodes[0]  # Follow first parent
                current = next(
                    (f for f in self._fragments if f.node_id == parent_id),
                    None
                )
            else:
                break
        return list(reversed(chain))
    
    def to_messages(self) -> list[Message]:
        """Convert all fragments to messages (for LLM calls)."""
        return [f.to_message() for f in self._fragments]
