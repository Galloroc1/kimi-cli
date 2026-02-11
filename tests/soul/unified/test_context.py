"""Tests for UnifiedContext."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from kosong.message import Message

from kimi_cli.soul.unified.context import (
    UnifiedContext,
    MemoryFragment,
    FragmentType,
)


class TestMemoryFragment:
    """Test MemoryFragment dataclass."""
    
    def test_fragment_creation(self):
        """Test creating a memory fragment."""
        fragment = MemoryFragment(
            role="user",
            content="Hello",
            fragment_type=FragmentType.USER,
            metadata={"key": "value"},
            tool_name="test_tool",
            tool_args={"arg": "value"},
            node_id="node_1",
            parent_nodes=["parent_1"],
        )
        
        assert fragment.role == "user"
        assert fragment.content == "Hello"
        assert fragment.fragment_type == FragmentType.USER
        assert fragment.metadata == {"key": "value"}
        assert fragment.tool_name == "test_tool"
        assert fragment.tool_args == {"arg": "value"}
        assert fragment.node_id == "node_1"
        assert fragment.parent_nodes == ["parent_1"]
    
    def test_fragment_defaults(self):
        """Test MemoryFragment default values."""
        fragment = MemoryFragment(
            role="assistant",
            content="Hi",
            fragment_type=FragmentType.ASSISTANT,
        )
        
        assert fragment.metadata == {}
        assert fragment.tool_name is None
        assert fragment.tool_args is None
        assert fragment.tool_result is None
        assert fragment.tool_success is None
        assert fragment.node_id is None
        assert fragment.parent_nodes == []
    
    def test_fragment_to_message(self):
        """Test converting fragment to Message."""
        fragment = MemoryFragment(
            role="user",
            content="Hello",
            fragment_type=FragmentType.USER,
        )
        
        message = fragment.to_message()
        
        assert isinstance(message, Message)
        assert message.role == "user"
        assert message.extract_text(" ") == "Hello"
    
    def test_fragment_from_message(self):
        """Test creating fragment from Message."""
        message = Message(role="assistant", content="Hello there")
        
        fragment = MemoryFragment.from_message(message, FragmentType.ASSISTANT)
        
        assert fragment.role == "assistant"
        assert fragment.content == "Hello there"
        assert fragment.fragment_type == FragmentType.ASSISTANT


class TestFragmentType:
    """Test FragmentType enum."""
    
    def test_fragment_types(self):
        """Test that all fragment types exist."""
        assert FragmentType.USER.name == "USER"
        assert FragmentType.ASSISTANT.name == "ASSISTANT"
        assert FragmentType.SYSTEM.name == "SYSTEM"
        assert FragmentType.TOOL_CALL.name == "TOOL_CALL"
        assert FragmentType.TOOL_RESULT.name == "TOOL_RESULT"
        assert FragmentType.OBSERVATION.name == "OBSERVATION"


class TestUnifiedContext:
    """Test UnifiedContext functionality."""
    
    @pytest.fixture
    def temp_context_file(self, tmp_path):
        """Create a temporary context file."""
        return tmp_path / "context.jsonl"
    
    @pytest.mark.asyncio
    async def test_context_creation(self, temp_context_file):
        """Test creating a UnifiedContext."""
        context = UnifiedContext(file_backend=temp_context_file)
        
        assert context.file_backend == temp_context_file
        assert len(context.fragments) == 0
        assert context.execution_graph is None
        assert context.token_count == 0
        assert context.n_checkpoints == 0
    
    @pytest.mark.asyncio
    async def test_append_fragment(self, temp_context_file):
        """Test appending a fragment."""
        context = UnifiedContext(file_backend=temp_context_file)
        
        fragment = MemoryFragment(
            role="user",
            content="Hello",
            fragment_type=FragmentType.USER,
        )
        
        await context.append_fragment(fragment)
        
        assert len(context.fragments) == 1
        assert context.fragments[0].content == "Hello"
    
    @pytest.mark.asyncio
    async def test_append_message(self, temp_context_file):
        """Test appending a Message."""
        context = UnifiedContext(file_backend=temp_context_file)
        
        message = Message(role="user", content="Test message")
        await context.append_message(message, FragmentType.USER)
        
        assert len(context.fragments) == 1
        assert context.fragments[0].role == "user"
        assert context.fragments[0].fragment_type == FragmentType.USER
    
    @pytest.mark.asyncio
    async def test_to_messages(self, temp_context_file):
        """Test converting fragments to Messages."""
        context = UnifiedContext(file_backend=temp_context_file)
        
        await context.append_fragment(MemoryFragment(
            role="user",
            content="Hello",
            fragment_type=FragmentType.USER,
        ))
        await context.append_fragment(MemoryFragment(
            role="assistant",
            content="Hi",
            fragment_type=FragmentType.ASSISTANT,
        ))
        
        messages = context.to_messages()
        
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
    
    @pytest.mark.asyncio
    async def test_get_fragments_by_type(self, temp_context_file):
        """Test getting fragments by type."""
        context = UnifiedContext(file_backend=temp_context_file)
        
        await context.append_fragment(MemoryFragment(
            role="user",
            content="Hello",
            fragment_type=FragmentType.USER,
        ))
        await context.append_fragment(MemoryFragment(
            role="assistant",
            content="Tool result",
            fragment_type=FragmentType.TOOL_RESULT,
        ))
        await context.append_fragment(MemoryFragment(
            role="assistant",
            content="Another result",
            fragment_type=FragmentType.TOOL_RESULT,
        ))
        
        tool_results = context.get_fragments_by_type(FragmentType.TOOL_RESULT)
        
        assert len(tool_results) == 2
        assert all(f.fragment_type == FragmentType.TOOL_RESULT for f in tool_results)
    
    @pytest.mark.asyncio
    async def test_execution_graph(self, temp_context_file):
        """Test execution graph getter/setter."""
        context = UnifiedContext(file_backend=temp_context_file)
        
        assert context.execution_graph is None
        
        graph = {"nodes": ["a", "b"], "edges": []}
        context.execution_graph = graph
        
        assert context.execution_graph == graph
    
    @pytest.mark.asyncio
    async def test_checkpoint(self, temp_context_file):
        """Test creating a checkpoint."""
        context = UnifiedContext(file_backend=temp_context_file)
        
        await context.checkpoint(add_user_message=False)
        
        assert context.n_checkpoints == 1
        
        # Check file was written
        assert temp_context_file.exists()
        content = temp_context_file.read_text()
        assert "_checkpoint" in content
    
    @pytest.mark.asyncio
    async def test_clear(self, temp_context_file):
        """Test clearing context."""
        context = UnifiedContext(file_backend=temp_context_file)
        
        await context.append_fragment(MemoryFragment(
            role="user",
            content="Hello",
            fragment_type=FragmentType.USER,
        ))
        context.execution_graph = {"test": "graph"}
        
        await context.clear()
        
        assert len(context.fragments) == 0
        assert context.execution_graph is None
        assert context.token_count == 0
    
    @pytest.mark.asyncio
    async def test_update_token_count(self, temp_context_file):
        """Test updating token count."""
        context = UnifiedContext(file_backend=temp_context_file)
        
        await context.update_token_count(100)
        
        assert context.token_count == 100
    
    @pytest.mark.asyncio
    async def test_get_tool_call_chain(self, temp_context_file):
        """Test getting tool call chain."""
        context = UnifiedContext(file_backend=temp_context_file)
        
        # Create a chain: parent -> child
        await context.append_fragment(MemoryFragment(
            role="assistant",
            content="Parent",
            fragment_type=FragmentType.TOOL_CALL,
            node_id="parent",
            parent_nodes=[],
        ))
        await context.append_fragment(MemoryFragment(
            role="assistant",
            content="Child",
            fragment_type=FragmentType.TOOL_CALL,
            node_id="child",
            parent_nodes=["parent"],
        ))
        
        chain = context.get_tool_call_chain("child")
        
        assert len(chain) == 2
        assert chain[0].node_id == "parent"
        assert chain[1].node_id == "child"
    
    @pytest.mark.asyncio
    async def test_restore(self, temp_context_file):
        """Test restoring context from file."""
        context = UnifiedContext(file_backend=temp_context_file)
        
        # Add some data
        await context.append_fragment(MemoryFragment(
            role="user",
            content="Hello",
            fragment_type=FragmentType.USER,
        ))
        await context.update_token_count(50)
        
        # Create new context pointing to same file
        context2 = UnifiedContext(file_backend=temp_context_file)
        restored = await context2.restore()
        
        assert restored is True
        assert len(context2.fragments) == 1
        assert context2.fragments[0].content == "Hello"
        assert context2.token_count == 50
    
    @pytest.mark.asyncio
    async def test_restore_empty_file(self, temp_context_file):
        """Test restoring from empty file."""
        context = UnifiedContext(file_backend=temp_context_file)
        
        # File doesn't exist
        restored = await context.restore()
        assert restored is False
        
        # Empty file
        temp_context_file.touch()
        restored = await context.restore()
        assert restored is False
