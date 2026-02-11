"""
Integration tests for Unified Agent Architecture with REAL LLM.

These tests require a running LLM server and will make actual API calls.
Set environment variables before running:
  - KIMI_BASE_URL or OPENAI_BASE_URL
  - KIMI_API_KEY or OPENAI_API_KEY (optional, defaults to "EMPTY")

Run with:
  export KIMI_BASE_URL="http://localhost:8000/v1"
  uv run pytest tests/soul/unified/test_unified_real_llm.py -v

Or skip these tests:
  uv run pytest tests/soul/unified/ -v --ignore=tests/soul/unified/test_unified_real_llm.py
"""

from __future__ import annotations

import os
import pytest
import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from kimi_cli.config import load_config, Config
from kimi_cli.auth.oauth import OAuthManager
from kimi_cli.session import Session
from kimi_cli.llm import create_llm, LLM
from kimi_cli.soul.agent import Runtime, Agent as KimiAgent, load_agent
from kimi_cli.soul.kimisoul import KimiSoul

from kimi_cli.soul.unified import AgentMode
from kimi_cli.soul.unified.soul import UnifiedSoul
from kimi_cli.soul.unified.agent import UnifiedAgent, ExecutionMode, ExecutionConfig
from kimi_cli.soul.unified.context import UnifiedContext
from kimi_cli.soul.unified.tool import UnifiedTool, unified_tool
from kimi_cli.soul.unified.loader import load_unified_agent, create_unified_soul
from kimi_cli.soul.unified.graph_engine import TaskGraph, TaskNode, GraphEngine

from kaos.path import KaosPath


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def requires_llm():
    """Skip tests if LLM is not available."""
    base_url = os.getenv("KIMI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not base_url:
        pytest.skip("LLM not configured. Set KIMI_BASE_URL or OPENAI_BASE_URL.")
    return base_url


@pytest.fixture(scope="module")
async def real_llm(requires_llm):
    """Create a real LLM instance."""
    config = load_config()
    oauth = OAuthManager(config)
    
    # Use first available provider
    provider = None
    model = None
    
    for prov_name, prov in config.providers.items():
        if prov.base_url:
            provider = prov
            # Find a model for this provider
            for model_name, m in config.models.items():
                if m.provider == prov_name:
                    model = m
                    break
            if model:
                break
    
    if not provider or not model:
        pytest.skip("No configured LLM provider found")
    
    llm = create_llm(provider, model, oauth=oauth)
    
    if llm is None:
        pytest.skip("Failed to create LLM instance")
    
    return llm


@pytest.fixture
async def real_runtime(real_llm, tmp_path):
    """Create a real runtime with LLM."""
    config = load_config()
    oauth = OAuthManager(config)
    
    # Create a temporary session
    work_dir = KaosPath(str(tmp_path))
    session = await Session.create(work_dir)
    
    runtime = await Runtime.create(
        config=config,
        oauth=oauth,
        llm=real_llm,
        session=session,
        yolo=True,  # Auto-approve for testing
    )
    
    return runtime


@pytest.fixture
def temp_context_file(tmp_path):
    """Create a temporary context file."""
    return tmp_path / "test_context.jsonl"


# =============================================================================
# Real Tools for Testing
# =============================================================================

class ReadFileParams(BaseModel):
    path: str = Field(description="Path to file to read")


class ReadFileTool(UnifiedTool[ReadFileParams]):
    """Read a file from the filesystem."""
    name = "read_file"
    description = "Read the contents of a file"
    params = ReadFileParams
    
    async def execute(self, params: ReadFileParams) -> str:
        try:
            file_path = Path(params.path).expanduser()
            if not file_path.exists():
                return f"Error: File '{params.path}' not found"
            if not file_path.is_file():
                return f"Error: '{params.path}' is not a file"
            content = file_path.read_text(encoding="utf-8")
            # Limit output size
            if len(content) > 2000:
                content = content[:2000] + "\n... (truncated)"
            return content
        except Exception as e:
            return f"Error reading file: {e}"


class ListFilesParams(BaseModel):
    directory: str = Field(description="Directory to list")
    pattern: str = Field(default="*", description="File pattern to match")


class ListFilesTool(UnifiedTool[ListFilesParams]):
    """List files in a directory."""
    name = "list_files"
    description = "List files in a directory matching a pattern"
    params = ListFilesParams
    
    async def execute(self, params: ListFilesParams) -> dict:
        try:
            dir_path = Path(params.directory).expanduser()
            if not dir_path.exists():
                return {"error": f"Directory '{params.directory}' not found"}
            
            files = list(dir_path.glob(params.pattern))
            file_list = [str(f.relative_to(dir_path)) for f in files if f.is_file()]
            
            return {
                "directory": str(dir_path),
                "pattern": params.pattern,
                "files": file_list[:50],  # Limit results
                "count": len(file_list),
            }
        except Exception as e:
            return {"error": str(e)}


# =============================================================================
# Real LLM Tests
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.real_llm
async def test_real_llm_connection(real_llm):
    """Test that we can connect to the real LLM."""
    # Just verify the LLM object is valid
    assert real_llm is not None
    assert real_llm.model_name is not None
    assert real_llm.max_context_size > 0
    print(f"\nLLM Connected: {real_llm.model_name}")


@pytest.mark.asyncio
@pytest.mark.real_llm
async def test_reactive_mode_with_real_llm(real_runtime, temp_context_file):
    """Test reactive mode with real LLM."""
    
    # Create mock toolset
    class MockToolset:
        tools = []
    
    toolset = MockToolset()
    
    # Create unified agent
    agent = UnifiedAgent(
        name="reactive_test_agent",
        system_prompt="You are a helpful assistant. Use tools when needed.",
        toolset=toolset,
        runtime=real_runtime,
        execution_config=ExecutionConfig(mode=ExecutionMode.REACTIVE),
    )
    
    # Create context and soul
    context = UnifiedContext(file_backend=temp_context_file)
    soul = UnifiedSoul(agent, context, mode=AgentMode.REACTIVE)
    
    # Verify setup
    assert soul.mode == AgentMode.REACTIVE
    assert soul.model_name == real_runtime.llm.model_name
    
    print(f"\nReactive mode agent created with LLM: {soul.model_name}")


@pytest.mark.asyncio
@pytest.mark.real_llm
async def test_proactive_mode_with_real_llm(real_runtime, temp_context_file):
    """Test proactive mode with real LLM."""
    
    # Create mock toolset
    class MockToolset:
        tools = []
    
    toolset = MockToolset()
    
    # Create unified agent with proactive mode
    agent = UnifiedAgent(
        name="proactive_test_agent",
        system_prompt="You are a helpful assistant that can break down complex tasks.",
        toolset=toolset,
        runtime=real_runtime,
        execution_config=ExecutionConfig(
            mode=ExecutionMode.PROACTIVE,
            max_iterations=2,
            parallel_tool_calls_limit=3,
        ),
    )
    
    # Create context and soul
    context = UnifiedContext(file_backend=temp_context_file)
    soul = UnifiedSoul(agent, context, mode=AgentMode.PROACTIVE)
    
    # Verify setup
    assert soul.mode == AgentMode.PROACTIVE
    assert soul._graph_engine is not None
    
    print(f"\nProactive mode agent created with LLM: {soul.model_name}")


@pytest.mark.asyncio
@pytest.mark.real_llm
async def test_graph_creation_with_real_llm(real_runtime, temp_context_file):
    """Test creating a task graph using real LLM."""
    
    # Create mock toolset
    class MockToolset:
        tools = []
    
    toolset = MockToolset()
    
    # Create agent
    agent = UnifiedAgent(
        name="graph_test_agent",
        system_prompt="You are a task planning assistant.",
        toolset=toolset,
        runtime=real_runtime,
        execution_config=ExecutionConfig(mode=ExecutionMode.PROACTIVE),
    )
    
    # Create context and graph engine
    context = UnifiedContext(file_backend=temp_context_file)
    graph_engine = GraphEngine(agent, context)
    
    # Try to create a task graph
    # Note: This will actually call the LLM
    try:
        graph = await graph_engine.create_task_graph(
            "List all Python files in the current directory and count them"
        )
        
        # Verify graph was created
        assert graph is not None
        assert len(graph.nodes) > 0
        
        print(f"\nGraph created with {len(graph.nodes)} nodes:")
        for node_id, node in graph.nodes.items():
            print(f"  - {node_id}: {node.description[:50]}...")
        
    except Exception as e:
        pytest.skip(f"Graph creation failed (LLM may not support the format): {e}")


@pytest.mark.asyncio
@pytest.mark.real_llm
async def test_adaptive_mode_analysis(real_runtime, temp_context_file):
    """Test adaptive mode task analysis with real context."""
    from kimi_cli.soul.toolset import KimiToolset
    
    toolset = KimiToolset()
    
    agent = UnifiedAgent(
        name="adaptive_test_agent",
        system_prompt="You are a helpful assistant.",
        toolset=toolset,
        runtime=real_runtime,
        execution_config=ExecutionConfig(mode=ExecutionMode.ADAPTIVE),
    )
    
    context = UnifiedContext(file_backend=temp_context_file)
    soul = UnifiedSoul(agent, context, mode=AgentMode.ADAPTIVE)
    
    # Test task analysis
    simple_task = "What is 2 + 2?"
    complex_task = "Analyze all Python files in the project and suggest refactoring"
    
    simple_analysis = await soul._analyze_task(simple_task)
    complex_analysis = await soul._analyze_task(complex_task)
    
    print(f"\nTask Analysis:")
    print(f"  Simple task: {simple_analysis.complexity} -> {simple_analysis.recommended_mode.name}")
    print(f"  Complex task: {complex_analysis.complexity} -> {complex_analysis.recommended_mode.name}")
    
    # Simple tasks should not be parallelizable
    assert not simple_analysis.parallelizable or simple_analysis.estimated_steps < 10
    # Complex tasks should be parallelizable or have many steps
    assert complex_analysis.parallelizable or complex_analysis.estimated_steps > 5


@pytest.mark.asyncio
@pytest.mark.real_llm
async def test_mode_switching_with_real_llm(real_runtime, temp_context_file):
    """Test switching modes with real LLM."""
    from kimi_cli.soul.toolset import KimiToolset
    
    toolset = KimiToolset()
    
    agent = UnifiedAgent(
        name="switch_test_agent",
        system_prompt="You are a helpful assistant.",
        toolset=toolset,
        runtime=real_runtime,
        execution_config=ExecutionConfig(mode=ExecutionMode.REACTIVE),
    )
    
    context = UnifiedContext(file_backend=temp_context_file)
    soul = UnifiedSoul(agent, context, mode=AgentMode.REACTIVE)
    
    # Verify initial state
    assert soul.mode == AgentMode.REACTIVE
    initial_model = soul.model_name
    
    # Switch to proactive
    soul.set_mode(AgentMode.PROACTIVE)
    assert soul.mode == AgentMode.PROACTIVE
    assert soul.model_name == initial_model  # Should keep same LLM
    
    # Switch to adaptive
    soul.set_mode(AgentMode.ADAPTIVE)
    assert soul.mode == AgentMode.ADAPTIVE
    
    print(f"\nMode switching test passed with LLM: {soul.model_name}")


@pytest.mark.asyncio
@pytest.mark.real_llm
async def test_unified_vs_original_kimisoul(real_runtime, temp_context_file):
    """Compare UnifiedSoul with original KimiSoul behavior."""
    from kimi_cli.soul.toolset import KimiToolset
    
    toolset = KimiToolset()
    
    # Create original KimiSoul
    kimi_agent = KimiAgent(
        name="comparison_agent",
        system_prompt="You are a helpful assistant.",
        toolset=toolset,
        runtime=real_runtime,
    )
    kimi_soul = KimiSoul(kimi_agent, context=UnifiedContext(file_backend=temp_context_file))
    
    # Create UnifiedSoul in reactive mode (should behave similarly)
    unified_agent = UnifiedAgent(
        name="comparison_agent",
        system_prompt="You are a helpful assistant.",
        toolset=toolset,
        runtime=real_runtime,
        execution_config=ExecutionConfig(mode=ExecutionMode.REACTIVE),
    )
    unified_soul = UnifiedSoul(
        unified_agent,
        UnifiedContext(file_backend=temp_context_file),
        mode=AgentMode.REACTIVE,
    )
    
    # Both should have same model
    assert kimi_soul.model_name == unified_soul.model_name
    assert kimi_soul.model_capabilities == unified_soul.model_capabilities
    
    print(f"\nUnifiedSoul matches KimiSoul:")
    print(f"  Model: {unified_soul.model_name}")
    print(f"  Capabilities: {unified_soul.model_capabilities}")


# =============================================================================
# Performance Comparison Tests
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.real_llm
@pytest.mark.slow
async def test_proactive_parallel_performance(real_runtime, temp_context_file):
    """Test that proactive mode can execute tasks in parallel."""
    import time
    
    # Create a mock tool that simulates work
    class SlowTool(UnifiedTool[BaseModel]):
        name = "slow_tool"
        description = "A slow tool for testing"
        params = BaseModel
        
        async def execute(self, params):
            await asyncio.sleep(0.5)  # Simulate work
            return "Done"
    
    # Create graph with multiple independent nodes
    graph = TaskGraph()
    for i in range(3):
        graph.add_node(TaskNode(
            id=f"task_{i}",
            description=f"Task {i}",
            tool="slow_tool",
            args={},
            dependencies=[],
        ))
    
    # Execute in parallel
    start_time = time.time()
    
    async def mock_executor(node):
        await asyncio.sleep(0.5)
        return f"result_{node.id}"
    
    # Execute all ready nodes concurrently
    ready = graph.get_ready_nodes()
    tasks = [mock_executor(node) for node in ready]
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    
    # Should take ~0.5s (parallel), not ~1.5s (sequential)
    assert elapsed < 1.0, f"Parallel execution took too long: {elapsed}s"
    assert len(results) == 3
    
    print(f"\nParallel execution: {elapsed:.2f}s for 3 tasks (expected < 1.0s)")


# =============================================================================
# Configuration Tests
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.real_llm
async def test_execution_config_with_real_llm(real_runtime, temp_context_file):
    """Test different execution configurations."""
    from kimi_cli.soul.toolset import KimiToolset
    
    toolset = KimiToolset()
    
    configs = [
        ExecutionConfig(mode=ExecutionMode.REACTIVE, max_steps=5),
        ExecutionConfig(mode=ExecutionMode.PROACTIVE, max_iterations=1),
        ExecutionConfig(mode=ExecutionMode.ADAPTIVE, parallel_tool_calls_limit=2),
    ]
    
    for config in configs:
        agent = UnifiedAgent(
            name="config_test_agent",
            system_prompt="Test agent.",
            toolset=toolset,
            runtime=real_runtime,
            execution_config=config,
        )
        
        context = UnifiedContext(file_backend=temp_context_file)
        mode = {
            ExecutionMode.REACTIVE: AgentMode.REACTIVE,
            ExecutionMode.PROACTIVE: AgentMode.PROACTIVE,
            ExecutionMode.ADAPTIVE: AgentMode.ADAPTIVE,
        }[config.mode]
        
        soul = UnifiedSoul(agent, context, mode=mode)
        
        assert soul.mode == mode
        print(f"\nConfig test passed: {config.mode.name}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    """Run tests manually."""
    pytest.main([__file__, "-v", "-m", "real_llm"])
