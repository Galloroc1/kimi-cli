"""
Example usage of the Unified Agent Architecture.

This file demonstrates how to use the unified architecture in various scenarios.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from kimi_cli.soul.unified import AgentMode
from kimi_cli.soul.unified.agent import ExecutionMode
from kimi_cli.soul.unified.loader import (
    load_unified_agent,
    create_unified_soul,
    UnifiedAgentConfig,
)
from kimi_cli.soul.unified.tool import unified_tool, UnifiedTool, register_tool
from pydantic import BaseModel, Field


# === Example 1: Basic Usage ===

async def example_basic():
    """Basic usage with default reactive mode."""
    from kimi_cli.config import load_config
    from kimi_cli.auth.oauth import OAuthManager
    from kimi_cli.session import Session
    from kaos.path import KaosPath
    
    # Load configuration
    config = load_config()
    oauth = OAuthManager()
    
    # Create session
    work_dir = KaosPath(".")
    session = await Session.create(work_dir)
    
    # Load unified agent with proactive mode
    agent = await load_unified_agent(
        agent_file=Path("src/kimi_cli/agents/default/agent.yaml"),
        config=config,
        oauth=oauth,
        llm=None,  # Will be loaded from config
        session=session,
        execution_mode=ExecutionMode.PROACTIVE,
    )
    
    # Create soul
    soul = create_unified_soul(
        agent=agent,
        context_file=session.context_file,
    )
    
    # Run
    await soul.run("Analyze the codebase structure and identify refactoring opportunities")


# === Example 2: Custom Tool ===

class SearchParams(BaseModel):
    """Parameters for search tool."""
    query: str = Field(description="Search query")
    max_results: int = Field(default=10, description="Maximum results")


@unified_tool(name="smart_search", description="Intelligent code search")
async def smart_search(params: SearchParams) -> dict:
    """
    Example unified tool that demonstrates the unified_tool decorator.
    """
    # Implementation would search codebase
    return {
        "query": params.query,
        "results": ["file1.py", "file2.py"],
        "count": 2,
    }


async def example_custom_tool():
    """Using custom unified tools."""
    config = UnifiedAgentConfig()
    config.with_tool(smart_search)
    
    # ... rest of setup


# === Example 3: Mode Switching ===

async def example_mode_switching():
    """Demonstrate switching between modes."""
    # Setup agent...
    agent = None  # Would be loaded
    context_file = Path("./context.jsonl")
    
    # Create soul in adaptive mode
    soul = create_unified_soul(
        agent=agent,
        context_file=context_file,
        mode=AgentMode.ADAPTIVE,
    )
    
    # Simple task - will use reactive mode
    await soul.run("What is 2 + 2?")
    
    # Complex task - adaptive mode will switch to proactive
    await soul.run("""
        Analyze all Python files in /src:
        1. Find code style issues
        2. Identify potential bugs
        3. Suggest performance improvements
        Provide a comprehensive report.
    """)
    
    # Manually switch to proactive mode
    soul.set_mode(AgentMode.PROACTIVE)
    await soul.run("Another complex task...")


# === Example 4: Graph Execution ===

async def example_graph_execution():
    """Direct graph execution."""
    from kimi_cli.soul.unified.graph_engine import TaskGraph, TaskNode, GraphScheduler
    
    # Create a task graph manually
    graph = TaskGraph()
    
    # Add nodes
    graph.add_node(TaskNode(
        id="analyze",
        description="Analyze requirements",
        tool=None,  # Direct LLM call
        args={},
        dependencies=[],
    ))
    
    graph.add_node(TaskNode(
        id="search",
        description="Search codebase",
        tool="Glob",
        args={"pattern": "**/*.py"},
        dependencies=["analyze"],
    ))
    
    graph.add_node(TaskNode(
        id="review",
        description="Review files",
        tool="ReadFile",
        args={"path": "main.py"},
        dependencies=["search"],
    ))
    
    # Execute with scheduler
    scheduler = GraphScheduler(max_parallel=3)
    # await scheduler.schedule(graph, executor_func)


# === Example 5: Configuration Builder ===

async def example_config_builder():
    """Using the configuration builder."""
    from kimi_cli.soul.agent import Runtime
    
    # Build configuration fluently
    config = (
        UnifiedAgentConfig()
        .with_mode(ExecutionMode.PROACTIVE)
        .with_parallel_limit(10)
        .with_max_iterations(5)
        .with_auto_summarize(True)
        .with_tool(smart_search)
    )
    
    # Load with configuration
    runtime = None  # Would be created
    agent = await config.load(
        agent_file=Path("agents/default/agent.yaml"),
        runtime=runtime,
    )


# === Example 6: Subagents ===

async def example_subagents():
    """Using subagents with different modes."""
    # Create main agent
    main_agent = None  # Would be loaded
    
    # Create specialized subagent
    coder_config = (
        UnifiedAgentConfig()
        .with_mode(ExecutionMode.REACTIVE)  # Coder uses reactive mode
        .with_max_steps(100)
    )
    
    coder_agent = await coder_config.load(
        agent_file=Path("agents/coder/agent.yaml"),
        runtime=main_agent.runtime,
    )
    
    # Add as subagent
    main_agent.add_subagent(
        name="coder",
        agent=coder_agent,
        description="Specialized in code generation and refactoring",
    )
    
    # Now main agent can delegate to coder subagent


# === Run Examples ===

if __name__ == "__main__":
    # Run an example
    # asyncio.run(example_basic())
    pass
