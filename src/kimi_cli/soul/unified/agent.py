"""
Unified Agent - Merges Kimi-CLI's Agent with Agent Framework's BaseAgent

Key design decisions:
1. Keep Kimi-CLI's Runtime and AgentSpec for configuration
2. Add Agent Framework's execution modes (reactive, proactive)
3. Support both single-step and graph-based execution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from kosong.tooling import Toolset
from kosong.message import Message

from kimi_cli.agentspec import load_agent_spec, ResolvedAgentSpec
from kimi_cli.soul.agent import Runtime, LaborMarket, Agent as KimiAgent
from kimi_cli.soul.unified.tool import UnifiedTool, ToolRegistry
from kimi_cli.soul.unified.types import ExecutionMode
from kimi_cli.utils.logging import logger

if TYPE_CHECKING:
    from kimi_cli.soul.unified.context import UnifiedContext


@dataclass
class ExecutionConfig:
    """Configuration for agent execution."""
    
    mode: ExecutionMode = ExecutionMode.REACTIVE
    """Execution mode."""
    
    max_steps: int = 50
    """Maximum steps for reactive mode."""
    
    max_iterations: int = 3
    """Maximum planning iterations for proactive mode."""
    
    parallel_tool_calls: bool = True
    """Whether to allow parallel tool calls."""
    
    parallel_tool_calls_limit: int = 5
    """Maximum parallel tool calls."""
    
    auto_summarize: bool = True
    """Whether to auto-summarize after tool execution."""


@dataclass
class UnifiedAgent:
    """
    Unified Agent that supports both reactive and proactive execution.
    
    This merges:
    - Kimi-CLI's Agent (configuration, toolset, runtime)
    - Agent Framework's BaseAgent (execution modes, memory management)
    """
    
    name: str
    system_prompt: str
    toolset: Toolset
    runtime: Runtime
    execution_config: ExecutionConfig = field(default_factory=ExecutionConfig)
    
    # Extended tool registry for unified tools
    unified_tools: ToolRegistry = field(default_factory=ToolRegistry)
    
    # Subagents (from both systems)
    subagents: dict[str, UnifiedAgent] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        # Register unified tools from toolset
        for tool in self.toolset.tools:
            if isinstance(tool, UnifiedTool):
                self.unified_tools.register(tool.__class__)
    
    @property
    def llm(self):
        """Get LLM from runtime."""
        return self.runtime.llm
    
    @property
    def session(self):
        """Get session from runtime."""
        return self.runtime.session
    
    # === Factory methods ===
    
    @staticmethod
    async def from_spec(
        agent_file: Path,
        runtime: Runtime,
        execution_config: ExecutionConfig | None = None,
        mcp_configs: list[Any] | None = None,
    ) -> UnifiedAgent:
        """
        Load unified agent from specification file.
        
        This extends Kimi-CLI's load_agent with unified features.
        """
        from kimi_cli.soul.agent import load_agent
        
        # Load base agent using Kimi-CLI's loader
        kimi_agent = await load_agent(agent_file, runtime, mcp_configs=mcp_configs or [])
        
        # Wrap in UnifiedAgent
        return UnifiedAgent(
            name=kimi_agent.name,
            system_prompt=kimi_agent.system_prompt,
            toolset=kimi_agent.toolset,
            runtime=kimi_agent.runtime,
            execution_config=execution_config or ExecutionConfig(),
        )
    
    def with_mode(self, mode: ExecutionMode) -> UnifiedAgent:
        """Create a copy with different execution mode."""
        new_config = ExecutionConfig(
            mode=mode,
            max_steps=self.execution_config.max_steps,
            max_iterations=self.execution_config.max_iterations,
            parallel_tool_calls=self.execution_config.parallel_tool_calls,
            parallel_tool_calls_limit=self.execution_config.parallel_tool_calls_limit,
            auto_summarize=self.execution_config.auto_summarize,
        )
        return UnifiedAgent(
            name=self.name,
            system_prompt=self.system_prompt,
            toolset=self.toolset,
            runtime=self.runtime,
            execution_config=new_config,
            unified_tools=self.unified_tools,
            subagents=self.subagents,
        )
    
    # === Tool management ===
    
    def add_unified_tool(self, tool_cls: type[UnifiedTool]) -> None:
        """Add a unified tool to the agent."""
        self.unified_tools.register(tool_cls)
        # Also add to toolset if possible
        # Note: KimiToolset may need extension to support this
    
    def get_tool(self, name: str) -> UnifiedTool | None:
        """Get a tool by name."""
        # First check unified tools
        tool = self.unified_tools.create_instance(name)
        if tool:
            return tool
        
        # Then check toolset
        for t in self.toolset.tools:
            if t.name == name:
                if isinstance(t, UnifiedTool):
                    return t
        
        return None
    
    # === Subagent management ===
    
    def add_subagent(self, name: str, agent: UnifiedAgent, description: str = "") -> None:
        """Add a subagent."""
        self.subagents[name] = agent
        # Also register with labor market for Task tool compatibility
        self.runtime.labor_market.add_fixed_subagent(name, agent.to_kimi_agent(), description)
    
    def to_kimi_agent(self) -> KimiAgent:
        """Convert to Kimi-CLI's Agent (for compatibility)."""
        return KimiAgent(
            name=self.name,
            system_prompt=self.system_prompt,
            toolset=self.toolset,
            runtime=self.runtime,
        )
