"""
Unified Agent Loader

This module provides loading functions that create UnifiedAgent instances
from configuration files, bridging Kimi-CLI's agent spec with unified features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kimi_cli.agentspec import load_agent_spec, ResolvedAgentSpec
from kimi_cli.auth.oauth import OAuthManager
from kimi_cli.config import Config
from kimi_cli.llm import LLM, create_llm
from kimi_cli.session import Session
from kimi_cli.soul.agent import Runtime, LaborMarket, Agent as KimiAgent
from kimi_cli.soul.unified.agent import UnifiedAgent, ExecutionConfig
from kimi_cli.soul.unified.types import ExecutionMode
from kimi_cli.soul.unified.tool import UnifiedTool
from kimi_cli.utils.logging import logger


async def load_unified_agent(
    agent_file: Path,
    config: Config,
    oauth: OAuthManager,
    llm: LLM | None,
    session: Session,
    execution_mode: ExecutionMode = ExecutionMode.REACTIVE,
    mcp_configs: list[Any] | None = None,
) -> UnifiedAgent:
    """
    Load a unified agent from specification file.
    
    This extends the standard Kimi-CLI agent loading with unified features:
    - Execution mode configuration
    - Unified tool registration
    - Graph execution capabilities
    
    Args:
        agent_file: Path to agent specification YAML
        config: Kimi-CLI configuration
        oauth: OAuth manager
        llm: LLM instance (optional)
        session: Session instance
        execution_mode: Default execution mode
        mcp_configs: MCP server configurations
        
    Returns:
        Configured UnifiedAgent instance
    """
    from kimi_cli.soul.agent import Runtime
    
    # Create runtime (same as Kimi-CLI)
    runtime = await Runtime.create(
        config=config,
        oauth=oauth,
        llm=llm,
        session=session,
        yolo=config.yolo,
    )
    
    # Load using unified loader
    return await UnifiedAgent.from_spec(
        agent_file=agent_file,
        runtime=runtime,
        execution_config=ExecutionConfig(mode=execution_mode),
        mcp_configs=mcp_configs,
    )


def create_unified_soul(
    agent: UnifiedAgent,
    context_file: Path,
    mode: ExecutionMode | None = None,
):
    """
    Create a UnifiedSoul from a UnifiedAgent.
    
    Args:
        agent: The unified agent
        context_file: Path to context file for persistence
        mode: Override execution mode (uses agent's default if None)
        
    Returns:
        Configured UnifiedSoul instance
    """
    from kimi_cli.soul.unified.context import UnifiedContext
    from kimi_cli.soul.unified.soul import UnifiedSoul
    from kimi_cli.soul.unified import AgentMode
    
    # Create unified context
    context = UnifiedContext(file_backend=context_file)
    
    # Determine mode
    effective_mode = mode or agent.execution_config.mode
    agent_mode = _execution_mode_to_agent_mode(effective_mode)
    
    # Create soul
    return UnifiedSoul(
        agent=agent,
        context=context,
        mode=agent_mode,
    )


def _execution_mode_to_agent_mode(mode: ExecutionMode) -> AgentMode:
    """Convert ExecutionMode to AgentMode."""
    from kimi_cli.soul.unified import AgentMode
    
    mapping = {
        ExecutionMode.REACTIVE: AgentMode.REACTIVE,
        ExecutionMode.PROACTIVE: AgentMode.PROACTIVE,
        ExecutionMode.ADAPTIVE: AgentMode.ADAPTIVE,
    }
    return mapping.get(mode, AgentMode.REACTIVE)


# === Configuration Helpers ===

class UnifiedAgentConfig:
    """
    Configuration builder for unified agents.
    
    Example:
        config = (
            UnifiedAgentConfig()
            .with_mode(ExecutionMode.PROACTIVE)
            .with_parallel_limit(10)
            .with_auto_summarize(True)
        )
        agent = await config.load(agent_file, runtime)
    """
    
    def __init__(self):
        self.mode = ExecutionMode.REACTIVE
        self.max_steps = 50
        self.max_iterations = 3
        self.parallel_limit = 5
        self.auto_summarize = True
        self.unified_tools: list[type[UnifiedTool]] = []
    
    def with_mode(self, mode: ExecutionMode) -> UnifiedAgentConfig:
        """Set execution mode."""
        self.mode = mode
        return self
    
    def with_max_steps(self, steps: int) -> UnifiedAgentConfig:
        """Set maximum steps for reactive mode."""
        self.max_steps = steps
        return self
    
    def with_max_iterations(self, iterations: int) -> UnifiedAgentConfig:
        """Set maximum planning iterations."""
        self.max_iterations = iterations
        return self
    
    def with_parallel_limit(self, limit: int) -> UnifiedAgentConfig:
        """Set parallel execution limit."""
        self.parallel_limit = limit
        return self
    
    def with_auto_summarize(self, enabled: bool) -> UnifiedAgentConfig:
        """Set auto-summarize option."""
        self.auto_summarize = enabled
        return self
    
    def with_tool(self, tool_cls: type[UnifiedTool]) -> UnifiedAgentConfig:
        """Register a unified tool."""
        self.unified_tools.append(tool_cls)
        return self
    
    def build_execution_config(self) -> ExecutionConfig:
        """Build ExecutionConfig from this configuration."""
        return ExecutionConfig(
            mode=self.mode,
            max_steps=self.max_steps,
            max_iterations=self.max_iterations,
            parallel_tool_calls=self.parallel_limit > 1,
            parallel_tool_calls_limit=self.parallel_limit,
            auto_summarize=self.auto_summarize,
        )
    
    async def load(
        self,
        agent_file: Path,
        runtime: Runtime,
        mcp_configs: list[Any] | None = None,
    ) -> UnifiedAgent:
        """Load agent with this configuration."""
        agent = await UnifiedAgent.from_spec(
            agent_file=agent_file,
            runtime=runtime,
            execution_config=self.build_execution_config(),
            mcp_configs=mcp_configs,
        )
        
        # Register additional tools
        for tool_cls in self.unified_tools:
            agent.add_unified_tool(tool_cls)
        
        return agent
