"""
Unified Soul - Merges KimiSoul with GraphAgent

Key design decisions:
1. Implement Soul Protocol for compatibility with existing Kimi-CLI
2. Add GraphAgent's proactive execution capabilities
3. Support mode switching (reactive/proactive/adaptive)
4. Maintain wire protocol compatibility for UI
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncGenerator

import kosong
from kosong import StepResult
from kosong.message import Message
from tenacity import retry_if_exception, stop_after_attempt, wait_exponential_jitter
from functools import partial

from kimi_cli.llm import ModelCapability, LLM
from kimi_cli.soul import (
    Soul,
    StatusSnapshot,
    MaxStepsReached,
    LLMNotSet,
    LLMNotSupported,
    wire_send,
)
from kimi_cli.soul.kimisoul import KimiSoul, StepOutcome, TurnOutcome, BackToTheFuture
from kimi_cli.soul.toolset import KimiToolset
from kimi_cli.soul.context import Context
from kimi_cli.soul.unified.agent import UnifiedAgent, ExecutionMode, ExecutionConfig
from kimi_cli.soul.unified.context import UnifiedContext, MemoryFragment, FragmentType
from kimi_cli.soul.unified.graph_engine import GraphEngine, TaskGraph, TaskNode, GraphScheduler
from kimi_cli.soul.unified.types import AgentMode
from kimi_cli.utils.logging import logger
from kimi_cli.utils.slashcmd import SlashCommand
from kimi_cli.wire.types import (
    TurnBegin,
    StepBegin,
    StatusUpdate,
    TextPart,
    ToolResult,
    CompactionBegin,
    CompactionEnd,
)

if TYPE_CHECKING:
    from kimi_cli.soul.unified.tool import UnifiedTool


@dataclass
class TaskAnalysis:
    """Result of task complexity analysis."""
    complexity: str  # "simple", "moderate", "complex"
    estimated_steps: int
    parallelizable: bool
    recommended_mode: AgentMode


class UnifiedSoul:
    """
    Unified Soul that combines KimiSoul and GraphAgent capabilities.
    
    Modes:
    - REACTIVE: Traditional step-by-step execution (KimiSoul)
    - PROACTIVE: Graph-based parallel execution (GraphAgent)
    - ADAPTIVE: Automatically choose based on task analysis
    """
    
    def __init__(
        self,
        agent: UnifiedAgent,
        context: UnifiedContext,
        mode: AgentMode = AgentMode.REACTIVE,
    ):
        self._agent = agent
        self._context = context
        self._mode = mode
        
        # Create underlying KimiSoul for reactive mode
        self._kimi_soul = KimiSoul(agent.to_kimi_agent(), context=context)
        
        # Create GraphEngine for proactive mode
        self._graph_engine: GraphEngine | None = None
        if mode in (AgentMode.PROACTIVE, AgentMode.ADAPTIVE):
            self._graph_engine = GraphEngine(agent, context)
        
        self._slash_commands = self._build_slash_commands()
    
    # === Soul Protocol Implementation ===
    
    @property
    def name(self) -> str:
        return self._agent.name
    
    @property
    def model_name(self) -> str:
        return self._kimi_soul.model_name
    
    @property
    def model_capabilities(self) -> set[ModelCapability] | None:
        return self._kimi_soul.model_capabilities
    
    @property
    def thinking(self) -> bool | None:
        return self._kimi_soul.thinking
    
    @property
    def status(self) -> StatusSnapshot:
        return self._kimi_soul.status
    
    @property
    def available_slash_commands(self) -> list[SlashCommand[Any]]:
        return self._slash_commands
    
    # === Mode Management ===
    
    @property
    def mode(self) -> AgentMode:
        return self._mode
    
    def set_mode(self, mode: AgentMode) -> None:
        """Change execution mode."""
        self._mode = mode
        if mode in (AgentMode.PROACTIVE, AgentMode.ADAPTIVE) and self._graph_engine is None:
            self._graph_engine = GraphEngine(self._agent, self._context)
    
    # === Main Execution ===
    
    async def run(self, user_input: str | list[Any]) -> None:
        """
        Run the soul with user input.
        
        This is the main entry point that dispatches to appropriate mode.
        """
        # Check for slash commands
        from kimi_cli.utils.slashcmd import parse_slash_command_call
        
        user_message = Message(role="user", content=user_input)
        text_input = user_message.extract_text(" ").strip()
        
        if command_call := parse_slash_command_call(text_input):
            wire_send(TurnBegin(user_input=user_input))
            command = self._find_slash_command(command_call.name)
            if command:
                ret = command.func(self, command_call.args)
                if asyncio.iscoroutine(ret):
                    await ret
                return
        
        # Determine execution mode
        if self._mode == AgentMode.ADAPTIVE:
            analysis = await self._analyze_task(text_input)
            effective_mode = analysis.recommended_mode
            logger.info(
                "Task analysis: complexity={complexity}, mode={mode}",
                complexity=analysis.complexity,
                mode=effective_mode.name,
            )
        else:
            effective_mode = self._mode
        
        # Execute in appropriate mode
        if effective_mode == AgentMode.PROACTIVE:
            await self._run_proactive(user_message)
        else:
            await self._run_reactive(user_message)
    
    async def _run_reactive(self, user_message: Message) -> None:
        """Run in reactive mode (delegate to KimiSoul)."""
        wire_send(TurnBegin(user_input=user_message.content))
        
        # Use KimiSoul's implementation
        result = await self._kimi_soul._turn(user_message)  # type: ignore
        
        if result.stop_reason == "tool_rejected":
            return
    
    async def _run_proactive(self, user_message: Message) -> None:
        """Run in proactive mode (use GraphEngine)."""
        wire_send(TurnBegin(user_input=user_message.content))
        
        if self._graph_engine is None:
            raise RuntimeError("GraphEngine not initialized")
        
        # Create task graph
        task_text = user_message.extract_text(" ")
        graph = await self._graph_engine.create_task_graph(task_text)
        
        # Execute graph
        await self._graph_engine.execute_graph(graph)
    
    async def _analyze_task(self, task: str) -> TaskAnalysis:
        """
        Analyze task complexity to determine execution mode.
        
        This uses a simple heuristic. Can be enhanced with LLM-based analysis.
        """
        # Simple heuristic-based analysis
        complexity_indicators = {
            "simple": ["simple", "quick", "basic", "single"],
            "complex": ["complex", "multiple", "parallel", "analyze all", "refactor"],
        }
        
        task_lower = task.lower()
        
        # Count indicators
        simple_score = sum(1 for w in complexity_indicators["simple"] if w in task_lower)
        complex_score = sum(1 for w in complexity_indicators["complex"] if w in task_lower)
        
        # Estimate steps
        estimated_steps = 5 + complex_score * 3 - simple_score
        
        # Determine complexity
        if complex_score > simple_score or estimated_steps > 10:
            complexity = "complex"
            recommended_mode = AgentMode.PROACTIVE
        elif simple_score > 0:
            complexity = "simple"
            recommended_mode = AgentMode.REACTIVE
        else:
            complexity = "moderate"
            recommended_mode = AgentMode.REACTIVE
        
        return TaskAnalysis(
            complexity=complexity,
            estimated_steps=estimated_steps,
            parallelizable=complex_score > 0,
            recommended_mode=recommended_mode,
        )
    
    # === Slash Commands ===
    
    def _build_slash_commands(self) -> list[SlashCommand[Any]]:
        """Build slash commands for unified soul."""
        commands = list(self._kimi_soul.available_slash_commands)
        
        # Add mode switching commands
        commands.append(
            SlashCommand(
                name="mode",
                func=self._cmd_mode,
                description="Switch execution mode: /mode reactive|proactive|adaptive",
                aliases=[],
            )
        )
        
        return commands
    
    def _find_slash_command(self, name: str) -> SlashCommand[Any] | None:
        """Find a slash command by name."""
        for cmd in self._slash_commands:
            if cmd.name == name or name in cmd.aliases:
                return cmd
        return None
    
    def _cmd_mode(self, soul: UnifiedSoul, args: str) -> None:
        """Handle /mode command."""
        mode_str = args.strip().lower()
        
        mode_map = {
            "reactive": AgentMode.REACTIVE,
            "proactive": AgentMode.PROACTIVE,
            "adaptive": AgentMode.ADAPTIVE,
        }
        
        if mode_str in mode_map:
            self.set_mode(mode_map[mode_str])
            wire_send(TextPart(text=f"Switched to {mode_str} mode."))
        else:
            wire_send(TextPart(text=f"Unknown mode: {mode_str}. Use: reactive, proactive, adaptive"))


