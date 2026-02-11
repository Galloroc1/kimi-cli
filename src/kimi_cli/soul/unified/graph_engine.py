"""
Graph Engine - Task graph representation and execution

This module provides the data structures for representing and executing
task graphs in proactive mode.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kimi_cli.soul.unified.agent import UnifiedAgent
    from kimi_cli.soul.unified.context import UnifiedContext, MemoryFragment, FragmentType


@dataclass
class TaskNode:
    """
    A node in the task graph representing a single subtask.
    """
    id: str
    """Unique identifier for this node."""
    
    description: str
    """Human-readable description of the task."""
    
    tool: str | None
    """Tool name to use, or None for direct LLM call."""
    
    args: dict[str, Any] = field(default_factory=dict)
    """Arguments for the tool."""
    
    dependencies: list[str] = field(default_factory=list)
    """IDs of nodes that must complete before this node can start."""
    
    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class TaskGraph:
    """
    A directed acyclic graph (DAG) of tasks.
    
    The graph represents a plan where:
    - Nodes are subtasks
    - Edges are dependencies (A -> B means B depends on A)
    """
    
    nodes: dict[str, TaskNode] = field(default_factory=dict)
    """All nodes in the graph, keyed by ID."""
    
    completed: set[str] = field(default_factory=set)
    """Set of completed node IDs."""
    
    results: dict[str, str] = field(default_factory=dict)
    """Results from completed nodes, keyed by node ID."""
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskGraph:
        """Create a TaskGraph from a dictionary."""
        graph = cls()
        
        for node_data in data.get("nodes", []):
            node = TaskNode(
                id=node_data["id"],
                description=node_data["description"],
                tool=node_data.get("tool"),
                args=node_data.get("args", {}),
                dependencies=node_data.get("dependencies", []),
            )
            graph.add_node(node)
        
        return graph
    
    def add_node(self, node: TaskNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
    
    def get_node(self, node_id: str) -> TaskNode | None:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_ready_nodes(self) -> list[TaskNode]:
        """
        Get all nodes that are ready to execute.
        
        A node is ready when:
        1. It hasn't been completed yet
        2. All its dependencies have been completed
        """
        ready = []
        for node in self.nodes.values():
            if node.id in self.completed:
                continue
            
            # Check if all dependencies are satisfied
            if all(dep_id in self.completed for dep_id in node.dependencies):
                ready.append(node)
        
        return ready
    
    def mark_complete(self, node_id: str, result: str) -> None:
        """Mark a node as complete with its result."""
        self.completed.add(node_id)
        self.results[node_id] = result
    
    def is_complete(self) -> bool:
        """Check if all nodes have been completed."""
        return len(self.completed) == len(self.nodes)
    
    def get_leaf_nodes(self) -> list[TaskNode]:
        """Get nodes with no dependents (leaf nodes)."""
        # Find all nodes that are dependencies of others
        has_dependents = set()
        for node in self.nodes.values():
            for dep_id in node.dependencies:
                has_dependents.add(dep_id)
        
        # Leaf nodes are those not in has_dependents
        return [
            node for node in self.nodes.values()
            if node.id not in has_dependents
        ]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary."""
        return {
            "nodes": [
                {
                    "id": node.id,
                    "description": node.description,
                    "tool": node.tool,
                    "args": node.args,
                    "dependencies": node.dependencies,
                }
                for node in self.nodes.values()
            ],
            "completed": list(self.completed),
            "results": self.results,
        }


class GraphScheduler:
    """
    Scheduler for executing task graphs with parallelization.
    
    Implements a simple work-stealing algorithm:
    1. Find all ready nodes (dependencies satisfied)
    2. Execute them in parallel (up to limit)
    3. Mark complete and repeat
    """
    
    def __init__(self, max_parallel: int = 5):
        self.max_parallel = max_parallel
    
    async def schedule(self, graph: TaskGraph, executor) -> None:
        """
        Schedule and execute a task graph.
        
        Args:
            graph: The task graph to execute
            executor: Async function that takes a TaskNode and returns result
        """
        semaphore = asyncio.Semaphore(self.max_parallel)
        
        async def execute_node(node: TaskNode) -> tuple[str, str]:
            async with semaphore:
                result = await executor(node)
                return node.id, result
        
        while not graph.is_complete():
            ready = graph.get_ready_nodes()
            
            if not ready:
                # Check for deadlock
                remaining = set(graph.nodes.keys()) - graph.completed
                if remaining:
                    raise RuntimeError(f"Deadlock detected: {remaining}")
                break
            
            # Execute all ready nodes in parallel
            tasks = [execute_node(node) for node in ready]
            completed_results = await asyncio.gather(*tasks)
            
            # Mark nodes complete
            for node_id, result in completed_results:
                graph.mark_complete(node_id, result)


@dataclass
class ExecutionPlan:
    """
    A high-level execution plan that can be converted to a TaskGraph.
    
    This is an intermediate representation that the LLM can generate,
    which is then converted to a concrete TaskGraph.
    """
    
    goal: str
    """The overall goal of the plan."""
    
    steps: list[PlanStep] = field(default_factory=list)
    """Sequential steps in the plan."""
    
    def to_task_graph(self) -> TaskGraph:
        """Convert this plan to a TaskGraph."""
        graph = TaskGraph()
        
        for i, step in enumerate(self.steps):
            node = TaskNode(
                id=f"step_{i}",
                description=step.description,
                tool=step.tool,
                args=step.args,
                dependencies=[f"step_{j}" for j in step.depends_on],
            )
            graph.add_node(node)
        
        return graph


@dataclass
class PlanStep:
    """A single step in an execution plan."""
    
    description: str
    """What this step does."""
    
    tool: str | None = None
    """Tool to use for this step."""
    
    args: dict[str, Any] = field(default_factory=dict)
    """Arguments for the tool."""
    
    depends_on: list[int] = field(default_factory=list)
    """Indices of steps that must complete before this one."""
    
    can_parallelize: bool = False
    """Whether this step can be parallelized internally."""


class GraphEngine:
    """
    Graph-based execution engine for proactive mode.
    
    This is a simplified version of Agent Framework's GraphAgent,
    adapted to work with Kimi-CLI's infrastructure.
    """
    
    def __init__(self, agent: UnifiedAgent, context: UnifiedContext):
        self._agent = agent
        self._context = context
        self._iteration = 0
    
    async def create_task_graph(self, task: str) -> TaskGraph:
        """
        Create a task graph from user input.
        
        This uses the LLM to decompose the task into a DAG.
        """
        from kosong.message import Message
        from kimi_cli.soul import LLMNotSet
        
        # Build planning prompt
        tools_desc = self._get_tools_description()
        
        prompt = f"""You are a task planner. Decompose the following task into a directed acyclic graph (DAG) of subtasks.

Available tools:
{tools_desc}

Task: {task}

Create a plan with:
1. Nodes: Each node is a subtask that can be executed
2. Edges: Dependencies between nodes (which must complete before others start)
3. For each node, specify the tool to use and arguments

IMPORTANT RULES:
- The "args" field must contain ONLY literal values (numbers, strings, booleans) that match the tool's parameter schema
- DO NOT use template placeholders like "{{{{node_1.result}}}}" or "{{{{result}}}}"
- If a node depends on a previous node's result, you have two options:
  a) Use literal values if you can compute them (e.g., if node_1 computes 5+3=8, use "a": 8 directly)
  b) Or set "tool": null and use the description to explain the computation, letting the LLM handle it

Output format (JSON):
{{
    "nodes": [
        {{
            "id": "node_1",
            "description": "What this node does",
            "tool": "tool_name",
            "args": {{"arg1": "value1"}},
            "dependencies": []
        }}
    ]
}}

Plan:"""
        
        # Call LLM to generate plan
        if self._agent.llm is None:
            raise LLMNotSet()
        
        import kosong
        from kosong import Message
        
        history = [
            Message(role="system", content="You are a task planning assistant."),
            Message(role="user", content=prompt),
        ]
        
        result = await kosong.generate(
            self._agent.llm.chat_provider,
            system_prompt=self._agent.system_prompt,
            tools=[],
            history=history,
        )
        response_content = result.message.extract_text()
        
        # Parse the plan into TaskGraph
        import json
        from kimi_cli.utils.logging import logger
        
        logger.debug("LLM response for task graph: {response}", response=response_content[:500])
        
        try:
            plan_data = json.loads(response_content)
            logger.info("Successfully parsed task graph with {num_nodes} nodes", num_nodes=len(plan_data.get("nodes", [])))
            for node_data in plan_data.get("nodes", []):
                logger.debug("  Node {id}: tool={tool}, args={args}", id=node_data.get("id"), tool=node_data.get("tool"), args=node_data.get("args"))
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON, using fallback")
            # Fallback: create a simple single-node graph
            plan_data = {
                "nodes": [{
                    "id": "node_1",
                    "description": task,
                    "tool": None,
                    "args": {},
                    "dependencies": [],
                }]
            }
        
        return TaskGraph.from_dict(plan_data)
    
    async def execute_graph(self, graph: TaskGraph) -> None:
        """
        Execute a task graph with parallel scheduling.
        """
        from kimi_cli.soul.unified.context import MemoryFragment, FragmentType
        from kimi_cli.soul import wire_send
        from kimi_cli.wire.types import StepBegin, TextPart
        from kimi_cli.utils.logging import logger
        
        max_iterations = self._agent.execution_config.max_iterations
        
        logger.info("Starting graph execution with {num_nodes} nodes", num_nodes=len(graph.nodes))
        
        for iteration in range(max_iterations):
            self._iteration = iteration
            logger.info("Graph execution iteration {iteration}", iteration=iteration)
            
            # Get ready nodes (dependencies satisfied)
            ready_nodes = graph.get_ready_nodes()
            logger.debug("Ready nodes: {ready_ids}", ready_ids=[n.id for n in ready_nodes])
            
            if not ready_nodes:
                if graph.is_complete():
                    logger.info("Graph execution complete")
                    break
                else:
                    logger.error("Deadlock detected in graph")
                    break
            
            # Execute ready nodes in parallel
            results = await self._execute_nodes_parallel(ready_nodes)
            
            # Update graph with results
            for node, result in zip(ready_nodes, results):
                logger.info("Node {node_id} completed with result: {result}", node_id=node.id, result=result[:100] if len(result) > 100 else result)
                graph.mark_complete(node.id, result)
                
                # Add to context
                fragment = MemoryFragment(
                    role="assistant",
                    content=f"Executed {node.description}: {result}",
                    fragment_type=FragmentType.TOOL_RESULT,
                    node_id=node.id,
                    parent_nodes=node.dependencies,
                )
                await self._context.append_fragment(fragment)
        
        # Generate final summary
        await self._generate_summary(graph)
    
    async def _execute_nodes_parallel(
        self, 
        nodes: list[TaskNode]
    ) -> list[str]:
        """Execute multiple nodes in parallel."""
        limit = self._agent.execution_config.parallel_tool_calls_limit
        
        # Limit concurrency
        semaphore = asyncio.Semaphore(limit)
        
        async def execute_with_limit(node: TaskNode) -> str:
            async with semaphore:
                return await self._execute_node(node)
        
        tasks = [execute_with_limit(node) for node in nodes]
        return await asyncio.gather(*tasks)
    
    async def _execute_node(self, node: TaskNode) -> str:
        """Execute a single node."""
        from kimi_cli.soul import get_wire_or_none, LLMNotSet
        from kimi_cli.wire.types import StepBegin
        from kosong.message import Message
        from kimi_cli.utils.logging import logger
        
        logger.debug("Executing node {node_id}: {description}", node_id=node.id, description=node.description)
        
        wire = get_wire_or_none()
        if wire is not None:
            wire.soul_side.send(StepBegin(n=self._iteration + 1))
        
        if node.tool is None:
            # Direct LLM call
            if self._agent.llm is None:
                raise LLMNotSet()
            
            import kosong
            
            logger.debug("Node {node_id}: Direct LLM call", node_id=node.id)
            result = await kosong.generate(
                self._agent.llm.chat_provider,
                system_prompt=self._agent.system_prompt,
                tools=[],
                history=self._context.history + [Message(role="user", content=node.description)],
            )
            return result.message.extract_text()
        
        # Tool execution
        logger.debug("Node {node_id}: Tool execution - {tool_name}", node_id=node.id, tool_name=node.tool)
        tool = self._agent.get_tool(node.tool)
        if tool is None:
            logger.error("Node {node_id}: Tool '{tool_name}' not found", node_id=node.id, tool_name=node.tool)
            return f"Error: Tool '{node.tool}' not found"
        
        # Create params from node.args
        logger.debug("Node {node_id}: Creating params with args: {args}", node_id=node.id, args=node.args)
        try:
            params = tool.params(**node.args)
        except Exception as e:
            logger.error("Node {node_id}: Failed to create params: {error}", node_id=node.id, error=str(e))
            logger.error("Expected params schema: {schema}", schema=tool.params.model_json_schema() if hasattr(tool.params, 'model_json_schema') else 'N/A')
            raise
        
        logger.debug("Node {node_id}: Calling tool with params: {params}", node_id=node.id, params=params.model_dump())
        result = await tool(params)
        
        # result is ToolOk or ToolError
        if result.is_error:
            logger.error("Node {node_id}: Tool execution failed: {error}", node_id=node.id, error=result.message)
            return f"Error: {result.message}"
        
        logger.debug("Node {node_id}: Tool execution successful", node_id=node.id)
        return result.output
    
    def _get_tools_description(self) -> str:
        """Get description of available tools with their parameter schemas."""
        import json
        descriptions = []
        for tool in self._agent.toolset.tools:
            # Get parameter schema from the tool's params model
            if hasattr(tool, 'params') and tool.params:
                schema = tool.params.model_json_schema()
                # Extract required fields and their types
                properties = schema.get('properties', {})
                required = schema.get('required', [])
                
                params_desc = []
                for field_name, field_info in properties.items():
                    field_type = field_info.get('type', 'any')
                    desc = field_info.get('description', '')
                    req_marker = "(required)" if field_name in required else "(optional)"
                    params_desc.append(f"      - {field_name}: {field_type} {req_marker} - {desc}")
                
                params_str = "\n".join(params_desc) if params_desc else "      (no parameters)"
            else:
                params_str = "      (no parameters)"
            
            descriptions.append(f"- {tool.name}: {tool.description}\n    Parameters:\n{params_str}")
        
        return "\n\n".join(descriptions) if descriptions else "(no tools available)"
    
    async def _generate_summary(self, graph: TaskGraph) -> None:
        """Generate final summary of graph execution."""
        from kimi_cli.soul import get_wire_or_none, LLMNotSet
        from kimi_cli.wire.types import TextPart
        from kosong.message import Message
        
        # Collect all results
        results = []
        for node_id, result in graph.results.items():
            node = graph.get_node(node_id)
            results.append(f"- {node.description}: {result}")
        
        summary_prompt = f"""Summarize the following task execution results:

{chr(10).join(results)}

Provide a concise summary for the user."""
        
        if self._agent.llm is None:
            raise LLMNotSet()
        
        import kosong
        from kosong import Message
        
        history = [Message(role="user", content=summary_prompt)]
        result = await kosong.generate(
            self._agent.llm.chat_provider,
            system_prompt=self._agent.system_prompt,
            tools=[],
            history=history,
        )
        summary = result.message.extract_text()
        
        wire = get_wire_or_none()
        if wire is not None:
            wire.soul_side.send(TextPart(text=summary))
