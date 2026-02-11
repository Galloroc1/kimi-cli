# Unified Agent Architecture

## Overview

This module provides a unified architecture that merges Kimi-CLI's reactive agent system with the Agent Framework's proactive graph-based execution.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        UnifiedSoul                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Reactive Mode  │  │  Proactive Mode │  │  Adaptive Mode  │  │
│  │   (KimiSoul)    │  │  (GraphEngine)  │  │  (Auto-switch)  │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                    │           │
│           └────────────────────┼────────────────────┘           │
│                                │                                │
│                    ┌───────────┴───────────┐                    │
│                    │     UnifiedAgent      │                    │
│                    │  ┌─────────────────┐  │                    │
│                    │  │  UnifiedContext │  │                    │
│                    │  │  ┌───────────┐  │  │                    │
│                    │  │  │  Kimi     │  │  │                    │
│                    │  │  │  Context  │  │  │                    │
│                    │  │  │ (persist) │  │  │                    │
│                    │  │  └───────────┘  │  │                    │
│                    │  │  ┌───────────┐  │  │                    │
│                    │  │  │  Memory   │  │  │                    │
│                    │  │  │ Fragments │  │  │                    │
│                    │  │  │  (graph)  │  │  │                    │
│                    │  │  └───────────┘  │  │                    │
│                    │  └─────────────────┘  │                    │
│                    │  ┌─────────────────┐  │                    │
│                    │  │  Unified Tools  │  │                    │
│                    │  │  ┌───────────┐  │  │                    │
│                    │  │  │  kosong   │  │  │                    │
│                    │  │  │  Callable │  │  │                    │
│                    │  │  │   Tool2   │  │  │                    │
│                    │  │  └───────────┘  │  │                    │
│                    │  │  ┌───────────┐  │  │                    │
│                    │  │  │  Agent    │  │  │                    │
│                    │  │  │ Framework │  │  │                    │
│                    │  │  │   Tool    │  │  │                    │
│                    │  │  └───────────┘  │  │                    │
│                    │  └─────────────────┘  │                    │
│                    └───────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. UnifiedSoul

The main entry point that implements the Soul Protocol.

**Features:**
- Three execution modes: REACTIVE, PROACTIVE, ADAPTIVE
- Automatic task complexity analysis
- Mode switching via `/mode` command
- Compatible with existing Kimi-CLI infrastructure

**Usage:**
```python
from kimi_cli.soul.unified import AgentMode
from kimi_cli.soul.unified.soul import UnifiedSoul

soul = UnifiedSoul(agent, context, mode=AgentMode.ADAPTIVE)
await soul.run("Complex task...")  # Will auto-choose mode
```

### 2. UnifiedAgent

Merges Kimi-CLI's Agent with Agent Framework's BaseAgent.

**Features:**
- Execution configuration (mode, limits, etc.)
- Unified tool registry
- Subagent management
- Backward compatible with KimiAgent

**Usage:**
```python
from kimi_cli.soul.unified.agent import UnifiedAgent, ExecutionMode

agent = await UnifiedAgent.from_spec(
    agent_file=path,
    runtime=runtime,
    execution_config=ExecutionConfig(mode=ExecutionMode.PROACTIVE),
)
```

### 3. UnifiedContext

Combines Kimi-CLI's file-backed persistence with Agent Framework's structured memory.

**Features:**
- Checkpoint and revert (from KimiContext)
- Memory fragments with metadata
- Execution graph tracking
- Tool call chain reconstruction

**Usage:**
```python
from kimi_cli.soul.unified.context import UnifiedContext, MemoryFragment, FragmentType

context = UnifiedContext(file_backend=path)
await context.append_fragment(MemoryFragment(
    role="assistant",
    content="Result...",
    fragment_type=FragmentType.TOOL_RESULT,
    node_id="node_1",
))
```

### 4. UnifiedTool

Base class for tools that work in both systems.

**Features:**
- Inherits from kosong's CallableTool2
- Agent Framework's result conversion
- MCP and LangChain compatibility
- Decorator for easy creation

**Usage:**
```python
from kimi_cli.soul.unified.tool import unified_tool
from pydantic import BaseModel

class Params(BaseModel):
    query: str

@unified_tool(name="search", description="Search tool")
async def search(params: Params) -> dict:
    return {"results": [...]}
```

### 5. GraphEngine

Proactive execution engine for parallel task scheduling.

**Features:**
- Task graph creation via LLM
- Parallel node execution
- Dependency resolution
- Deadlock detection

**Usage:**
```python
from kimi_cli.soul.unified.graph_engine import TaskGraph, TaskNode

graph = TaskGraph()
graph.add_node(TaskNode(
    id="node_1",
    description="Task description",
    tool="tool_name",
    args={"key": "value"},
    dependencies=[],
))
```

## Execution Modes

### Reactive Mode (Original KimiSoul)

```
User Input → LLM Call → Tool Calls → Results → LLM Call → ... → Final Answer
```

- Step-by-step execution
- Tools called sequentially
- Good for simple, linear tasks
- Lower latency for simple queries

### Proactive Mode (GraphAgent)

```
User Input → Plan Creation → Task Graph → Parallel Execution → Summary
                ↓
         ┌──────┴──────┐
         ↓             ↓
    [Task A]       [Task B]  (parallel)
         ↓             ↓
         └──────┬──────┘
                ↓
           [Task C]      (depends on A and B)
                ↓
           Summary
```

- Task decomposition
- Parallel execution
- Better for complex, multi-step tasks
- Higher throughput for parallelizable work

### Adaptive Mode

```
User Input → Complexity Analysis → Mode Selection → Execution
                    ↓
            Simple → Reactive
            Complex → Proactive
```

- Analyzes task complexity
- Automatically chooses mode
- Best of both worlds

## Integration Points

### With Existing Kimi-CLI Code

1. **Agent Loading**: Use `load_unified_agent()` instead of `load_agent()`
2. **Soul Creation**: Use `create_unified_soul()` instead of `KimiSoul()`
3. **Tool Registration**: Use `@unified_tool` decorator
4. **Configuration**: Use `UnifiedAgentConfig` builder

### Migration Path

```python
# Before (Kimi-CLI)
from kimi_cli.soul.agent import load_agent
from kimi_cli.soul.kimisoul import KimiSoul

agent = await load_agent(agent_file, runtime)
soul = KimiSoul(agent, context=context)

# After (Unified)
from kimi_cli.soul.unified.loader import load_unified_agent, create_unified_soul

agent = await load_unified_agent(agent_file, runtime, execution_mode=ExecutionMode.PROACTIVE)
soul = create_unified_soul(agent, context_file)
```

## Configuration

### Agent YAML

```yaml
version: 1
agent:
  name: "unified-agent"
  system_prompt_path: ./system.md
  tools:
    - "kimi_cli.tools.shell:Shell"
    - "kimi_cli.tools.file:ReadFile"
    # Unified tools work alongside existing tools
  
  # New: Execution configuration
  execution:
    mode: proactive  # reactive | proactive | adaptive
    max_iterations: 3
    parallel_limit: 5
```

### Programmatic Configuration

```python
from kimi_cli.soul.unified.loader import UnifiedAgentConfig
from kimi_cli.soul.unified.agent import ExecutionMode

config = (
    UnifiedAgentConfig()
    .with_mode(ExecutionMode.PROACTIVE)
    .with_parallel_limit(10)
    .with_max_iterations(5)
)

agent = await config.load(agent_file, runtime)
```

## Future Enhancements

1. **Hierarchical Planning**: Multi-level task graphs
2. **Dynamic Replanning**: Adjust plan based on intermediate results
3. **Tool Result Caching**: Avoid redundant tool calls
4. **Execution Visualization**: Show graph execution in UI
5. **Learning**: Remember successful plans for similar tasks

## Testing

```python
# Test reactive mode
async def test_reactive():
    soul = create_unified_soul(agent, context_file, mode=AgentMode.REACTIVE)
    await soul.run("Simple question")

# Test proactive mode
async def test_proactive():
    soul = create_unified_soul(agent, context_file, mode=AgentMode.PROACTIVE)
    await soul.run("Complex multi-step task")

# Test adaptive mode
async def test_adaptive():
    soul = create_unified_soul(agent, context_file, mode=AgentMode.ADAPTIVE)
    await soul.run("Simple")  # Should use reactive
    await soul.run("Complex parallel task")  # Should use proactive
```
