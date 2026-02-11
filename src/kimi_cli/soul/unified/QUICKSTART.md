# Quick Start: Unified Agent Architecture

## Installation

The unified module is part of kimi-cli. No additional installation needed.

```bash
# Ensure you're in the kimi-cli directory
cd /Users/ran/pyproject/kimi-cli

# Install dependencies
uv sync
```

## Basic Usage

### 1. Simple Reactive Mode (Like Original KimiSoul)

```python
import asyncio
from pathlib import Path
from kimi_cli.soul.unified.loader import load_unified_agent, create_unified_soul
from kimi_cli.soul.unified.agent import ExecutionMode

async def main():
    # Setup (same as before)
    from kimi_cli.config import load_config
    from kimi_cli.auth.oauth import OAuthManager
    from kimi_cli.session import Session
    from kaos.path import KaosPath
    
    config = load_config()
    oauth = OAuthManager()
    session = await Session.create(KaosPath("."))
    
    # Load agent in reactive mode
    agent = await load_unified_agent(
        agent_file=Path("src/kimi_cli/agents/default/agent.yaml"),
        config=config,
        oauth=oauth,
        llm=None,
        session=session,
        execution_mode=ExecutionMode.REACTIVE,  # Step-by-step
    )
    
    # Create soul
    soul = create_unified_soul(agent, session.context_file)
    
    # Run
    await soul.run("What files are in the current directory?")

asyncio.run(main())
```

### 2. Proactive Mode (Graph-Based Execution)

```python
# Just change the mode!
agent = await load_unified_agent(
    agent_file=Path("src/kimi_cli/agents/default/agent.yaml"),
    config=config,
    oauth=oauth,
    llm=None,
    session=session,
    execution_mode=ExecutionMode.PROACTIVE,  # Graph-based
)

soul = create_unified_soul(agent, session.context_file)

# This will create a task graph and execute in parallel
await soul.run("""
    Analyze the codebase:
    1. Find all Python files
    2. Check for syntax errors
    3. Identify imports
    4. Generate a summary report
""")
```

### 3. Adaptive Mode (Auto-Select)

```python
agent = await load_unified_agent(
    agent_file=Path("src/kimi_cli/agents/default/agent.yaml"),
    config=config,
    oauth=oauth,
    llm=None,
    session=session,
    execution_mode=ExecutionMode.ADAPTIVE,  # Auto-select
)

soul = create_unified_soul(agent, session.context_file)

# Simple task → reactive mode
await soul.run("What is 2 + 2?")

# Complex task → proactive mode
await soul.run("Refactor all Python files to use type hints")
```

## Creating Custom Tools

### Using Decorator

```python
from pydantic import BaseModel, Field
from kimi_cli.soul.unified.tool import unified_tool

class CalculatorParams(BaseModel):
    a: int = Field(description="First number")
    b: int = Field(description="Second number")
    operation: str = Field(description="Operation: add, subtract, multiply, divide")

@unified_tool(name="calculator", description="Perform calculations")
async def calculator(params: CalculatorParams) -> dict:
    """Simple calculator tool."""
    if params.operation == "add":
        result = params.a + params.b
    elif params.operation == "subtract":
        result = params.a - params.b
    elif params.operation == "multiply":
        result = params.a * params.b
    elif params.operation == "divide":
        result = params.a / params.b if params.b != 0 else "Error: Division by zero"
    else:
        result = f"Unknown operation: {params.operation}"
    
    return {"result": result}

# Register with agent
config = UnifiedAgentConfig()
config.with_tool(calculator)
```

### Using Class

```python
from kimi_cli.soul.unified.tool import UnifiedTool, register_tool
from pydantic import BaseModel

class WeatherParams(BaseModel):
    city: str
    units: str = "celsius"

@register_tool
class WeatherTool(UnifiedTool[WeatherParams]):
    name = "weather"
    description = "Get weather information"
    params = WeatherParams
    
    async def execute(self, params: WeatherParams) -> dict:
        # Implementation
        return {
            "city": params.city,
            "temperature": 22,
            "conditions": "sunny"
        }
    
    def result_to_nl(self, result: dict) -> str:
        """Custom result formatting."""
        return f"Weather in {result['city']}: {result['temperature']}°{result.get('units', 'C')}, {result['conditions']}"
```

## Runtime Mode Switching

```python
from kimi_cli.soul.unified import AgentMode

# Create soul in adaptive mode
soul = create_unified_soul(agent, context_file, mode=AgentMode.ADAPTIVE)

# Check current mode
print(soul.mode)  # AgentMode.ADAPTIVE

# Switch modes dynamically
soul.set_mode(AgentMode.PROACTIVE)
await soul.run("Complex task...")

soul.set_mode(AgentMode.REACTIVE)
await soul.run("Simple question...")
```

## Using Slash Commands

The unified soul adds new slash commands:

```
/mode reactive     - Switch to reactive mode
/mode proactive    - Switch to proactive mode
/mode adaptive     - Switch to adaptive mode
```

Usage in chat:
```
User: /mode proactive
User: Analyze all files in parallel
```

## Configuration Options

### Using Config Builder

```python
from kimi_cli.soul.unified.loader import UnifiedAgentConfig
from kimi_cli.soul.unified.agent import ExecutionMode

config = (
    UnifiedAgentConfig()
    .with_mode(ExecutionMode.PROACTIVE)
    .with_max_steps(100)           # For reactive mode
    .with_max_iterations(5)        # For proactive mode
    .with_parallel_limit(10)       # Max parallel executions
    .with_auto_summarize(True)     # Auto-generate summaries
    .with_tool(calculator)         # Add custom tool
    .with_tool(weather_tool)
)

agent = await config.load(agent_file, runtime)
```

### Using Agent YAML

```yaml
version: 1
agent:
  name: "my-agent"
  system_prompt_path: ./system.md
  tools:
    - "kimi_cli.tools.shell:Shell"
    - "my_tools.calculator:calculator"
  
  # Execution configuration
  execution:
    mode: proactive
    max_iterations: 3
    parallel_limit: 5
```

## Working with Task Graphs

### Manual Graph Creation

```python
from kimi_cli.soul.unified.graph_engine import TaskGraph, TaskNode

# Create graph
graph = TaskGraph()

# Add independent tasks
graph.add_node(TaskNode(
    id="task_a",
    description="Analyze file A",
    tool="ReadFile",
    args={"path": "file_a.py"},
    dependencies=[],
))

graph.add_node(TaskNode(
    id="task_b",
    description="Analyze file B",
    tool="ReadFile",
    args={"path": "file_b.py"},
    dependencies=[],
))

# Add dependent task
graph.add_node(TaskNode(
    id="summary",
    description="Summarize analyses",
    tool=None,  # Direct LLM call
    args={},
    dependencies=["task_a", "task_b"],
))

# Check status
print(f"Ready nodes: {[n.id for n in graph.get_ready_nodes()]}")
print(f"Is complete: {graph.is_complete()}")
```

### Automatic Graph Creation

In proactive mode, the graph is created automatically by the LLM:

```python
# The LLM will:
# 1. Analyze your task
# 2. Decompose into subtasks
# 3. Create dependency graph
# 4. Execute in parallel

await soul.run("""
    Review this codebase:
    1. Check code style in all Python files
    2. Find potential bugs
    3. Check test coverage
    4. Generate a comprehensive report
""")
```

## Debugging

### Enable Verbose Logging

```python
import logging
logging.getLogger("kimi_cli.soul.unified").setLevel(logging.DEBUG)
```

### Inspect Context

```python
# Get all fragments
for fragment in context.fragments:
    print(f"{fragment.role}: {fragment.content[:50]}...")

# Get tool calls
tool_fragments = context.get_fragments_by_type(FragmentType.TOOL_CALL)

# Get execution chain
chain = context.get_tool_call_chain("node_5")
```

### Check Graph Status

```python
# In proactive mode
if soul._graph_engine:
    graph = soul._graph_engine._current_graph
    print(f"Completed: {len(graph.completed)}/{len(graph.nodes)}")
    print(f"Results: {graph.results}")
```

## Common Patterns

### Pattern 1: Code Review

```python
# Proactive mode is perfect for parallel code review
soul.set_mode(AgentMode.PROACTIVE)

await soul.run("""
    Review all Python files in /src:
    - Check PEP 8 compliance
    - Find potential bugs
    - Check docstring coverage
    - Identify security issues
    Generate a detailed report with file-by-file analysis.
""")
```

### Pattern 2: Interactive Debugging

```python
# Use reactive mode for step-by-step debugging
soul.set_mode(AgentMode.REACTIVE)

await soul.run("Run the tests")
await soul.run("The test failed with X error, investigate")
await soul.run("Fix the issue in file Y")
await soul.run("Run tests again to verify")
```

### Pattern 3: Research Task

```python
# Adaptive mode for mixed complexity
soul.set_mode(AgentMode.ADAPTIVE)

# Simple lookup → reactive
await soul.run("What is the license of this project?")

# Complex analysis → proactive
await soul.run("""
    Research the dependencies:
    1. List all dependencies
    2. Check for known vulnerabilities
    3. Check for outdated packages
    4. Suggest alternatives
""")
```

## Troubleshooting

### Issue: Tool not found

**Solution:** Ensure tool is registered:
```python
from kimi_cli.soul.unified.tool import unified_tool_registry

# Check registration
print(unified_tool_registry.list_tools())

# Register if needed
unified_tool_registry.register(MyTool)
```

### Issue: Mode not switching

**Solution:** Check mode after switch:
```python
soul.set_mode(AgentMode.PROACTIVE)
assert soul.mode == AgentMode.PROACTIVE
```

### Issue: Graph execution stuck

**Solution:** Check for dependency cycles:
```python
# Graph should be a DAG (no cycles)
# Check that dependencies exist
for node in graph.nodes.values():
    for dep in node.dependencies:
        assert dep in graph.nodes, f"Missing dependency: {dep}"
```

## Next Steps

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design
2. Read [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) to port existing code
3. Check [example_usage.py](example_usage.py) for more examples
4. Explore the source code in the `unified/` directory
