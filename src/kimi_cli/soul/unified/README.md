# Unified Agent Architecture

This module provides a unified architecture that merges Kimi-CLI's reactive agent system with the Agent Framework's proactive graph-based execution.

## 📁 File Structure

```
unified/
├── __init__.py           # Public API exports
├── agent.py              # UnifiedAgent - merged Agent + BaseAgent
├── context.py            # UnifiedContext - merged Context + Memory
├── tool.py               # UnifiedTool - merged CallableTool2 + Tool
├── soul.py               # UnifiedSoul - merged KimiSoul + GraphAgent
├── graph_engine.py       # Task graph representation and execution
├── loader.py             # Loading utilities and configuration
├── example_usage.py      # Usage examples
├── ARCHITECTURE.md       # Detailed architecture documentation
├── MIGRATION_GUIDE.md    # Step-by-step migration guide
├── QUICKSTART.md         # Quick start guide
└── README.md             # This file
```

## 🚀 Quick Start

```python
from kimi_cli.soul.unified.loader import load_unified_agent, create_unified_soul
from kimi_cli.soul.unified.agent import ExecutionMode

# Load agent with proactive mode
agent = await load_unified_agent(
    agent_file=Path("agents/default/agent.yaml"),
    config=config,
    oauth=oauth,
    llm=None,
    session=session,
    execution_mode=ExecutionMode.PROACTIVE,
)

# Create soul
soul = create_unified_soul(agent, session.context_file)

# Run with graph-based execution
await soul.run("Analyze all Python files in parallel")
```

## 🏗️ Architecture

### Three Execution Modes

1. **Reactive Mode** (`AgentMode.REACTIVE`)
   - Step-by-step execution (original KimiSoul)
   - Good for simple, linear tasks
   - Lower latency

2. **Proactive Mode** (`AgentMode.PROACTIVE`)
   - Graph-based parallel execution (GraphAgent)
   - Task decomposition and scheduling
   - Better for complex, parallelizable tasks

3. **Adaptive Mode** (`AgentMode.ADAPTIVE`)
   - Automatically chooses mode based on task complexity
   - Best of both worlds

### Key Components

| Component | Purpose | Merges |
|-----------|---------|--------|
| `UnifiedSoul` | Main entry point | KimiSoul + GraphAgent |
| `UnifiedAgent` | Agent configuration | Agent + BaseAgent |
| `UnifiedContext` | State management | Context + Memory |
| `UnifiedTool` | Tool interface | CallableTool2 + Tool |
| `GraphEngine` | Parallel execution | Graph scheduling logic |

## 📖 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed design and internals
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Migrate existing code
- **[example_usage.py](example_usage.py)** - Code examples

## 🔧 Integration Status

### ✅ Completed

- [x] Core unified abstractions (Soul, Agent, Context, Tool)
- [x] Three execution modes (Reactive, Proactive, Adaptive)
- [x] Graph-based task scheduling
- [x] Tool result conversion
- [x] Configuration system
- [x] Documentation

### 🚧 Next Steps

1. **Testing**
   - [ ] Unit tests for each component
   - [ ] Integration tests with real LLM
   - [ ] Performance benchmarks

2. **Enhancements**
   - [ ] LLM-based task complexity analysis
   - [ ] Hierarchical task graphs
   - [ ] Dynamic replanning
   - [ ] Tool result caching

3. **Integration**
   - [ ] Wire protocol extensions for graph visualization
   - [ ] UI updates for proactive mode feedback
   - [ ] Agent YAML schema extension

## 🔄 Migration from Agent Framework

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed steps.

Quick overview:
1. Copy `unified/` module to `kimi-cli/src/kimi_cli/soul/`
2. Update imports in your code
3. Convert tools to `UnifiedTool`
4. Switch from `GraphAgent` to `UnifiedSoul` with proactive mode
5. Test and iterate

## 📝 Example: Converting a Tool

**Before (Agent Framework):**
```python
from ran.tools.tool import Tool

class MyTool(Tool):
    name = "my_tool"
    des = "Does something"
    input_params = MyParams
    
    async def call(self, params):
        return result
```

**After (Unified):**
```python
from kimi_cli.soul.unified.tool import UnifiedTool, register_tool

@register_tool
class MyTool(UnifiedTool[MyParams]):
    name = "my_tool"
    description = "Does something"
    params = MyParams
    
    async def execute(self, params):
        return result
```

## 🤝 Contributing

When modifying this module:

1. Maintain backward compatibility with Kimi-CLI
2. Follow existing code style
3. Add tests for new features
4. Update documentation

## 📄 License

Same as Kimi-CLI project.
