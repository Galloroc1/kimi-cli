# Migration Guide: Integrating Agent Framework into Kimi-CLI

## Phase 1: Preparation (No Code Changes)

### 1.1 Understand the Current State

**Kimi-CLI Architecture:**
```
KimiSoul (implements Soul Protocol)
├── Agent (configuration + toolset)
│   ├── Runtime (LLM, Session, Config)
│   └── Toolset (kosong CallableTool2 tools)
├── Context (file-backed message history)
└── kosong.step() (reactive execution)
```

**Agent Framework Architecture:**
```
GraphAgent (extends BaseFunctionCallingAgent)
├── RunnerEnv (LLM, tools, config)
├── Memory (structured fragments)
├── Tool system (Tool class + @tool decorator)
└── Graph scheduling (parallel execution)
```

### 1.2 Identify Integration Points

| Component | Kimi-CLI | Agent Framework | Unified Approach |
|-----------|----------|-----------------|------------------|
| LLM | `kosong.ChatProvider` | `OpenAiModel` | Use kosong, adapt interface |
| Tools | `CallableTool2` | `Tool` | UnifiedTool inherits CallableTool2 |
| Context | `Context` (file) | `Memory` (fragments) | UnifiedContext merges both |
| Execution | `kosong.step()` | Graph scheduling | UnifiedSoul dispatches to both |

## Phase 2: Core Integration (Week 1)

### Step 2.1: Copy Unified Module

Copy the `unified/` directory to `kimi-cli/src/kimi_cli/soul/`:

```bash
cp -r /path/to/unified /Users/ran/pyproject/kimi-cli/src/kimi_cli/soul/
```

### Step 2.2: Update Imports

In `kimi-cli/src/kimi_cli/soul/__init__.py`, add:

```python
# Add unified exports
from kimi_cli.soul.unified import (
    AgentMode,
    UnifiedSoul,
    UnifiedAgent,
    UnifiedContext,
)

__all__ = [
    # ... existing exports
    "AgentMode",
    "UnifiedSoul",
    "UnifiedAgent",
    "UnifiedContext",
]
```

### Step 2.3: Test Basic Integration

Create a test file:

```python
# tests/test_unified_integration.py
import pytest
from kimi_cli.soul.unified import AgentMode
from kimi_cli.soul.unified.context import UnifiedContext, MemoryFragment, FragmentType

@pytest.mark.asyncio
async def test_unified_context():
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp:
        context_file = Path(tmp) / "context.jsonl"
        context = UnifiedContext(context_file)
        
        # Test basic operations
        await context.append_fragment(MemoryFragment(
            role="user",
            content="Hello",
            fragment_type=FragmentType.USER,
        ))
        
        assert len(context.fragments) == 1
        assert context.fragments[0].content == "Hello"
```

## Phase 3: Gradual Migration (Week 2-3)

### Step 3.1: Create Hybrid Agent

Modify your existing agent loading to support both modes:

```python
# In your CLI or main entry point

async def create_agent(
    agent_file: Path,
    runtime: Runtime,
    use_unified: bool = False,  # Feature flag
):
    if use_unified:
        from kimi_cli.soul.unified.loader import load_unified_agent
        from kimi_cli.soul.unified.agent import ExecutionMode
        
        return await load_unified_agent(
            agent_file=agent_file,
            runtime=runtime,
            execution_mode=ExecutionMode.ADAPTIVE,  # Start with adaptive
        )
    else:
        from kimi_cli.soul.agent import load_agent
        return await load_agent(agent_file, runtime)
```

### Step 3.2: Port Your Tools

Convert your Agent Framework tools to UnifiedTools:

**Before (Agent Framework):**
```python
# In your ran/tools/tool.py
from ran.tools.tool import Tool
from pydantic import BaseModel

class SearchParams(BaseModel):
    query: str

class SearchTool(Tool):
    name = "search"
    des = "Search the codebase"
    input_params = SearchParams
    
    async def call(self, params: SearchParams):
        results = await search_code(params.query)
        return results
    
    def result_to_nl(self, result):
        return f"Found {len(result)} results"
```

**After (Unified):**
```python
# In kimi-cli
from kimi_cli.soul.unified.tool import UnifiedTool, register_tool
from pydantic import BaseModel

class SearchParams(BaseModel):
    query: str

@register_tool
class SearchTool(UnifiedTool[SearchParams]):
    name = "search"
    description = "Search the codebase"
    params = SearchParams
    
    async def execute(self, params: SearchParams):
        results = await search_code(params.query)
        return results
    
    def result_to_nl(self, result):
        return f"Found {len(result)} results"
```

### Step 3.3: Migrate GraphAgent Logic

Port your GraphAgent's core logic to GraphEngine:

**Before (Agent Framework):**
```python
# In your ran/agent/graph/agent.py
class GraphAgent(BasePSAFunctionCallingAgent):
    async def call(self, input_params):
        # Create plan
        self.task_graph = self.planer.call(input_params)
        
        # Execute with scheduler
        scheduler = TaskSchedule(self.task_graph, workers=self.tools_cls)
        async for chunk in scheduler.call():
            yield chunk
```

**After (Unified):**
```python
# The GraphEngine in unified/soul.py already implements this
# Just ensure your task graph format is compatible

# In your usage:
from kimi_cli.soul.unified.soul import UnifiedSoul
from kimi_cli.soul.unified import AgentMode

soul = UnifiedSoul(agent, context, mode=AgentMode.PROACTIVE)
await soul.run("Your task")  # Uses GraphEngine internally
```

## Phase 4: Full Migration (Week 4)

### Step 4.1: Remove Feature Flag

Once you're confident, make unified the default:

```python
async def create_agent(
    agent_file: Path,
    runtime: Runtime,
    use_legacy: bool = False,  # Inverted flag
):
    if use_legacy:
        # Keep old path for emergencies
        from kimi_cli.soul.agent import load_agent
        return await load_agent(agent_file, runtime)
    
    # Default to unified
    from kimi_cli.soul.unified.loader import load_unified_agent
    return await load_unified_agent(agent_file, runtime)
```

### Step 4.2: Update Agent Specs

Add execution configuration to your agent YAML files:

```yaml
# agents/default/agent.yaml
version: 1
agent:
  name: "default"
  system_prompt_path: ./system.md
  tools:
    - "kimi_cli.tools.shell:Shell"
    - "kimi_cli.tools.file:ReadFile"
    # Add your unified tools
    - "my_tools.search:SearchTool"
  
  # Optional: execution configuration
  execution:
    mode: adaptive  # reactive | proactive | adaptive
    max_iterations: 3
    parallel_limit: 5
```

### Step 4.3: Port Remaining Features

**Memory/Context:**
```python
# Before
from ran.schema.memory import Memory, RoleFragment
memory = Memory()
memory.add_memory(RoleFragment(role="user", content="Hello"))

# After
from kimi_cli.soul.unified.context import UnifiedContext, MemoryFragment, FragmentType
context = UnifiedContext(file_backend=path)
await context.append_fragment(MemoryFragment(
    role="user",
    content="Hello",
    fragment_type=FragmentType.USER,
))
```

**Tool Calls:**
```python
# Before
from ran.schema.memory import ToolFragment
fragment = ToolFragment(name="tool", arguments={"arg": "value"})

# After
from kimi_cli.soul.unified.context import MemoryFragment, FragmentType
fragment = MemoryFragment(
    role="assistant",
    content="Using tool...",
    fragment_type=FragmentType.TOOL_CALL,
    tool_name="tool",
    tool_args={"arg": "value"},
)
```

## Phase 5: Cleanup (Week 5)

### Step 5.1: Remove Legacy Code

Once everything works:
1. Remove the `use_legacy` flag
2. Delete old Agent Framework code (keep backup)
3. Update imports throughout codebase

### Step 5.2: Update Tests

```python
# Before
from ran.agent.graph.agent import GraphAgent

async def test_graph_agent():
    agent = GraphAgent()
    result = await agent.call(params)

# After
from kimi_cli.soul.unified.soul import UnifiedSoul
from kimi_cli.soul.unified import AgentMode

async def test_unified_soul():
    soul = UnifiedSoul(agent, context, mode=AgentMode.PROACTIVE)
    await soul.run("Task")
```

### Step 5.3: Documentation

Update your project documentation:
- README.md with new architecture
- API docs for UnifiedSoul
- Migration guide for other developers

## Common Issues and Solutions

### Issue 1: Import Cycles

**Problem:** Circular imports between old and new code.

**Solution:** Use lazy imports or move shared code to a common module.

```python
# Instead of top-level import
def get_tool():
    from kimi_cli.soul.unified.tool import UnifiedTool
    return UnifiedTool
```

### Issue 2: Different Message Formats

**Problem:** kosong.Message vs your Message format.

**Solution:** Use the conversion methods in UnifiedContext.

```python
# Convert kosong Message to Fragment
fragment = MemoryFragment.from_message(kosong_msg, FragmentType.USER)

# Convert Fragment back to Message
msg = fragment.to_message()
```

### Issue 3: Async Compatibility

**Problem:** Different async patterns.

**Solution:** UnifiedTool handles both sync and async execution.

```python
class MyTool(UnifiedTool[Params]):
    async def execute(self, params: Params):
        # Always async
        result = await some_async_operation()
        return result
```

### Issue 4: Tool Result Format

**Problem:** Different tool result expectations.

**Solution:** UnifiedTool normalizes results to ToolOk/ToolError.

```python
async def __call__(self, params: Params) -> ToolReturnValue:
    try:
        result = await self.execute(params)
        return ToolOk(output=self.result_to_nl(result))
    except Exception as e:
        return ToolError(message=str(e), brief="Failed")
```

## Testing Strategy

### Unit Tests

```python
@pytest.mark.asyncio
async def test_unified_tool():
    tool = SearchTool()
    result = await tool(SearchParams(query="test"))
    assert isinstance(result, ToolOk)
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_reactive_mode():
    soul = create_test_soul(mode=AgentMode.REACTIVE)
    await soul.run("Simple task")
    # Verify step-by-step execution

@pytest.mark.asyncio
async def test_proactive_mode():
    soul = create_test_soul(mode=AgentMode.PROACTIVE)
    await soul.run("Complex task")
    # Verify graph execution
```

### End-to-End Tests

```python
@pytest.mark.asyncio
async def test_full_workflow():
    # Load real agent
    agent = await load_unified_agent(...)
    soul = create_unified_soul(agent, context_file)
    
    # Run real task
    await soul.run("Refactor this codebase")
    
    # Verify results
    assert context.n_checkpoints > 0
```

## Rollback Plan

If issues arise:

1. **Immediate:** Set `use_unified=False` to use legacy code
2. **Short-term:** Fix issues in unified module
3. **Long-term:** Consider keeping both paths if needed

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| 1: Preparation | 2-3 days | Understanding, planning |
| 2: Core Integration | 1 week | Unified module, basic tests |
| 3: Gradual Migration | 2 weeks | Hybrid mode, tool porting |
| 4: Full Migration | 1 week | Remove feature flags |
| 5: Cleanup | 1 week | Remove legacy, docs |

**Total: 5-6 weeks**

## Success Metrics

- [ ] All existing tests pass
- [ ] New unified tests pass
- [ ] Performance comparable or better
- [ ] No regressions in user-facing features
- [ ] Documentation updated
