# Unified Module Tests

This directory contains tests for the unified agent architecture.

## Test Files

| File | Description | Tests |
|------|-------------|-------|
| `test_types.py` | Tests for enums (AgentMode, ExecutionMode) | 4 |
| `test_graph_engine.py` | Tests for graph engine components | 32 |
| `test_context.py` | Tests for UnifiedContext | 18 |
| `test_tool.py` | Tests for UnifiedTool | 20 |
| `test_agent.py` | Tests for UnifiedAgent | 10 |
| `test_soul.py` | Tests for UnifiedSoul basics | 14 |

**Total: 78 tests**

## Running Tests

```bash
# Run all unified tests
cd /Users/ran/pyproject/kimi-cli
uv run pytest tests/soul/unified/ -v

# Run specific test file
uv run pytest tests/soul/unified/test_graph_engine.py -v

# Run with coverage
uv run pytest tests/soul/unified/ --cov=kimi_cli.soul.unified
```

## Test Coverage

### Types (`test_types.py`)
- AgentMode enum values
- ExecutionMode enum values
- Auto-incremented values

### Graph Engine (`test_graph_engine.py`)
- TaskNode creation and hashing
- TaskGraph operations (add, get, ready nodes, completion)
- GraphScheduler parallel execution
- Deadlock detection
- ExecutionPlan conversion

### Context (`test_context.py`)
- MemoryFragment creation and conversion
- FragmentType enum
- UnifiedContext persistence
- Fragment management
- Execution graph tracking

### Tool (`test_tool.py`)
- UnifiedTool execution
- Error handling
- Result conversion (dict, pydantic, string, None)
- @unified_tool decorator
- ToolRegistry operations

### Agent (`test_agent.py`)
- ExecutionConfig defaults
- UnifiedAgent creation
- Mode switching
- Tool management

### Soul (`test_soul.py`)
- TaskAnalysis
- Mode analysis heuristics
- AgentMode/ExecutionMode enums
- Basic agent and context functionality

## Key Test Patterns

### Async Tests
```python
@pytest.mark.asyncio
async def test_async_operation():
    result = await some_async_function()
    assert result == expected
```

### Fixtures
```python
@pytest.fixture
def mock_context(tmp_path):
    context_file = tmp_path / "context.jsonl"
    return UnifiedContext(file_backend=context_file)
```

### Mock Objects
```python
class MockRuntime:
    llm = None
    skills = {}
    # ... other attributes
```

## Notes

- Tests use `tmp_path` fixture for temporary files
- Mock objects avoid complex KimiSoul initialization
- Focus on testing unified module logic, not integration
