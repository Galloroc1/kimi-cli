# Real LLM Testing Guide

## Overview

This guide explains how to test the unified agent architecture with a **real LLM**.

## Test Files

| File | Description | Requires LLM |
|------|-------------|--------------|
| `test_unified_real_llm.py` | Full integration tests with real LLM | ✅ Yes |
| `example_real_llm_usage.py` | Example usage with real LLM | ✅ Yes |
| `demo_simple.py` | Simple demo (no LLM required) | ❌ No |
| `demo_with_mock_llm.py` | Demo with simulated LLM | ❌ No |

## Prerequisites

### 1. Set Environment Variables

```bash
# Option 1: Kimi
export KIMI_BASE_URL="http://localhost:8000/v1"
export KIMI_API_KEY="your-api-key"  # Optional

# Option 2: OpenAI-compatible
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="your-api-key"  # Optional
```

### 2. Start LLM Server

Example using vLLM:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000
```

Or using llama.cpp:
```bash
./server -m model.gguf --port 8000
```

## Running Tests

### Without LLM (Default)

```bash
# Run all unified tests (mock-based)
uv run pytest tests/soul/unified/ -v

# Run simple demo
uv run python tests/soul/unified/demo_simple.py
```

### With Real LLM

```bash
# Set LLM endpoint
export KIMI_BASE_URL="http://localhost:8000/v1"

# Run real LLM tests
uv run pytest tests/soul/unified/test_unified_real_llm.py -v

# Run example usage
uv run python tests/soul/unified/example_real_llm_usage.py
```

## Test Coverage

### With Mock/No LLM (101 tests)

- ✅ Unit tests for all components
- ✅ Graph engine tests
- ✅ Tool system tests
- ✅ Context management tests
- ✅ Mode switching tests

### With Real LLM (Additional tests)

- ✅ LLM connection test
- ✅ Reactive mode with real LLM
- ✅ Proactive mode with real LLM
- ✅ Graph creation via LLM
- ✅ Adaptive mode analysis
- ✅ Mode switching with real LLM
- ✅ Performance comparison

## Example: Using Real LLM

```python
import asyncio
from pathlib import Path

from kimi_cli.config import load_config
from kimi_cli.auth.oauth import OAuthManager
from kimi_cli.session import Session
from kimi_cli.llm import create_llm
from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.toolset import KimiToolset

from kimi_cli.soul.unified import AgentMode
from kimi_cli.soul.unified.soul import UnifiedSoul
from kimi_cli.soul.unified.agent import UnifiedAgent, ExecutionMode, ExecutionConfig
from kimi_cli.soul.unified.context import UnifiedContext
from kimi_cli.soul.unified.loader import create_unified_soul

from kaos.path import KaosPath

async def main():
    # Setup
    config = load_config()
    oauth = OAuthManager(config)
    
    # Create LLM
    provider = config.llm.providers[0]  # Use first provider
    model = config.llm.models[0]  # Use first model
    llm = create_llm(provider, model, oauth=oauth)
    
    # Create session
    work_dir = KaosPath(".")
    session = await Session.create(work_dir)
    
    # Create runtime
    runtime = await Runtime.create(
        config=config,
        oauth=oauth,
        llm=llm,
        session=session,
        yolo=True,
    )
    
    # Create agent with tools
    toolset = KimiToolset()
    toolset.load_tools([
        "kimi_cli.tools.shell:Shell",
        "kimi_cli.tools.file:ReadFile",
    ], {})
    
    agent = UnifiedAgent(
        name="my_agent",
        system_prompt="You are a helpful assistant.",
        toolset=toolset,
        runtime=runtime,
        execution_config=ExecutionConfig(
            mode=ExecutionMode.ADAPTIVE,  # Auto-select mode
        ),
    )
    
    # Create soul
    soul = create_unified_soul(agent, session.context_file)
    
    # Run task
    await soul.run("Analyze the codebase and suggest improvements")

asyncio.run(main())
```

## Expected Output

### Reactive Mode
```
Agent: reactive_agent
Mode: REACTIVE
Model: Qwen2.5-7B-Instruct

Task: What is 2 + 2?
Execution: Single LLM call
Result: 2 + 2 = 4
```

### Proactive Mode
```
Agent: proactive_agent
Mode: PROACTIVE
Graph Engine: True

Task: Analyze Python files
Graph created with 3 nodes:
  - find_files
  - analyze
  - report

Execution:
  1. find_files (parallel)
  2. analyze (depends on find_files)
  3. report (depends on analyze)
```

### Adaptive Mode
```
Agent: adaptive_agent
Mode: ADAPTIVE

Task Analysis:
  'What is 2 + 2?' -> simple -> REACTIVE
  'Analyze all files' -> complex -> PROACTIVE
```

## Troubleshooting

### LLM Connection Failed
```
Error: Failed to create LLM instance
```
**Solution**: Check `KIMI_BASE_URL` or `OPENAI_BASE_URL` is set correctly.

### No Model Found
```
Error: No configured LLM provider found
```
**Solution**: Configure a provider in `~/.config/kimi-cli/config.toml`.

### Tests Skipped
```
SKIPPED: LLM not configured
```
**Solution**: Set environment variables before running tests.

## Performance Notes

| Mode | Simple Task | Complex Task |
|------|-------------|--------------|
| Reactive | ~1s | ~5s (sequential) |
| Proactive | ~2s (planning overhead) | ~2s (parallel) |
| Adaptive | ~1s (selects reactive) | ~2s (selects proactive) |

## Next Steps

1. ✅ Run mock-based tests: `uv run pytest tests/soul/unified/`
2. ⏭️ Set up LLM server
3. ⏭️ Run real LLM tests: `uv run pytest tests/soul/unified/test_unified_real_llm.py`
4. ⏭️ Integrate into your application
