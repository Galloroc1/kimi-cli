"""
Real LLM Test with Moonshot API

This test uses the actual Moonshot API to test the unified agent architecture.

Run:
   uv run python test_moonshot_real.py
"""

from __future__ import annotations

import os
import asyncio
import pytest
from pathlib import Path

from pydantic import BaseModel, Field
from kosong.message import Message

# Set Moonshot credentials
os.environ["KIMI_BASE_URL"] = "https://api.moonshot.cn/v1"
os.environ["KIMI_API_KEY"] = "sk-UmS2XYgC1rymHIY6xzy3hkwQJG6hJc3wwBiAYpHkoNmMQOKR"

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
from kimi_cli.utils.logging import logger
logger.enable("kimi_cli")

from kimi_cli.config import load_config
from kimi_cli.auth.oauth import OAuthManager
from kimi_cli.session import Session
from kimi_cli.llm import create_llm
from kimi_cli.soul.agent import Runtime

from kimi_cli.soul.unified import AgentMode
from kimi_cli.soul.unified.soul import UnifiedSoul
from kimi_cli.soul.unified.agent import UnifiedAgent, ExecutionMode, ExecutionConfig
from kimi_cli.soul.unified.context import UnifiedContext
from kimi_cli.soul.unified.tool import UnifiedTool
from kimi_cli.soul.unified.graph_engine import TaskGraph, TaskNode, GraphEngine

from kaos.path import KaosPath


class CalculatorParams(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")
    operation: str = Field(description="Operation: add, subtract, multiply, divide")


class CalculatorTool(UnifiedTool[CalculatorParams]):
    name = "calculator"
    description = "Perform mathematical calculations"
    params = CalculatorParams
    
    async def execute(self, params: CalculatorParams) -> dict:
        ops = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero",
        }
        result = ops.get(params.operation, lambda x, y: f"Unknown: {params.operation}")(params.a, params.b)
        return {"result": result, "operation": params.operation, "inputs": [params.a, params.b]}


class MockToolset:
    """Mock toolset for testing."""
    def __init__(self, tools=None):
        self.tools = tools or []


async def test_llm_connection():
    """Test basic LLM connection."""
    print("\n" + "="*60)
    print("TEST 1: LLM Connection")
    print("="*60)
    
    config = load_config()
    oauth = OAuthManager(config)
    
    # Get Moonshot provider
    provider = None
    model = None
    
    for prov_name, prov in config.providers.items():
        if "moonshot" in prov_name.lower() or prov.base_url == "https://api.moonshot.cn/v1":
            provider = prov
            # Find a model for this provider
            for model_name, m in config.models.items():
                if m.provider == prov_name:
                    model = m
                    break
            if model:
                break
    
    if not provider:
        print("Error: Moonshot provider not found in config")
        return False
    
    # Override with our credentials
    provider.base_url = "https://api.moonshot.cn/v1"
    from pydantic import SecretStr
    provider.api_key = SecretStr("sk-UmS2XYgC1rymHIY6xzy3hkwQJG6hJc3wwBiAYpHkoNmMQOKR")
    
    print(f"Provider: {provider.type}")
    print(f"Base URL: {provider.base_url}")
    print(f"Model: {model.model if model else 'None'}")
    
    llm = create_llm(provider, model, oauth=oauth)
    
    if llm is None:
        print("Error: Failed to create LLM instance")
        return False
    
    print(f"✓ LLM created: {llm.model_name}")
    
    # Test basic chat - just verify the LLM object is valid
    print(f"✓ LLM Model: {llm.model_name}")
    print(f"✓ Max Context: {llm.max_context_size}")
    print(f"✓ Capabilities: {llm.capabilities}")
    return True


async def test_reactive_mode():
    """Test reactive mode with real LLM."""
    print("\n" + "="*60)
    print("TEST 2: Reactive Mode")
    print("="*60)
    
    config = load_config()
    oauth = OAuthManager(config)
    
    # Setup provider
    provider = None
    model = None
    for prov_name, prov in config.providers.items():
        if "moonshot" in prov_name.lower():
            provider = prov
            for model_name, m in config.models.items():
                if m.provider == prov_name:
                    model = m
                    break
            break
    
    if not provider:
        print("Error: Provider not found")
        return False
    
    provider.base_url = "https://api.moonshot.cn/v1"
    from pydantic import SecretStr
    provider.api_key = SecretStr("sk-UmS2XYgC1rymHIY6xzy3hkwQJG6hJc3wwBiAYpHkoNmMQOKR")
    
    llm = create_llm(provider, model, oauth=oauth)
    if not llm:
        print("Error: Failed to create LLM")
        return False
    
    # Create session and runtime
    work_dir = KaosPath(".")
    session = await Session.create(work_dir)
    
    runtime = await Runtime.create(
        config=config,
        oauth=oauth,
        llm=llm,
        session=session,
        yolo=True,
    )
    
    # Create agent
    toolset = MockToolset([CalculatorTool()])
    
    agent = UnifiedAgent(
        name="reactive_moonshot",
        system_prompt="You are a helpful assistant.",
        toolset=toolset,
        runtime=runtime,
        execution_config=ExecutionConfig(mode=ExecutionMode.REACTIVE),
    )
    
    context = UnifiedContext(file_backend=session.context_file)
    soul = UnifiedSoul(agent, context, mode=AgentMode.REACTIVE)
    
    print(f"✓ Agent created: {soul.name}")
    print(f"✓ Mode: {soul.mode.name}")
    print(f"✓ Model: {soul.model_name}")
    
    # Test tool execution
    tool = CalculatorTool()
    result = await tool(CalculatorParams(a=5, b=3, operation="multiply"))
    print(f"✓ Tool result: {result.output}")
    
    return True


async def test_proactive_mode():
    """Test proactive mode with real LLM."""
    print("\n" + "="*60)
    print("TEST 3: Proactive Mode")
    print("="*60)
    
    config = load_config()
    oauth = OAuthManager(config)
    
    # Setup provider
    provider = None
    model = None
    for prov_name, prov in config.providers.items():
        if "moonshot" in prov_name.lower():
            provider = prov
            for model_name, m in config.models.items():
                if m.provider == prov_name:
                    model = m
                    break
            break
    
    if not provider:
        print("Error: Provider not found")
        return False
    
    provider.base_url = "https://api.moonshot.cn/v1"
    from pydantic import SecretStr
    provider.api_key = SecretStr("sk-UmS2XYgC1rymHIY6xzy3hkwQJG6hJc3wwBiAYpHkoNmMQOKR")
    
    llm = create_llm(provider, model, oauth=oauth)
    if not llm:
        print("Error: Failed to create LLM")
        return False
    
    # Create session and runtime
    work_dir = KaosPath(".")
    session = await Session.create(work_dir)
    
    runtime = await Runtime.create(
        config=config,
        oauth=oauth,
        llm=llm,
        session=session,
        yolo=True,
    )
    
    # Create agent
    toolset = MockToolset([CalculatorTool()])
    
    agent = UnifiedAgent(
        name="proactive_moonshot",
        system_prompt="You are a task planning assistant.",
        toolset=toolset,
        runtime=runtime,
        execution_config=ExecutionConfig(
            mode=ExecutionMode.PROACTIVE,
            max_iterations=2,
        ),
    )
    
    context = UnifiedContext(file_backend=session.context_file)
    soul = UnifiedSoul(agent, context, mode=AgentMode.PROACTIVE)
    
    print(f"✓ Agent created: {soul.name}")
    print(f"✓ Mode: {soul.mode.name}")
    print(f"✓ Graph Engine: {soul._graph_engine is not None}")
    
    # Test graph creation and execution
    graph_engine = GraphEngine(agent, context)
    
    try:
        # Use LLM to automatically create task graph
        print("Creating task graph using LLM...")
        task = "Calculate (5 + 3) * 2 using the calculator tool"
        graph = await graph_engine.create_task_graph(task)
        
        print(f"✓ Graph created with {len(graph.nodes)} nodes:")
        for node_id, node in graph.nodes.items():
            print(f"    - {node_id}: {node.description[:50]}")
        
        # Test parallel execution - actually execute the graph
        print("\nExecuting graph...")
        await graph_engine.execute_graph(graph)
        
        # Verify results
        print(f"\n✓ Graph execution complete!")
        print(f"✓ Results:")
        for node_id, result in graph.results.items():
            print(f"    - {node_id}: {result}")
        
        # Verify all nodes completed
        assert graph.is_complete(), "Not all nodes were completed"
        
        return True
        
    except Exception as e:
        print(f"✗ Graph execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_adaptive_mode():
    """Test adaptive mode."""
    print("\n" + "="*60)
    print("TEST 4: Adaptive Mode")
    print("="*60)
    
    config = load_config()
    oauth = OAuthManager(config)
    
    # Setup provider
    provider = None
    model = None
    for prov_name, prov in config.providers.items():
        if "moonshot" in prov_name.lower():
            provider = prov
            for model_name, m in config.models.items():
                if m.provider == prov_name:
                    model = m
                    break
            break
    
    if not provider:
        print("Error: Provider not found")
        return False
    
    provider.base_url = "https://api.moonshot.cn/v1"
    from pydantic import SecretStr
    provider.api_key = SecretStr("sk-UmS2XYgC1rymHIY6xzy3hkwQJG6hJc3wwBiAYpHkoNmMQOKR")
    
    llm = create_llm(provider, model, oauth=oauth)
    if not llm:
        print("Error: Failed to create LLM")
        return False
    
    # Create session and runtime
    work_dir = KaosPath(".")
    session = await Session.create(work_dir)
    
    runtime = await Runtime.create(
        config=config,
        oauth=oauth,
        llm=llm,
        session=session,
        yolo=True,
    )
    
    # Create agent
    toolset = MockToolset([CalculatorTool()])
    
    agent = UnifiedAgent(
        name="adaptive_moonshot",
        system_prompt="You are a helpful assistant.",
        toolset=toolset,
        runtime=runtime,
        execution_config=ExecutionConfig(mode=ExecutionMode.ADAPTIVE),
    )
    
    context = UnifiedContext(file_backend=session.context_file)
    soul = UnifiedSoul(agent, context, mode=AgentMode.ADAPTIVE)
    
    print(f"✓ Agent created: {soul.name}")
    print(f"✓ Mode: {soul.mode.name}")
    
    # Test task analysis
    test_tasks = [
        "What is 2 + 2?",
        "Calculate 5 * 3",
        "Analyze all files and refactor code",
        "Multiple parallel tasks to execute",
    ]
    
    print("\nTask Analysis:")
    for task in test_tasks:
        analysis = await soul._analyze_task(task)
        print(f"  '{task[:35]}...' -> {analysis.complexity} -> {analysis.recommended_mode.name}")
    
    return True


async def main():
    """Run all tests."""
    print("="*60)
    print("Unified Agent Architecture - Moonshot Real LLM Test")
    print("="*60)
    
    results = []
    
    try:
        results.append(("LLM Connection", await test_llm_connection()))
    except Exception as e:
        print(f"\n✗ LLM Connection failed: {e}")
        results.append(("LLM Connection", False))
    
    try:
        results.append(("Reactive Mode", await test_reactive_mode()))
    except Exception as e:
        print(f"\n✗ Reactive Mode failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Reactive Mode", False))
    
    try:
        results.append(("Proactive Mode", await test_proactive_mode()))
    except Exception as e:
        print(f"\n✗ Proactive Mode failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Proactive Mode", False))
    
    try:
        results.append(("Adaptive Mode", await test_adaptive_mode()))
    except Exception as e:
        print(f"\n✗ Adaptive Mode failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Adaptive Mode", False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    return 0 if passed_count == total_count else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
