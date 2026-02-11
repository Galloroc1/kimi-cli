"""
Example: Using Unified Agent Architecture with REAL LLM

This script demonstrates how to use the unified architecture with a real LLM.

Prerequisites:
1. Set environment variable:
   export KIMI_BASE_URL="http://localhost:8000/v1"
   # or
   export OPENAI_BASE_URL="http://localhost:8000/v1"

2. Or modify the code below with your LLM endpoint

Run:
   uv run python example_real_llm_usage.py
"""

from __future__ import annotations

import os
import asyncio
from pathlib import Path

from kaos.path import KaosPath

from kimi_cli.config import load_config
from kimi_cli.auth.oauth import OAuthManager
from kimi_cli.session import Session
from kimi_cli.llm import create_llm
from kimi_cli.soul.toolset import KimiToolset

from kimi_cli.soul.unified import AgentMode
from kimi_cli.soul.unified.soul import UnifiedSoul
from kimi_cli.soul.unified.agent import UnifiedAgent, ExecutionMode, ExecutionConfig
from kimi_cli.soul.unified.context import UnifiedContext
from kimi_cli.soul.unified.loader import load_unified_agent, create_unified_soul


async def setup_llm():
    """Setup LLM connection."""
    config = load_config()
    oauth = OAuthManager(config)
    
    # Try to find a configured provider
    provider = None
    model = None
    
    for prov in config.providers:
        if prov.base_url or os.getenv("KIMI_BASE_URL") or os.getenv("OPENAI_BASE_URL"):
            provider = prov
            # Find a model for this provider
            for m in config.models:
                if m.provider == prov.name:
                    model = m
                    break
            if model:
                break
    
    if not provider:
        print("Error: No LLM provider configured.")
        print("Please set KIMI_BASE_URL or OPENAI_BASE_URL environment variable.")
        print("Or configure a provider in ~/.config/kimi-cli/config.toml")
        return None
    
    # Override with environment variable if set
    if os.getenv("KIMI_BASE_URL"):
        provider.base_url = os.getenv("KIMI_BASE_URL")
    if os.getenv("OPENAI_BASE_URL"):
        provider.base_url = os.getenv("OPENAI_BASE_URL")
    
    llm = create_llm(provider, model, oauth=oauth)
    
    if llm is None:
        print("Error: Failed to create LLM instance.")
        return None
    
    print(f"✓ LLM connected: {llm.model_name}")
    return llm


async def test_reactive_mode(llm):
    """Test reactive mode with real LLM."""
    print("\n" + "="*60)
    print("TEST 1: Reactive Mode (Step-by-step)")
    print("="*60)
    
    config = load_config()
    oauth = OAuthManager(config)
    
    # Create session
    work_dir = KaosPath(".")
    session = await Session.create(work_dir)
    
    # Create runtime
    runtime = await Runtime.create(
        config=config,
        oauth=oauth,
        llm=llm,
        session=session,
        yolo=True,  # Auto-approve for testing
    )
    
    # Create toolset with real tools
    toolset = KimiToolset()
    toolset.load_tools([
        "kimi_cli.tools.shell:Shell",
        "kimi_cli.tools.file:ReadFile",
        "kimi_cli.tools.file:Glob",
    ])
    
    # Create unified agent
    agent = UnifiedAgent(
        name="reactive_agent",
        system_prompt="You are a helpful coding assistant. Use tools when needed.",
        toolset=toolset,
        runtime=runtime,
        execution_config=ExecutionConfig(
            mode=ExecutionMode.REACTIVE,
            max_steps=10,
        ),
    )
    
    # Create soul
    context = UnifiedContext(file_backend=session.context_file)
    soul = UnifiedSoul(agent, context, mode=AgentMode.REACTIVE)
    
    print(f"Agent: {soul.name}")
    print(f"Mode: {soul.mode.name}")
    print(f"Model: {soul.model_name}")
    print(f"Tools: {[t.name for t in agent.toolset.tools]}")
    
    print("\n✓ Reactive mode setup complete")
    return soul


async def test_proactive_mode(llm):
    """Test proactive mode with real LLM."""
    print("\n" + "="*60)
    print("TEST 2: Proactive Mode (Graph-based)")
    print("="*60)
    
    config = load_config()
    oauth = OAuthManager()
    
    work_dir = KaosPath(".")
    session = await Session.create(work_dir)
    
    runtime = await Runtime.create(
        config=config,
        oauth=oauth,
        llm=llm,
        session=session,
        yolo=True,
    )
    
    toolset = KimiToolset()
    toolset.load_tools([
        "kimi_cli.tools.shell:Shell",
        "kimi_cli.tools.file:ReadFile",
        "kimi_cli.tools.file:Glob",
    ])
    
    agent = UnifiedAgent(
        name="proactive_agent",
        system_prompt="You are a helpful assistant that can break down complex tasks.",
        toolset=toolset,
        runtime=runtime,
        execution_config=ExecutionConfig(
            mode=ExecutionMode.PROACTIVE,
            max_iterations=3,
            parallel_tool_calls_limit=5,
        ),
    )
    
    context = UnifiedContext(file_backend=session.context_file)
    soul = UnifiedSoul(agent, context, mode=AgentMode.PROACTIVE)
    
    print(f"Agent: {soul.name}")
    print(f"Mode: {soul.mode.name}")
    print(f"Model: {soul.model_name}")
    print(f"Graph Engine: {soul._graph_engine is not None}")
    
    print("\n✓ Proactive mode setup complete")
    return soul


async def test_adaptive_mode(llm):
    """Test adaptive mode with real LLM."""
    print("\n" + "="*60)
    print("TEST 3: Adaptive Mode (Auto-select)")
    print("="*60)
    
    config = load_config()
    oauth = OAuthManager()
    
    work_dir = KaosPath(".")
    session = await Session.create(work_dir)
    
    runtime = await Runtime.create(
        config=config,
        oauth=oauth,
        llm=llm,
        session=session,
        yolo=True,
    )
    
    toolset = KimiToolset()
    toolset.load_tools([
        "kimi_cli.tools.shell:Shell",
        "kimi_cli.tools.file:ReadFile",
    ])
    
    agent = UnifiedAgent(
        name="adaptive_agent",
        system_prompt="You are a helpful assistant.",
        toolset=toolset,
        runtime=runtime,
        execution_config=ExecutionConfig(mode=ExecutionMode.ADAPTIVE),
    )
    
    context = UnifiedContext(file_backend=session.context_file)
    soul = UnifiedSoul(agent, context, mode=AgentMode.ADAPTIVE)
    
    print(f"Agent: {soul.name}")
    print(f"Mode: {soul.mode.name}")
    print(f"Model: {soul.model_name}")
    
    # Test task analysis
    print("\nTask Analysis:")
    
    test_tasks = [
        "What is 2 + 2?",
        "Read the README.md file",
        "Analyze all Python files and suggest improvements",
        "List files and check for errors",
    ]
    
    for task in test_tasks:
        analysis = await soul._analyze_task(task)
        print(f"  '{task[:40]}...' -> {analysis.complexity} -> {analysis.recommended_mode.name}")
    
    print("\n✓ Adaptive mode setup complete")
    return soul


async def test_mode_switching(llm):
    """Test mode switching at runtime."""
    print("\n" + "="*60)
    print("TEST 4: Runtime Mode Switching")
    print("="*60)
    
    config = load_config()
    oauth = OAuthManager()
    
    work_dir = KaosPath(".")
    session = await Session.create(work_dir)
    
    runtime = await Runtime.create(
        config=config,
        oauth=oauth,
        llm=llm,
        session=session,
        yolo=True,
    )
    
    toolset = KimiToolset()
    toolset.load_tools(["kimi_cli.tools.shell:Shell"])
    
    agent = UnifiedAgent(
        name="switchable_agent",
        system_prompt="You are a helpful assistant.",
        toolset=toolset,
        runtime=runtime,
        execution_config=ExecutionConfig(mode=ExecutionMode.REACTIVE),
    )
    
    context = UnifiedContext(file_backend=session.context_file)
    soul = UnifiedSoul(agent, context, mode=AgentMode.REACTIVE)
    
    print(f"Initial mode: {soul.mode.name}")
    
    # Switch modes
    soul.set_mode(AgentMode.PROACTIVE)
    print(f"After switch: {soul.mode.name}")
    
    soul.set_mode(AgentMode.ADAPTIVE)
    print(f"After switch: {soul.mode.name}")
    
    soul.set_mode(AgentMode.REACTIVE)
    print(f"After switch: {soul.mode.name}")
    
    print("\n✓ Mode switching works")


async def main():
    """Main test runner."""
    print("="*60)
    print("Unified Agent Architecture - Real LLM Test")
    print("="*60)
    
    # Setup LLM
    llm = await setup_llm()
    if not llm:
        print("\nFailed to setup LLM. Exiting.")
        return 1
    
    try:
        # Run tests
        await test_reactive_mode(llm)
        await test_proactive_mode(llm)
        await test_adaptive_mode(llm)
        await test_mode_switching(llm)
        
        print("\n" + "="*60)
        print("All tests completed successfully!")
        print("="*60)
        print("\nYou can now use the unified architecture:")
        print("  soul = UnifiedSoul(agent, context, mode=AgentMode.PROACTIVE)")
        print("  await soul.run('Your task here')")
        
        return 0
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
