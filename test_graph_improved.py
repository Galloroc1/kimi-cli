#!/usr/bin/env python3
"""
改进后的 Graph 模式测试 - 验证新的提示词效果
"""

import asyncio
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from kaos.path import KaosPath
from kimi_cli.soul.unified.soul import UnifiedSoul
from kimi_cli.soul.unified.agent import UnifiedAgent, ExecutionConfig
from kimi_cli.soul.unified.context import UnifiedContext
from kimi_cli.soul.unified.types import AgentMode, ExecutionMode
from kimi_cli.soul import run_soul
from kimi_cli.soul.agent import Runtime
from kimi_cli.session import Session
from kimi_cli.config import load_config
from kimi_cli.auth.oauth import OAuthManager
from kimi_cli.llm import create_llm
from kimi_cli.wire import Wire
from kimi_cli.wire.types import TextPart, ToolCall, ToolResult, StepBegin
from kimi_cli.agentspec import DEFAULT_AGENT_FILE


async def test_improved_graph():
    """测试改进后的 Graph 生成"""
    
    print("=" * 70)
    print("改进后的 Graph 模式测试")
    print("=" * 70)
    
    # 创建 Agent
    config = load_config()
    session = await Session.create(KaosPath.cwd())
    
    model = config.models[config.default_model]
    provider = config.providers[model.provider]
    oauth = OAuthManager(config)
    llm = create_llm(provider, model, session_id=session.id, oauth=oauth)
    
    runtime = await Runtime.create(config, oauth, llm, session, yolo=True)
    
    execution_config = ExecutionConfig(
        mode=ExecutionMode.PROACTIVE,
        max_iterations=10,
        parallel_tool_calls_limit=5,
    )
    
    unified_agent = await UnifiedAgent.from_spec(
        DEFAULT_AGENT_FILE, runtime, execution_config, []
    )
    
    context = UnifiedContext(session.context_file)
    await context.restore()
    
    soul = UnifiedSoul(unified_agent, context, mode=AgentMode.PROACTIVE)
    
    print(f"\n✅ Agent 创建完成")
    print(f"   GraphEngine: {'已初始化' if soul._graph_engine else '未初始化'}")
    
    # 直接测试 GraphEngine
    if soul._graph_engine:
        engine = soul._graph_engine
        
        # 测试任务 - 明确需要并行处理
        task = "并行读取 README.md 和 pyproject.toml，然后总结项目"
        
        print(f"\n{'─' * 70}")
        print(f"📝 任务: {task}")
        print('─' * 70)
        
        print("\n⏳ 生成任务图...")
        graph = await engine.create_task_graph(task)
        
        print(f"\n✅ 任务图生成完成!")
        print(f"   节点数: {len(graph.nodes)}")
        
        # 打印图结构
        print(f"\n{'─' * 70}")
        print("📊 任务图结构")
        print('─' * 70)
        
        for node_id, node in graph.nodes.items():
            print(f"\n  📦 {node_id}")
            print(f"     描述: {node.description}")
            print(f"     工具: {node.tool or 'None (LLM)'}")
            print(f"     参数: {json.dumps(node.args, ensure_ascii=False)}")
            print(f"     依赖: {node.dependencies if node.dependencies else '无'}")
        
        # 分析可并行性
        ready = graph.get_ready_nodes()
        print(f"\n{'─' * 70}")
        print("🚀 可并行执行的节点")
        print('─' * 70)
        print(f"   数量: {len(ready)}")
        for node in ready:
            print(f"   - {node.id}: {node.description[:40]}...")
        
        # 执行
        print(f"\n{'─' * 70}")
        print("⏳ 执行任务图...")
        print('─' * 70)
        
        await engine.execute_graph(graph)
        
        print(f"\n✅ 执行完成!")
        print(f"   完成节点: {len(graph.completed)}/{len(graph.nodes)}")
        
        # 打印结果
        print(f"\n{'─' * 70}")
        print("📋 执行结果")
        print('─' * 70)
        for node_id, result in graph.results.items():
            preview = str(result)[:120].replace('\n', ' ')
            print(f"\n  {node_id}:")
            print(f"     {preview}...")


async def test_stream_proactive():
    """流式测试 PROACTIVE 模式"""
    
    print("\n" + "=" * 70)
    print("流式输出测试")
    print("=" * 70)
    
    # 创建 Agent
    config = load_config()
    session = await Session.create(KaosPath.cwd())
    model = config.models[config.default_model]
    provider = config.providers[model.provider]
    oauth = OAuthManager(config)
    llm = create_llm(provider, model, session_id=session.id, oauth=oauth)
    runtime = await Runtime.create(config, oauth, llm, session, yolo=True)
    
    execution_config = ExecutionConfig(mode=ExecutionMode.PROACTIVE)
    unified_agent = await UnifiedAgent.from_spec(
        DEFAULT_AGENT_FILE, runtime, execution_config, []
    )
    context = UnifiedContext(session.context_file)
    await context.restore()
    soul = UnifiedSoul(unified_agent, context, mode=AgentMode.PROACTIVE)
    
    prompt = "读取 README.md 和 pyproject.toml，总结项目"
    print(f"\n📝 提示: {prompt}\n")
    print("🤖 回复: ", end="", flush=True)
    
    cancel_event = asyncio.Event()
    text_queue = asyncio.Queue()
    
    async def stream_ui(wire: Wire):
        ui_side = wire.ui_side(merge=True)
        try:
            while True:
                msg = await ui_side.receive()
                if isinstance(msg, TextPart):
                    await text_queue.put(msg.text)
                elif isinstance(msg, StepBegin):
                    await text_queue.put(f"\n[步骤 {msg.n}] ")
                elif isinstance(msg, ToolCall):
                    await text_queue.put(f"\n[工具: {msg.name}] ")
        except:
            pass
        finally:
            await text_queue.put(None)
    
    soul_task = asyncio.create_task(
        run_soul(soul, prompt, stream_ui, cancel_event)
    )
    
    try:
        while True:
            text = await text_queue.get()
            if text is None:
                break
            print(text, end="", flush=True)
    finally:
        if not soul_task.done():
            cancel_event.set()
            try:
                await soul_task
            except:
                pass
    
    print("\n")


async def main():
    """主函数"""
    import sys
    
    test = sys.argv[1] if len(sys.argv) > 1 else "graph"
    
    match test:
        case "graph":
            await test_improved_graph()
        case "stream":
            await test_stream_proactive()
        case _:
            print(f"未知测试: {test}")
            print("可用: graph, stream")


if __name__ == "__main__":
    asyncio.run(main())
