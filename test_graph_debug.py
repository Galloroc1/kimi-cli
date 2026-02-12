#!/usr/bin/env python3
"""
Graph 模式调试 - 展示任务图生成和执行过程
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


async def test_graph_generation():
    """测试 Graph 生成过程"""
    
    print("=" * 70)
    print("Graph 模式 - 任务规划与执行")
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
        agent_file=DEFAULT_AGENT_FILE,
        runtime=runtime,
        execution_config=execution_config,
        mcp_configs=[],
    )
    
    context = UnifiedContext(session.context_file)
    await context.restore()
    
    soul = UnifiedSoul(unified_agent, context, mode=AgentMode.PROACTIVE)
    
    print(f"\n✅ Agent 创建完成")
    print(f"   模式: PROACTIVE")
    print(f"   GraphEngine: {'已初始化' if soul._graph_engine else '未初始化'}")
    
    # 直接测试 GraphEngine
    if soul._graph_engine:
        engine = soul._graph_engine
        
        # 测试任务
        task = """分析项目结构：
1. 读取 README.md
2. 读取 pyproject.toml  
3. 列出 src/kimi_cli 目录
4. 总结项目"""
        
        print(f"\n{'─' * 70}")
        print("📝 任务分解")
        print('─' * 70)
        print(task)
        
        print(f"\n{'─' * 70}")
        print("⏳ 生成任务图 (TaskGraph)...")
        print('─' * 70)
        
        try:
            graph = await engine.create_task_graph(task)
            
            print(f"\n✅ 任务图生成成功!")
            print(f"   节点总数: {len(graph.nodes)}")
            print(f"   已完成: {len(graph.completed)}")
            
            # 打印节点结构
            print(f"\n{'─' * 70}")
            print("📊 任务图结构")
            print('─' * 70)
            
            for node_id, node in graph.nodes.items():
                print(f"\n  📦 节点: {node_id}")
                print(f"     描述: {node.description}")
                print(f"     工具: {node.tool or 'None (LLM直接回答)'}")
                print(f"     参数: {json.dumps(node.args, ensure_ascii=False)}")
                print(f"     依赖: {node.dependencies if node.dependencies else '无'}")
            
            # 可并行节点
            ready = graph.get_ready_nodes()
            print(f"\n{'─' * 70}")
            print("🚀 可并行执行的节点")
            print('─' * 70)
            for node in ready:
                print(f"   - {node.id}: {node.description[:50]}...")
            
            # 执行图
            print(f"\n{'─' * 70}")
            print("⏳ 执行任务图...")
            print('─' * 70)
            
            await engine.execute_graph(graph)
            
            print(f"\n✅ 执行完成!")
            print(f"   完成节点数: {len(graph.completed)}")
            
            # 打印结果
            print(f"\n{'─' * 70}")
            print("📋 执行结果")
            print('─' * 70)
            for node_id, result in graph.results.items():
                preview = str(result)[:150].replace('\n', ' ')
                print(f"\n  {node_id}:")
                print(f"     {preview}...")
                
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 流式运行完整对话
    print(f"\n{'=' * 70}")
    print("💬 流式对话测试")
    print('=' * 70)
    
    prompt = "并行读取 README.md 和 pyproject.toml，然后总结项目"
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
                    await text_queue.put(f"\n[步骤 {msg.n}]\n")
                elif isinstance(msg, ToolCall):
                    await text_queue.put(f"\n[工具: {msg.name}]\n")
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


async def compare_reactive_proactive():
    """对比 REACTIVE 和 PROACTIVE 模式"""
    
    print("\n" + "=" * 70)
    print("REACTIVE vs PROACTIVE 对比")
    print("=" * 70)
    
    prompt = "读取 README.md 和 pyproject.toml，总结项目"
    
    for mode_name, mode in [("REACTIVE", ExecutionMode.REACTIVE), ("PROACTIVE", ExecutionMode.PROACTIVE)]:
        print(f"\n{'─' * 70}")
        print(f"【{mode_name} 模式】")
        print('─' * 70)
        
        # 创建 Agent
        config = load_config()
        session = await Session.create(KaosPath.cwd())
        model = config.models[config.default_model]
        provider = config.providers[model.provider]
        oauth = OAuthManager(config)
        llm = create_llm(provider, model, session_id=session.id, oauth=oauth)
        runtime = await Runtime.create(config, oauth, llm, session, yolo=True)
        
        execution_config = ExecutionConfig(mode=mode)
        unified_agent = await UnifiedAgent.from_spec(
            DEFAULT_AGENT_FILE, runtime, execution_config, []
        )
        context = UnifiedContext(session.context_file)
        await context.restore()
        
        agent_mode = AgentMode.REACTIVE if mode == ExecutionMode.REACTIVE else AgentMode.PROACTIVE
        soul = UnifiedSoul(unified_agent, context, mode=agent_mode)
        
        # 运行
        cancel_event = asyncio.Event()
        texts = []
        steps = 0
        tools = 0
        
        async def collect(wire: Wire):
            ui_side = wire.ui_side(merge=True)
            try:
                while True:
                    msg = await ui_side.receive()
                    if isinstance(msg, TextPart):
                        texts.append(msg.text)
                    elif isinstance(msg, StepBegin):
                        steps += 1
                    elif isinstance(msg, ToolCall):
                        tools += 1
            except:
                pass
        
        await run_soul(soul, prompt, collect, cancel_event)
        
        print(f"   步骤数: {steps}")
        print(f"   工具调用: {tools}")
        print(f"   回复长度: {len(''.join(texts))}")
        print(f"   回复预览: {''.join(texts)[:200]}...")


async def main():
    """主函数"""
    import sys
    
    test = sys.argv[1] if len(sys.argv) > 1 else "graph"
    
    match test:
        case "graph":
            await test_graph_generation()
        case "compare":
            await compare_reactive_proactive()
        case _:
            print(f"未知测试: {test}")
            print("可用: graph, compare")


if __name__ == "__main__":
    asyncio.run(main())
