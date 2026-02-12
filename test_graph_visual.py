#!/usr/bin/env python3
"""
Graph 可视化测试 - 展示 PROACTIVE 模式的内部工作原理
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

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
from kimi_cli.wire.types import TextPart, ToolCall, ToolResult, StepBegin, TurnBegin
from kimi_cli.agentspec import DEFAULT_AGENT_FILE


def print_section(title: str, content: str = ""):
    """打印章节"""
    print(f"\n{'='*70}")
    print(f"🎯 {title}")
    print('='*70)
    if content:
        print(content)


def print_subsection(title: str):
    """打印子章节"""
    print(f"\n{'─'*70}")
    print(f"📌 {title}")
    print('─'*70)


class GraphVisualizer:
    """Graph 执行可视化器"""
    
    def __init__(self, soul: UnifiedSoul):
        self.soul = soul
        self.events = []
        self.start_time = None
    
    def log_event(self, phase: str, detail: str, data: dict = None):
        """记录事件"""
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        event = {
            "time": f"{elapsed:.2f}s",
            "phase": phase,
            "detail": detail,
            "data": data,
        }
        self.events.append(event)
        
        # 实时打印
        data_str = f" {json.dumps(data, ensure_ascii=False)}" if data else ""
        print(f"[{event['time']}] [{phase:12}] {detail}{data_str}")
    
    async def run_with_visualization(self, prompt: str):
        """带可视化的运行"""
        self.start_time = datetime.now()
        cancel_event = asyncio.Event()
        message_queue = asyncio.Queue()
        
        print_section("开始执行", f"提示: {prompt[:80]}...")
        self.log_event("INPUT", "用户输入", {"prompt": prompt[:100]})
        
        async def visual_ui_loop(wire: Wire):
            """可视化 UI 循环"""
            ui_side = wire.ui_side(merge=False)
            self.log_event("SETUP", "UI 循环启动")
            
            try:
                while True:
                    msg = await ui_side.receive()
                    
                    match msg:
                        case TurnBegin():
                            self.log_event("TURN", "新回合开始")
                            await message_queue.put(("turn", msg))
                            
                        case StepBegin(n=n):
                            self.log_event("STEP", f"步骤 {n} 开始", {"step": n})
                            await message_queue.put(("step", msg))
                            
                        case TextPart(text=text):
                            preview = text[:40].replace('\n', ' ')
                            self.log_event("TEXT", f"文本: {preview}...", {"len": len(text)})
                            await message_queue.put(("text", msg))
                            
                        case ToolCall(name=name, arguments=args):
                            self.log_event("TOOL_CALL", f"调用 {name}", {"args": str(args)[:60]})
                            await message_queue.put(("tool_call", msg))
                            
                        case ToolResult(tool_call_id=tcid, return_value=rv):
                            result = str(rv)[:50]
                            self.log_event("TOOL_RESULT", f"结果 {tcid}", {"result": result})
                            await message_queue.put(("tool_result", msg))
                            
                        case _:
                            self.log_event("OTHER", f"{type(msg).__name__}")
                            await message_queue.put(("other", msg))
                            
            except Exception as e:
                self.log_event("ERROR", f"UI 异常: {e}")
            finally:
                self.log_event("END", "UI 循环结束")
                await message_queue.put(("end", None))
        
        # 启动 Soul
        self.log_event("EXEC", "启动 Soul")
        soul_task = asyncio.create_task(
            run_soul(self.soul, prompt, visual_ui_loop, cancel_event)
        )
        
        # 收集输出
        full_text = []
        try:
            while True:
                msg_type, msg = await message_queue.get()
                if msg_type == "end":
                    break
                if msg_type == "text":
                    full_text.append(msg.text)
                yield msg_type, msg
        finally:
            if not soul_task.done():
                cancel_event.set()
                try:
                    await soul_task
                except:
                    pass
            self.log_event("COMPLETE", "运行完成")
        
        # 打印摘要
        print_section("执行摘要")
        print(f"总事件数: {len(self.events)}")
        print(f"总耗时: {(datetime.now() - self.start_time).total_seconds():.2f}s")
        print(f"生成文本: {len(''.join(full_text))} 字符")
        
        # 统计
        counts = {}
        for e in self.events:
            counts[e['phase']] = counts.get(e['phase'], 0) + 1
        print(f"\n事件统计:")
        for phase, count in sorted(counts.items()):
            print(f"   {phase:12}: {count}")


async def test_proactive_visual():
    """可视化测试 PROACTIVE 模式"""
    
    print_section("PROACTIVE 模式可视化测试")
    
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
    
    print(f"✅ Agent 创建完成")
    print(f"   GraphEngine: {'已初始化' if soul._graph_engine else '未初始化'}")
    
    # 创建可视化器并运行
    visualizer = GraphVisualizer(soul)
    
    prompt = "读取 README.md 和 pyproject.toml，总结项目信息"
    print(f"\n📝 提示: {prompt}\n")
    
    print("🤖 流式输出:\n")
    async for msg_type, msg in visualizer.run_with_visualization(prompt):
        if msg_type == "text":
            print(msg.text, end="", flush=True)
    print("\n")


async def inspect_graph_engine():
    """检查 GraphEngine 内部"""
    
    print_section("GraphEngine 内部检查")
    
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
    
    if not soul._graph_engine:
        print("❌ GraphEngine 未初始化")
        return
    
    engine = soul._graph_engine
    
    print_subsection("GraphEngine 配置")
    print(f"Agent: {engine._agent.name}")
    print(f"当前迭代: {engine._iteration}")
    print(f"最大并行: {engine._agent.execution_config.parallel_tool_calls_limit}")
    
    # 工具列表
    print_subsection("可用工具")
    for tool in engine._agent.toolset.tools[:5]:
        print(f"   - {tool.name}: {tool.description[:50]}...")
    
    # 生成任务图
    print_subsection("生成任务图")
    task = "并行读取 README.md 和 pyproject.toml"
    print(f"任务: {task}")
    print("⏳ 调用 LLM 生成 TaskGraph...\n")
    
    graph = await engine.create_task_graph(task)
    
    print(f"\n✅ 生成完成!")
    print(f"   节点数: {len(graph.nodes)}")
    
    # 打印图结构
    print_subsection("TaskGraph 结构")
    for node_id, node in graph.nodes.items():
        print(f"\n  📦 {node_id}")
        print(f"     描述: {node.description}")
        print(f"     工具: {node.tool or 'None'}")
        print(f"     参数: {node.args}")
        print(f"     依赖: {node.dependencies or '[]'}")
    
    # 执行顺序
    print_subsection("执行分析")
    ready = graph.get_ready_nodes()
    print(f"可立即执行: {[n.id for n in ready]}")
    print(f"是否完成: {graph.is_complete()}")
    
    # 执行
    print_subsection("执行任务图")
    await engine.execute_graph(graph)
    
    print(f"\n✅ 执行完成")
    print(f"完成节点: {list(graph.completed)}")
    print(f"结果数: {len(graph.results)}")
    
    for node_id, result in graph.results.items():
        preview = str(result)[:100].replace('\n', ' ')
        print(f"\n  {node_id}:")
        print(f"     {preview}...")


async def main():
    """主函数"""
    import sys
    
    test = sys.argv[1] if len(sys.argv) > 1 else "visual"
    
    match test:
        case "visual":
            await test_proactive_visual()
        case "inspect":
            await inspect_graph_engine()
        case _:
            print(f"未知测试: {test}")
            print("可用: visual, inspect")


if __name__ == "__main__":
    asyncio.run(main())
