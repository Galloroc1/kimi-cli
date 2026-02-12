#!/usr/bin/env python3
"""
UnifiedSoul (Graph 模式) 详细日志版本 - 展示规划图和执行过程
"""

import asyncio
import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent / "src"))

from kaos.path import KaosPath
from kimi_cli.soul.unified.soul import UnifiedSoul
from kimi_cli.soul.unified.agent import UnifiedAgent, ExecutionConfig
from kimi_cli.soul.unified.context import UnifiedContext
from kimi_cli.soul.unified.types import AgentMode, ExecutionMode
from kimi_cli.soul.unified.graph_engine import TaskGraph, TaskNode, GraphEngine
from kimi_cli.soul import run_soul
from kimi_cli.soul.agent import Runtime
from kimi_cli.session import Session
from kimi_cli.config import load_config
from kimi_cli.auth.oauth import OAuthManager
from kimi_cli.llm import create_llm
from kimi_cli.wire import Wire
from kimi_cli.wire.types import (
    TextPart, ToolCall, ToolResult, 
    StepBegin, StatusUpdate, TurnBegin,
    WireMessage
)
from kimi_cli.agentspec import DEFAULT_AGENT_FILE
from kimi_cli.utils.logging import logger


# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@dataclass
class ExecutionLog:
    """执行日志"""
    phase: str
    content: str
    data: dict | None = None
    
    def __str__(self):
        if self.data:
            return f"[{self.phase}] {self.content}\n{json.dumps(self.data, indent=2, ensure_ascii=False)}"
        return f"[{self.phase}] {self.content}"


class VerboseUnifiedAgent:
    """带详细日志的 Unified Agent"""
    
    def __init__(self, soul: UnifiedSoul, session: Session, verbose: bool = True):
        self.soul = soul
        self.session = session
        self.verbose = verbose
        self.logs: list[ExecutionLog] = []
    
    @classmethod
    async def create(
        cls,
        mode: ExecutionMode = ExecutionMode.PROACTIVE,
        yolo: bool = True,
        verbose: bool = True,
    ) -> "VerboseUnifiedAgent":
        """创建 Agent"""
        
        print("🚀 创建 UnifiedSoul...")
        print(f"   模式: {mode.name}")
        print(f"   工作目录: {Path.cwd()}")
        
        config = load_config()
        session = await Session.create(KaosPath.cwd())
        print(f"   Session ID: {session.id}")
        
        model = config.models[config.default_model]
        provider = config.providers[model.provider]
        oauth = OAuthManager(config)
        llm = create_llm(provider, model, session_id=session.id, oauth=oauth)
        
        print(f"   模型: {model.model}")
        
        if llm is None:
            raise RuntimeError("LLM 创建失败")
        
        runtime = await Runtime.create(config, oauth, llm, session, yolo=yolo)
        
        execution_config = ExecutionConfig(
            mode=mode,
            max_iterations=10,
            parallel_tool_calls_limit=5,
        )
        
        print(f"   执行配置: {asdict(execution_config)}")
        
        unified_agent = await UnifiedAgent.from_spec(
            agent_file=DEFAULT_AGENT_FILE,
            runtime=runtime,
            execution_config=execution_config,
            mcp_configs=[],
        )
        
        print(f"   Agent: {unified_agent.name}")
        print(f"   工具数量: {len(unified_agent.toolset.tools)}")
        for tool in unified_agent.toolset.tools[:5]:
            print(f"      - {tool.name}")
        if len(unified_agent.toolset.tools) > 5:
            print(f"      ... 等共 {len(unified_agent.toolset.tools)} 个工具")
        
        context = UnifiedContext(session.context_file)
        await context.restore()
        
        agent_mode = {
            ExecutionMode.REACTIVE: AgentMode.REACTIVE,
            ExecutionMode.PROACTIVE: AgentMode.PROACTIVE,
            ExecutionMode.ADAPTIVE: AgentMode.ADAPTIVE,
        }.get(mode, AgentMode.REACTIVE)
        
        soul = UnifiedSoul(unified_agent, context, mode=agent_mode)
        
        instance = cls(soul, session, verbose)
        
        # 如果是 PROACTIVE 模式，打印 GraphEngine 信息
        if mode == ExecutionMode.PROACTIVE and soul._graph_engine:
            print(f"   GraphEngine: 已初始化")
        
        print("✅ Agent 创建完成\n")
        
        return instance
    
    def _log(self, phase: str, content: str, data: dict | None = None):
        """记录日志"""
        log = ExecutionLog(phase, content, data)
        self.logs.append(log)
        if self.verbose:
            print(f"\n{'='*60}")
            print(log)
            print('='*60)
    
    async def run_stream(self, prompt: str):
        """
        流式运行，展示详细执行过程
        
        Yields:
            tuple[str, WireMessage]: (类型, 消息)
        """
        cancel_event = asyncio.Event()
        message_queue = asyncio.Queue()
        
        # 记录任务
        self._log("TASK", f"用户输入: {prompt}", {"prompt": prompt, "mode": self.soul.mode.name})
        
        async def verbose_ui_loop(wire: Wire):
            """详细的 UI 循环 - 记录所有消息"""
            ui_side = wire.ui_side(merge=False)  # 不合并，看原始消息
            
            self._log("SETUP", "UI 循环启动，开始接收消息...")
            
            try:
                while True:
                    msg = await ui_side.receive()
                    
                    # 记录消息
                    match msg:
                        case TurnBegin(user_input=ui):
                            self._log("TURN", "新回合开始", {"user_input": str(ui)[:100]})
                            await message_queue.put(("turn", msg))
                            
                        case StepBegin(n=n):
                            self._log("STEP", f"步骤 {n} 开始", {"step_number": n})
                            await message_queue.put(("step", msg))
                            
                        case TextPart(text=text):
                            # 文本片段不打印完整日志，只记录前50字符
                            preview = text[:50].replace('\n', ' ')
                            if len(text) > 50:
                                preview += "..."
                            self._log("TEXT", f"文本片段: {preview}", {"length": len(text)})
                            await message_queue.put(("text", msg))
                            
                        case ToolCall(name=name, arguments=args):
                            self._log("TOOL_CALL", f"工具调用: {name}", {
                                "name": name,
                                "arguments": args,
                            })
                            await message_queue.put(("tool_call", msg))
                            
                        case ToolResult(tool_call_id=tcid, return_value=rv):
                            result_str = str(rv)
                            self._log("TOOL_RESULT", f"工具结果: {tcid}", {
                                "tool_call_id": tcid,
                                "result_preview": result_str[:200],
                                "result_length": len(result_str),
                            })
                            await message_queue.put(("tool_result", msg))
                            
                        case StatusUpdate(token_usage=usage, context_usage=ctx):
                            data = {"context_usage": ctx}
                            if usage:
                                data["tokens"] = {
                                    "input": usage.input,
                                    "output": usage.output,
                                    "total": usage.total,
                                }
                            self._log("STATUS", "状态更新", data)
                            await message_queue.put(("status", msg))
                            
                        case _:
                            self._log("OTHER", f"其他消息: {type(msg).__name__}", {
                                "type": type(msg).__name__,
                            })
                            await message_queue.put(("other", msg))
                            
            except Exception as e:
                self._log("ERROR", f"UI 循环异常: {e}", {"error": str(e)})
            finally:
                self._log("END", "UI 循环结束")
                await message_queue.put(("end", None))
        
        # 启动 Soul 运行
        self._log("EXEC", "启动 Soul 运行...")
        soul_task = asyncio.create_task(
            run_soul(self.soul, prompt, verbose_ui_loop, cancel_event)
        )
        
        # 流式产出消息
        try:
            while True:
                msg_type, msg = await message_queue.get()
                if msg_type == "end":
                    break
                yield msg_type, msg
        finally:
            if not soul_task.done():
                cancel_event.set()
                try:
                    await soul_task
                except:
                    pass
            self._log("COMPLETE", "运行完成")
    
    def print_summary(self):
        """打印执行摘要"""
        print(f"\n{'='*60}")
        print("执行摘要")
        print('='*60)
        
        # 统计各类消息
        counts = {}
        for log in self.logs:
            counts[log.phase] = counts.get(log.phase, 0) + 1
        
        print(f"\n消息统计:")
        for phase, count in sorted(counts.items()):
            print(f"   {phase}: {count}")
        
        print(f"\n详细日志数: {len(self.logs)}")


async def test_proactive_verbose():
    """详细测试 PROACTIVE 模式"""
    
    print("\n" + "🧪 " * 30)
    print("PROACTIVE 模式详细测试")
    print("🧪 " * 30 + "\n")
    
    agent = await VerboseUnifiedAgent.create(
        mode=ExecutionMode.PROACTIVE,
        yolo=True,
        verbose=True,
    )
    
    # 一个适合并行处理的任务
    prompt = """请帮我完成以下任务：
1. 读取 README.md 了解项目
2. 查看 pyproject.toml 了解依赖
3. 列出 src/kimi_cli 目录结构
4. 总结这是一个什么项目
"""
    
    print(f"\n{'#'*60}")
    print(f"开始执行: {prompt[:80]}...")
    print(f"{'#'*60}\n")
    
    full_text = []
    
    async for msg_type, msg in agent.run_stream(prompt):
        match msg_type:
            case "text":
                # 流式打印文本
                print(msg.text, end="", flush=True)
                full_text.append(msg.text)
            case "tool_call":
                print(f"\n[🔧 调用: {msg.name}]")
            case "tool_result":
                print(f"\n[✅ 完成: {msg.tool_call_id}]")
            case "step":
                print(f"\n\n{'─'*40}")
                print(f"[步骤 {msg.n}]")
                print('─'*40)
    
    print("\n")
    agent.print_summary()
    
    # 打印完整回复
    print(f"\n{'='*60}")
    print("完整回复:")
    print('='*60)
    print("".join(full_text))


async def test_graph_details():
    """测试 Graph 细节 - 直接操作 GraphEngine"""
    
    print("\n" + "🔍 " * 30)
    print("GraphEngine 细节测试")
    print("🔍 " * 30 + "\n")
    
    agent = await VerboseUnifiedAgent.create(
        mode=ExecutionMode.PROACTIVE,
        yolo=True,
        verbose=True,
    )
    
    # 直接访问 GraphEngine
    if agent.soul._graph_engine:
        engine = agent.soul._graph_engine
        
        print("\n📊 GraphEngine 信息:")
        print(f"   Agent: {engine._agent.name}")
        print(f"   迭代次数: {engine._iteration}")
        
        # 创建任务图
        task = "分析项目结构并找出所有 Python 文件"
        print(f"\n📝 任务: {task}")
        print("\n⏳ 正在生成任务图...")
        
        try:
            graph = await engine.create_task_graph(task)
            
            print(f"\n✅ 任务图生成完成!")
            print(f"   节点数: {len(graph.nodes)}")
            print(f"   已完成: {len(graph.completed)}")
            
            # 打印节点详情
            print(f"\n📋 节点详情:")
            for node_id, node in graph.nodes.items():
                print(f"\n   节点: {node_id}")
                print(f"      描述: {node.description}")
                print(f"      工具: {node.tool or 'None (直接LLM)'}")
                print(f"      参数: {node.args}")
                print(f"      依赖: {node.dependencies or '无'}")
                
            # 找出可并行执行的节点
            ready = graph.get_ready_nodes()
            print(f"\n🚀 可并行执行的节点: {[n.id for n in ready]}")
            
            # 执行图
            print(f"\n⏳ 开始执行任务图...")
            await engine.execute_graph(graph)
            
            print(f"\n✅ 任务图执行完成!")
            print(f"   最终结果数: {len(graph.results)}")
            for node_id, result in graph.results.items():
                preview = str(result)[:100].replace('\n', ' ')
                print(f"   {node_id}: {preview}...")
                
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ GraphEngine 未初始化")


async def test_compare_modes():
    """对比三种模式的执行过程"""
    
    print("\n" + "⚖️  " * 30)
    print("三种模式对比")
    print("⚖️  " * 30 + "\n")
    
    prompt = "查看当前目录的 README.md 文件"
    
    for mode in [ExecutionMode.REACTIVE, ExecutionMode.PROACTIVE]:
        print(f"\n{'='*60}")
        print(f"模式: {mode.name}")
        print('='*60)
        
        agent = await VerboseUnifiedAgent.create(mode=mode, verbose=False)
        
        step_count = 0
        tool_count = 0
        text_parts = []
        
        async for msg_type, msg in agent.run_stream(prompt):
            match msg_type:
                case "step":
                    step_count += 1
                case "tool_call":
                    tool_count += 1
                    print(f"[工具调用: {msg.name}]")
                case "text":
                    text_parts.append(msg.text)
        
        print(f"\n统计:")
        print(f"   步骤: {step_count}")
        print(f"   工具调用: {tool_count}")
        print(f"   回复长度: {len(''.join(text_parts))}")


async def main():
    """主函数"""
    import sys
    
    test = sys.argv[1] if len(sys.argv) > 1 else "proactive"
    
    match test:
        case "proactive":
            await test_proactive_verbose()
        case "graph":
            await test_graph_details()
        case "compare":
            await test_compare_modes()
        case _:
            print(f"未知测试: {test}")
            print("可用: proactive, graph, compare")


if __name__ == "__main__":
    asyncio.run(main())
