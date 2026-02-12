#!/usr/bin/env python3
"""
纯后台运行的 KimiSoul - 无 UI，直接获取结果
"""

import asyncio
import sys
from pathlib import Path
from typing import AsyncIterator
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent / "src"))

from kaos.path import KaosPath
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.soul.agent import Runtime, load_agent
from kimi_cli.soul.context import Context
from kimi_cli.soul import run_soul
from kimi_cli.session import Session
from kimi_cli.config import load_config
from kimi_cli.auth.oauth import OAuthManager
from kimi_cli.llm import create_llm
from kimi_cli.wire import Wire
from kimi_cli.wire.types import (
    TextPart, ToolCall, ToolResult, 
    TurnBegin, StepBegin, StatusUpdate,
    WireMessage
)
from kimi_cli.agentspec import DEFAULT_AGENT_FILE


@dataclass
class AgentResult:
    """Agent 运行结果"""
    text: str                    # 完整文本回复
    tool_calls: list[dict]       # 工具调用记录
    token_usage: dict            # Token 使用情况
    step_count: int              # 执行步数


class BackgroundAgent:
    """
    纯后台 Agent - 无 UI 交互，直接返回结果
    
    用法:
        agent = await BackgroundAgent.create()
        result = await agent.run("你好")
        print(result.text)
    """
    
    def __init__(self, soul: KimiSoul, session: Session):
        self.soul = soul
        self.session = session
    
    @classmethod
    async def create(
        cls,
        work_dir: Path | None = None,
        yolo: bool = True,
    ) -> "BackgroundAgent":
        """创建 Agent 实例"""
        
        config = load_config()
        work_dir = KaosPath.unsafe_from_local_path(work_dir) if work_dir else KaosPath.cwd()
        session = await Session.create(work_dir)
        
        if not config.default_model or config.default_model not in config.models:
            raise RuntimeError("未配置模型，请先运行 `kimi login`")
        
        model = config.models[config.default_model]
        provider = config.providers[model.provider]
        oauth = OAuthManager(config)
        llm = create_llm(provider, model, session_id=session.id, oauth=oauth)
        
        if llm is None:
            raise RuntimeError("LLM 创建失败")
        
        runtime = await Runtime.create(config, oauth, llm, session, yolo=yolo)
        agent = await load_agent(DEFAULT_AGENT_FILE, runtime, mcp_configs=[])
        
        context = Context(session.context_file)
        await context.restore()
        soul = KimiSoul(agent, context=context)
        
        return cls(soul, session)
    
    async def run(self, prompt: str) -> AgentResult:
        """
        运行 Agent，直接返回结果（无流式输出）
        
        Args:
            prompt: 用户输入
            
        Returns:
            AgentResult: 包含完整回复和元信息
        """
        cancel_event = asyncio.Event()
        
        # 收集结果
        text_parts = []
        tool_calls = []
        token_usage = {"input": 0, "output": 0, "total": 0}
        step_count = 0
        
        async def dummy_ui_loop(wire: Wire):
            """
            虚拟 UI 循环 - 只收集信息，不做任何显示
            这是 run_soul 必需的，但我们只用来收集数据
            """
            ui_side = wire.ui_side(merge=True)
            try:
                while True:
                    msg = await ui_side.receive()
                    
                    # 只收集，不显示
                    match msg:
                        case TextPart(text=text):
                            text_parts.append(text)
                        case ToolCall(name=name, arguments=args):
                            tool_calls.append({
                                "name": name,
                                "arguments": args,
                            })
                        case StepBegin(n=n):
                            step_count = n
                        case StatusUpdate(token_usage=usage):
                            if usage:
                                token_usage["input"] = usage.input
                                token_usage["output"] = usage.output
                                token_usage["total"] = usage.total
                                
            except Exception:
                pass  # Wire 关闭，结束循环
        
        # 运行 Soul
        await run_soul(self.soul, prompt, dummy_ui_loop, cancel_event)
        
        return AgentResult(
            text="".join(text_parts),
            tool_calls=tool_calls,
            token_usage=token_usage,
            step_count=step_count,
        )
    
    async def run_stream(self, prompt: str) -> AsyncIterator[str]:
        """
        流式运行 - 只产出文本片段（无其他消息类型）
        
        Yields:
            str: 文本片段
        """
        cancel_event = asyncio.Event()
        text_queue = asyncio.Queue()
        
        async def stream_ui_loop(wire: Wire):
            """只提取文本，放入队列"""
            ui_side = wire.ui_side(merge=True)
            try:
                while True:
                    msg = await ui_side.receive()
                    if isinstance(msg, TextPart):
                        await text_queue.put(msg.text)
            except:
                pass
            finally:
                await text_queue.put(None)  # 结束标记
        
        # 后台运行 Soul
        soul_task = asyncio.create_task(
            run_soul(self.soul, prompt, stream_ui_loop, cancel_event)
        )
        
        # 流式产出文本
        try:
            while True:
                text = await text_queue.get()
                if text is None:
                    break
                yield text
        finally:
            if not soul_task.done():
                cancel_event.set()
                try:
                    await soul_task
                except:
                    pass


# ============ 使用示例 ============

async def basic_run():
    """基础用法 - 直接获取结果"""
    print("🚀 创建后台 Agent...")
    agent = await BackgroundAgent.create(yolo=True)
    
    print("📝 发送: 你好\n")
    result = await agent.run("你好，请介绍一下自己")
    
    print(f"🤖 回复:\n{result.text}\n")
    print(f"📊 统计: {result.step_count} 步, {result.token_usage['total']} tokens")
    if result.tool_calls:
        print(f"🔧 工具调用: {len(result.tool_calls)} 次")


async def stream_run():
    """流式获取文本（仅文本，无其他消息）"""
    print("🚀 创建后台 Agent...")
    agent = await BackgroundAgent.create(yolo=True)
    
    print("📝 发送: 讲个故事\n")
    print("🤖 ", end="", flush=True)
    
    async for text in agent.run_stream("讲一个100字的短故事"):
        print(text, end="", flush=True)
    
    print("\n")


async def multi_turn():
    """多轮对话 - 每轮都是独立的，但共享上下文"""
    print("🚀 创建后台 Agent...")
    agent = await BackgroundAgent.create(yolo=True)
    
    prompts = [
        "你好，我叫小明",
        "我叫什么名字？",
        "2+2等于几？",
    ]
    
    for prompt in prompts:
        print(f"👤 {prompt}")
        result = await agent.run(prompt)
        print(f"🤖 {result.text}\n")


async def batch_process():
    """批量处理多个任务"""
    print("🚀 创建后台 Agent...")
    agent = await BackgroundAgent.create(yolo=True)
    
    tasks = [
        "总结：Python 是一种高级编程语言",
        "翻译：Hello World -> 中文",
        "计算：100的平方根是多少",
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n📋 任务 {i}: {task}")
        result = await agent.run(task)
        print(f"✅ 结果: {result.text[:100]}...")
        print(f"   使用: {result.token_usage['total']} tokens")


async def with_tools():
    """使用工具的场景"""
    print("🚀 创建后台 Agent...")
    agent = await BackgroundAgent.create(yolo=True)
    
    print("📝 发送: 查看当前目录\n")
    result = await agent.run("请列出当前目录的文件，并总结项目结构")
    
    print(f"🤖 回复:\n{result.text}\n")
    if result.tool_calls:
        print(f"🔧 使用的工具:")
        for tc in result.tool_calls:
            print(f"   - {tc['name']}")


if __name__ == "__main__":
    import sys
    
    example = sys.argv[1] if len(sys.argv) > 1 else "basic"
    
    match example:
        case "basic":
            asyncio.run(basic_run())
        case "stream":
            asyncio.run(stream_run())
        case "multi":
            asyncio.run(multi_turn())
        case "batch":
            asyncio.run(batch_process())
        case "tool":
            asyncio.run(with_tools())
        case _:
            print(f"未知示例: {example}")
            print("可用: basic, stream, multi, batch, tool")
