#!/usr/bin/env python3
"""
极简 KimiSoul 使用示例 - 纯 Agent，无 CLI
"""

import asyncio
import sys
from pathlib import Path
from typing import AsyncIterator

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
from kimi_cli.wire.types import TextPart, ToolCall, ToolResult, WireMessage
from kimi_cli.agentspec import DEFAULT_AGENT_FILE


class SimpleAgent:
    """
    极简 Agent 封装 - 直接使用 KimiSoul
    
    用法:
        agent = await SimpleAgent.create()
        async for msg in agent.chat("你好"):
            print(msg)
    """
    
    def __init__(self, soul: KimiSoul, session: Session):
        self.soul = soul
        self.session = session
        self._cancel_event = asyncio.Event()
    
    @classmethod
    async def create(
        cls,
        work_dir: Path | None = None,
        yolo: bool = True,
    ) -> "SimpleAgent":
        """创建 Agent 实例"""
        
        # 加载配置
        config = load_config()
        
        # 创建 Session
        work_dir = KaosPath.unsafe_from_local_path(work_dir) if work_dir else KaosPath.cwd()
        session = await Session.create(work_dir)
        
        # 创建 LLM
        if not config.default_model or config.default_model not in config.models:
            raise RuntimeError("未配置模型，请先运行 `kimi login`")
        
        model = config.models[config.default_model]
        provider = config.providers[model.provider]
        oauth = OAuthManager(config)
        llm = create_llm(provider, model, session_id=session.id, oauth=oauth)
        
        if llm is None:
            raise RuntimeError("LLM 创建失败")
        
        # 创建 Runtime
        runtime = await Runtime.create(config, oauth, llm, session, yolo=yolo)
        
        # 加载 Agent
        agent = await load_agent(DEFAULT_AGENT_FILE, runtime, mcp_configs=[])
        
        # 创建 Context 和 Soul
        context = Context(session.context_file)
        await context.restore()
        soul = KimiSoul(agent, context=context)
        
        return cls(soul, session)
    
    async def chat(self, prompt: str) -> AsyncIterator[WireMessage]:
        """
        发送消息并流式获取响应
        
        Yields:
            WireMessage: 各种消息类型
            - TextPart: 文本内容
            - ToolCall: 工具调用
            - ToolResult: 工具结果
            - ... 其他消息类型
        """
        self._cancel_event.clear()
        
        # 用于传递消息的队列
        queue: asyncio.Queue[WireMessage] = asyncio.Queue()
        
        async def ui_loop(wire: Wire):
            """后台任务：接收消息并放入队列"""
            ui_side = wire.ui_side(merge=True)
            try:
                while True:
                    msg = await ui_side.receive()
                    await queue.put(msg)
            except:
                pass  # Wire 关闭
            finally:
                await queue.put(None)  # 结束标记
        
        # 启动 Soul 运行（后台任务）
        soul_task = asyncio.create_task(
            run_soul(self.soul, prompt, ui_loop, self._cancel_event)
        )
        
        # 从队列中消费消息
        try:
            while True:
                msg = await queue.get()
                if msg is None:
                    break
                yield msg
        finally:
            # 确保 Soul 任务完成
            if not soul_task.done():
                self._cancel_event.set()
                try:
                    await soul_task
                except:
                    pass
    
    def cancel(self):
        """取消当前对话"""
        self._cancel_event.set()


# ============ 使用示例 ============

async def basic_usage():
    """基础用法"""
    print("🚀 创建 Agent...")
    agent = await SimpleAgent.create(yolo=True)
    
    print("📝 发送: 你好\n")
    async for msg in agent.chat("你好，请介绍一下自己"):
        match msg:
            case TextPart(text=text):
                print(text, end="", flush=True)
            case ToolCall(name=name):
                print(f"\n[调用工具: {name}]")
            case ToolResult():
                print(f"\n[工具完成]")
    print("\n")


async def multi_turn_chat():
    """多轮对话"""
    agent = await SimpleAgent.create(yolo=True)
    
    prompts = [
        "你好，我叫小明",
        "我叫什么名字？",
        "1+1等于几？",
    ]
    
    for prompt in prompts:
        print(f"👤 {prompt}")
        print("🤖 ", end="", flush=True)
        
        async for msg in agent.chat(prompt):
            if isinstance(msg, TextPart):
                print(msg.text, end="", flush=True)
        
        print("\n")


async def with_tool_usage():
    """使用工具的对话"""
    agent = await SimpleAgent.create(yolo=True)
    
    print("👤 查看当前目录文件\n")
    async for msg in agent.chat("请列出当前目录的文件"):
        match msg:
            case TextPart(text=text):
                print(text, end="", flush=True)
            case ToolCall(name=name, arguments=args):
                print(f"\n[工具: {name}({args})]")
            case ToolResult(return_value=rv):
                print(f"\n[结果: {str(rv)[:100]}...]")
    print("\n")


async def cancellable_chat():
    """可取消的对话"""
    agent = await SimpleAgent.create(yolo=True)
    
    # 5秒后取消
    asyncio.get_event_loop().call_later(5, agent.cancel)
    
    try:
        async for msg in agent.chat("请写一个很长很长的故事"):
            if isinstance(msg, TextPart):
                print(msg.text, end="", flush=True)
    except Exception as e:
        print(f"\n[已取消: {e}]")


if __name__ == "__main__":
    import sys
    
    # 默认运行基础示例
    example = sys.argv[1] if len(sys.argv) > 1 else "basic"
    
    match example:
        case "basic":
            asyncio.run(basic_usage())
        case "multi":
            asyncio.run(multi_turn_chat())
        case "tool":
            asyncio.run(with_tool_usage())
        case "cancel":
            asyncio.run(cancellable_chat())
        case _:
            print(f"未知示例: {example}")
            print("可用: basic, multi, tool, cancel")
