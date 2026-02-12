#!/usr/bin/env python3
"""
直接使用 KimiSoul，绕过 CLI 封装
直接运行: python test_soul_direct.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from kaos.path import KaosPath
from kosong.message import Message

# 直接导入 Soul 相关组件
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.soul.agent import Runtime, Agent, load_agent
from kimi_cli.soul.context import Context
from kimi_cli.session import Session
from kimi_cli.config import Config, load_config
from kimi_cli.auth.oauth import OAuthManager
from kimi_cli.llm import create_llm


async def run_soul_direct(prompt: str):
    """直接运行 KimiSoul，不使用 CLI 封装"""
    
    print(f"🚀 直接启动 KimiSoul...")
    print(f"📝 输入: {prompt}")
    print("-" * 50)
    
    # 1. 加载配置（从默认位置 ~/.kimi/config.toml）
    try:
        config = load_config()
    except Exception as e:
        print(f"⚠️  加载配置失败: {e}")
        print("使用默认配置...")
        config = Config()
    
    # 2. 创建 Session
    work_dir = KaosPath.cwd()
    session = await Session.create(work_dir)
    print(f"📁 Session: {session.id}")
    
    # 3. 创建 OAuth 管理器
    oauth = OAuthManager(config)
    
    # 4. 创建 LLM（从配置）
    if config.default_model and config.default_model in config.models:
        model = config.models[config.default_model]
        provider = config.providers[model.provider]
        llm = create_llm(provider, model, session_id=session.id, oauth=oauth)
        print(f"🤖 模型: {model.model}")
    else:
        print("❌ 未配置模型，请先运行 `kimi login` 或配置模型")
        return
    
    if llm is None:
        print("❌ LLM 创建失败")
        return
    
    # 5. 创建 Runtime
    runtime = await Runtime.create(
        config=config,
        oauth=oauth,
        llm=llm,
        session=session,
        yolo=True,  # 自动审批
    )
    
    # 6. 加载 Agent
    from kimi_cli.agentspec import DEFAULT_AGENT_FILE
    try:
        agent = await load_agent(
            agent_file=DEFAULT_AGENT_FILE,
            runtime=runtime,
            mcp_configs=[],  # 不加载 MCP
        )
        print(f"🎭 Agent: {agent.name}")
    except Exception as e:
        print(f"❌ 加载 Agent 失败: {e}")
        return
    
    # 7. 创建 Context
    context = Context(session.context_file)
    await context.restore()
    
    # 8. 创建 KimiSoul
    soul = KimiSoul(agent, context=context)
    print(f"✨ Soul 创建成功")
    print("-" * 50)
    
    # 9. 直接调用 soul.run() - 这是最简单的接口！
    # 但 soul.run() 不会 yield 消息，它内部使用 wire_send
    # 所以我们需要设置一个 Wire 来接收消息
    
    from kimi_cli.soul import run_soul
    from kimi_cli.wire import Wire
    from kimi_cli.wire.types import TextPart, ToolCall, ToolResult
    
    cancel_event = asyncio.Event()
    messages = []
    
    async def ui_loop(wire: Wire):
        """UI 循环：接收 Soul 发送的消息"""
        ui_side = wire.ui_side(merge=True)
        try:
            while True:
                msg = await ui_side.receive()
                messages.append(msg)
                
                # 实时打印
                match msg:
                    case TextPart(text=text):
                        print(text, end="", flush=True)
                    case ToolCall(name=name):
                        print(f"\n🔧 [工具: {name}]")
                    case ToolResult():
                        print(f"\n✅ [工具完成]")
                        
        except Exception:
            pass  # Wire 关闭时会抛出异常
    
    # 10. 运行 Soul
    try:
        await run_soul(soul, prompt, ui_loop, cancel_event)
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
    
    print(f"\n{'-' * 50}")
    print(f"✨ 完成! 共收到 {len(messages)} 条消息")
    
    return messages


async def simple_chat_example():
    """更简单的聊天示例 - 手动管理上下文"""
    
    print("\n" + "=" * 50)
    print("简单聊天示例（手动管理上下文）")
    print("=" * 50)
    
    # 加载配置
    config = load_config()
    work_dir = KaosPath.cwd()
    session = await Session.create(work_dir)
    oauth = OAuthManager(config)
    
    # 创建 LLM
    model = config.models[config.default_model]
    provider = config.providers[model.provider]
    llm = create_llm(provider, model, session_id=session.id, oauth=oauth)
    
    # 创建 Runtime 和 Agent
    runtime = await Runtime.create(config, oauth, llm, session, yolo=True)
    agent = await load_agent(DEFAULT_AGENT_FILE, runtime, mcp_configs=[])
    
    # 创建 Context
    context = Context(session.context_file)
    await context.restore()
    
    # 创建 Soul
    soul = KimiSoul(agent, context=context)
    
    # 多轮对话
    prompts = [
        "你好，我叫小明",
        "我叫什么名字？",
        "1+1等于几？",
    ]
    
    for prompt in prompts:
        print(f"\n👤 用户: {prompt}")
        print("🤖 AI: ", end="", flush=True)
        
        # 使用 run_soul 运行一轮
        cancel_event = asyncio.Event()
        
        async def ui_loop(wire: Wire):
            ui_side = wire.ui_side(merge=True)
            try:
                while True:
                    msg = await ui_side.receive()
                    if isinstance(msg, TextPart):
                        print(msg.text, end="", flush=True)
            except:
                pass
        
        await run_soul(soul, prompt, ui_loop, cancel_event)
        print()  # 换行


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "你好，请介绍一下自己"
    
    # 运行单次对话
    asyncio.run(run_soul_direct(prompt))
    
    # 或者运行多轮对话示例
    # asyncio.run(simple_chat_example())
