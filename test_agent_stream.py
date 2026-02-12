#!/usr/bin/env python3
"""
简单测试 Kimi CLI Agent 的异步生成器接口
直接运行: python test_agent_stream.py
"""

import asyncio
import sys
from pathlib import Path

# 确保能导入 kimi_cli
sys.path.insert(0, str(Path(__file__).parent / "src"))

from kaos.path import KaosPath
from kimi_cli.session import Session
from kimi_cli.app import KimiCLI
from kimi_cli.wire.types import (
    TextPart,
    ToolResult,
    ToolCall,
    ApprovalRequest,
    TurnBegin,
    StepBegin,
    StepInterrupted,
    StatusUpdate,
    CompactionBegin,
    CompactionEnd,
)


async def test_agent_stream(prompt: str = "你好，请介绍一下自己"):
    """测试 Agent 流式输出"""
    
    print(f"🚀 启动 Agent...")
    print(f"📝 输入: {prompt}")
    print("-" * 50)
    
    # 1. 创建 Session
    work_dir = KaosPath.cwd()
    session = await Session.create(work_dir)
    print(f"📁 Session: {session.id}")
    
    # 2. 创建 CLI 实例（需要配置，否则会提示未设置模型）
    try:
        instance = await KimiCLI.create(
            session,
            yolo=True,  # 自动审批，方便测试
        )
    except Exception as e:
        print(f"❌ 创建失败: {e}")
        print("提示: 请先运行 `kimi login` 登录，或设置 KIMI_API_KEY 环境变量")
        return
    
    # 3. 创建取消事件
    cancel_event = asyncio.Event()
    
    # 4. 运行并流式获取输出
    step_count = 0
    token_count = 0
    
    try:
        async for msg in instance.run(
            user_input=prompt,
            cancel_event=cancel_event,
            merge_wire_messages=True,
        ):
            match msg:
                case TurnBegin():
                    print(f"\n🟢 [回合开始]")
                    
                case StepBegin(n=n):
                    step_count += 1
                    print(f"\n📍 [步骤 {n} 开始]")
                    
                case TextPart(text=text):
                    # 流式打印文本
                    print(text, end="", flush=True)
                    
                case ToolCall(name=name, arguments=args):
                    print(f"\n🔧 [工具调用] {name}({args})")
                    
                case ToolResult(tool_call_id=tcid, return_value=rv):
                    print(f"\n✅ [工具结果] {tcid}: {str(rv)[:200]}...")
                    
                case StatusUpdate(token_usage=usage, context_usage=ctx):
                    if usage:
                        token_count = usage.total
                        print(f"\n📊 [Token: {usage.input}/{usage.output}, 上下文: {ctx:.1%}]")
                        
                case CompactionBegin():
                    print(f"\n🗜️  [上下文压缩开始]")
                    
                case CompactionEnd():
                    print(f"\n🗜️  [上下文压缩结束]")
                    
                case StepInterrupted():
                    print(f"\n⛔ [步骤中断]")
                    
                case ApprovalRequest():
                    # yolo=True 时不会收到这个，但以防万一
                    print(f"\n🤔 [需要审批] {msg.action}")
                    msg.resolve("approve")
                    
                case _:
                    # 其他消息类型
                    print(f"\n[{type(msg).__name__}]")
                    
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"\n{'-' * 50}")
    print(f"✨ 完成! 步骤: {step_count}, Token: {token_count}")


async def test_multi_turn():
    """测试多轮对话"""
    
    print("\n" + "=" * 50)
    print("测试多轮对话")
    print("=" * 50)
    
    work_dir = KaosPath.cwd()
    session = await Session.create(work_dir)
    
    try:
        instance = await KimiCLI.create(session, yolo=True)
    except Exception as e:
        print(f"❌ 创建失败: {e}")
        return
    
    cancel_event = asyncio.Event()
    
    # 第一轮
    print("\n📝 第一轮: 你好，我叫小明")
    print("-" * 30)
    async for msg in instance.run("你好，我叫小明", cancel_event):
        if isinstance(msg, TextPart):
            print(msg.text, end="", flush=True)
    print()
    
    # 第二轮（同一会话，应该记得名字）
    print("\n📝 第二轮: 我叫什么名字？")
    print("-" * 30)
    async for msg in instance.run("我叫什么名字？", cancel_event):
        if isinstance(msg, TextPart):
            print(msg.text, end="", flush=True)
    print()


async def test_with_tools():
    """测试带工具调用的场景"""
    
    print("\n" + "=" * 50)
    print("测试工具调用")
    print("=" * 50)
    
    work_dir = KaosPath.cwd()
    session = await Session.create(work_dir)
    
    try:
        instance = await KimiCLI.create(session, yolo=True)
    except Exception as e:
        print(f"❌ 创建失败: {e}")
        return
    
    cancel_event = asyncio.Event()
    
    # 测试文件操作工具
    prompt = "请查看当前目录下有哪些文件"
    print(f"\n📝 {prompt}")
    print("-" * 30)
    
    async for msg in instance.run(prompt, cancel_event):
        match msg:
            case TextPart(text=text):
                print(text, end="", flush=True)
            case ToolCall(name=name):
                print(f"\n🔧 [调用工具: {name}]")
            case ToolResult(return_value=rv):
                result = str(rv)
                print(f"\n📦 [结果: {result[:150]}...]")
    print()


async def main():
    """主函数"""
    
    # 测试 1: 基本流式输出
    await test_agent_stream("你好，请简单介绍一下自己")
    
    # 测试 2: 多轮对话
    # await test_multi_turn()
    
    # 测试 3: 工具调用
    # await test_with_tools()
    
    # 可以取消注释上面的测试来运行


if __name__ == "__main__":
    asyncio.run(main())
