#!/usr/bin/env python3
"""
极简测试 Kimi CLI Agent
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from kaos.path import KaosPath
from kimi_cli.session import Session
from kimi_cli.app import KimiCLI
from kimi_cli.wire.types import TextPart, ToolCall, ToolResult


async def chat(prompt: str):
    """最简单的聊天接口"""
    session = await Session.create(KaosPath.cwd())
    instance = await KimiCLI.create(session, yolo=True)
    cancel_event = asyncio.Event()
    
    async for msg in instance.run(prompt, cancel_event):
        match msg:
            case TextPart(text=text):
                print(text, end="", flush=True)
            case ToolCall(name=name):
                print(f"\n[工具: {name}]")
            case ToolResult():
                print(f"\n[工具完成]")


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "你好"
    print(f"👤 用户: {prompt}\n🤖 AI: ", end="", flush=True)
    asyncio.run(chat(prompt))
    print()  # 最后换行
