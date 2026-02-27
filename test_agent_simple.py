#!/usr/bin/env python3
"""
极简测试 Kimi CLI Agent
使用自定义 LLM 配置
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from kaos.path import KaosPath
from kimi_cli.session import Session
from kimi_cli.app import KimiCLI
from kimi_cli.llm import create_llm
from kimi_cli.config import LLMModel,  LLMProvider
from kimi_cli.wire.types import TextPart, ToolCall, ToolResult
from pydantic import SecretStr


# 直接写死的配置
BASE_URL = "https://coding.dashscope.aliyuncs.com/apps/anthropic"
API_KEY = "sk-sp-0a0d748817eb422ca0ec9a253b2eca74"
MODEL_NAME = "kimi-k2.5"


def create_custom_llm():
    """创建自定义 LLM 实例"""
    provider = LLMProvider(
        type="anthropic",
        base_url=BASE_URL,
        api_key=SecretStr(API_KEY),
    )
    model = LLMModel(
        provider="anthropic",
        model=MODEL_NAME,
        max_context_size=100_000,
    )
    return create_llm(provider, model)


async def chat(prompt: str):
    """最简单的聊天接口"""
    # 创建自定义 LLM
    llm = create_custom_llm()

    session = await Session.create(KaosPath.cwd())
    instance = await KimiCLI.create(session, llm=llm, yolo=True)
    cancel_event = asyncio.Event()

    async for msg in instance.run(prompt, cancel_event):
        print(msg)
        match msg:
            case TextPart(text=text):
                print(text, end="", flush=True)
            case ToolCall(name=name):
                print(f"\n[工具: {name}]")
            case ToolResult():
                print(f"\n[工具完成]")


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "你好"

    # 显示当前配置
    print(f"📍 Base URL: {BASE_URL}", file=sys.stderr)
    masked = API_KEY[:10] + "..." + API_KEY[-4:] if len(API_KEY) > 14 else "***"
    print(f"🔑 Auth Token: {masked}", file=sys.stderr)
    print(f"🤖 Model: {MODEL_NAME}", file=sys.stderr)
    print(f"👤 用户: {prompt}\n🤖 AI: ", end="", flush=True)

    asyncio.run(chat(prompt))
    print()  # 最后换行
