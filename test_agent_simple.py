#!/usr/bin/env python3
"""
极简测试 Kimi CLI Agent
使用自定义 LLM 配置，支持聊天记录和 Command+Q 退出
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
from kimi_cli.config import LLMModel, LLMProvider
from kimi_cli.wire.types import TextPart, ToolCall, ToolResult, ToolCallPart, WireMessage
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


async def chat_loop():
    """聊天循环，支持聊天记录和 Command+Q 退出"""
    # 创建自定义 LLM
    llm = create_custom_llm()

    session = await Session.create(KaosPath.cwd())
    instance = await KimiCLI.create(session, llm=llm, yolo=True)

    # 聊天记录
    history = []

    print("🤖 Kimi CLI Agent 已启动")
    print("💡 输入你的消息开始对话，按 Ctrl+Q 或输入 'exit'/'quit' 退出\n")

    while True:
        try:
            # 获取用户输入
            prompt = input("👤 你: ").strip()

            # 检查退出命令
            if prompt.lower() in ("exit", "quit", "q"):
                print("👋 再见！")
                break

            if not prompt:
                continue

            # 添加到历史记录
            history.append({"role": "user", "content": prompt})

            # 构建带上下文的提示
            context_prompt = build_context_prompt(history)

            cancel_event = asyncio.Event()
            response_parts = []

            print("🤖 Kimi: ", end="", flush=True)

            msg: WireMessage
            async for msg in instance.run(context_prompt, cancel_event):
                match msg:
                    case TextPart(text=text):
                        print(text, end="", flush=True)
                        response_parts.append(text)
                    case ToolCall(name=name):
                        print(f"\n[工具: {name}]")
                        print("*"*100)
                        input()
                    case ToolCallPart(arguments_part=arguments_part):
                        print(arguments_part, end="", flush=True)
                    case ToolResult():
                        print(f"\n[工具完成]{msg}")
                        print("*" * 100)
                        input()

            print()  # 换行

            # 保存助手回复到历史
            full_response = "".join(response_parts)
            history.append({"role": "assistant", "content": full_response})

            # 显示历史轮数
            print(f"\n📜 当前对话轮数: {len(history) // 2}\n")

        except KeyboardInterrupt:
            print("\n👋 检测到中断，再见！")
            break
        except EOFError:
            print("\n👋 检测到 EOF，再见！")
            break


def build_context_prompt(history):
    """构建带上下文的提示"""
    if len(history) <= 1:
        return history[-1]["content"] if history else ""

    # 构建上下文
    lines = ["以下是之前的对话历史：\n"]

    for msg in history[:-1]:
        role = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"{role}: {msg['content']}\n")

    lines.append("\n现在请回答用户的最新问题：")
    lines.append(f"用户: {history[-1]['content']}")

    return "\n".join(lines)


if __name__ == '__main__':
    asyncio.run(chat_loop())
