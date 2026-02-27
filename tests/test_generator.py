import asyncio
from kaos.path import KaosPath
from kimi_cli.session import Session
from kimi_cli.app import KimiCLI
from kimi_cli.wire.types import TextPart, ToolCall, ToolResult

async def chat(prompt: str):
    session = await Session.create(KaosPath.cwd())
    instance = await KimiCLI.create(session, yolo=True)
    cancel_event = asyncio.Event()
    
    async for msg in instance.run(prompt, cancel_event): print(msg)
#        match msg:
#            case TextPart(text=text):
#                print(text, end="", flush=True)
#            case ToolCall(name=name):
#                print(f"\n[工具: {name}]")
#            case ToolResult():
#                print(f"\n[工具完成]")

# 运行
# asyncio.run(chat("你好"))
while True:
	prompt = input()
	asyncio.run(chat(prompt))
