#!/usr/bin/env python3
"""
简化版 UnifiedSoul 测试 - 专注于三种执行模式的区别
"""

import asyncio
import sys
from pathlib import Path
from dataclasses import dataclass

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
from kimi_cli.wire.types import TextPart, ToolCall, ToolResult, StepBegin, StatusUpdate
from kimi_cli.agentspec import DEFAULT_AGENT_FILE


@dataclass
class RunResult:
    """运行结果"""
    text: str
    steps: int
    tools: int
    tokens: dict
    mode: str


class SimpleUnifiedAgent:
    """简化版 Unified Agent 封装"""
    
    def __init__(self, soul: UnifiedSoul, session: Session):
        self.soul = soul
        self.session = session
    
    @classmethod
    async def create(
        cls,
        mode: ExecutionMode = ExecutionMode.REACTIVE,
        yolo: bool = True,
    ) -> "SimpleUnifiedAgent":
        """创建 Agent"""
        
        config = load_config()
        session = await Session.create(KaosPath.cwd())
        
        model = config.models[config.default_model]
        provider = config.providers[model.provider]
        oauth = OAuthManager(config)
        llm = create_llm(provider, model, session_id=session.id, oauth=oauth)
        
        if llm is None:
            raise RuntimeError("LLM 创建失败")
        
        runtime = await Runtime.create(config, oauth, llm, session, yolo=yolo)
        
        execution_config = ExecutionConfig(
            mode=mode,
            max_iterations=10,
            parallel_tool_calls_limit=5,
        )
        
        unified_agent = await UnifiedAgent.from_spec(
            agent_file=DEFAULT_AGENT_FILE,
            runtime=runtime,
            execution_config=execution_config,
            mcp_configs=[],
        )
        
        context = UnifiedContext(session.context_file)
        await context.restore()
        
        agent_mode = {
            ExecutionMode.REACTIVE: AgentMode.REACTIVE,
            ExecutionMode.PROACTIVE: AgentMode.PROACTIVE,
            ExecutionMode.ADAPTIVE: AgentMode.ADAPTIVE,
        }.get(mode, AgentMode.REACTIVE)
        
        soul = UnifiedSoul(unified_agent, context, mode=agent_mode)
        
        return cls(soul, session)
    
    async def run(self, prompt: str) -> RunResult:
        """运行并返回结果"""
        cancel_event = asyncio.Event()
        
        texts = []
        steps = 0
        tools = 0
        tokens = {"input": 0, "output": 0, "total": 0}
        
        async def collect(wire: Wire):
            ui_side = wire.ui_side(merge=True)
            try:
                while True:
                    msg = await ui_side.receive()
                    match msg:
                        case TextPart(text=text):
                            texts.append(text)
                        case StepBegin():
                            steps += 1
                        case ToolCall():
                            tools += 1
                        case StatusUpdate(token_usage=usage):
                            if usage:
                                tokens["input"] = usage.input
                                tokens["output"] = usage.output
                                tokens["total"] = usage.total
            except:
                pass
        
        await run_soul(self.soul, prompt, collect, cancel_event)
        
        return RunResult(
            text="".join(texts),
            steps=steps,
            tools=tools,
            tokens=tokens,
            mode=self.soul.mode.name,
        )


async def compare_modes():
    """对比三种执行模式"""
    
    print("=" * 70)
    print("对比三种执行模式")
    print("=" * 70)
    
    prompt = "分析当前目录的 Python 项目结构"
    
    # REACTIVE 模式
    print("\n" + "-" * 70)
    print("【REACTIVE 模式】逐步执行")
    print("-" * 70)
    agent1 = await SimpleUnifiedAgent.create(mode=ExecutionMode.REACTIVE)
    result1 = await agent1.run(prompt)
    print(f"回复: {result1.text[:300]}...")
    print(f"统计: {result1.steps} 步, {result1.tools} 次工具调用, {result1.tokens['total']} tokens")
    
    # PROACTIVE 模式
    print("\n" + "-" * 70)
    print("【PROACTIVE 模式】Graph 并行执行")
    print("-" * 70)
    agent2 = await SimpleUnifiedAgent.create(mode=ExecutionMode.PROACTIVE)
    result2 = await agent2.run(prompt)
    print(f"回复: {result2.text[:300]}...")
    print(f"统计: {result2.steps} 步, {result2.tools} 次工具调用, {result2.tokens['total']} tokens")
    
    # ADAPTIVE 模式
    print("\n" + "-" * 70)
    print("【ADAPTIVE 模式】自动选择")
    print("-" * 70)
    agent3 = await SimpleUnifiedAgent.create(mode=ExecutionMode.ADAPTIVE)
    result3 = await agent3.run(prompt)
    print(f"回复: {result3.text[:300]}...")
    print(f"实际模式: {result3.mode}")
    print(f"统计: {result3.steps} 步, {result3.tools} 次工具调用, {result3.tokens['total']} tokens")
    
    # 总结
    print("\n" + "=" * 70)
    print("对比总结")
    print("=" * 70)
    print(f"REACTIVE:  {result1.steps} 步,  {result1.tools} 工具, {result1.tokens['total']:>6} tokens")
    print(f"PROACTIVE: {result2.steps} 步,  {result2.tools} 工具, {result2.tokens['total']:>6} tokens")
    print(f"ADAPTIVE:  {result3.steps} 步,  {result3.tools} 工具, {result3.tokens['total']:>6} tokens (实际: {result3.mode})")


async def test_task_analysis():
    """测试任务分析（ADAPTIVE 模式的核心）"""
    
    print("\n" + "=" * 70)
    print("测试 ADAPTIVE 模式的任务分析")
    print("=" * 70)
    
    agent = await SimpleUnifiedAgent.create(mode=ExecutionMode.ADAPTIVE)
    
    test_cases = [
        ("简单任务", "你好"),
        ("中等任务", "解释 Python 的 asyncio"),
        ("复杂任务", "分析这个复杂项目的所有 Python 文件，找出潜在的性能问题和改进建议"),
    ]
    
    for label, prompt in test_cases:
        print(f"\n【{label}】: {prompt[:50]}...")
        result = await agent.run(prompt)
        print(f"   实际使用模式: {result.mode}")
        print(f"   执行步骤: {result.steps}")


async def test_streaming():
    """测试流式输出"""
    
    print("\n" + "=" * 70)
    print("测试流式输出 (PROACTIVE)")
    print("=" * 70)
    
    agent = await SimpleUnifiedAgent.create(mode=ExecutionMode.PROACTIVE)
    
    prompt = "写一首关于编程的短诗"
    print(f"\n提示: {prompt}")
    print("回复: ", end="", flush=True)
    
    cancel_event = asyncio.Event()
    text_queue = asyncio.Queue()
    
    async def stream_ui(wire: Wire):
        ui_side = wire.ui_side(merge=True)
        try:
            while True:
                msg = await ui_side.receive()
                if isinstance(msg, TextPart):
                    await text_queue.put(msg.text)
        except:
            pass
        finally:
            await text_queue.put(None)
    
    soul_task = asyncio.create_task(
        run_soul(agent.soul, prompt, stream_ui, cancel_event)
    )
    
    try:
        while True:
            text = await text_queue.get()
            if text is None:
                break
            print(text, end="", flush=True)
    finally:
        if not soul_task.done():
            cancel_event.set()
            try:
                await soul_task
            except:
                pass
    
    print("\n")


async def main():
    """主函数"""
    import sys
    
    test = sys.argv[1] if len(sys.argv) > 1 else "compare"
    
    match test:
        case "compare":
            await compare_modes()
        case "analysis":
            await test_task_analysis()
        case "stream":
            await test_streaming()
        case _:
            print(f"未知测试: {test}")
            print("可用: compare, analysis, stream")


if __name__ == "__main__":
    asyncio.run(main())
