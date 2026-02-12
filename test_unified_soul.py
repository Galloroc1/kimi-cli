#!/usr/bin/env python3
"""
UnifiedSoul (Graph 模式) 测试 - 支持 Reactive/Proactive/Adaptive 三种模式
"""

import asyncio
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import AsyncIterator

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
class GraphResult:
    """Graph 执行结果"""
    text: str
    steps: list[dict]
    tool_calls: list[dict]
    tokens: dict
    mode: str


class UnifiedBackgroundAgent:
    """
    UnifiedSoul 后台运行封装 - 支持 Graph 并行执行
    
    三种模式:
    - REACTIVE: 传统逐步执行 (KimiSoul)
    - PROACTIVE: Graph 并行执行
    - ADAPTIVE: 自动选择模式
    """
    
    def __init__(self, soul: UnifiedSoul, session: Session):
        self.soul = soul
        self.session = session
    
    @classmethod
    async def create(
        cls,
        work_dir: Path | None = None,
        mode: ExecutionMode = ExecutionMode.REACTIVE,
        yolo: bool = True,
    ) -> "UnifiedBackgroundAgent":
        """创建 Unified Agent"""
        
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
        
        # 创建 Runtime
        runtime = await Runtime.create(config, oauth, llm, session, yolo=yolo)
        
        # 使用 from_spec 加载 UnifiedAgent
        execution_config = ExecutionConfig(
            mode=mode,
            max_iterations=50,
            parallel_tool_calls_limit=5,
        )
        
        unified_agent = await UnifiedAgent.from_spec(
            agent_file=DEFAULT_AGENT_FILE,
            runtime=runtime,
            execution_config=execution_config,
            mcp_configs=[],
        )
        
        # 创建 UnifiedContext
        context = UnifiedContext(session.context_file)
        await context.restore()
        
        # 创建 UnifiedSoul
        agent_mode = {
            ExecutionMode.REACTIVE: AgentMode.REACTIVE,
            ExecutionMode.PROACTIVE: AgentMode.PROACTIVE,
            ExecutionMode.ADAPTIVE: AgentMode.ADAPTIVE,
        }.get(mode, AgentMode.REACTIVE)
        
        soul = UnifiedSoul(unified_agent, context, mode=agent_mode)
        
        return cls(soul, session)
    
    async def run(self, prompt: str) -> GraphResult:
        """
        运行 Agent，返回完整结果
        
        Args:
            prompt: 用户输入
            
        Returns:
            GraphResult: 包含执行结果和元信息
        """
        cancel_event = asyncio.Event()
        
        # 收集器
        text_parts = []
        steps = []
        tool_calls = []
        tokens = {"input": 0, "output": 0, "total": 0}
        
        async def collect_ui_loop(wire: Wire):
            """UI 循环 - 只收集数据"""
            ui_side = wire.ui_side(merge=True)
            try:
                while True:
                    msg = await ui_side.receive()
                    
                    match msg:
                        case TextPart(text=text):
                            text_parts.append(text)
                        case StepBegin(n=n):
                            steps.append({"step": n, "status": "begin"})
                        case ToolCall(name=name, arguments=args):
                            tool_calls.append({
                                "name": name,
                                "arguments": args,
                            })
                        case StatusUpdate(token_usage=usage):
                            if usage:
                                tokens["input"] = usage.input
                                tokens["output"] = usage.output
                                tokens["total"] = usage.total
                                
            except Exception:
                pass
        
        # 运行 Soul
        await run_soul(self.soul, prompt, collect_ui_loop, cancel_event)
        
        return GraphResult(
            text="".join(text_parts),
            steps=steps,
            tool_calls=tool_calls,
            tokens=tokens,
            mode=self.soul.mode.name,
        )
    
    async def run_stream(self, prompt: str) -> AsyncIterator[str]:
        """流式获取文本输出"""
        cancel_event = asyncio.Event()
        text_queue = asyncio.Queue()
        
        async def stream_ui_loop(wire: Wire):
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
            run_soul(self.soul, prompt, stream_ui_loop, cancel_event)
        )
        
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
    
    def set_mode(self, mode: AgentMode):
        """切换执行模式"""
        self.soul.set_mode(mode)


# ============ 测试示例 ============

async def test_reactive_mode():
    """测试 REACTIVE 模式（传统逐步执行）"""
    print("=" * 60)
    print("测试 REACTIVE 模式")
    print("=" * 60)
    
    agent = await UnifiedBackgroundAgent.create(
        mode=ExecutionMode.REACTIVE,
        yolo=True,
    )
    
    prompt = "你好，请介绍一下自己"
    print(f"\n📝 输入: {prompt}\n")
    
    result = await agent.run(prompt)
    
    print(f"🤖 回复:\n{result.text[:500]}...")
    print(f"\n📊 统计:")
    print(f"   模式: {result.mode}")
    print(f"   步骤: {len(result.steps)}")
    print(f"   Token: {result.tokens}")


async def test_proactive_mode():
    """测试 PROACTIVE 模式（Graph 并行执行）"""
    print("\n" + "=" * 60)
    print("测试 PROACTIVE 模式 (Graph 并行)")
    print("=" * 60)
    
    agent = await UnifiedBackgroundAgent.create(
        mode=ExecutionMode.PROACTIVE,
        yolo=True,
    )
    
    # 一个适合并行处理的任务
    prompt = "请并行分析当前目录：1) 列出所有Python文件 2) 统计代码行数 3) 找出最大的文件"
    print(f"\n📝 输入: {prompt}\n")
    
    result = await agent.run(prompt)
    
    print(f"🤖 回复:\n{result.text}")
    print(f"\n📊 统计:")
    print(f"   模式: {result.mode}")
    print(f"   步骤: {len(result.steps)}")
    print(f"   工具调用: {len(result.tool_calls)} 次")
    for tc in result.tool_calls:
        print(f"      - {tc['name']}")
    print(f"   Token: {result.tokens}")


async def test_adaptive_mode():
    """测试 ADAPTIVE 模式（自动选择）"""
    print("\n" + "=" * 60)
    print("测试 ADAPTIVE 模式 (自动选择)")
    print("=" * 60)
    
    agent = await UnifiedBackgroundAgent.create(
        mode=ExecutionMode.ADAPTIVE,
        yolo=True,
    )
    
    # 复杂任务，应该自动选择 PROACTIVE
    prompt = "分析这个复杂项目的所有Python文件，找出潜在的性能问题"
    print(f"\n📝 输入: {prompt}\n")
    
    result = await agent.run(prompt)
    
    print(f"🤖 回复:\n{result.text[:800]}...")
    print(f"\n📊 统计:")
    print(f"   实际使用模式: {result.mode}")
    print(f"   步骤: {len(result.steps)}")
    print(f"   工具调用: {len(result.tool_calls)} 次")


async def test_mode_switching():
    """测试模式切换"""
    print("\n" + "=" * 60)
    print("测试模式切换")
    print("=" * 60)
    
    agent = await UnifiedBackgroundAgent.create(
        mode=ExecutionMode.REACTIVE,
        yolo=True,
    )
    
    # 第一轮：REACTIVE
    print("\n📝 第一轮 (REACTIVE): 你好")
    result1 = await agent.run("你好")
    print(f"   模式: {result1.mode}, 回复长度: {len(result1.text)}")
    
    # 切换到 PROACTIVE
    agent.set_mode(AgentMode.PROACTIVE)
    print("\n📝 第二轮 (PROACTIVE): 分析项目结构")
    result2 = await agent.run("分析当前目录结构")
    print(f"   模式: {result2.mode}, 步骤: {len(result2.steps)}")
    
    # 切换到 ADAPTIVE
    agent.set_mode(AgentMode.ADAPTIVE)
    print("\n📝 第三轮 (ADAPTIVE): 简单问候")
    result3 = await agent.run("今天天气如何？")
    print(f"   实际模式: {result3.mode}")


async def test_stream_output():
    """测试流式输出"""
    print("\n" + "=" * 60)
    print("测试流式输出 (PROACTIVE)")
    print("=" * 60)
    
    agent = await UnifiedBackgroundAgent.create(
        mode=ExecutionMode.PROACTIVE,
        yolo=True,
    )
    
    prompt = "讲一个短故事"
    print(f"\n📝 输入: {prompt}")
    print("🤖 回复: ", end="", flush=True)
    
    async for text in agent.run_stream(prompt):
        print(text, end="", flush=True)
    
    print("\n")


async def test_parallel_tools():
    """测试并行工具调用"""
    print("\n" + "=" * 60)
    print("测试并行工具调用")
    print("=" * 60)
    
    agent = await UnifiedBackgroundAgent.create(
        mode=ExecutionMode.PROACTIVE,
        yolo=True,
    )
    
    # 明确需要并行处理的任务
    prompt = """请并行执行以下任务：
1. 查看 README.md 文件内容
2. 查看 pyproject.toml 文件内容  
3. 列出 src/kimi_cli 目录下的文件
然后总结项目结构。
"""
    print(f"\n📝 输入: {prompt}\n")
    
    import time
    start = time.time()
    
    result = await agent.run(prompt)
    
    elapsed = time.time() - start
    
    print(f"🤖 回复:\n{result.text}")
    print(f"\n📊 统计:")
    print(f"   执行时间: {elapsed:.2f}s")
    print(f"   工具调用: {len(result.tool_calls)} 次")
    print(f"   步骤: {len(result.steps)}")


async def main():
    """主函数 - 运行所有测试"""
    import sys
    
    # 获取命令行参数
    test_name = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    tests = {
        "reactive": test_reactive_mode,
        "proactive": test_proactive_mode,
        "adaptive": test_adaptive_mode,
        "switch": test_mode_switching,
        "stream": test_stream_output,
        "parallel": test_parallel_tools,
    }
    
    if test_name == "all":
        # 运行所有测试
        for name, test_func in tests.items():
            try:
                await test_func()
            except Exception as e:
                print(f"\n❌ 测试 {name} 失败: {e}")
                import traceback
                traceback.print_exc()
    elif test_name in tests:
        await tests[test_name]()
    else:
        print(f"未知测试: {test_name}")
        print(f"可用测试: {', '.join(tests.keys())}, all")


if __name__ == "__main__":
    asyncio.run(main())
