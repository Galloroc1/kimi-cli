# Kimi CLI - Agent 实现与 AI 算法深度解析

> 本文档面向算法工程师，深入剖析 Kimi CLI 的 Agent 架构、上下文管理、压缩算法等核心 AI 算法

---

## 目录

1. [架构总览](#1-架构总览)
2. [Agent 核心架构](#2-agent-核心架构)
3. [ReAct 循环实现](#3-react-循环实现)
4. [上下文管理系统](#4-上下文管理系统)
5. [上下文压缩算法](#5-上下文压缩算法)
6. [工具调用系统](#6-工具调用系统)
7. [审批与安全机制](#7-审批与安全机制)
8. [D-Mail 时间旅行机制](#8-d-mail-时间旅行机制)
9. [重试与容错机制](#9-重试与容错机制)
10. [多 Agent 协作](#10-多-agent-协作)

---

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                        User Input                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  KimiSoul (Agent Loop)                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   _turn()   │──│  _step()    │──│   _agent_loop()     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Context   │ │  Toolset    │ │   Runtime   │
│  (历史管理)  │ │  (工具执行)  │ │  (运行时环境)│
└─────────────┘ └─────────────┘ └─────────────┘
```

---

## 2. Agent 核心架构

### 2.1 核心类层次

```python
# agent.py - Runtime 运行时环境
@dataclass(slots=True, kw_only=True)
class Runtime:
    config: Config                    # 配置管理
    oauth: OAuthManager              # OAuth 认证
    llm: LLM | None                  # LLM 实例
    session: Session                 # 会话管理
    builtin_args: BuiltinSystemPromptArgs  # 系统提示词参数
    denwa_renji: DenwaRenji          # "电话留言"系统 (D-Mail)
    approval: Approval               # 审批系统
    labor_market: LaborMarket        # 子 Agent 劳动力市场
    environment: Environment         # 环境检测
    skills: dict[str, Skill]         # 技能系统
```

```python
# agent.py - Agent 定义
@dataclass(frozen=True, slots=True, kw_only=True)
class Agent:
    name: str                        # Agent 名称
    system_prompt: str               # 系统提示词
    toolset: Toolset                 # 工具集
    runtime: Runtime                 # 运行时环境
```

### 2.2 Runtime 创建流程

```python
@staticmethod
async def create(config, oauth, llm, session, yolo, skills_dir) -> Runtime:
    # 1. 并行加载环境信息
    ls_output, agents_md, environment = await asyncio.gather(
        list_directory(session.work_dir),      # 工作目录列表
        load_agents_md(session.work_dir),      # AGENTS.md
        Environment.detect(),                   # 环境检测
    )

    # 2. 发现 Skills
    skills_roots = await resolve_skills_roots(session.work_dir, skills_dir)
    skills = await discover_skills_from_roots(skills_roots)

    # 3. 组装 Runtime
    return Runtime(
        config=config,
        oauth=oauth,
        llm=llm,
        session=session,
        builtin_args=BuiltinSystemPromptArgs(
            KIMI_NOW=datetime.now().astimezone().isoformat(),
            KIMI_WORK_DIR=session.work_dir,
            KIMI_WORK_DIR_LS=ls_output,
            KIMI_AGENTS_MD=agents_md or "",
            KIMI_SKILLS=skills_formatted,
        ),
        denwa_renji=DenwaRenji(),
        approval=Approval(yolo=yolo),
        labor_market=LaborMarket(),
        environment=environment,
        skills=skills_by_name,
    )
```

**算法要点**:
- 使用 `asyncio.gather` 并行化独立 IO 操作
- 延迟加载策略：Skills 按需发现
- 依赖注入：通过 `builtin_args` 将环境信息注入系统提示词

---

## 3. ReAct 循环实现

### 3.1 ReAct 模式概述

Kimi CLI 实现了经典的 **ReAct (Reasoning + Acting)** 模式：

```
Thought (推理) → Action (行动/工具调用) → Observation (观察/结果) → ... → Answer
```

### 3.2 主循环实现 (kimisoul.py)

```python
class KimiSoul:
    async def _agent_loop(self) -> TurnOutcome:
        """The main agent loop for one run."""
        step_no = 0
        while True:
            step_no += 1
            if step_no > self._loop_control.max_steps_per_turn:
                raise MaxStepsReached(...)

            wire_send(StepBegin(n=step_no))

            # 1. 检查上下文长度，必要时压缩
            if self._context.token_count + reserved >= self._runtime.llm.max_context_size:
                await self.compact_context()

            # 2. 创建检查点
            await self._checkpoint()

            # 3. 执行单步
            step_outcome = await self._step()

            # 4. 处理结果
            if step_outcome.stop_reason == "no_tool_calls":
                return TurnOutcome(...)

            # 5. 处理 D-Mail (时间旅行)
            if dmail := self._denwa_renji.fetch_pending_dmail():
                raise BackToTheFuture(dmail.checkpoint_id, dmail.messages)
```

### 3.3 单步执行详解

```python
async def _step(self) -> StepOutcome | None:
    """Run a single step and return a stop outcome, or None to continue."""

    # 3.3.1 带重试的 LLM 调用
    @tenacity.retry(
        retry=retry_if_exception(self._is_retryable_error),
        wait=wait_exponential_jitter(initial=0.3, max=5, jitter=0.5),
        stop=stop_after_attempt(self._loop_control.max_retries_per_step),
    )
    async def _kosong_step_with_retry() -> StepResult:
        return await kosong.step(
            chat_provider,              # LLM 提供者
            self._agent.system_prompt,  # 系统提示词
            self._agent.toolset,        # 可用工具
            self._context.history,      # 对话历史
            on_message_part=wire_send,  # 流式回调
            on_tool_result=wire_send,   # 工具结果回调
        )

    result = await _kosong_step_with_retry()

    # 3.3.2 等待工具执行结果
    results = await result.tool_results()

    # 3.3.3 更新上下文 (shield 保护防止中断)
    await asyncio.shield(self._grow_context(result, results))

    # 3.3.4 判断停止条件
    if result.tool_calls:
        return None  # 有工具调用，继续循环
    return StepOutcome(stop_reason="no_tool_calls", ...)
```

### 3.4 上下文增长管理

```python
async def _grow_context(self, result: StepResult, tool_results: list[ToolResult]):
    """将 LLM 输出和工具结果加入上下文"""

    # 1. 转换工具结果为消息
    tool_messages = [tool_result_to_message(tr) for tr in tool_results]

    # 2. 检查模型能力兼容性
    for tm in tool_messages:
        if missing_caps := check_message(tm, self._runtime.llm.capabilities):
            raise LLMNotSupported(self._runtime.llm, list(missing_caps))

    # 3. 追加助手消息
    await self._context.append_message(result.message)

    # 4. 更新 Token 计数
    if result.usage is not None:
        await self._context.update_token_count(result.usage.total)

    # 5. 追加工具消息
    await self._context.append_message(tool_messages)
```

---

## 4. 上下文管理系统

### 4.1 Context 类实现 (context.py)

```python
class Context:
    def __init__(self, file_backend: Path):
        self._file_backend = file_backend  # 持久化文件
        self._history: list[Message] = []   # 内存中的历史
        self._token_count: int = 0          # 当前 Token 数
        self._next_checkpoint_id: int = 0   # 下一个检查点 ID
```

### 4.2 检查点机制

```python
async def checkpoint(self, add_user_message: bool):
    """创建检查点，支持时间旅行回退"""
    checkpoint_id = self._next_checkpoint_id
    self._next_checkpoint_id += 1

    # 写入检查点标记
    async with aiofiles.open(self._file_backend, "a") as f:
        await f.write(json.dumps({"role": "_checkpoint", "id": checkpoint_id}) + "\n")

    if add_user_message:
        await self.append_message(
            Message(role="user", content=[system(f"CHECKPOINT {checkpoint_id}")])
        )
```

### 4.3 时间旅行回退

```python
async def revert_to(self, checkpoint_id: int):
    """回退到指定检查点"""

    # 1. 轮转当前文件 (备份)
    rotated_file_path = await next_available_rotation(self._file_backend)
    await aiofiles.os.replace(self._file_backend, rotated_file_path)

    # 2. 重置内存状态
    self._history.clear()
    self._token_count = 0
    self._next_checkpoint_id = 0

    # 3. 从备份恢复直到目标检查点
    async with (
        aiofiles.open(rotated_file_path) as old_file,
        aiofiles.open(self._file_backend, "w") as new_file,
    ):
        async for line in old_file:
            line_json = json.loads(line)
            # 遇到目标检查点停止
            if line_json["role"] == "_checkpoint" and line_json["id"] == checkpoint_id:
                break
            await new_file.write(line)
            # 恢复内存状态
            if line_json["role"] == "_usage":
                self._token_count = line_json["token_count"]
            elif line_json["role"] == "_checkpoint":
                self._next_checkpoint_id = line_json["id"] + 1
            else:
                message = Message.model_validate(line_json)
                self._history.append(message)
```

**算法要点**:
- 文件轮转：保留历史版本，支持审计
- 增量恢复：只恢复到指定点，丢弃后续
- 状态同步：内存和文件保持一致

---

## 5. 上下文压缩算法

### 5.1 压缩策略概述

当上下文接近模型限制时，触发压缩：

```python
# 触发条件
if self._context.token_count + reserved >= self._runtime.llm.max_context_size:
    await self.compact_context()
```

### 5.2 SimpleCompaction 实现

```python
class SimpleCompaction:
    def __init__(self, max_preserved_messages: int = 2):
        """
        Args:
            max_preserved_messages: 保留最近 N 条消息不压缩
        """
        self.max_preserved_messages = max_preserved_messages
```

### 5.3 消息分割算法

```python
def prepare(self, messages: Sequence[Message]) -> PrepareResult:
    """将消息分为"待压缩"和"保留"两部分"""

    history = list(messages)
    preserve_start_index = len(history)
    n_preserved = 0

    # 从后向前遍历，找到需要保留的消息
    for index in range(len(history) - 1, -1, -1):
        if history[index].role in {"user", "assistant"}:
            n_preserved += 1
            if n_preserved == self.max_preserved_messages:
                preserve_start_index = index
                break

    to_compact = history[:preserve_start_index]   # 待压缩的历史
    to_preserve = history[preserve_start_index:]  # 保留的最近消息

    return PrepareResult(
        compact_message=self._build_compact_message(to_compact),
        to_preserve=to_preserve
    )
```

**算法要点**:
- 滑动窗口：始终保留最近 N 条对话
- 角色过滤：只计算 user/assistant 消息
- 渐进压缩：老历史压缩，保留上下文连贯性

### 5.4 LLM 驱动的压缩

```python
async def compact(self, messages: Sequence[Message], llm: LLM) -> Sequence[Message]:
    compact_message, to_preserve = self.prepare(messages)
    if compact_message is None:
        return to_preserve

    # 调用 LLM 进行摘要
    result = await kosong.step(
        chat_provider=llm.chat_provider,
        system_prompt="You are a helpful assistant that compacts conversation context.",
        toolset=EmptyToolset(),  # 压缩时不使用工具
        history=[compact_message],
    )

    # 构建压缩后的消息序列
    content = [
        system("Previous context has been compacted. Here is the compaction output:")
    ]
    # 过滤掉思考部分，保留核心内容
    content.extend(part for part in result.message.content if not isinstance(part, ThinkPart))

    compacted_messages = [Message(role="user", content=content)]
    compacted_messages.extend(to_preserve)
    return compacted_messages
```

### 5.5 压缩提示词设计

```markdown
# prompts/compact.md

**Compression Priorities (in order):**
1. **Current Task State**: What is being worked on RIGHT NOW
2. **Errors & Solutions**: All encountered errors and their resolutions
3. **Code Evolution**: Final working versions only (remove intermediate attempts)
4. **System Context**: Project structure, dependencies, environment setup
5. **Design Decisions**: Architectural choices and their rationale
6. **TODO Items**: Unfinished tasks and known issues

**Compression Rules:**
- MUST KEEP: Error messages, stack traces, working solutions, current task
- MERGE: Similar discussions into single summary points
- REMOVE: Redundant explanations, failed attempts (keep lessons learned)
- CONDENSE: Long code blocks → keep signatures + key logic only
```

**算法要点**:
- 结构化输出：使用 XML 标签组织压缩结果
- 优先级排序：确保关键信息不丢失
- 代码特殊处理：长代码保留签名+关键逻辑

---

## 6. 工具调用系统

### 6.1 工具集架构

```python
class KimiToolset:
    def __init__(self):
        self._tool_dict: dict[str, ToolType] = {}
        self._mcp_servers: dict[str, MCPServerInfo] = {}
        self._mcp_loading_task: asyncio.Task | None = None
```

### 6.2 工具加载与依赖注入

```python
def load_tools(self, tool_paths: list[str], dependencies: dict[type, Any]) -> None:
    for tool_path in tool_paths:
        tool = self._load_tool(tool_path, dependencies)
        if tool:
            self.add(tool)

@staticmethod
def _load_tool(tool_path: str, dependencies: dict[type, Any]) -> ToolType | None:
    module_name, class_name = tool_path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    tool_cls = getattr(module, class_name)

    # 依赖注入：根据参数类型自动注入
    args = []
    for param in inspect.signature(tool_cls).parameters.values():
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            break  # 遇到 keyword-only 参数停止注入
        if param.annotation not in dependencies:
            raise ValueError(f"Tool dependency not found: {param.annotation}")
        args.append(dependencies[param.annotation])

    return tool_cls(*args)
```

### 6.3 工具调用处理

```python
def handle(self, tool_call: ToolCall) -> HandleResult:
    # 设置当前工具调用上下文
    token = current_tool_call.set(tool_call)
    try:
        if tool_call.function.name not in self._tool_dict:
            return ToolResult(
                tool_call_id=tool_call.id,
                return_value=ToolNotFoundError(tool_call.function.name)
            )

        tool = self._tool_dict[tool_call.function.name]

        # 解析参数
        arguments = json.loads(tool_call.function.arguments or "{}")

        # 异步执行工具
        async def _call():
            ret = await tool.call(arguments)
            return ToolResult(tool_call_id=tool_call.id, return_value=ret)

        return asyncio.create_task(_call())
    finally:
        current_tool_call.reset(token)
```

### 6.4 MCP 工具集成

```python
async def load_mcp_tools(self, mcp_configs: list[MCPConfig], runtime: Runtime) -> None:
    """加载 MCP (Model Context Protocol) 工具"""

    async def _connect_server(server_name, server_info):
        async with server_info.client as client:
            # 获取工具列表
            for tool in await client.list_tools():
                server_info.tools.append(
                    MCPTool(server_name, tool, client, runtime=runtime)
                )

            # 注册到工具集
            for tool in server_info.tools:
                self.add(tool)

    # 并行连接所有服务器
    tasks = [
        asyncio.create_task(_connect_server(name, info))
        for name, info in self._mcp_servers.items()
    ]
    await asyncio.gather(*tasks)
```

---

## 7. 审批与安全机制

### 7.1 Approval 系统

```python
class Approval:
    def __init__(self, yolo: bool = False, state: ApprovalState | None = None):
        self._request_queue = Queue[Request]()  # 审批请求队列
        self._requests: dict[str, tuple[Request, asyncio.Future[bool]]] = {}
        self._state = state or ApprovalState(yolo=yolo)

    async def request(self, sender, action, description, display) -> bool:
        """工具调用前请求审批"""

        # YOLO 模式：自动通过
        if self._state.yolo:
            return True

        # 已自动批准的动作
        if action in self._state.auto_approve_actions:
            return True

        # 创建审批请求
        request = Request(
            id=str(uuid.uuid4()),
            tool_call_id=tool_call.id,
            sender=sender,
            action=action,
            description=description,
            display=display or [],
        )

        # 放入队列等待用户响应
        approved_future = asyncio.Future[bool]()
        self._request_queue.put_nowait(request)
        self._requests[request.id] = (request, approved_future)

        return await approved_future  # 阻塞等待审批
```

### 7.2 审批流程

```python
# kimisoul.py 中的审批管道
async def _pipe_approval_to_wire():
    while True:
        request = await self._approval.fetch_request()

        # 发送到 UI 层
        wire_request = ApprovalRequest(...)
        wire_send(wire_request)

        # 等待用户响应
        resp = await wire_request.wait()
        self._approval.resolve_request(request.id, resp)
        wire_send(ApprovalResponse(request_id=request.id, response=resp))
```

### 7.3 响应类型

```python
type Response = Literal["approve", "approve_for_session", "reject"]

def resolve_request(self, request_id: str, response: Response):
    match response:
        case "approve":
            future.set_result(True)
        case "approve_for_session":
            self._state.auto_approve_actions.add(request.action)  # 会话内自动批准
            future.set_result(True)
        case "reject":
            future.set_result(False)
```

---

## 8. D-Mail 时间旅行机制

### 8.1 概念

D-Mail (电话留言) 是一种**时间旅行**机制，允许 Agent 向过去的自己发送消息：

```python
class DMail(BaseModel):
    message: str        # 消息内容
    checkpoint_id: int  # 目标检查点 (必须 >= 0)
```

### 8.2 实现

```python
class DenwaRenji:
    def __init__(self):
        self._pending_dmail: DMail | None = None
        self._n_checkpoints: int = 0

    def send_dmail(self, dmail: DMail):
        """发送 D-Mail (由 SendDMail 工具调用)"""
        if self._pending_dmail is not None:
            raise DenwaRenjiError("Only one D-Mail can be sent at a time")
        if dmail.checkpoint_id >= self._n_checkpoints:
            raise DenwaRenjiError("Checkpoint does not exist")
        self._pending_dmail = dmail

    def fetch_pending_dmail(self) -> DMail | None:
        """获取待处理的 D-Mail (由 Soul 调用)"""
        pending = self._pending_dmail
        self._pending_dmail = None
        return pending
```

### 8.3 时间旅行触发

```python
# kimisoul.py _step() 方法
if dmail := self._denwa_renji.fetch_pending_dmail():
    raise BackToTheFuture(
        dmail.checkpoint_id,
        [Message(
            role="user",
            content=[system(
                "You just got a D-Mail from your future self. "
                f"D-Mail content:\n\n{dmail.message.strip()}"
            )]
        )]
    )

# _agent_loop() 捕获并处理
try:
    step_outcome = await self._step()
except BackToTheFuture as e:
    await self._context.revert_to(e.checkpoint_id)  # 回退
    await self._checkpoint()
    await self._context.append_message(e.messages)   # 添加 D-Mail
```

**应用场景**:
- 子 Agent 完成工作后通知父 Agent
- 长时间任务的阶段性汇报
- 错误恢复和重试策略

---

## 9. 重试与容错机制

### 9.1 指数退避重试

```python
@tenacity.retry(
    retry=retry_if_exception(self._is_retryable_error),
    wait=wait_exponential_jitter(initial=0.3, max=5, jitter=0.5),
    stop=stop_after_attempt(self._loop_control.max_retries_per_step),
    reraise=True,
)
async def _kosong_step_with_retry() -> StepResult:
    return await kosong.step(...)
```

### 9.2 可重试错误判断

```python
@staticmethod
def _is_retryable_error(exception: BaseException) -> bool:
    # 网络层错误
    if isinstance(exception, (APIConnectionError, APITimeoutError, APIEmptyResponseError)):
        return True

    # HTTP 状态码判断
    return isinstance(exception, APIStatusError) and exception.status_code in (
        429,  # Too Many Requests
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
    )
```

### 9.3 日志记录

```python
@staticmethod
def _retry_log(name: str, retry_state: RetryCallState):
    logger.info(
        "Retrying {name} for the {n} time. Waiting {sleep} seconds.",
        name=name,
        n=retry_state.attempt_number,
        sleep=retry_state.next_action.sleep if retry_state.next_action else "unknown",
    )
```

---

## 10. 多 Agent 协作

### 10.1 LaborMarket (劳动力市场)

```python
class LaborMarket:
    def __init__(self):
        self.fixed_subagents: dict[str, Agent] = {}      # 固定子 Agent
        self.fixed_subagent_descs: dict[str, str] = {}
        self.dynamic_subagents: dict[str, Agent] = {}    # 动态子 Agent

    def add_fixed_subagent(self, name: str, agent: Agent, description: str):
        """添加固定子 Agent (预定义)"""
        self.fixed_subagents[name] = agent
        self.fixed_subagent_descs[name] = description

    def add_dynamic_subagent(self, name: str, agent: Agent):
        """添加动态子 Agent (运行时创建)"""
        self.dynamic_subagents[name] = agent
```

### 10.2 子 Agent 运行时克隆

```python
def copy_for_fixed_subagent(self) -> Runtime:
    """为固定子 Agent 克隆运行时"""
    return Runtime(
        config=self.config,
        oauth=self.oauth,
        llm=self.llm,
        session=self.session,
        builtin_args=self.builtin_args,
        denwa_renji=DenwaRenji(),           # 独立的 D-Mail 系统
        approval=self.approval.share(),      # 共享审批状态
        labor_market=LaborMarket(),          # 独立的劳动力市场
        environment=self.environment,
        skills=self.skills,
    )

def copy_for_dynamic_subagent(self) -> Runtime:
    """为动态子 Agent 克隆运行时"""
    return Runtime(
        ...
        labor_market=self.labor_market,      # 共享劳动力市场
        ...
    )
```

### 10.3 Agent 加载流程

```python
async def load_agent(agent_file: Path, runtime: Runtime, mcp_configs) -> Agent:
    # 1. 加载 Agent 规范
    agent_spec = load_agent_spec(agent_file)

    # 2. 加载系统提示词
    system_prompt = _load_system_prompt(
        agent_spec.system_prompt_path,
        agent_spec.system_prompt_args,
        runtime.builtin_args,
    )

    # 3. 递归加载子 Agent
    for subagent_name, subagent_spec in agent_spec.subagents.items():
        subagent = await load_agent(
            subagent_spec.path,
            runtime.copy_for_fixed_subagent(),
            mcp_configs=mcp_configs,
        )
        runtime.labor_market.add_fixed_subagent(
            subagent_name, subagent, subagent_spec.description
        )

    # 4. 加载工具集
    toolset = KimiToolset()
    tool_deps = {...}
    toolset.load_tools(agent_spec.tools, tool_deps)

    # 5. 加载 MCP 工具
    if mcp_configs:
        await toolset.load_mcp_tools(mcp_configs, runtime)

    return Agent(...)
```

---

## 11. 关键算法总结

| 算法/机制 | 文件 | 核心思想 |
|-----------|------|----------|
| **ReAct 循环** | `kimisoul.py` | Thought → Action → Observation 循环 |
| **上下文压缩** | `compaction.py` | 滑动窗口 + LLM 摘要 |
| **检查点/回退** | `context.py` | 文件轮转 + 增量恢复 |
| **指数退避** | `kimisoul.py` | 指数增长 + 随机抖动 |
| **依赖注入** | `toolset.py` | 类型注解自动注入 |
| **审批队列** | `approval.py` | 异步 Future + 队列 |
| **D-Mail** | `denwarenji.py` | 异常驱动的时间旅行 |

---

## 12. 学习建议

### 初级 (1-2周)
1. 理解 `KimiSoul._agent_loop()` 主循环逻辑
2. 学习 `Context` 的持久化和恢复机制
3. 掌握 `Approval` 的异步审批流程

### 中级 (3-4周)
1. 深入 `SimpleCompaction` 的压缩策略
2. 理解 `DenwaRenji` 的时间旅行机制
3. 学习 `KimiToolset` 的工具管理

### 高级 (5-6周)
1. 分析 `kosong.step()` 的集成 (外部库)
2. 研究 MCP 协议的实现细节
3. 探索多 Agent 协作的复杂场景

---

*文档生成时间: 2026-02-03*
*项目版本: kimi-cli v1.5*
