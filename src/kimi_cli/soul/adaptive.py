"""Adaptive execution with dynamic DAG generation.

This module implements a hierarchical planning and execution system:
- Level 1: Planner generates a high-level stage DAG
- Level 2: StageExecutor uses ReAct to execute each stage
- Level 3: ExecutionController coordinates and handles transitions
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from kosong.message import ContentPart, Message
from kosong.tooling import Toolset

from kimi_cli.soul import get_wire_or_none
from kimi_cli.soul.context import Context
from kimi_cli.utils.logging import logger
from kimi_cli.wire.types import (
    StatusUpdate,
    StepBegin,
    TextPart,
    ToolResult,
)


def _safe_wire_send(msg) -> None:
    """Send message to wire if available, otherwise log."""
    wire = get_wire_or_none()
    if wire is not None:
        wire.soul_side.send(msg)
    else:
        # Fallback: just log the message
        from kimi_cli.wire.types import TextPart
        if isinstance(msg, TextPart):
            logger.info(msg.text)
            print(f"[ADAPTIVE] {msg.text}")
        else:
            logger.debug(f"Wire message (no wire): {type(msg).__name__}")


def _print_debug(title: str, content: str) -> None:
    """Print debug information with clear formatting."""
    print(f"\n{'='*60}")
    print(f"[ADAPTIVE] {title}")
    print(f"{'='*60}")
    print(content)
    print(f"{'='*60}\n")

T = TypeVar("T")

StageStatus = Literal["pending", "running", "completed", "failed", "skipped"]
TransitionType = Literal["next", "retry", "backtrack", "replan", "finish"]


@dataclass(frozen=True, slots=True)
class Stage:
    """A stage in the execution plan."""

    id: str
    name: str
    goal: str
    exit_criteria: str
    max_iterations: int = 10
    tools_hint: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Transition:
    """Transition between stages."""

    from_stage: str
    to_stage: str
    condition: str
    transition_type: TransitionType = "next"


@dataclass(slots=True)
class StageResult:
    """Result of executing a stage."""

    stage_id: str
    status: StageStatus
    summary: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)


@dataclass(slots=True)
class ExecutionPlan:
    """The execution plan as a DAG of stages."""

    stages: dict[str, Stage]
    transitions: dict[str, list[Transition]]
    entry: str
    exits: list[str]


@dataclass(slots=True)
class StepRecord:
    """Record of a single step within a stage."""

    step_no: int
    thought: str
    action: str | None
    action_input: dict[str, Any] | None
    observation: str


class Planner:
    """Generates high-level execution plans."""

    PLANNER_PROMPT = """你是一个任务规划专家。分析用户请求并创建一个结构化的执行计划。

用户请求: {user_input}

上下文:
- 工作目录: {work_dir}
- 可用工具: {tools}

创建一个包含3-7个阶段的执行计划。每个阶段应该包含:
- 一个清晰、具体的目标
- 明确的退出标准
- 相关的工具提示

输出一个JSON对象，结构如下:
{{
    "stages": [
        {{
            "id": "analyze",
            "name": "分析需求",
            "goal": "理解需要完成的任务",
            "exit_criteria": "对任务范围有清晰的理解",
            "tools_hint": ["read_file", "list_dir"],
            "max_iterations": 10
        }}
    ],
    "transitions": [
        {{
            "from": "analyze",
            "to": "implement",
            "condition": "需求已理解"
        }}
    ],
    "entry": "analyze",
    "exits": ["verify"]
}}

指导原则:
1. 从分析/理解阶段开始
2. 包含验证/测试阶段
3. 允许迭代（转换可以向后进行）
4. 保持阶段目标具体且可验证"""

    REPLAN_PROMPT = """当前执行计划需要调整。

当前计划:
{current_plan}

失败的阶段: {failed_stage}
失败原因: {failure_reason}

执行历史:
{history}

调整计划。你可以:
- 添加新阶段
- 修改现有阶段
- 更改转换路径
- 拆分复杂阶段

以相同的JSON格式输出更新后的计划。"""

    def __init__(
        self,
        llm_call: Callable[[str, list[Message]], Awaitable[Message]],
        tools: list[str],
        work_dir: str,
    ):
        self._llm_call = llm_call
        self._tools = tools
        self._work_dir = work_dir

    async def plan(self, user_input: str, context: dict[str, Any]) -> ExecutionPlan:
        """Generate initial execution plan."""
        prompt = self.PLANNER_PROMPT.format(
            user_input=user_input,
            work_dir=self._work_dir,
            tools=", ".join(self._tools),
        )

        _print_debug("[Planner] 输入提示词", f"用户输入: {user_input}\n\n完整提示词:\n{prompt}")

        msgs = [Message(role="user", content=prompt)]
        print(f"\n[ADAPTIVE] [DEBUG] Planner 消息历史 (共 {len(msgs)} 条):")
        for i, msg in enumerate(msgs):
            content_preview = str(msg.content)[:200] if msg.content else "(空)"
            print(f"  [{i}] role={msg.role}, content={content_preview}...")

        import time
        print(f"\n{'='*60}")
        print(f"[ADAPTIVE] [REQUEST] >>> 准备调用模型 (Planner.plan)")
        print(f"[ADAPTIVE] [REQUEST] 任务: 生成执行计划")
        print(f"[ADAPTIVE] [REQUEST] llm_call 类型: {type(self._llm_call)}")
        print(f"[ADAPTIVE] [REQUEST] 时间: {time.strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        start_time = time.time()
        print(f"[ADAPTIVE] [WAITING] 正在等待模型响应，请稍候...")
        try:
            response = await self._llm_call(prompt, msgs)
            elapsed = time.time() - start_time

            print(f"\n{'='*60}")
            print(f"[ADAPTIVE] [RESPONSE] <<< 模型响应已返回 (Planner.plan)")
            print(f"[ADAPTIVE] [RESPONSE] 耗时: {elapsed:.2f} 秒")
            print(f"[ADAPTIVE] [RESPONSE] 响应类型: {type(response)}")
            response_text = response.extract_text()
            print(f"[ADAPTIVE] [RESPONSE] 文本长度: {len(response_text)} 字符")
            print(f"{'='*60}")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"[ADAPTIVE] [ERROR] 模型调用失败! 耗时: {elapsed:.2f} 秒")
            print(f"[ADAPTIVE] [ERROR] 异常: {type(e).__name__}: {e}")
            print(f"{'='*60}")
            import traceback
            traceback.print_exc()
            raise

        print(f"[ADAPTIVE] <<< 模型输出 (Planner.plan) - 输出长度: {len(response_text)} 字符")
        _print_debug("[Planner] 模型原始输出", response_text)

        plan_data = self._parse_plan_json(response_text)

        return self._build_plan(plan_data)

    async def replan(
        self,
        current_plan: ExecutionPlan,
        failed_stage: str,
        failure_reason: str,
        execution_history: list[dict],
    ) -> ExecutionPlan:
        """Adjust plan based on execution feedback."""
        prompt = self.REPLAN_PROMPT.format(
            current_plan=self._plan_to_string(current_plan),
            failed_stage=failed_stage,
            failure_reason=failure_reason,
            history=json.dumps(execution_history, indent=2, ensure_ascii=False),
        )

        _print_debug("[Replanner] 输入提示词", f"失败阶段: {failed_stage}\n失败原因: {failure_reason}\n\n完整提示词:\n{prompt}")

        msgs = [Message(role="user", content=prompt)]
        print(f"\n[ADAPTIVE] [DEBUG] Replanner 消息历史 (共 {len(msgs)} 条):")
        for i, msg in enumerate(msgs):
            content_preview = str(msg.content)[:200] if msg.content else "(空)"
            print(f"  [{i}] role={msg.role}, content={content_preview}...")

        import time
        print(f"\n{'='*60}")
        print(f"[ADAPTIVE] [REQUEST] >>> 准备调用模型 (Planner.replan)")
        print(f"[ADAPTIVE] [REQUEST] 任务: 重新规划")
        print(f"[ADAPTIVE] [REQUEST] 失败阶段: {failed_stage}")
        print(f"[ADAPTIVE] [REQUEST] llm_call 类型: {type(self._llm_call)}")
        print(f"[ADAPTIVE] [REQUEST] 时间: {time.strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        start_time = time.time()
        print(f"[ADAPTIVE] [WAITING] 正在等待模型响应，请稍候...")
        try:
            response = await self._llm_call(prompt, msgs)
            elapsed = time.time() - start_time

            print(f"\n{'='*60}")
            print(f"[ADAPTIVE] [RESPONSE] <<< 模型响应已返回 (Planner.replan)")
            print(f"[ADAPTIVE] [RESPONSE] 耗时: {elapsed:.2f} 秒")
            print(f"[ADAPTIVE] [RESPONSE] 响应类型: {type(response)}")
            response_text = response.extract_text()
            print(f"[ADAPTIVE] [RESPONSE] 文本长度: {len(response_text)} 字符")
            print(f"{'='*60}")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"[ADAPTIVE] [ERROR] 模型调用失败! 耗时: {elapsed:.2f} 秒")
            print(f"[ADAPTIVE] [ERROR] 异常: {type(e).__name__}: {e}")
            print(f"{'='*60}")
            import traceback
            traceback.print_exc()
            raise

        print(f"[ADAPTIVE] <<< 模型输出 (Planner.replan) - 输出长度: {len(response_text)} 字符")
        _print_debug("[Replanner] 模型原始输出", response_text)

        plan_data = self._parse_plan_json(response_text)

        return self._build_plan(plan_data)

    def _parse_plan_json(self, text: str) -> dict:
        """Extract JSON from LLM response."""
        # Try to find JSON in code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        # Try to find JSON object directly
        if not text.strip().startswith("{"):
            json_match = re.search(r"(\{.*\})", text, re.DOTALL)
            if json_match:
                text = json_match.group(1)

        return json.loads(text)

    def _build_plan(self, data: dict) -> ExecutionPlan:
        """Build ExecutionPlan from parsed data."""
        stages = {
            s["id"]: Stage(
                id=s["id"],
                name=s["name"],
                goal=s["goal"],
                exit_criteria=s["exit_criteria"],
                max_iterations=s.get("max_iterations", 10),
                tools_hint=s.get("tools_hint", []),
            )
            for s in data["stages"]
        }

        transitions: dict[str, list[Transition]] = {}
        for t in data.get("transitions", []):
            from_id = t["from"]
            if from_id not in transitions:
                transitions[from_id] = []
            transitions[from_id].append(
                Transition(
                    from_stage=from_id,
                    to_stage=t["to"],
                    condition=t["condition"],
                    transition_type=t.get("transition_type", "next"),
                )
            )

        return ExecutionPlan(
            stages=stages,
            transitions=transitions,
            entry=data["entry"],
            exits=data.get("exits", []),
        )

    def _plan_to_string(self, plan: ExecutionPlan) -> str:
        """Convert plan to string for replanning prompt."""
        stages_str = "\n".join(
            f"  - {s.id}: {s.name} (goal: {s.goal})" for s in plan.stages.values()
        )
        transitions_str = "\n".join(
            f"  - {t.from_stage} -> {t.to_stage}: {t.condition}"
            for ts in plan.transitions.values()
            for t in ts
        )
        return f"Stages:\n{stages_str}\n\nTransitions:\n{transitions_str}"


class StageExecutor:
    """Executes a single stage using ReAct pattern."""

    STAGE_PROMPT = """你正在执行以下阶段:

阶段: {stage_name}
目标: {stage_goal}
退出标准: {exit_criteria}

可用工具:
{tools_list}

工具调用格式:
使用以下格式调用工具:
```tool
tool_name
--param1=value1
--param2=value2
```

重要提示:
- Glob 工具: pattern 不能以 '**' 开头（这是安全限制）。如需递归搜索，请使用 Grep 工具或指定子目录。
- 如果工具返回错误，请阅读错误信息并调整你的调用方式。

步骤历史:
{step_history}

当前状态:
{current_state}

决定你的下一步行动。你可以:
1. 使用工具来取得进展（使用上面的格式）
2. 如果满足退出标准，则完成该阶段

如果完成，输出: <stage_complete>已完成的工作总结</stage_complete>
如果使用工具，请逐步思考并使用正确的工具调用格式。"""

    def __init__(
        self,
        llm_call: Callable[[str, list[Message]], Awaitable[Message]],
        tool_executor: Callable[[str, dict[str, Any]], Awaitable[ToolResult]],
        on_message: Callable[[Any], None] | None = None,
        tools: list[str] | None = None,
    ):
        self._llm_call = llm_call
        self._tool_executor = tool_executor
        self._on_message = on_message
        self._tools = tools or []

    async def execute(
        self,
        stage: Stage,
        context: Context,
        previous_results: dict[str, StageResult],
    ) -> StageResult:
        """Execute a stage using ReAct."""
        step_records: list[StepRecord] = []

        for step_no in range(1, stage.max_iterations + 1):
            print(f"\n[ADAPTIVE] === 阶段 '{stage.name}' - 步骤 {step_no}/{stage.max_iterations} ===")

            if self._on_message:
                self._on_message(StepBegin(n=step_no))
            else:
                _safe_wire_send(StepBegin(n=step_no))

            # Build prompt - 包含完整的步骤历史
            tools_list = "\n".join(f"  - {t}" for t in (stage.tools_hint or self._tools))
            prompt = self.STAGE_PROMPT.format(
                stage_name=stage.name,
                stage_goal=stage.goal,
                exit_criteria=stage.exit_criteria,
                tools_list=tools_list if tools_list else "  (无)",
                step_history=self._format_step_history(step_records),
                current_state="正在执行阶段...",
            )

            _print_debug(f"[StageExecutor] 步骤 {step_no} 输入提示词", prompt)

            # 每次只发送当前 prompt，历史通过 step_history 在 prompt 中体现
            msgs_for_model = [Message(role="user", content=prompt)]
            print(f"\n[ADAPTIVE] [DEBUG] 消息 (共 {len(msgs_for_model)} 条，无历史累积):")
            for i, msg in enumerate(msgs_for_model):
                content_preview = str(msg.content)[:200] if msg.content else "(空)"
                print(f"  [{i}] role={msg.role}, content={content_preview}...")

            import time
            print(f"\n{'='*60}")
            print(f"[ADAPTIVE] [REQUEST] >>> 准备调用模型 (StageExecutor Step {step_no})")
            print(f"[ADAPTIVE] [REQUEST] 阶段: {stage.name}")
            print(f"[ADAPTIVE] [REQUEST] llm_call 类型: {type(self._llm_call)}")
            print(f"[ADAPTIVE] [REQUEST] 时间: {time.strftime('%H:%M:%S')}")
            print(f"{'='*60}")

            # Get LLM response
            start_time = time.time()
            print(f"[ADAPTIVE] [WAITING] 正在等待模型响应，请稍候...")
            try:
                response = await self._llm_call(prompt, msgs_for_model)
                elapsed = time.time() - start_time

                print(f"\n{'='*60}")
                print(f"[ADAPTIVE] [RESPONSE] <<< 模型响应已返回 (StageExecutor Step {step_no})")
                print(f"[ADAPTIVE] [RESPONSE] 耗时: {elapsed:.2f} 秒")
                print(f"[ADAPTIVE] [RESPONSE] 响应类型: {type(response)}")
                response_text = response.extract_text()
                print(f"[ADAPTIVE] [RESPONSE] 文本长度: {len(response_text)} 字符")
                print(f"{'='*60}")
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"\n{'='*60}")
                print(f"[ADAPTIVE] [ERROR] 模型调用失败! 耗时: {elapsed:.2f} 秒")
                print(f"[ADAPTIVE] [ERROR] 异常: {type(e).__name__}: {e}")
                print(f"{'='*60}")
                import traceback
                traceback.print_exc()
                raise

            print(f"[ADAPTIVE] <<< 模型输出 (StageExecutor.execute Step {step_no}) - 输出长度: {len(response_text)} 字符")
            _print_debug(f"[StageExecutor] 步骤 {step_no} 模型原始输出", response_text)

            # Check for stage completion
            complete_match = re.search(
                r"<stage_complete>(.*?)</stage_complete>", response_text, re.DOTALL
            )
            if complete_match:
                summary = complete_match.group(1).strip()
                print(f"[ADAPTIVE] 阶段 '{stage.name}' 已完成: {summary}")
                return StageResult(
                    stage_id=stage.id,
                    status="completed",
                    summary=summary,
                    artifacts=self._extract_artifacts(step_records),
                    messages=[],
                )

            # Parse thought and action
            thought, action_name, action_input = self._parse_action(response_text)
            print(f"[ADAPTIVE] 解析动作: action_name={action_name}, action_input={action_input}")
            if action_name is None:
                print(f"[ADAPTIVE] [WARNING] 未能从响应中解析出工具调用")
                print(f"[ADAPTIVE] [DEBUG] 响应前200字符: {response_text[:200]}...")

            # Execute tool if action specified
            observation = ""
            if action_name and action_name != "think":
                print(f"[ADAPTIVE] 执行工具: {action_name}({action_input})")
                try:
                    tool_result = await self._tool_executor(action_name, action_input or {})
                    # 格式化工具结果
                    from kimi_cli.wire.types import ToolReturnValue
                    ret_val = tool_result.return_value
                    if hasattr(ret_val, 'is_error') and ret_val.is_error:
                        observation = f"ERROR: {getattr(ret_val, 'message', str(ret_val))}\nOutput: {getattr(ret_val, 'output', 'N/A')}"
                    else:
                        observation = f"OK: {getattr(ret_val, 'output', str(ret_val))}"
                    _print_debug(f"[StageExecutor] 工具结果 ({action_name})", observation)
                    if self._on_message:
                        self._on_message(tool_result)
                    else:
                        _safe_wire_send(tool_result)
                except Exception as e:
                    observation = f"Error: {e}"
            else:
                observation = "Thinking..."

            step_records.append(
                StepRecord(
                    step_no=step_no,
                    thought=thought,
                    action=action_name,
                    action_input=action_input,
                    observation=observation,
                )
            )

        # Max iterations reached
        return StageResult(
            stage_id=stage.id,
            status="failed",
            summary=f"Stage exceeded max iterations ({stage.max_iterations})",
            artifacts={"history": step_records},
            messages=[],
        )

    def _format_step_history(self, records: list[StepRecord]) -> str:
        """Format step history for prompt."""
        if not records:
            return "No steps yet."
        lines = []
        for r in records:
            lines.append(f"Step {r.step_no}:")
            # 提取 Thought（工具调用前的内容）
            thought_text = r.thought
            if "```tool" in thought_text:
                thought_text = thought_text.split("```tool")[0].strip()
            lines.append(f"  Thought: {thought_text[:200]}...")
            if r.action:
                lines.append(f"  Action: {r.action}")
                # 检查是否是错误结果
                if "is_error=True" in r.observation:
                    lines.append(f"  Result: ERROR - {r.observation[:300]}...")
                else:
                    lines.append(f"  Result: {r.observation[:200]}...")
            else:
                lines.append(f"  Result: {r.observation[:200]}...")
        return "\n".join(lines)

    def _parse_action(
        self, text: str
    ) -> tuple[str, str | None, dict[str, Any] | None]:
        """Parse thought and action from response.

        Supports multiple tool call formats:
        1. Action: tool_name({"param": "value"})
        2. ```tool\ntool_name\n--param=value\n```
        3. <tool>tool_name</tool>
        """
        # Extract thought (before any tool call)
        thought = text
        action_name = None
        action_input = None

        # Pattern 1: Action: tool_name({"param": "value"})
        tool_match = re.search(
            r"(?:Action:|Tool:|Calling)\s*(\w+)\s*\((.*?)\)", text, re.DOTALL
        )
        if tool_match:
            action_name = tool_match.group(1)
            try:
                action_input = json.loads(tool_match.group(2))
            except json.JSONDecodeError:
                action_input = {"input": tool_match.group(2).strip()}
            return thought, action_name, action_input

        # Pattern 2: ```tool\ntool_name\n--param=value\n``` (kosong format)
        tool_block_match = re.search(
            r"```tool\s*\n(\w+)\s*\n(.*?)\n```", text, re.DOTALL
        )
        if tool_block_match:
            action_name = tool_block_match.group(1)
            params_text = tool_block_match.group(2)
            # Parse --param=value format
            action_input = {}
            for line in params_text.strip().split('\n'):
                line = line.strip()
                if line.startswith('--'):
                    # Format: --param=value or --param value
                    if '=' in line:
                        key, value = line[2:].split('=', 1)
                        action_input[key] = value
                    else:
                        parts = line[2:].split(' ', 1)
                        key = parts[0]
                        value = parts[1] if len(parts) > 1 else ""
                        action_input[key] = value
            return thought, action_name, action_input

        # Pattern 3: <tool>tool_name</tool> (XML format)
        xml_tool_match = re.search(
            r"<tool>(\w+)</tool>", text, re.DOTALL
        )
        if xml_tool_match:
            action_name = xml_tool_match.group(1)
            # Try to extract params from XML attributes or content
            params_match = re.search(
                r"<params>(.*?)</params>", text, re.DOTALL
            )
            if params_match:
                try:
                    action_input = json.loads(params_match.group(1))
                except json.JSONDecodeError:
                    action_input = {"input": params_match.group(1).strip()}
            else:
                action_input = {}
            return thought, action_name, action_input

        return thought, action_name, action_input

    def _extract_artifacts(self, records: list[StepRecord]) -> dict[str, Any]:
        """Extract useful artifacts from execution."""
        artifacts: dict[str, Any] = {"steps": len(records)}
        # Could extract file changes, test results, etc.
        return artifacts


class ExecutionController:
    """Coordinates plan execution with dynamic adaptation."""

    def __init__(
        self,
        planner: Planner,
        executor: StageExecutor,
        max_global_iterations: int = 50,
        max_replans: int = 3,
    ):
        self._planner = planner
        self._executor = executor
        self._max_global_iterations = max_global_iterations
        self._max_replans = max_replans

    async def execute(
        self,
        user_input: str,
        context: Context,
    ) -> StageResult:
        """Execute the full adaptive workflow."""
        _print_debug("[ExecutionController] 执行开始", f"用户输入: {user_input}")

        # Initial planning
        plan = await self._planner.plan(user_input, {})
        _safe_wire_send(TextPart(text=f"已生成包含 {len(plan.stages)} 个阶段的执行计划"))

        # Print plan overview
        plan_overview = "\n".join([
            f"  {i+1}. {s.id}: {s.name} (goal: {s.goal})"
            for i, s in enumerate(plan.stages.values())
        ])
        _print_debug("[ExecutionController] 执行计划", f"入口: {plan.entry}\n出口: {plan.exits}\n\n阶段:\n{plan_overview}")

        # Execution state
        current_stage_id = plan.entry
        stage_results: dict[str, StageResult] = {}
        execution_trace: list[dict] = []
        global_iteration = 0
        replan_count = 0

        while global_iteration < self._max_global_iterations:
            global_iteration += 1

            if current_stage_id not in plan.stages:
                return StageResult(
                    stage_id="error",
                    status="failed",
                    summary=f"Invalid stage: {current_stage_id}",
                )

            stage = plan.stages[current_stage_id]
            _print_debug(f"[ExecutionController] 阶段开始 ({global_iteration}/{self._max_global_iterations})",
                        f"阶段: {stage.name}\n目标: {stage.goal}\n退出标准: {stage.exit_criteria}")
            _safe_wire_send(
                TextPart(text=f"\n[阶段 {global_iteration}] {stage.name}: {stage.goal}")
            )

            # Execute stage
            result = await self._executor.execute(stage, context, stage_results)
            stage_results[current_stage_id] = result

            _print_debug(f"[ExecutionController] 阶段结果 ({stage.name})",
                        f"状态: {result.status}\n总结: {result.summary}")

            execution_trace.append({
                "iteration": global_iteration,
                "stage": current_stage_id,
                "status": result.status,
                "summary": result.summary,
            })

            # Decide next transition
            transition = await self._decide_transition(
                plan, current_stage_id, result, stage_results
            )

            _print_debug("[ExecutionController] 转换决策",
                        f"从: {current_stage_id}\n类型: {transition.transition_type}\n到: {transition.to_stage}\n条件: {transition.condition}")
            _safe_wire_send(TextPart(text=f"转换: {transition.transition_type}"))

            # Handle transition
            if transition.transition_type == "finish":
                _print_debug("[ExecutionController] 执行完成", f"总共执行阶段数: {len(stage_results)}")
                _safe_wire_send(TextPart(text="\n执行成功完成!"))
                return StageResult(
                    stage_id="final",
                    status="completed",
                    summary=self._build_final_summary(stage_results),
                    artifacts={"trace": execution_trace, "results": stage_results},
                )

            elif transition.transition_type == "replan":
                replan_count += 1
                _print_debug("[ExecutionController] 重新规划", f"尝试 {replan_count}/{self._max_replans}")
                if replan_count > self._max_replans:
                    return StageResult(
                        stage_id="error",
                        status="failed",
                        summary="超过最大重新规划尝试次数",
                        artifacts={"trace": execution_trace},
                    )

                _safe_wire_send(TextPart(text=f"\n重新规划中 (尝试 {replan_count})..."))
                plan = await self._planner.replan(
                    plan,
                    current_stage_id,
                    result.summary,
                    execution_trace,
                )
                current_stage_id = plan.entry
                _print_debug("[ExecutionController] 重新规划完成", f"新入口: {plan.entry}")

            elif transition.transition_type == "retry":
                _print_debug("[ExecutionController] 重试阶段", f"将重试: {current_stage_id}")
                # Stay on same stage (will re-execute)
                pass

            elif transition.transition_type == "backtrack":
                _print_debug("[ExecutionController] 回退", f"从: {current_stage_id} -> 到: {transition.to_stage}")
                current_stage_id = transition.to_stage

            else:  # next
                _print_debug("[ExecutionController] 下一阶段", f"从: {current_stage_id} -> 到: {transition.to_stage}")
                current_stage_id = transition.to_stage

        _print_debug("[ExecutionController] 执行失败", f"超过最大迭代次数: {self._max_global_iterations}")
        return StageResult(
            stage_id="error",
            status="failed",
            summary=f"超过最大全局迭代次数 ({self._max_global_iterations})",
            artifacts={"trace": execution_trace},
        )

    async def _decide_transition(
        self,
        plan: ExecutionPlan,
        current_stage: str,
        result: StageResult,
        all_results: dict[str, StageResult],
    ) -> Transition:
        """Decide which transition to take."""
        # If stage failed
        if result.status == "failed":
            # Check if we've tried this stage before
            retry_count = sum(1 for r in all_results.values() if r.stage_id == current_stage)
            if retry_count < 2:
                return Transition(current_stage, current_stage, "Retry failed stage", "retry")
            else:
                return Transition(current_stage, "", "Multiple failures, need replan", "replan")

        # If stage completed and is an exit
        if result.status == "completed" and current_stage in plan.exits:
            return Transition(current_stage, "", "Reached exit stage", "finish")

        # Find matching transition based on result
        available = plan.transitions.get(current_stage, [])

        # Simple heuristic: take first available transition
        # Could be enhanced with LLM-based decision
        if available:
            return available[0]

        # No outgoing transitions - treat as finish
        return Transition(current_stage, "", "No outgoing transitions", "finish")

    def _build_final_summary(self, results: dict[str, StageResult]) -> str:
        """Build final execution summary."""
        lines = ["Execution Summary:", ""]
        for stage_id, result in results.items():
            lines.append(f"- {stage_id}: {result.status}")
            lines.append(f"  {result.summary[:200]}...")
        return "\n".join(lines)
