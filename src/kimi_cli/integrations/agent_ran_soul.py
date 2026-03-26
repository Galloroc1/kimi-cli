from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from kosong.message import TextPart, ThinkPart, ToolCall
from kosong.tooling import ToolError, ToolOk, ToolResult

from kimi_cli.llm import ALL_MODEL_CAPABILITIES, ModelCapability
from kimi_cli.soul import StatusSnapshot, wire_send
from kimi_cli.tools.display import TodoDisplayBlock, TodoDisplayItem
from kimi_cli.wire.types import ContentPart, StepBegin, TurnBegin, TurnEnd


def _ensure_agent_ran_on_path() -> None:
    try:
        import agents  # noqa: F401
        import ran  # noqa: F401

        return
    except Exception:
        pass

    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "agent-ran"
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            return


_ensure_agent_ran_on_path()

from ran.ran.agent.base_agent.base import AgentBaseInputParams, BaseFunctionCallingAgent
from ran.ran.model.llm import OpenAiModel
from ran.ran.runner.env import RunnerEnv


class AgentRanSoul:
    def __init__(
        self,
        agent_cls: type[BaseFunctionCallingAgent],
        *,
        base_url: str,
        model_name: str,
        api_key: str,
    ) -> None:
        model = OpenAiModel(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
        )
        env = RunnerEnv(llm=model, is_think=False, is_stream=True, show_logger=False)
        self._agent = agent_cls(runner_env=env)

    @property
    def name(self) -> str:
        return "AgentRan"

    @property
    def model_name(self) -> str:
        return "agent-ran"

    @property
    def model_capabilities(self) -> set[ModelCapability]:
        return ALL_MODEL_CAPABILITIES

    @property
    def thinking(self) -> bool | None:
        return None

    @property
    def status(self) -> StatusSnapshot:
        return StatusSnapshot(context_usage=0.0)

    @property
    def available_slash_commands(self):
        return []

    async def run(self, user_input: str | list[ContentPart]) -> None:
        sent_tool_calls: set[str] = set()

        if isinstance(user_input, list):
            user_text = "".join(
                part.text for part in user_input if isinstance(part, TextPart)
            )
        else:
            user_text = user_input

        wire_send(TurnBegin(user_input=user_input))
        wire_send(StepBegin(n=1))

        async for msg in self._agent.call(AgentBaseInputParams(task=user_text)):
            if msg.reasoning_content:
                wire_send(ThinkPart(think=msg.reasoning_content))

            if msg.content:
                wire_send(TextPart(text=msg.content))

            tool_calls = msg.tool_calls
            if not tool_calls:
                continue

            if not isinstance(tool_calls, list):
                continue

            for tool in tool_calls:
                if not tool:
                    continue
                tool_id = tool.id or ""

                args = tool.arguments
                if hasattr(args, "system_log"):
                    args = getattr(args, "system_log", "")
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)

                if tool_id and tool_id not in sent_tool_calls:
                    wire_send(
                        ToolCall(
                            id=tool_id,
                            function=ToolCall.FunctionBody(
                                name=tool.name,
                                arguments=args,
                            ),
                        )
                    )
                    sent_tool_calls.add(tool_id)

                if not tool.status:
                    continue

                if tool.result is None:
                    continue

                result_output = tool.result
                if hasattr(result_output, "system_log"):
                    result_output = getattr(result_output, "system_log", "")
                if isinstance(result_output, (dict, list)):
                    result_output = json.dumps(result_output, ensure_ascii=False)

                if tool.is_success:
                    rv = ToolOk(output=str(result_output), brief=f"{tool.name} ok")
                else:
                    rv = ToolError(message=str(result_output), brief=f"{tool.name} failed")

                if tool.is_success and tool.name == "ToDoList":
                    items: list[TodoDisplayItem] = []
                    args = tool.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = None
                    if isinstance(args, dict):
                        tasks = args.get("tasks")
                        if isinstance(tasks, list):
                            for task in tasks:
                                if isinstance(task, str) and task.strip():
                                    items.append(
                                        TodoDisplayItem(
                                            title=task.strip(),
                                            status="pending",
                                        )
                                    )
                    if items:
                        rv.display.append(TodoDisplayBlock(items=items))

                wire_send(ToolResult(tool_call_id=tool_id, return_value=rv))

        wire_send(TurnEnd())


def build_agent_ran_soul() -> AgentRanSoul:
    base_url = os.getenv("AGENT_RAN_BASE_URL", "").strip()
    model_name = os.getenv("AGENT_RAN_MODEL", "").strip()
    api_key = os.getenv("AGENT_RAN_API_KEY", "").strip()

    missing = [
        name
        for name, value in [
            ("AGENT_RAN_BASE_URL", base_url),
            ("AGENT_RAN_MODEL", model_name),
            ("AGENT_RAN_API_KEY", api_key),
        ]
        if not value
    ]
    if missing:
        raise ValueError(
            "Missing required environment variables for agent-ran: "
            + ", ".join(missing)
        )

    from agents.agent import Pi

    return AgentRanSoul(
        Pi,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
    )
