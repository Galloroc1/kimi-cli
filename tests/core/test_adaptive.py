"""Tests for adaptive planning with dynamic DAG generation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from kosong.message import Message
from kosong.tooling.empty import EmptyToolset

from kimi_cli.soul.adaptive import (
    ExecutionController,
    ExecutionPlan,
    Planner,
    Stage,
    StageExecutor,
    StageResult,
    Transition,
)
from kimi_cli.soul.context import Context


@pytest.fixture
def mock_llm_call():
    """Create a mock LLM call function."""

    async def _mock_llm(prompt: str, messages: list[Message]) -> Message:
        # Return a simple response for testing
        return Message(role="assistant", content="Test response")

    return AsyncMock(side_effect=_mock_llm)


@pytest.fixture
def mock_tool_executor():
    """Create a mock tool executor."""

    async def _mock_tool(name: str, params: dict) -> MagicMock:
        result = MagicMock()
        result.return_value = f"Result of {name}"
        return result

    return AsyncMock(side_effect=_mock_tool)


@pytest.fixture
def planner(mock_llm_call):
    """Create a Planner instance with mock LLM."""
    return Planner(
        llm_call=mock_llm_call,
        tools=["read_file", "write_file", "shell"],
        work_dir="/tmp/test",
    )


@pytest.fixture
def stage_executor(mock_llm_call, mock_tool_executor):
    """Create a StageExecutor instance."""
    return StageExecutor(
        llm_call=mock_llm_call,
        tool_executor=mock_tool_executor,
    )


class TestPlanner:
    """Tests for the Planner class."""

    async def test_plan_generation(self, planner, mock_llm_call):
        """Test that planner generates a valid execution plan."""
        # Mock the LLM to return a valid plan
        plan_json = {
            "stages": [
                {
                    "id": "analyze",
                    "name": "Analyze Task",
                    "goal": "Understand the requirements",
                    "exit_criteria": "Requirements are clear",
                    "tools_hint": ["read_file"],
                    "max_iterations": 5,
                },
                {
                    "id": "implement",
                    "name": "Implement Solution",
                    "goal": "Write the code",
                    "exit_criteria": "Code is written",
                    "tools_hint": ["write_file"],
                    "max_iterations": 10,
                },
            ],
            "transitions": [
                {"from": "analyze", "to": "implement", "condition": "Analysis complete"}
            ],
            "entry": "analyze",
            "exits": ["implement"],
        }
        mock_llm_call.return_value = Message(
            role="assistant",
            content=f"```json\n{json.dumps(plan_json)}\n```",
        )

        plan = await planner.plan("Test task", {})

        assert isinstance(plan, ExecutionPlan)
        assert len(plan.stages) == 2
        assert "analyze" in plan.stages
        assert "implement" in plan.stages
        assert plan.entry == "analyze"
        assert plan.exits == ["implement"]

    async def test_plan_stages_have_correct_types(self, planner, mock_llm_call):
        """Test that generated stages have correct types."""
        plan_json = {
            "stages": [
                {
                    "id": "test",
                    "name": "Test Stage",
                    "goal": "Test goal",
                    "exit_criteria": "Test exit",
                    "max_iterations": 5,
                    "tools_hint": ["tool1", "tool2"],
                }
            ],
            "transitions": [],
            "entry": "test",
            "exits": ["test"],
        }
        mock_llm_call.return_value = Message(
            role="assistant",
            content=json.dumps(plan_json),
        )

        plan = await planner.plan("Test", {})
        stage = plan.stages["test"]

        assert isinstance(stage, Stage)
        assert stage.id == "test"
        assert stage.name == "Test Stage"
        assert stage.goal == "Test goal"
        assert stage.exit_criteria == "Test exit"
        assert stage.max_iterations == 5
        assert stage.tools_hint == ["tool1", "tool2"]

    async def test_replan_after_failure(self, planner, mock_llm_call):
        """Test that replan modifies the plan based on failure."""
        original_plan = ExecutionPlan(
            stages={
                "stage1": Stage(id="stage1", name="Stage 1", goal="Goal 1", exit_criteria="Exit 1"),
            },
            transitions={},
            entry="stage1",
            exits=["stage1"],
        )

        new_plan_json = {
            "stages": [
                {
                    "id": "stage1",
                    "name": "Stage 1",
                    "goal": "Goal 1",
                    "exit_criteria": "Exit 1",
                },
                {
                    "id": "stage2",
                    "name": "Stage 2",
                    "goal": "Goal 2",
                    "exit_criteria": "Exit 2",
                },
            ],
            "transitions": [
                {"from": "stage1", "to": "stage2", "condition": "Stage 1 done"}
            ],
            "entry": "stage1",
            "exits": ["stage2"],
        }
        mock_llm_call.return_value = Message(
            role="assistant",
            content=json.dumps(new_plan_json),
        )

        result = await planner.replan(
            original_plan,
            failed_stage="stage1",
            failure_reason="Need more steps",
            execution_history=[],
        )

        assert len(result.stages) == 2
        assert "stage2" in result.stages


class TestStageExecutor:
    """Tests for the StageExecutor class."""

    async def test_stage_completion(self, stage_executor, mock_llm_call):
        """Test that stage completes when LLM signals completion."""
        mock_llm_call.return_value = Message(
            role="assistant",
            content="<stage_complete>Task completed successfully</stage_complete>",
        )

        stage = Stage(
            id="test",
            name="Test Stage",
            goal="Test goal",
            exit_criteria="Test exit",
            max_iterations=5,
        )

        result = await stage_executor.execute(
            stage=stage,
            context=MagicMock(spec=Context),
            previous_results={},
        )

        assert result.status == "completed"
        assert result.summary == "Task completed successfully"
        assert result.stage_id == "test"

    async def test_stage_max_iterations(self, stage_executor, mock_llm_call):
        """Test that stage fails after max iterations."""
        # LLM never signals completion
        mock_llm_call.return_value = Message(
            role="assistant",
            content="Still working on it...",
        )

        stage = Stage(
            id="test",
            name="Test Stage",
            goal="Test goal",
            exit_criteria="Test exit",
            max_iterations=3,
        )

        result = await stage_executor.execute(
            stage=stage,
            context=MagicMock(spec=Context),
            previous_results={},
        )

        assert result.status == "failed"
        assert "max iterations" in result.summary.lower()

    async def test_stage_tool_execution(self, stage_executor, mock_llm_call, mock_tool_executor):
        """Test that stage executor calls tools correctly."""
        responses = [
            Message(role="assistant", content="Action: read_file({'path': '/tmp/test.txt'})"),
            Message(role="assistant", content="<stage_complete>Done</stage_complete>"),
        ]
        mock_llm_call.side_effect = responses

        stage = Stage(
            id="test",
            name="Test Stage",
            goal="Test goal",
            exit_criteria="Test exit",
            max_iterations=5,
        )

        await stage_executor.execute(
            stage=stage,
            context=MagicMock(spec=Context),
            previous_results={},
        )

        # Check that tool was called
        mock_tool_executor.assert_called_once()
        call_args = mock_tool_executor.call_args
        assert call_args[0][0] == "read_file"


class TestExecutionController:
    """Tests for the ExecutionController class."""

    @pytest.fixture
    def mock_planner(self):
        """Create a mock planner."""
        planner = MagicMock(spec=Planner)
        planner.plan = AsyncMock()
        planner.replan = AsyncMock()
        return planner

    @pytest.fixture
    def mock_executor(self):
        """Create a mock stage executor."""
        executor = MagicMock(spec=StageExecutor)
        executor.execute = AsyncMock()
        return executor

    @pytest.fixture
    def controller(self, mock_planner, mock_executor):
        """Create an ExecutionController with mocks."""
        return ExecutionController(
            planner=mock_planner,
            executor=mock_executor,
            max_global_iterations=10,
            max_replans=2,
        )

    async def test_successful_execution(self, controller, mock_planner, mock_executor):
        """Test successful execution through all stages."""
        # Setup plan
        plan = ExecutionPlan(
            stages={
                "stage1": Stage(id="stage1", name="Stage 1", goal="Goal 1", exit_criteria="Exit 1"),
            },
            transitions={},
            entry="stage1",
            exits=["stage1"],
        )
        mock_planner.plan.return_value = plan

        # Setup stage result
        mock_executor.execute.return_value = StageResult(
            stage_id="stage1",
            status="completed",
            summary="Stage 1 done",
        )

        result = await controller.execute("Test task", MagicMock(spec=Context))

        assert result.status == "completed"
        mock_planner.plan.assert_called_once()
        mock_executor.execute.assert_called_once()

    async def test_stage_retry_on_failure(self, controller, mock_planner, mock_executor):
        """Test that failed stages are retried."""
        plan = ExecutionPlan(
            stages={
                "stage1": Stage(id="stage1", name="Stage 1", goal="Goal 1", exit_criteria="Exit 1"),
            },
            transitions={},
            entry="stage1",
            exits=["stage1"],
        )
        mock_planner.plan.return_value = plan

        # First call fails, second succeeds
        mock_executor.execute.side_effect = [
            StageResult(stage_id="stage1", status="failed", summary="Error"),
            StageResult(stage_id="stage1", status="completed", summary="Success"),
        ]

        result = await controller.execute("Test task", MagicMock(spec=Context))

        assert result.status == "completed"
        assert mock_executor.execute.call_count == 2

    async def test_replan_after_multiple_failures(self, controller, mock_planner, mock_executor):
        """Test that controller replans after multiple failures."""
        original_plan = ExecutionPlan(
            stages={
                "stage1": Stage(id="stage1", name="Stage 1", goal="Goal 1", exit_criteria="Exit 1"),
            },
            transitions={},
            entry="stage1",
            exits=[],
        )
        mock_planner.plan.return_value = original_plan

        # Always fail
        mock_executor.execute.return_value = StageResult(
            stage_id="stage1",
            status="failed",
            summary="Persistent error",
        )

        # New plan after replan
        new_plan = ExecutionPlan(
            stages={
                "stage1": Stage(id="stage1", name="Stage 1", goal="Goal 1", exit_criteria="Exit 1"),
                "stage2": Stage(id="stage2", name="Stage 2", goal="Goal 2", exit_criteria="Exit 2"),
            },
            transitions={"stage1": [Transition("stage1", "stage2", "Done")]},
            entry="stage1",
            exits=["stage2"],
        )
        mock_planner.replan.return_value = new_plan

        # Second stage succeeds
        mock_executor.execute.side_effect = [
            StageResult(stage_id="stage1", status="failed", summary="Error"),  # Try 1
            StageResult(stage_id="stage1", status="failed", summary="Error"),  # Try 2 - triggers replan
            StageResult(stage_id="stage1", status="completed", summary="Success"),  # After replan
        ]

        result = await controller.execute("Test task", MagicMock(spec=Context))

        mock_planner.replan.assert_called_once()

    async def test_max_replans_reached(self, controller, mock_planner, mock_executor):
        """Test that execution fails after max replans."""
        plan = ExecutionPlan(
            stages={
                "stage1": Stage(id="stage1", name="Stage 1", goal="Goal 1", exit_criteria="Exit 1"),
            },
            transitions={},
            entry="stage1",
            exits=[],
        )
        mock_planner.plan.return_value = plan
        mock_planner.replan.return_value = plan  # Same plan after replan

        # Always fail
        mock_executor.execute.return_value = StageResult(
            stage_id="stage1",
            status="failed",
            summary="Always fails",
        )

        result = await controller.execute("Test task", MagicMock(spec=Context))

        assert result.status == "failed"
        assert "replan" in result.summary.lower()


class TestIntegration:
    """Integration tests for the adaptive system."""

    async def test_full_workflow(self):
        """Test the full adaptive workflow end-to-end."""
        # This would be a more comprehensive test with real (mocked) components
        pass
