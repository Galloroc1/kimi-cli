"""Tests for graph engine components."""

from __future__ import annotations

import pytest

from kimi_cli.soul.unified.graph_engine import (
    TaskNode,
    TaskGraph,
    GraphScheduler,
    ExecutionPlan,
    PlanStep,
)


class TestTaskNode:
    """Test TaskNode dataclass."""
    
    def test_task_node_creation(self):
        """Test creating a TaskNode."""
        node = TaskNode(
            id="node_1",
            description="Test node",
            tool="test_tool",
            args={"arg1": "value1"},
            dependencies=["dep1", "dep2"],
        )
        
        assert node.id == "node_1"
        assert node.description == "Test node"
        assert node.tool == "test_tool"
        assert node.args == {"arg1": "value1"}
        assert node.dependencies == ["dep1", "dep2"]
    
    def test_task_node_defaults(self):
        """Test TaskNode default values."""
        node = TaskNode(
            id="node_1",
            description="Test node",
            tool=None,
        )
        
        assert node.args == {}
        assert node.dependencies == []
    
    def test_task_node_hash(self):
        """Test TaskNode hashing (based on id)."""
        node1 = TaskNode(id="node_1", description="Test", tool=None)
        node2 = TaskNode(id="node_1", description="Different", tool=None)
        node3 = TaskNode(id="node_2", description="Test", tool=None)
        
        # Same id should have same hash
        assert hash(node1) == hash(node2)
        # Different id should have different hash
        assert hash(node1) != hash(node3)


class TestTaskGraph:
    """Test TaskGraph functionality."""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample task graph."""
        graph = TaskGraph()
        
        # Add independent nodes
        graph.add_node(TaskNode(
            id="a",
            description="Task A",
            tool=None,
            dependencies=[],
        ))
        graph.add_node(TaskNode(
            id="b",
            description="Task B",
            tool=None,
            dependencies=[],
        ))
        
        # Add dependent node
        graph.add_node(TaskNode(
            id="c",
            description="Task C",
            tool=None,
            dependencies=["a", "b"],
        ))
        
        return graph
    
    def test_graph_creation(self, sample_graph):
        """Test graph creation."""
        assert len(sample_graph.nodes) == 3
        assert "a" in sample_graph.nodes
        assert "b" in sample_graph.nodes
        assert "c" in sample_graph.nodes
    
    def test_get_node(self, sample_graph):
        """Test getting nodes by id."""
        node = sample_graph.get_node("a")
        assert node is not None
        assert node.description == "Task A"
        
        missing = sample_graph.get_node("missing")
        assert missing is None
    
    def test_get_ready_nodes_initial(self, sample_graph):
        """Test getting ready nodes (no dependencies)."""
        ready = sample_graph.get_ready_nodes()
        
        # Initially a and b should be ready (no dependencies)
        ready_ids = {n.id for n in ready}
        assert ready_ids == {"a", "b"}
        assert "c" not in ready_ids  # c depends on a and b
    
    def test_get_ready_nodes_after_completion(self, sample_graph):
        """Test getting ready nodes after marking some complete."""
        # Mark a as complete
        sample_graph.mark_complete("a", "result_a")
        
        ready = sample_graph.get_ready_nodes()
        ready_ids = {n.id for n in ready}
        
        # b should still be ready
        assert "b" in ready_ids
        # c is not ready yet (still needs b)
        assert "c" not in ready_ids
        
        # Mark b as complete
        sample_graph.mark_complete("b", "result_b")
        
        ready = sample_graph.get_ready_nodes()
        ready_ids = {n.id for n in ready}
        
        # Now c should be ready
        assert "c" in ready_ids
    
    def test_mark_complete(self, sample_graph):
        """Test marking nodes as complete."""
        sample_graph.mark_complete("a", "result_a")
        
        assert "a" in sample_graph.completed
        assert sample_graph.results["a"] == "result_a"
    
    def test_is_complete(self, sample_graph):
        """Test checking if graph is complete."""
        assert not sample_graph.is_complete()
        
        sample_graph.mark_complete("a", "r1")
        sample_graph.mark_complete("b", "r2")
        sample_graph.mark_complete("c", "r3")
        
        assert sample_graph.is_complete()
    
    def test_get_leaf_nodes(self, sample_graph):
        """Test getting leaf nodes (no dependents)."""
        leaves = sample_graph.get_leaf_nodes()
        leaf_ids = {n.id for n in leaves}
        
        # c is the only leaf (no one depends on it)
        assert leaf_ids == {"c"}
    
    def test_from_dict(self):
        """Test creating graph from dictionary."""
        data = {
            "nodes": [
                {
                    "id": "node_1",
                    "description": "Test node",
                    "tool": "test_tool",
                    "args": {"arg": "value"},
                    "dependencies": [],
                }
            ]
        }
        
        graph = TaskGraph.from_dict(data)
        
        assert len(graph.nodes) == 1
        node = graph.get_node("node_1")
        assert node.description == "Test node"
        assert node.tool == "test_tool"
        assert node.args == {"arg": "value"}
    
    def test_to_dict(self, sample_graph):
        """Test converting graph to dictionary."""
        sample_graph.mark_complete("a", "result_a")
        
        data = sample_graph.to_dict()
        
        assert "nodes" in data
        assert "completed" in data
        assert "results" in data
        assert "a" in data["completed"]
        assert data["results"]["a"] == "result_a"


class TestGraphScheduler:
    """Test GraphScheduler functionality."""
    
    @pytest.mark.asyncio
    async def test_schedule_simple_graph(self):
        """Test scheduling a simple graph."""
        graph = TaskGraph()
        graph.add_node(TaskNode(id="a", description="Task A", tool=None))
        graph.add_node(TaskNode(id="b", description="Task B", tool=None, dependencies=["a"]))
        
        execution_order = []
        
        async def executor(node: TaskNode) -> str:
            execution_order.append(node.id)
            return f"result_{node.id}"
        
        scheduler = GraphScheduler(max_parallel=2)
        await scheduler.schedule(graph, executor)
        
        # Both should be executed
        assert "a" in execution_order
        assert "b" in execution_order
        # a should come before b (dependency)
        assert execution_order.index("a") < execution_order.index("b")
    
    @pytest.mark.asyncio
    async def test_schedule_parallel_execution(self):
        """Test that independent nodes execute in parallel."""
        graph = TaskGraph()
        graph.add_node(TaskNode(id="a", description="Task A", tool=None))
        graph.add_node(TaskNode(id="b", description="Task B", tool=None))
        graph.add_node(TaskNode(id="c", description="Task C", tool=None))
        
        execution_times = {}
        import asyncio
        
        async def executor(node: TaskNode) -> str:
            execution_times[node.id] = asyncio.get_event_loop().time()
            await asyncio.sleep(0.01)  # Small delay
            return f"result_{node.id}"
        
        scheduler = GraphScheduler(max_parallel=3)
        start_time = asyncio.get_event_loop().time()
        await scheduler.schedule(graph, executor)
        end_time = asyncio.get_event_loop().time()
        
        # All should complete (parallel execution)
        assert len(execution_times) == 3
        # Should be faster than sequential (3 * 0.01 = 0.03)
        assert end_time - start_time < 0.05
    
    @pytest.mark.asyncio
    async def test_schedule_deadlock_detection(self):
        """Test deadlock detection."""
        graph = TaskGraph()
        # Create circular dependency
        graph.add_node(TaskNode(id="a", description="Task A", tool=None, dependencies=["b"]))
        graph.add_node(TaskNode(id="b", description="Task B", tool=None, dependencies=["a"]))
        
        async def executor(node: TaskNode) -> str:
            return f"result_{node.id}"
        
        scheduler = GraphScheduler(max_parallel=2)
        
        with pytest.raises(RuntimeError, match="Deadlock detected"):
            await scheduler.schedule(graph, executor)


class TestExecutionPlan:
    """Test ExecutionPlan functionality."""
    
    def test_plan_creation(self):
        """Test creating an execution plan."""
        plan = ExecutionPlan(
            goal="Test goal",
            steps=[
                PlanStep(description="Step 1", tool="tool1"),
                PlanStep(description="Step 2", tool="tool2", depends_on=[0]),
            ],
        )
        
        assert plan.goal == "Test goal"
        assert len(plan.steps) == 2
    
    def test_plan_to_task_graph(self):
        """Test converting plan to task graph."""
        plan = ExecutionPlan(
            goal="Test goal",
            steps=[
                PlanStep(description="Step 1", tool="tool1"),
                PlanStep(description="Step 2", tool="tool2", depends_on=[0]),
            ],
        )
        
        graph = plan.to_task_graph()
        
        assert len(graph.nodes) == 2
        # Step 0 has no dependencies
        assert graph.get_node("step_0").dependencies == []
        # Step 1 depends on step 0
        assert graph.get_node("step_1").dependencies == ["step_0"]


class TestPlanStep:
    """Test PlanStep dataclass."""
    
    def test_step_creation(self):
        """Test creating a plan step."""
        step = PlanStep(
            description="Test step",
            tool="test_tool",
            args={"arg": "value"},
            depends_on=[0, 1],
            can_parallelize=True,
        )
        
        assert step.description == "Test step"
        assert step.tool == "test_tool"
        assert step.args == {"arg": "value"}
        assert step.depends_on == [0, 1]
        assert step.can_parallelize is True
    
    def test_step_defaults(self):
        """Test PlanStep default values."""
        step = PlanStep(description="Test step")
        
        assert step.tool is None
        assert step.args == {}
        assert step.depends_on == []
        assert step.can_parallelize is False
