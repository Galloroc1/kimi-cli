"""Tests for UnifiedTool."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from kosong.tooling import ToolOk, ToolError

from kimi_cli.soul.unified.tool import (
    UnifiedTool,
    unified_tool,
    ToolRegistry,
    unified_tool_registry,
    register_tool,
)


class SearchParams(BaseModel):
    """Test parameters model."""
    query: str = Field(description="Search query")
    limit: int = Field(default=10, description="Result limit")


class TestUnifiedTool:
    """Test UnifiedTool base class."""
    
    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test basic tool execution."""
        class TestTool(UnifiedTool[SearchParams]):
            name = "test_tool"
            description = "A test tool"
            params = SearchParams
            
            async def execute(self, params: SearchParams) -> dict:
                return {"query": params.query, "results": ["a", "b"]}
        
        tool = TestTool()
        params = SearchParams(query="test", limit=5)
        
        result = await tool(params)
        
        assert isinstance(result, ToolOk)
        assert "query" in result.output
        assert "results" in result.output
    
    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        """Test tool error handling."""
        class ErrorTool(UnifiedTool[SearchParams]):
            name = "error_tool"
            description = "A tool that errors"
            params = SearchParams
            
            async def execute(self, params: SearchParams) -> dict:
                raise ValueError("Something went wrong")
        
        tool = ErrorTool()
        params = SearchParams(query="test")
        
        result = await tool(params)
        
        assert isinstance(result, ToolError)
        assert "Something went wrong" in result.message
    
    @pytest.mark.asyncio
    async def test_tool_result_conversion(self):
        """Test result to natural language conversion."""
        class TestTool(UnifiedTool[SearchParams]):
            name = "test_tool"
            description = "A test tool"
            params = SearchParams
            
            async def execute(self, params: SearchParams) -> dict:
                return {"count": 5, "items": ["a", "b", "c"]}
        
        tool = TestTool()
        params = SearchParams(query="test")
        
        result = await tool(params)
        
        # Should be converted to string
        assert isinstance(result.output, str)
        assert "count" in result.output
    
    @pytest.mark.asyncio
    async def test_tool_string_result(self):
        """Test tool that returns string directly."""
        class StringTool(UnifiedTool[SearchParams]):
            name = "string_tool"
            description = "Returns string"
            params = SearchParams
            
            async def execute(self, params: SearchParams) -> str:
                return f"Results for {params.query}"
        
        tool = StringTool()
        params = SearchParams(query="test")
        
        result = await tool(params)
        
        assert result.output == "Results for test"
    
    @pytest.mark.asyncio
    async def test_tool_none_result(self):
        """Test tool that returns None."""
        class NoneTool(UnifiedTool[SearchParams]):
            name = "none_tool"
            description = "Returns None"
            params = SearchParams
            
            async def execute(self, params: SearchParams) -> None:
                return None
        
        tool = NoneTool()
        params = SearchParams(query="test")
        
        result = await tool(params)
        
        assert "successfully" in result.output.lower()
    
    def test_result_to_nl_pydantic(self):
        """Test result_to_nl with Pydantic model."""
        class ResultModel(BaseModel):
            name: str
            value: int
        
        class TestTool(UnifiedTool[SearchParams]):
            name = "test_tool"
            description = "Test"
            params = SearchParams
            
            async def execute(self, params: SearchParams):
                return ResultModel(name="test", value=42)
        
        tool = TestTool()
        result = ResultModel(name="test", value=42)
        
        nl = tool.result_to_nl(result)
        
        assert "name" in nl
        assert "test" in nl
        assert "value" in nl
        assert "42" in nl
    
    def test_result_to_nl_dict(self):
        """Test result_to_nl with dict."""
        class TestTool(UnifiedTool[SearchParams]):
            name = "test_tool"
            description = "Test"
            params = SearchParams
            
            async def execute(self, params: SearchParams):
                return {"key": "value"}
        
        tool = TestTool()
        result = {"key": "value", "number": 123}
        
        nl = tool.result_to_nl(result)
        
        assert "key" in nl
        assert "value" in nl


class TestUnifiedToolDecorator:
    """Test @unified_tool decorator."""
    
    @pytest.mark.asyncio
    async def test_decorator_basic(self):
        """Test basic decorator usage."""
        @unified_tool(name="search", description="Search tool")
        async def search(query: str, limit: int = 10) -> dict:
            """Search for items."""
            return {"results": ["a", "b"], "count": 2}
        
        # Should return a class
        assert isinstance(search, type)
        assert issubclass(search, UnifiedTool)
        
        # Create instance and test
        tool = search()
        params = tool.params(query="test", limit=5)
        
        result = await tool(params)
        
        assert isinstance(result, ToolOk)
    
    @pytest.mark.asyncio
    async def test_decorator_infer_name(self):
        """Test decorator infers name from function."""
        @unified_tool()
        async def my_custom_tool(query: str) -> str:
            return f"Result for {query}"
        
        assert my_custom_tool.name == "my_custom_tool"
    
    @pytest.mark.asyncio
    async def test_decorator_infer_description(self):
        """Test decorator infers description from docstring."""
        @unified_tool()
        async def documented_tool(query: str) -> str:
            """This is the description."""
            return "result"
        
        assert documented_tool.description == "This is the description."


class TestToolRegistry:
    """Test ToolRegistry."""
    
    def test_registry_creation(self):
        """Test creating a registry."""
        registry = ToolRegistry()
        
        assert len(registry.list_tools()) == 0
    
    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        
        @register_tool
        class TestTool(UnifiedTool[SearchParams]):
            name = "test_tool"
            description = "Test"
            params = SearchParams
            
            async def execute(self, params: SearchParams):
                return {}
        
        # Should be in global registry
        assert unified_tool_registry.get("test_tool") is not None
    
    def test_get_tool(self):
        """Test getting a tool from registry."""
        registry = ToolRegistry()
        
        class TestTool(UnifiedTool[SearchParams]):
            name = "gettable_tool"
            description = "Test"
            params = SearchParams
            
            async def execute(self, params: SearchParams):
                return {}
        
        registry.register(TestTool)
        
        found = registry.get("gettable_tool")
        assert found is TestTool
        
        not_found = registry.get("missing")
        assert not_found is None
    
    def test_create_instance(self):
        """Test creating tool instance from registry."""
        registry = ToolRegistry()
        
        class TestTool(UnifiedTool[SearchParams]):
            name = "instance_tool"
            description = "Test"
            params = SearchParams
            
            async def execute(self, params: SearchParams):
                return {}
        
        registry.register(TestTool)
        
        instance = registry.create_instance("instance_tool")
        assert instance is not None
        assert isinstance(instance, TestTool)
        
        missing = registry.create_instance("missing")
        assert missing is None
    
    def test_list_tools(self):
        """Test listing registered tools."""
        registry = ToolRegistry()
        
        class Tool1(UnifiedTool[SearchParams]):
            name = "tool1"
            description = "Tool 1"
            params = SearchParams
            
            async def execute(self, params: SearchParams):
                return {}
        
        class Tool2(UnifiedTool[SearchParams]):
            name = "tool2"
            description = "Tool 2"
            params = SearchParams
            
            async def execute(self, params: SearchParams):
                return {}
        
        registry.register(Tool1)
        registry.register(Tool2)
        
        tools = registry.list_tools()
        
        assert len(tools) == 2
        assert Tool1 in tools
        assert Tool2 in tools
