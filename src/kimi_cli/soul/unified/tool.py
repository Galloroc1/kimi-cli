"""
Unified Tool System - Merges Kimi-CLI's CallableTool2 with Agent Framework's Tool

Key design decisions:
1. Use kosong's CallableTool2 as the base interface (it's more mature)
2. Add Agent Framework's metadata and conversion methods as mixins
3. Support both sync and async tool execution
4. Enable tool result to natural language conversion
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, get_type_hints

from kosong.tooling import CallableTool2, ToolOk, ToolError, ToolReturnValue
from pydantic import BaseModel, create_model


T = TypeVar('T', bound=BaseModel)


@dataclass
class ToolMetadata:
    """Extended metadata for unified tools."""
    name: str
    description: str
    
    # For Agent Framework compatibility
    input_schema: type[BaseModel] | None = None
    output_schema: type[BaseModel] | None = None
    
    # For result conversion
    result_converter: Callable[[Any], str] | None = None
    
    # For MCP compatibility
    register_mcp: bool = True


class UnifiedTool(CallableTool2[T], ABC):
    """
    Abstract base class for unified tools.
    
    Combines:
    - kosong's CallableTool2 (execution interface)
    - Agent Framework's metadata and conversion methods
    
    Subclasses must define:
    - params: type[T] (the Pydantic model for parameters)
    - name: str
    - description: str
    """
    
    # Tool metadata
    register_mcp: bool = True
    """Whether to register this tool with MCP."""
    
    output_schema: type[BaseModel] | None = None
    """Optional output schema for structured results."""
    
    def __init__(self):
        super().__init__(description=self.description)
    
    @abstractmethod
    async def execute(self, params: T) -> Any:
        """
        Execute the tool with validated parameters.
        
        Args:
            params: Validated Pydantic model
            
        Returns:
            Tool execution result (any type)
        """
        raise NotImplementedError
    
    async def __call__(self, params: T) -> ToolReturnValue:
        """
        Main entry point - wraps execute() with error handling.
        
        This method:
        1. Calls execute()
        2. Converts result to string if needed
        3. Returns ToolOk or ToolError
        """
        try:
            result = await self.execute(params)
            
            # Convert result to string for LLM consumption
            if isinstance(result, str):
                output = result
            elif isinstance(result, BaseModel):
                output = self._convert_pydantic_result(result)
            elif result is None:
                output = "Tool executed successfully."
            else:
                output = self.result_to_nl(result)
            
            return ToolOk(output=output)
            
        except Exception as e:
            return ToolError(
                message=f"Tool execution failed: {e}",
                brief=str(e),
            )
    
    def result_to_nl(self, result: Any) -> str:
        """
        Convert tool result to natural language.
        
        Override this method for custom result formatting.
        Default implementation uses str() or json.dumps().
        
        Args:
            result: Tool execution result
            
        Returns:
            Natural language string for LLM
        """
        if isinstance(result, BaseModel):
            return self._convert_pydantic_result(result)
        
        try:
            import json
            return json.dumps(result, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(result)
    
    def _convert_pydantic_result(self, result: BaseModel) -> str:
        """Convert Pydantic model to natural language."""
        # If the model has a custom __str__, use it
        if result.__class__.__str__ is not BaseModel.__str__:
            return str(result)
        
        # Otherwise, format as key-value pairs
        fields = []
        for key, value in result.model_dump().items():
            if isinstance(value, (dict, list)):
                import json
                value_str = json.dumps(value, ensure_ascii=False)
            else:
                value_str = str(value)
            fields.append(f"{key}: {value_str}")
        
        return "\n".join(fields)
    
    # === Conversion methods for different frameworks ===
    
    def to_mcp_tool(self) -> dict[str, Any]:
        """Convert to MCP tool format."""
        from mcp.types import Tool as MCPTool
        
        return MCPTool(
            name=self.name,
            description=self.description,
            inputSchema=self.params.model_json_schema(),
        )
    
    def to_langchain_tool(self) -> Any:
        """Convert to LangChain tool format."""
        from langchain_core.tools import StructuredTool
        
        async def _func(**kwargs):
            params = self.params(**kwargs)
            result = await self.execute(params)
            return self.result_to_nl(result)
        
        return StructuredTool.from_function(
            coroutine=_func,
            name=self.name,
            description=self.description,
            args_schema=self.params,
        )


def unified_tool(
    name: str | None = None,
    description: str | None = None,
    register_mcp: bool = True,
) -> Callable[[Callable], type[UnifiedTool]]:
    """
    Decorator to create a UnifiedTool from a function.
    
    Example:
        @unified_tool(name="calculator", description="Perform calculations")
        async def calculator(params: CalculatorParams) -> int:
            return params.a + params.b
    """
    def decorator(func: Callable) -> type[UnifiedTool]:
        # Build input model from function signature
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        fields = {}
        for param_name, param in sig.parameters.items():
            ann = type_hints.get(param_name, str)
            default = param.default if param.default != inspect.Parameter.empty else ...
            fields[param_name] = (ann, default)
        
        ParamsModel = create_model(f"{func.__name__}Params", **fields)
        
        # Determine return type for output schema
        return_type = type_hints.get('return', Any)
        
        # Capture values in local variables to avoid closure issues
        _func = func
        _name = name or func.__name__
        _description = description or (func.__doc__ or "")
        _register_mcp = register_mcp
        
        class ConcreteTool(UnifiedTool):
            params = ParamsModel
            name = _name
            description = _description
            register_mcp = _register_mcp
            
            async def execute(self, params: ParamsModel) -> Any:
                # Convert params back to kwargs
                kwargs = params.model_dump()
                result = await _func(**kwargs)
                return result
        
        # Copy function metadata
        ConcreteTool.__module__ = _func.__module__
        ConcreteTool.__doc__ = _func.__doc__
        
        return ConcreteTool
    
    return decorator


# === Tool Registry ===

class ToolRegistry:
    """Registry for unified tools."""
    
    def __init__(self):
        self._tools: dict[str, type[UnifiedTool]] = {}
    
    def register(self, tool_cls: type[UnifiedTool]) -> type[UnifiedTool]:
        """Register a tool class."""
        self._tools[tool_cls.name] = tool_cls
        return tool_cls
    
    def get(self, name: str) -> type[UnifiedTool] | None:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> list[type[UnifiedTool]]:
        """List all registered tools."""
        return list(self._tools.values())
    
    def create_instance(self, name: str) -> UnifiedTool | None:
        """Create a tool instance by name."""
        tool_cls = self._tools.get(name)
        if tool_cls:
            return tool_cls()
        return None


# Global registry
unified_tool_registry = ToolRegistry()


def register_tool(tool_cls: type[UnifiedTool]) -> type[UnifiedTool]:
    """Decorator to register a tool in the global registry."""
    return unified_tool_registry.register(tool_cls)
