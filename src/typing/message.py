from typing import Any, Dict, List, TypedDict
from typing_extensions import NotRequired


class ToolFunction(TypedDict):
    name: str
    arguments: str


class ToolCall(TypedDict):
    id: str
    type: str
    function: ToolFunction


class Message(TypedDict):
    role: str
    content: str | None
    name: NotRequired[str]
    tool_calls: NotRequired[List[ToolCall]]
    tool_call_id: NotRequired[str]
    meta: NotRequired[Dict[str, Any]]
