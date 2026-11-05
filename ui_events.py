from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter


class FileUpdatePayload(BaseModel):
    filepath: str
    content: str


class NodeFinishedPayload(BaseModel):
    node: str


class DonePayload(BaseModel):
    status: str = "done"
    message: str | None = None
    run_plan: dict[str, Any] | None = None
    structure: list[dict[str, Any]] | None = None


class StatusEvent(BaseModel):
    type: Literal["status"]
    data: str
    agent: str | None = None


class ThoughtsEvent(BaseModel):
    type: Literal["thoughts"]
    data: str
    agent: str | None = None


class ToolCallsEvent(BaseModel):
    type: Literal["tool_calls"]
    data: list[dict[str, Any]]
    agent: str | None = None


class NodeFinishedEvent(BaseModel):
    type: Literal["node_finished"]
    data: NodeFinishedPayload
    agent: str | None = None


class FileUpdateEvent(BaseModel):
    type: Literal["file_update"]
    data: FileUpdatePayload
    agent: str | None = None


class TerminalOutputEvent(BaseModel):
    type: Literal["terminal_output"]
    data: str
    agent: str | None = None


class DoneEvent(BaseModel):
    type: Literal["done"]
    data: DonePayload
    agent: str | None = None


UIEvent = Annotated[
    Union[
        StatusEvent,
        ThoughtsEvent,
        ToolCallsEvent,
        NodeFinishedEvent,
        FileUpdateEvent,
        TerminalOutputEvent,
        DoneEvent,
    ],
    Field(discriminator="type"),
]

_UI_EVENT_ADAPTER = TypeAdapter(UIEvent)


def build_ui_event(event_type: str, data: Any, agent: str | None = None) -> dict:
    payload = {"type": event_type, "data": data}
    if agent:
        payload["agent"] = agent
    return _UI_EVENT_ADAPTER.validate_python(payload).model_dump()
