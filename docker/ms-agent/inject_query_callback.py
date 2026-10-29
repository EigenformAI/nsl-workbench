"""ms-agent workflow callback for per-step prompt injection."""

from __future__ import annotations

from ms_agent.callbacks.base import Callback
from ms_agent.llm.utils import Message


class InjectStepQuery(Callback):
    """Append the current step's query before ms-agent's first LLM call.

    Workaround for ms-agent 1.6.0: ``LLMAgent.create_messages`` preserves
    list inputs but does not add ``prompt.query`` for workflow steps after the
    first. ``on_task_begin`` runs after message creation and before generation.
    """

    async def on_task_begin(self, runtime, messages):
        query = getattr(getattr(self.config, "prompt", None), "query", None)
        if not query:
            return
        if messages and messages[-1].role == "user" and messages[-1].content == query:
            return
        messages.append(Message(role="user", content=query))
