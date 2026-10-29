from typing import Any, List

from openai import OpenAI

from src.genner.OAI import OAIGenner
from src.genner.config import SglangServerConfig
from src.typing.message import Message


class SglangServerGenner(OAIGenner):
    def __init__(self, client: OpenAI, config: SglangServerConfig):
        super().__init__(client=client, config=config, identifier="sglang")
        self._lora_routing = config.lora_routing_enabled
        self._default_lora = config.default_lora_name

    def _resolve_model(self, messages: List[Message]) -> str:
        base_model = self.config.model.strip()
        if not self._lora_routing:
            return base_model

        meta = self._first_meta(messages)
        requested_model = meta.get("model") if meta else None
        if isinstance(requested_model, str) and requested_model.startswith(
            f"{base_model}:"
        ):
            return requested_model

        adapter = meta.get("lora_adapter") if meta else None
        if not adapter:
            adapter = self._default_lora
        return f"{base_model}:{adapter}" if adapter else base_model

    def _prepare_messages(self, messages: List[Message]) -> List[Message]:
        prepared: list[Message] = []
        for message in messages:
            payload: dict[str, Any] = dict(message)
            payload.pop("meta", None)
            prepared.append(payload)  # type: ignore[arg-type]
        return prepared

    @staticmethod
    def _first_meta(messages: List[Message]) -> dict[str, Any]:
        for message in messages:
            meta = message.get("meta")
            if isinstance(meta, dict):
                return meta
        return {}
