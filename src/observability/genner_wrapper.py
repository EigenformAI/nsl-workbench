import time
from typing import Callable, Optional

from loguru import logger
from result import Err, Ok, Result

from src.genner.Base import Genner
from src.helper import nanoid
from src.observability.collector import MetricsCollector
from src.observability.types import InferenceMetric, InferenceResult, UsageInfo
from src.typing.message import Message


class MetricsGenner(Genner):
    def __init__(
        self,
        inner: Genner,
        collector: MetricsCollector,
        phase_name_provider: Optional[Callable[[], Optional[str]]] = None,
        episode_id_provider: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        super().__init__(inner.identifier)
        self.inner = inner
        self.collector = collector
        self.phase_name_provider = phase_name_provider
        self.episode_id_provider = episode_id_provider

    def plist_completion(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, object]] | None = None,
        tool_choice: object = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Result[InferenceResult, str]:
        inference_id = nanoid()
        phase_name = self._resolve_phase_name(messages)
        episode_id = self._resolve_episode_id(messages)
        started_at = time.perf_counter()

        with logger.contextualize(inference_id=inference_id):
            if (
                tools is None
                and tool_choice is None
                and max_tokens is None
                and temperature is None
            ):
                result = self.inner.plist_completion(messages)
            else:
                kwargs = {"tools": tools, "tool_choice": tool_choice}
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                if temperature is not None:
                    kwargs["temperature"] = temperature
                result = self.inner.plist_completion(messages, **kwargs)

        latency_ms = (time.perf_counter() - started_at) * 1000
        resources = self.collector.snapshot_resources()

        match result:
            case Ok(inference_result):
                usage = inference_result.usage
                generation_rates = self._compute_generation_rates(usage, latency_ms)
                self.collector.record_inference_safe(
                    InferenceMetric(
                        inference_id=inference_id,
                        run_id=self.collector.run_id,
                        backend=self.inner.identifier,
                        episode_id=episode_id,
                        phase=phase_name,
                        success=True,
                        content=inference_result.content,
                        usage=usage,
                        model=usage.model if usage is not None else None,
                        latency_ms=latency_ms,
                        prompt_tokens_per_second=generation_rates[
                            "prompt_tokens_per_second"
                        ],
                        output_tokens_per_second=generation_rates[
                            "output_tokens_per_second"
                        ],
                        total_tokens_per_second=generation_rates[
                            "total_tokens_per_second"
                        ],
                        gpu_memory_mb=resources.gpu_memory_mb,
                        host_memory_mb=resources.host_memory_mb,
                    )
                )
                return Ok(inference_result)
            case Err(error_message):
                self.collector.record_inference_safe(
                    InferenceMetric(
                        inference_id=inference_id,
                        run_id=self.collector.run_id,
                        backend=self.inner.identifier,
                        episode_id=episode_id,
                        phase=phase_name,
                        success=False,
                        error_message=error_message,
                        latency_ms=latency_ms,
                        gpu_memory_mb=resources.gpu_memory_mb,
                        host_memory_mb=resources.host_memory_mb,
                    )
                )
                return Err(error_message)

    @staticmethod
    def get_usage_info(response) -> UsageInfo:
        return UsageInfo()

    def _resolve_phase_name(self, messages: list[Message]) -> str:
        for message in messages:
            meta = message.get("meta")
            if meta:
                phase_name = meta.get("phase")
                if phase_name:
                    return phase_name

        if self.phase_name_provider is not None:
            phase_name = self.phase_name_provider()
            if phase_name:
                return phase_name

        return "unknown"

    def _resolve_episode_id(self, messages: list[Message]) -> Optional[str]:
        for message in messages:
            meta = message.get("meta")
            if meta:
                episode_id = meta.get("episode_id")
                if episode_id:
                    return episode_id

        if self.episode_id_provider is None:
            return None

        episode_id = self.episode_id_provider()
        if episode_id:
            return episode_id

        return None

    def _compute_generation_rates(
        self,
        usage: Optional[UsageInfo],
        latency_ms: float,
    ) -> dict[str, Optional[float]]:
        if usage is None or latency_ms <= 0:
            return {
                "prompt_tokens_per_second": None,
                "output_tokens_per_second": None,
                "total_tokens_per_second": None,
            }

        latency_seconds = latency_ms / 1000
        if latency_seconds <= 0:
            return {
                "prompt_tokens_per_second": None,
                "output_tokens_per_second": None,
                "total_tokens_per_second": None,
            }

        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        if total_tokens is None and (
            prompt_tokens is not None or completion_tokens is not None
        ):
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

        return {
            "prompt_tokens_per_second": (
                float(prompt_tokens) / latency_seconds
                if prompt_tokens is not None
                else None
            ),
            "output_tokens_per_second": (
                float(completion_tokens) / latency_seconds
                if completion_tokens is not None
                else None
            ),
            "total_tokens_per_second": (
                float(total_tokens) / latency_seconds
                if total_tokens is not None
                else None
            ),
        }
