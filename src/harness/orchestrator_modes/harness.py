"""OrchestratorModeHarness — the default in-process reference harness.

The orchestrator-LLM turn picks a *mode* (a harness-internal name like
``analyzer`` / ``exploiter``); the delegated-LLM turn emits content for that
mode and, when ``mode.runs_code`` is set, a fenced code block. The harness
extracts the fence and dispatches it to a task-owned MCP capability
(``mode.code_capability``) — execution lives task-side, never inline in the
harness. Tasks are free of any "modes" vocabulary; modes are purely a
harness-side delegation primitive.

All configuration is read from ``ctx.config.settings`` (the
``OrchestratorModesConfig`` Pydantic model surfaced as a dict).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from result import Err, Ok

from src.framework.capability_bridge import CapabilityMcpBridge
from src.genner.Base import (
    CONTEXT_OVERFLOW_PREFIX,
    INFERENCE_TIMEOUT_PREFIX,
    INFERENCE_UNAVAILABLE_PREFIX,
    InferenceTimeoutError,
)
from src.harness.base import HarnessError, HarnessSpec
from src.harness.context import HarnessContext
from src.harness.orchestrator_modes.memory import CrossEpisodeScratchpad
from src.harness.transcript import HarnessTranscript, TerminationCategory
from src.observability.types import PhaseMetric
from src.parsing.code_extraction import WhenNoMatch, extract_code_block
from src.parsing.repetition_guard import (
    RepetitionAction,
    RepetitionCollapseError,
    RepetitionDetector,
    RepetitionGuardConfig,
)
from src.task.types import (
    CapabilityInvocation,
    CapabilityResult,
    EpisodeArtifacts,
)
from src.typing.config import ModeConfig
from src.typing.message import Message


class ContextOverflowError(RuntimeError):
    """Raised when the prompt exceeds the model's available context."""


# ---------------------------------------------------------------------------
# Internal per-episode state types
# ---------------------------------------------------------------------------


class _ActionBudget:
    def __init__(self, total: int) -> None:
        self.total = total
        self.remaining = total

    def consume(self, amount: int = 1) -> bool:
        if self.remaining >= amount:
            self.remaining -= amount
            return True
        return False

    def is_exhausted(self) -> bool:
        return self.remaining <= 0


@dataclass
class _OrchestratorDecision:
    target_capability: str
    instruction: str
    reasoning: str
    raw_response: str
    duration_ms: float = 0.0


@dataclass
class _ModeTurnResult:
    capability: str
    success: bool
    content: str
    code_executed: str | None
    execution_result: dict[str, Any] | None
    invocations: list[CapabilityInvocation]
    results: list[CapabilityResult]
    error_message: str | None = None
    duration_ms: float = 0.0


@dataclass
class _EpisodeState:
    episode_id: str
    current_step: int = 1
    mode_history: list[dict[str, Any]] = field(default_factory=list)


def _edit_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n < m:
        a, b = b, a
        n, m = m, n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def _extract_capability_name(
    raw_response: str,
    delegated_caps: list[str],
) -> str:
    """Resolve the orchestrator's target capability with ranked fallbacks."""
    pattern = "|".join(re.escape(c) for c in delegated_caps)
    fallback = delegated_caps[0]

    match = re.search(rf"MODE:\s*({pattern})", raw_response, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    match = re.search(
        rf"(?:Next\s+)?Mode:\s*\**?\s*({pattern})\s*\**?",
        raw_response,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).lower()

    cleaned = re.sub(r"\*+", "", raw_response)
    match = re.search(
        rf"^({pattern})\b",
        cleaned,
        re.IGNORECASE | re.MULTILINE,
    )
    if match:
        return match.group(1).lower()

    top_section = raw_response[:500]
    for word in re.findall(r"\b[a-zA-Z]{4,12}\b", top_section):
        lower_word = word.lower()
        for cap in delegated_caps:
            distance = _edit_distance(lower_word, cap)
            if distance <= 2:
                if distance > 0:
                    logger.info(
                        f"Fuzzy-matched capability: {lower_word!r} -> {cap!r} "
                        f"(edit distance {distance})"
                    )
                return cap

    logger.warning(
        f"No valid MODE found in orchestrator response: {raw_response[:100]}"
    )
    return fallback


def _compose_system_prompt(prompt_spec: Any) -> str:
    """Fold ``TaskPromptSpec.environment_context`` into the system message.

    Tasks that fold everything into ``system_instruction`` (today's three
    shipped tasks) see no change — their ``environment_context`` is empty.
    Tasks that split episode-specific context out (addresses, fork blocks,
    etc.) get it appended to the system turn so the harness honors the
    advertised TaskPromptSpec contract.
    """
    system = prompt_spec.system_instruction or ""
    env = getattr(prompt_spec, "environment_context", "") or ""
    if env.strip():
        return f"{system}\n\n{env}" if system.strip() else env
    return system


def _extract_label(raw_response: str, label: str) -> str:
    match = re.search(
        rf"{label}:\s*(.+?)$",
        raw_response,
        re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return raw_response
    return match.group(1).strip()


def _safe_artifact_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return sanitized or "unknown"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _resolve_modes(settings: dict[str, Any]) -> dict[str, ModeConfig]:
    """Coerce ``settings["modes"]`` into ``dict[str, ModeConfig]``.

    Tolerates two shapes: dict-of-ModeConfig (when programmatically set) and
    dict-of-dict (when the AppConfig is dumped before reaching the harness).
    """
    raw = settings.get("modes", {}) or {}
    out: dict[str, ModeConfig] = {}
    for name, value in dict(raw).items():
        if isinstance(value, ModeConfig):
            out[name] = value
        else:
            out[name] = ModeConfig(**dict(value))
    return out


class OrchestratorModeHarness(HarnessSpec):
    """Orchestrator-delegates-to-modes in-process reference harness.

    Reads settings from ``ctx.config.settings`` (the
    ``OrchestratorModesConfig`` Pydantic model exposed as a dict):

    - ``max_harness_iterations``: int (default 12). Safety bound on orchestrator
      decision iterations. This is a harness-level guard, not a task tool call
      count or LLM turn count.
    - ``scratchpad_max_chars``: int (default 32000).
    - ``tool_output_max_chars``: int (default 20000).
    - ``orchestrator_prompt``: str — user template with placeholders
      ``{scratchpad_content}``, ``{budget_remaining}``, ``{total_budget}``.
    - ``modes``: ``dict[str, ModeConfig]`` — typed mode declarations.
    - ``repetition_guard``: dict mapping to :class:`RepetitionGuardConfig`.

    Per-mode dispositions (code exec, scratchpad writes, labels,
    metric-scoped scratchpad) are owned wholly by ``ModeConfig`` — no
    capability-side annotations are read. When ``mode.runs_code=True`` the
    harness extracts the fenced code block from the delegated turn and
    invokes ``mode.code_capability`` on the task; the task is responsible
    for running the code.
    """

    name = "orchestrator_modes"
    description = "Orchestrator chooses from named modes each step"

    def __init__(self, harness_config: dict[str, Any]) -> None:
        super().__init__(harness_config)
        self._ctx: HarnessContext | None = None
        self._episode_state: _EpisodeState | None = None
        self._episode_budget: _ActionBudget | None = None

    def telemetry(self) -> dict[str, str]:
        if self._episode_state is None or self._episode_budget is None:
            return {}
        return {
            "step": str(self._episode_state.current_step),
            "budget_left": str(self._episode_budget.remaining),
        }

    def telemetry_columns(self) -> list[str]:
        return ["step", "budget_left"]

    def failure_extras(self) -> dict[str, Any]:
        extras: dict[str, Any] = {}
        if self._ctx is not None:
            try:
                extras["scratchpad_final"] = self._scratchpad(self._ctx).get_content()
            except Exception:
                pass
        if self._episode_state is not None:
            extras["mode_history"] = list(self._episode_state.mode_history)
        return extras

    # --- HarnessSpec API ---

    def run_episode(self, *, ctx: HarnessContext) -> HarnessTranscript:
        self._ctx = ctx
        try:
            settings = ctx.config.settings

            max_harness_iterations = int(settings.get("max_harness_iterations", 12))
            tool_output_max = int(settings.get("tool_output_max_chars", 20000))

            scratchpad = self._scratchpad(ctx)

            # Mode declarations are owned wholly by config — capabilities are
            # task MCP tools, not modes. Failure modes (no modes / no scratchpad
            # writer / unknown code_capability) surface here, before the loop.
            modes = _resolve_modes(settings)
            if not modes:
                raise HarnessError(
                    "OrchestratorModeHarness: harness.orchestrator_modes.modes "
                    "is empty; declare at least one mode."
                )
            if not any(m.writes_scratchpad for m in modes.values()):
                raise HarnessError(
                    "OrchestratorModeHarness: no mode has writes_scratchpad=True. "
                    "At least one mode must opt into scratchpad writes to avoid "
                    "silently running empty."
                )
            cap_names = {c.name for c in ctx.prompt_spec.capabilities}
            for mode_name, mode in modes.items():
                if mode.runs_code and mode.code_capability not in cap_names:
                    raise HarnessError(
                        f"OrchestratorModeHarness: mode {mode_name!r} declares "
                        f"code_capability={mode.code_capability!r} but the task "
                        f"does not declare a capability of that name "
                        f"(declared: {sorted(cap_names)})"
                    )

            code_exec_modes = {n for n, m in modes.items() if m.runs_code}
            metric_modes = {n for n, m in modes.items() if m.publishes_metric}
            scratchpad_modes = {n for n, m in modes.items() if m.writes_scratchpad}
            content_labels = {
                n: m.scratchpad_label for n, m in modes.items() if m.scratchpad_label
            }

            rep_config_raw: Any = settings.get("repetition_guard", {}) or {}
            if isinstance(rep_config_raw, RepetitionGuardConfig):
                rep_config = rep_config_raw
            else:
                rep_config = RepetitionGuardConfig(**dict(rep_config_raw))
            repetition = RepetitionDetector(config=rep_config)

            budget = _ActionBudget(total=max_harness_iterations)
            state = _EpisodeState(episode_id=ctx.episode_id)
            self._episode_budget = budget
            self._episode_state = state
            scratchpad.append(
                f"Started with budget {budget.total}",
                ctx.episode_id,
            )

            orchestrator_prompt = settings.get("orchestrator_prompt")
            if not orchestrator_prompt:
                raise HarnessError(
                    "harness.orchestrator_modes.orchestrator_prompt is required "
                    "for OrchestratorModeHarness"
                )

            mode_names = list(modes.keys())

            invocations: list[CapabilityInvocation] = []
            results: list[CapabilityResult] = []
            last_content: str | None = None
            bridge = CapabilityMcpBridge(ctx, token="")

            termination_category: TerminationCategory = "agent_failure"
            termination_reason = "loop exited without explicit termination"

            while not budget.is_exhausted():
                if ctx.cancel_event.is_set():
                    termination_category = "wall_clock"
                    termination_reason = "cancel_event set"
                    break

                ctx.recorder.log_state(
                    "step_start",
                    {
                        "step": state.current_step,
                        "budget_remaining": budget.remaining,
                        "budget_total": budget.total,
                    },
                )

                try:
                    decision = self._run_orchestrator(
                        ctx=ctx,
                        budget=budget,
                        state=state,
                        orchestrator_prompt_template=orchestrator_prompt,
                        mode_names=mode_names,
                        repetition=repetition,
                    )

                    ctx.recorder.log_decision(
                        "mode_selected",
                        {
                            "mode": decision.target_capability,
                            "instruction_preview": decision.instruction[:200],
                        },
                    )

                    turn = self._run_delegated(
                        ctx=ctx,
                        state=state,
                        mode_name=decision.target_capability,
                        instruction=decision.instruction,
                        modes=modes,
                        code_exec_modes=code_exec_modes,
                        content_labels=content_labels,
                        scratchpad_modes=scratchpad_modes,
                        metric_modes=metric_modes,
                        tool_output_max=tool_output_max,
                        repetition=repetition,
                        bridge=bridge,
                    )

                    invocations.extend(turn.invocations)
                    results.extend(turn.results)
                    last_content = turn.content

                    state.mode_history.append(
                        {
                            "step": state.current_step,
                            "mode": decision.target_capability,
                            "instruction": decision.instruction,
                            "success": turn.success,
                            "duration_ms": turn.duration_ms,
                            "orchestrator_duration_ms": decision.duration_ms,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    state.current_step += 1

                    if not turn.success:
                        logger.warning(
                            f"Mode {decision.target_capability} failed: "
                            f"{turn.error_message}"
                        )

                except ContextOverflowError as exc:
                    termination_category = "context_overflow"
                    termination_reason = str(exc)
                    ctx.recorder.log_warning("context_overflow", {"detail": str(exc)})
                    scratchpad.append("Episode aborted: context overflow")
                    break
                except RepetitionCollapseError as exc:
                    termination_category = "repetition_collapse"
                    termination_reason = str(exc)
                    ctx.recorder.log_warning(
                        "repetition_collapse", {"detail": str(exc)}
                    )
                    scratchpad.append("Episode aborted: repetition collapse")
                    break
                except InferenceTimeoutError:
                    raise
                except HarnessError:
                    raise
                except Exception as exc:
                    logger.error(f"Harness step error: {exc}")
                    ctx.recorder.log_warning("step_error", {"detail": str(exc)})
                    scratchpad.append(f"Episode failed: {exc}")
                    termination_category = "harness_error"
                    termination_reason = str(exc)
                    break
            else:
                # budget exhausted without break
                termination_category = "budget_exhausted"
                termination_reason = (
                    f"action budget of {budget.total} decisions consumed"
                )

            # Reaching the natural loop end (budget exhaustion) sets the
            # above; any break/raise already set termination_*.
            if budget.is_exhausted() and termination_category == "agent_failure":
                termination_category = "budget_exhausted"
                termination_reason = (
                    f"action budget of {budget.total} decisions consumed"
                )

            artifacts = EpisodeArtifacts(
                capability_invocations=invocations,
                capability_results=results,
                final_response=last_content,
            )

            # Any non-error termination AND at least one successful capability
            # turn → success for transcript classification purposes (final
            # reward still comes from compute_reward).
            if termination_category == "agent_failure" and any(
                r.success for r in results
            ):
                termination_category = "success"

            return HarnessTranscript(
                artifacts=artifacts,
                llm_turns=0,  # ledger-owned in Phase 2; harness iterations go to extra
                termination_reason=termination_reason,
                termination_category=termination_category,
                extra={
                    "scratchpad_final": scratchpad.get_content(),
                    "mode_history": state.mode_history,
                    "harness_iterations_count": state.current_step - 1,
                },
            )
        finally:
            self._ctx = None
            self._episode_state = None
            self._episode_budget = None

    def _scratchpad(self, ctx: HarnessContext) -> CrossEpisodeScratchpad:
        scratchpad = ctx.harness_session.get("scratchpad")
        if scratchpad is None:
            scratchpad = CrossEpisodeScratchpad(
                max_chars=int(ctx.config.settings.get("scratchpad_max_chars", 64_000)),
            )
            ctx.harness_session["scratchpad"] = scratchpad
        return scratchpad

    # --- Orchestrator turn ---

    def _run_orchestrator(
        self,
        *,
        ctx: HarnessContext,
        budget: _ActionBudget,
        state: _EpisodeState,
        orchestrator_prompt_template: str,
        mode_names: list[str],
        repetition: RepetitionDetector,
    ) -> _OrchestratorDecision:
        user_prompt = orchestrator_prompt_template.format(
            scratchpad_content=self._scratchpad(ctx).get_content(),
            budget_remaining=budget.remaining,
            total_budget=budget.total,
        )
        messages: list[Message] = [
            {
                "role": "system",
                "content": _compose_system_prompt(ctx.prompt_spec),
                "meta": {"episode_id": ctx.episode_id, "phase": "orchestrator"},
            },
            {
                "role": "user",
                "content": user_prompt,
                "meta": {"episode_id": ctx.episode_id, "phase": "orchestrator"},
            },
        ]

        started = time.perf_counter()
        result = ctx.genner.plist_completion(
            messages,
            phase="orchestrator",
            meta={"workflow_step": ctx.workflow_step},
        )
        duration_ms = (time.perf_counter() - started) * 1000
        match result:
            case Ok(inference_result):
                raw_response = inference_result.content
                raw_response = self._apply_repetition_guard(
                    repetition, "orchestrator", raw_response
                )
                decision = self._parse_orchestrator_response(raw_response, mode_names)
                decision.duration_ms = duration_ms
                budget.consume(1)
                self._record_phase_metric(ctx, "orchestrator", duration_ms, True)
                return decision
            case Err(error):
                self._record_phase_metric(
                    ctx, "orchestrator", duration_ms, False, str(error)
                )
                if str(error).startswith(CONTEXT_OVERFLOW_PREFIX):
                    raise ContextOverflowError(str(error))
                if str(error).startswith(INFERENCE_TIMEOUT_PREFIX):
                    raise InferenceTimeoutError(str(error))
                if str(error).startswith(INFERENCE_UNAVAILABLE_PREFIX):
                    raise HarnessError(str(error))
                raise RuntimeError(f"Orchestrator execution failed: {error}")
        raise RuntimeError("unreachable: Result must be Ok or Err")

    @staticmethod
    def _parse_orchestrator_response(
        raw_response: str,
        mode_names: list[str],
    ) -> _OrchestratorDecision:
        fallback = mode_names[0]
        try:
            cap_name = _extract_capability_name(raw_response, mode_names)

            instruction_match = re.search(
                r"INSTRUCTION:\s*(.+?)(?=\n(?:REASONING|REASON|NOTE):|$)",
                raw_response,
                re.IGNORECASE | re.DOTALL,
            )
            instruction = (
                instruction_match.group(1).strip()
                if instruction_match
                else f"Default {cap_name} task"
            )

            reasoning_match = re.search(
                r"(?:REASONING|REASON|NOTE):\s*(.+?)$",
                raw_response,
                re.IGNORECASE | re.DOTALL,
            )
            reasoning = (
                reasoning_match.group(1).strip()
                if reasoning_match
                else "No reasoning provided"
            )

            return _OrchestratorDecision(
                target_capability=cap_name,
                instruction=instruction,
                reasoning=reasoning,
                raw_response=raw_response,
            )
        except Exception as exc:
            logger.error(f"Failed to parse orchestrator response: {exc}")
            return _OrchestratorDecision(
                target_capability=fallback,
                instruction="Analyze current system state",
                reasoning=f"Parser error, defaulting to {fallback}: {exc}",
                raw_response=raw_response,
            )

    # --- Delegated-capability turn ---

    def _run_delegated(
        self,
        *,
        ctx: HarnessContext,
        state: _EpisodeState,
        mode_name: str,
        instruction: str,
        modes: dict[str, ModeConfig],
        code_exec_modes: set[str],
        content_labels: dict[str, str],
        scratchpad_modes: set[str],
        metric_modes: set[str],
        tool_output_max: int,
        repetition: RepetitionDetector,
        bridge: CapabilityMcpBridge,
    ) -> _ModeTurnResult:
        mode = modes.get(mode_name)
        if mode is None:
            raise HarnessError(
                f"Orchestrator selected unknown mode {mode_name!r}; "
                f"declared: {sorted(modes)}"
            )

        user_prompt = mode.prompt.format(
            instruction=instruction,
            scratchpad_content=self._scratchpad(ctx).get_content(),
        )
        messages: list[Message] = [
            {
                "role": "system",
                "content": _compose_system_prompt(ctx.prompt_spec),
                "meta": {
                    "episode_id": ctx.episode_id,
                    "phase": mode_name,
                },
            },
            {
                "role": "user",
                "content": user_prompt,
                "meta": {
                    "episode_id": ctx.episode_id,
                    "phase": mode_name,
                },
            },
        ]
        exec_timeout = int(mode.timeout_s)

        started = time.perf_counter()
        result = ctx.genner.plist_completion(
            messages,
            phase=mode_name,
            meta={"workflow_step": ctx.workflow_step},
        )
        match result:
            case Ok(inference_result):
                raw_response = inference_result.content
                raw_response = self._apply_repetition_guard(
                    repetition, mode_name, raw_response
                )
                duration_ms = (time.perf_counter() - started) * 1000

                # Code-block extraction (only meaningful for runs_code modes).
                block = extract_code_block(
                    raw_response,
                    accepted_langs=[
                        "python",
                        "py",
                        "python3",
                        "bash",
                        "shell",
                        "sh",
                        "",
                    ],
                    fallback=WhenNoMatch.NONE,
                )
                code: str | None = None
                code_lang: str = ""
                if block is not None:
                    code_lang = block.lang
                    code = block.body

                execution_result: dict[str, Any] | None = None
                turn_invocations: list[CapabilityInvocation] = []
                turn_results: list[CapabilityResult] = []

                # Code execution path: route through the task-owned MCP
                # capability declared on the mode. The harness no longer
                # runs python in the container directly — execution lives
                # task-side, behind a declared capability.
                if mode_name in code_exec_modes:
                    if code is not None and mode.code_capability:
                        ctx.recorder.log_action(
                            "exec_code",
                            {
                                "mode": mode_name,
                                "capability": mode.code_capability,
                                "lang": code_lang,
                                "bytes": len(code),
                            },
                        )
                        code_inv = CapabilityInvocation(
                            name=mode.code_capability,
                            input={
                                "code": code,
                                "lang": code_lang or "python",
                                "timeout_s": exec_timeout,
                            },
                        )
                        try:
                            cap_result = bridge.invoke(code_inv)
                        except Exception as exc:
                            cap_result = CapabilityResult(
                                name=code_inv.name,
                                output={},
                                success=False,
                                error=str(exc),
                            )
                        turn_invocations.append(code_inv)
                        turn_results.append(cap_result)
                        execution_result = {
                            "success": cap_result.success,
                            "stdout": cap_result.output.get("stdout", "") or "",
                            "stderr": cap_result.output.get(
                                "stderr", cap_result.error or ""
                            )
                            or "",
                            "return_code": cap_result.output.get(
                                "return_code", 0 if cap_result.success else -1
                            ),
                            "executed_code": code,
                        }
                    elif code is None:
                        logger.warning(
                            f"[{mode_name}] expected executable code block "
                            "but none was extracted"
                        )
                        execution_result = {
                            "success": False,
                            "stdout": "",
                            "stderr": (
                                "No executable code block was found. Wrap "
                                "commands in a ```python, ```bash, ```shell, "
                                "```sh, or bare fenced block."
                            ),
                            "return_code": -1,
                            "executed_code": "",
                        }

                # Task-side parse_response — preserves the harness-phase-tag
                # routing for tasks that emit structured invocations from
                # delegated turns (e.g. forked_exploit's deploy_attack_sol
                # extraction). Documented as a harness phase tag, not a
                # Capability.name.
                parsed = (
                    ctx.task.parse_response(
                        raw_response,
                        invoked_capability=mode_name,
                    )
                    or []
                )
                for inv in parsed:
                    try:
                        cap_result = bridge.invoke(inv)
                    except Exception as exc:
                        cap_result = CapabilityResult(
                            name=inv.name,
                            output={},
                            success=False,
                            error=str(exc),
                        )
                    turn_invocations.append(inv)
                    turn_results.append(cap_result)
                    if (
                        execution_result is None
                        and cap_result.output
                        and "stdout" in cap_result.output
                        and "stderr" in cap_result.output
                    ):
                        execution_result = {
                            "success": cap_result.success,
                            "stdout": cap_result.output.get("stdout", ""),
                            "stderr": cap_result.output.get("stderr", ""),
                            "return_code": 0 if cap_result.success else 1,
                            "executed_code": "",
                        }

                if execution_result is not None:
                    execution_result = self._truncate_execution_result(
                        ctx, state, mode_name, execution_result, tool_output_max
                    )
                    self._emit_tool_output_observation(ctx, mode_name, execution_result)

                # Scratchpad update.
                content_label = content_labels.get(mode_name)
                if content_label is not None:
                    extracted = _extract_label(raw_response, content_label)
                    if execution_result is not None:
                        if execution_result["success"] and execution_result["stdout"]:
                            extracted += (
                                f"\n\nExecution Output:\n{execution_result['stdout']}"
                            )
                        elif not execution_result["success"]:
                            extracted += (
                                f"\n\nExecution Error:\n{execution_result['stderr']}"
                            )
                    if mode_name in scratchpad_modes:
                        if mode_name in metric_modes:
                            self._update_metric_scratchpad(
                                ctx, mode_name, extracted, turn_invocations
                            )
                        else:
                            self._update_scratchpad(ctx, mode_name, extracted)
                    content = extracted
                else:
                    content = raw_response

                self._record_phase_metric(ctx, mode_name, duration_ms, True)
                return _ModeTurnResult(
                    capability=mode_name,
                    success=True,
                    content=content,
                    code_executed=code,
                    execution_result=execution_result,
                    invocations=turn_invocations,
                    results=turn_results,
                    duration_ms=duration_ms,
                )

            case Err(error):
                duration_ms = (time.perf_counter() - started) * 1000
                self._record_phase_metric(
                    ctx, mode_name, duration_ms, False, str(error)
                )
                if str(error).startswith(CONTEXT_OVERFLOW_PREFIX):
                    raise ContextOverflowError(str(error))
                if str(error).startswith(INFERENCE_TIMEOUT_PREFIX):
                    raise InferenceTimeoutError(str(error))
                if str(error).startswith(INFERENCE_UNAVAILABLE_PREFIX):
                    raise HarnessError(str(error))
                logger.error(f"{mode_name} mode failed: {error}")
                return _ModeTurnResult(
                    capability=mode_name,
                    success=False,
                    content=f"{mode_name.title()} failed: {error}",
                    code_executed=None,
                    execution_result=None,
                    invocations=[],
                    results=[],
                    error_message=str(error),
                    duration_ms=duration_ms,
                )
        raise RuntimeError("unreachable: Result must be Ok or Err")

    # --- Tool-output truncation ---

    def _truncate_execution_result(
        self,
        ctx: HarnessContext,
        state: _EpisodeState,
        capability_name: str,
        execution_result: dict[str, Any],
        max_chars: int,
    ) -> dict[str, Any]:
        out = dict(execution_result)
        any_truncated = False
        for stream_name in ("stdout", "stderr"):
            stream = out.get(stream_name)
            if isinstance(stream, str):
                new_text, was_truncated = self._truncate_tool_output(
                    ctx, state, capability_name, stream_name, stream, max_chars
                )
                out[stream_name] = new_text
                any_truncated = any_truncated or was_truncated
        out["truncated"] = any_truncated
        return out

    def _emit_tool_output_observation(
        self,
        ctx: HarnessContext,
        capability_name: str,
        execution_result: dict[str, Any],
    ) -> None:
        """Single post-truncation observation — byte counts reflect what the
        agent actually saw, and ``truncated`` is accurate."""
        ctx.recorder.log_observation(
            "tool_output",
            {
                "capability": capability_name,
                "stdout_bytes": len(execution_result.get("stdout", "") or ""),
                "stderr_bytes": len(execution_result.get("stderr", "") or ""),
                "exit_code": execution_result.get("return_code"),
                "success": execution_result.get("success", False),
                "truncated": bool(execution_result.get("truncated", False)),
            },
        )

    def _truncate_tool_output(
        self,
        ctx: HarnessContext,
        state: _EpisodeState,
        capability_name: str,
        stream_name: str,
        content: str,
        max_chars: int,
    ) -> tuple[str, bool]:
        """Returns (possibly-truncated-content, was_truncated).

        If we cannot persist the full-artifact file, we do NOT tell the agent
        it was saved — the prior behavior pointed debugging at a nonexistent
        path. Instead the truncation marker just notes the byte count.
        """
        if max_chars <= 0 or len(content) <= max_chars:
            return content, False

        artifacts_dir = Path(ctx.config.train_data_save_folder) / "artifacts"
        safe_mode = _safe_artifact_component(capability_name)
        safe_stream = _safe_artifact_component(stream_name)
        artifact_name = (
            f"tool_output_{state.episode_id}_{state.current_step}_"
            f"{safe_mode}_{safe_stream}.txt"
        )
        artifact_path = artifacts_dir / artifact_name
        artifact_written = False
        try:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(content, encoding="utf-8")
            artifact_written = True
        except OSError as exc:
            logger.warning(
                f"tool_output artifact persist failed "
                f"({capability_name}:{stream_name}): {exc}"
            )
            ctx.recorder.log_warning(
                "tool_output_artifact_failed",
                {
                    "capability": capability_name,
                    "stream": stream_name,
                    "error": str(exc),
                },
            )

        truncated_text = content[:max_chars]
        truncated_bytes = max(
            0,
            len(content.encode("utf-8")) - len(truncated_text.encode("utf-8")),
        )
        event_message = (
            f"tool={capability_name}:{stream_name} "
            f"bytes={truncated_bytes} cap={max_chars}"
        )
        logger.warning(f"tool_output_truncated: {event_message}")
        warning_payload: dict[str, Any] = {
            "capability": capability_name,
            "stream": stream_name,
            "bytes": truncated_bytes,
            "cap": max_chars,
            "artifact_written": artifact_written,
        }
        if artifact_written:
            warning_payload["artifact_path"] = str(artifact_path)
        ctx.recorder.log_warning("tool_output_truncated", warning_payload)
        self._record_phase_metric(
            ctx, "tool_output_truncated", 0.0, True, event_message
        )
        suffix = (
            f"\n...[truncated {truncated_bytes} bytes; "
            f"full output at artifacts/{artifact_name}]"
            if artifact_written
            else f"\n...[truncated {truncated_bytes} bytes; full output not persisted]"
        )
        return truncated_text + suffix, True

    # --- Repetition guard ---

    @staticmethod
    def _apply_repetition_guard(
        repetition: RepetitionDetector,
        phase: str,
        raw_response: str,
    ) -> str:
        check = repetition.check(raw_response)
        if not check.triggered:
            return raw_response
        if check.action is RepetitionAction.END_EPISODE:
            raise RepetitionCollapseError(
                f"{phase}: repetition_collapse hit={repetition.hits} "
                f"preview={check.repeated_segment_preview!r}"
            )
        if check.action is RepetitionAction.TRUNCATE and check.truncated_response:
            return check.truncated_response
        return raw_response

    # --- Scratchpad ---

    def _update_scratchpad(
        self,
        ctx: HarnessContext,
        capability_name: str,
        content: str,
    ) -> None:
        if not content:
            return
        self._scratchpad(ctx).append(f"[{capability_name.title()}] {content}")

    def _update_metric_scratchpad(
        self,
        ctx: HarnessContext,
        capability_name: str,
        content: str,
        invocations: list[CapabilityInvocation],
    ) -> None:
        metric_name = ctx.task.metric_name
        metric_unit = ctx.task.metric_unit
        value = 0.0
        for inv in invocations:
            raw = inv.input.get(metric_name)
            if raw is None:
                continue
            try:
                value = max(value, float(raw))
            except (TypeError, ValueError):
                continue
        if metric_name and value > 0:
            self._scratchpad(ctx).append(
                f"[{capability_name.title()}] {metric_name}={value:.1f}"
                f"{metric_unit}\n{content}"
            )
            return
        self._update_scratchpad(ctx, capability_name, content)

    # --- Metrics ---

    @staticmethod
    def _record_phase_metric(
        ctx: HarnessContext,
        phase_name: str,
        duration_ms: float,
        success: bool,
        error_message: str | None = None,
    ) -> None:
        if ctx.metrics is None:
            return
        ctx.metrics.record_phase_safe(
            PhaseMetric(
                phase_name=phase_name,
                run_id=ctx.metrics.run_id,
                episode_id=ctx.episode_id,
                duration_ms=duration_ms,
                success=success,
                error_message=error_message,
            )
        )
