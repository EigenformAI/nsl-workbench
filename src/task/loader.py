"""Task loader — resolves a dotted import path to an instantiated TaskSpec."""

import importlib
from typing import Any

from src.task.base import TaskSpec


def load_task(
    dotted_path: str,
    task_config: dict[str, Any] | None = None,
) -> TaskSpec:
    """Load a TaskSpec subclass by dotted import path and instantiate it.

    Validation chain:
        1. issubclass — ensures the class inherits from TaskSpec.
        2. ABC enforcement — task_class(...) raises TypeError if any
           abstract method is unimplemented.
        3. instance.validate() — checks variations and capability uniqueness.

    Phase 2: prompt placeholder validation moved to the harness-config
    layer (templates live in ``[harness.orchestrator_modes]`` now, not on
    the task's prompt_spec). That check fires when the harness runs, not
    here.
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    task_class = getattr(module, class_name)

    if not isinstance(task_class, type) or not issubclass(task_class, TaskSpec):
        raise TypeError(f"{class_name} must be a subclass of TaskSpec")

    instance = task_class(task_config or {})

    instance.validate()

    return instance
