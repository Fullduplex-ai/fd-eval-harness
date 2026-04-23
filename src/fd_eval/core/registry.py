"""Plugin registry for tasks and adapters."""

from __future__ import annotations

from importlib.metadata import entry_points

from fd_eval.core.adapter import FDModelAdapter
from fd_eval.core.task import Task


def get_task(name: str) -> type[Task]:
    """Retrieve a Task subclass by its registered entry point name."""
    eps = entry_points(group="fd_eval.tasks")
    for ep in eps:
        if ep.name == name:
            task_cls = ep.load()
            if not issubclass(task_cls, Task):
                raise TypeError(f"Entry point '{name}' is not a Task subclass.")
            return task_cls
    raise KeyError(f"Task '{name}' not found in registry.")


def get_adapter(name: str) -> type[FDModelAdapter]:
    """Retrieve an FDModelAdapter subclass by its registered entry point name."""
    eps = entry_points(group="fd_eval.adapters")
    for ep in eps:
        if ep.name == name:
            adapter_cls = ep.load()
            if not issubclass(adapter_cls, FDModelAdapter):
                raise TypeError(f"Entry point '{name}' is not an FDModelAdapter subclass.")
            return adapter_cls
    raise KeyError(f"Adapter '{name}' not found in registry.")


def list_tasks() -> list[str]:
    """List all registered task names."""
    eps = entry_points(group="fd_eval.tasks")
    return sorted(ep.name for ep in eps)


def list_adapters() -> list[str]:
    """List all registered adapter names."""
    eps = entry_points(group="fd_eval.adapters")
    return sorted(ep.name for ep in eps)
