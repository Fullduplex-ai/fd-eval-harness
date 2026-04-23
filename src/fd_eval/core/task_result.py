"""Shared return type for ``Task.evaluate``.

Promoted from ``fd_eval.tasks._types`` to ``fd_eval.core`` per D011
so third-party task plugins can import the shape from the public core
surface without depending on the in-tree example tasks module.

``score`` is the task's single primary number. ``details`` carries the
secondary metrics a reader would want in a per-run report. The value
type union is intentionally narrow (float | int | str) for v0.1; it
widens in a minor release once a concrete task needs richer nesting.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskResult:
    """Uniform return shape from ``Task.evaluate``.

    Frozen so downstream reporters can treat the object as a value.
    """

    score: float
    details: dict[str, float | int | str] = field(default_factory=dict)
