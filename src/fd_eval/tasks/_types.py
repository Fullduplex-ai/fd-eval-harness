"""Shared result and event types for v0.1 reference tasks.

These types are intentionally task-local rather than core-wide. The core
module exposes the abstract ``PredictionEvent`` and ``Task`` contracts;
the concrete event subclasses and the ``TaskResult`` return shape live
under ``fd_eval.tasks`` while the ``evaluate()`` signature is still being
finalized (expected in D011). Once stable, these may be promoted into
``fd_eval.core``.

Note on ``@dataclass`` vs ``@dataclass(frozen=True)``:
``PredictionEvent`` in the core is an unfrozen dataclass, and Python's
dataclass machinery forbids a frozen subclass of an unfrozen parent. The
event subclasses below therefore inherit the unfrozen stance. ``TaskResult``
has no such constraint and is frozen for safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from fd_eval.core import PredictionEvent

TurnTakingEventKind = Literal["onset", "offset"]


@dataclass(frozen=True)
class TaskResult:
    """Uniform return type for ``Task.evaluate``.

    ``score`` is the task's single primary number (for VAD: F1; for
    turn-taking latency: mean latency in seconds). ``details`` carries
    the secondary metrics a reader would want in a per-run report
    (precision, recall, median, P95, count buckets, etc.). Keeping the
    shape flat avoids committing to a deeper schema before we have more
    than two reference tasks to generalize over.
    """

    score: float
    details: dict[str, float | int | str] = field(default_factory=dict)


@dataclass
class VADPredictionEvent(PredictionEvent):
    """Predicted speech-activity state change for a single channel.

    Emitted at the boundary between speech and non-speech, not per chunk.
    ``channel`` indexes into the AudioSession's original audio array
    (not into the input/target sub-lists), so callers can scope the
    reference set the same way.
    """

    channel: int = 0
    is_speech: bool = False


@dataclass
class TurnTakingPredictionEvent(PredictionEvent):
    """Predicted participant turn boundary.

    Participant-mode tasks only. ``event_kind="onset"`` is the moment
    the model began emitting audio in response; ``event_kind="offset"``
    is the moment it stopped. For the v0.1 turn-taking latency task,
    only ``onset`` events are consumed by the scorer, but the offset
    variant is reserved so adapters can report both without a schema
    change later.
    """

    channel: int = 0
    event_kind: TurnTakingEventKind = "onset"
