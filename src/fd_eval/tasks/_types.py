"""Shared event types for v0.1 reference tasks.

``TaskResult`` was promoted to ``fd_eval.core`` by D011 so third-party
task plugins can import the return shape from the public core surface.
This module re-exports it for backward compatibility with in-tree
imports that reached here before the promotion.

The concrete event subclasses below stay task-local: they are event
shapes specific to each task's semantics, and promoting them would force
the core to ship a taxonomy of event kinds it does not need to own.

Note on ``@dataclass`` vs ``@dataclass(frozen=True)``:
``PredictionEvent`` in the core is an unfrozen dataclass, and Python's
dataclass machinery forbids a frozen subclass of an unfrozen parent. The
event subclasses below therefore inherit the unfrozen stance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fd_eval.core import PredictionEvent, TaskResult

__all__ = [
    "TaskResult",
    "ToolCallPredictionEvent",
    "TranscriptPredictionEvent",
    "TurnTakingEventKind",
    "TurnTakingPredictionEvent",
    "VADPredictionEvent",
]

TurnTakingEventKind = Literal["onset", "offset"]


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


@dataclass
class ToolCallPredictionEvent(PredictionEvent):
    """Predicted tool call emitted by the model.

    ``channel`` is the target channel on which the model generated this payload.
    ``tool_name`` is the string identifier of the tool.
    ``arguments`` is the JSON payload containing the arguments.
    """

    channel: int = 0
    tool_name: str = ""
    arguments: dict = None  # type: ignore

    def __post_init__(self):
        if self.arguments is None:
            self.arguments = {}


@dataclass
class TranscriptPredictionEvent(PredictionEvent):
    """Predicted text transcript emitted by the model.

    ``channel`` is the target channel on which the model generated this text.
    ``text`` is the actual transcribed text string.
    """

    channel: int = 0
    text: str = ""
