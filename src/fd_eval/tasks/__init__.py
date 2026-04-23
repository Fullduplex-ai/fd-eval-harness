"""fd-eval-harness reference task plugins (v0.1 skeleton).

This package hosts the reference task plugins shipped with the v0.1
harness (see ``docs/TASKS.md`` § "v0.1 reference tasks") and the
importable module paths the ``fd_eval.tasks`` entry-point group will
resolve against.

Current v0.1 inventory:

    voice_activity_detection  -> fd_eval.tasks.voice_activity          (shipped)
    turn_taking_latency       -> fd_eval.tasks.turn_taking_latency    (shipped)
    speaker_change_detection  -> fd_eval.tasks.speaker_change         (reserved)
    laughter_detection        -> fd_eval.tasks.laughter               (reserved)
    disfluency_detection      -> fd_eval.tasks.disfluency             (reserved)

See ``docs/TASKS.md`` for the authoring contract and
``_internal/DECISIONS.md`` D008 / D009 for the scoring_method literal
and the participant / observer mode split.
"""

from ._types import (
    TaskResult,
    TurnTakingEventKind,
    TurnTakingPredictionEvent,
    VADPredictionEvent,
)
from .turn_taking_latency import PartnerOffsetReference, TurnTakingLatency
from .voice_activity import VADReferenceEvent, VoiceActivityDetection

__all__ = [
    "PartnerOffsetReference",
    "TaskResult",
    "TurnTakingEventKind",
    "TurnTakingLatency",
    "TurnTakingPredictionEvent",
    "VADPredictionEvent",
    "VADReferenceEvent",
    "VoiceActivityDetection",
]
