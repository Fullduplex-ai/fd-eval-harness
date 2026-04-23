"""Turn-Taking Latency — participant-mode reference task (v0.1).

Measures how long the model-under-test waits after the conversation
partner stops speaking before it begins emitting audio. This is the
participant-mode counterpart to VAD: the harness needs to see the
session's *input* channel activity (the partner) and the adapter's
*onset* events on a target channel (the model speaking).

Scoring: for each reference partner-speech offset event, pair it with
the nearest later adapter onset event on the target channel inside a
``max_latency_s`` window. Compute latency deltas, then report the mean
(primary score), median, and P95. A reference offset that has no
adapter onset within the window counts as a miss and is tracked
separately; misses do not contribute to the latency distribution to
avoid inflating the mean with the window cap.

See ``docs/TASKS.md`` for the observer/participant split and
``_internal/DECISIONS.md`` D009.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from fd_eval.core import AudioSession, PredictionStream, Task

from ._types import TaskResult, TurnTakingPredictionEvent


@dataclass(frozen=True)
class PartnerOffsetReference:
    """Ground-truth partner-speech offset on an input channel.

    ``timestamp_s`` is the moment the partner stopped talking.
    ``channel`` is the input channel the partner is on. The adapter's
    response onset is expected on a different (target) channel.
    """

    timestamp_s: float
    channel: int


class TurnTakingLatency(Task):
    """Participant-mode turn-taking latency.

    Parameters
    ----------
    max_latency_s:
        Cap in seconds beyond which an adapter onset is no longer
        considered a response to the given partner offset. Default
        3.0 s, which is well beyond conversational norm (Stivers 2009
        puts cross-linguistic mode near 200 ms) while still giving slow
        models enough room to be scored rather than dismissed as misses.
    """

    mode = "participant"
    scoring_method = "algorithmic"

    def __init__(self, max_latency_s: float = 3.0) -> None:
        if max_latency_s <= 0:
            raise ValueError(f"max_latency_s must be positive, got {max_latency_s}")
        self.max_latency_s = max_latency_s

    def evaluate(
        self,
        session: AudioSession,
        predictions: PredictionStream,
        references: Sequence[PartnerOffsetReference],
    ) -> TaskResult:
        target_channels = set(session.target_channel_indices)

        onset_events: list[TurnTakingPredictionEvent] = []
        for ev in predictions:
            if not isinstance(ev, TurnTakingPredictionEvent):
                raise TypeError(
                    f"TurnTakingLatency expects TurnTakingPredictionEvent, got {type(ev).__name__}"
                )
            if ev.event_kind != "onset":
                # Offsets are reserved for future tasks; ignored here.
                continue
            if ev.channel not in target_channels:
                # Only onsets on declared target channels count as a
                # response. Ignore adapter events on input channels.
                continue
            onset_events.append(ev)

        # Sort onsets by time so we can walk forward from each reference.
        onset_events.sort(key=lambda e: e.timestamp_s)
        used_onsets: set[int] = set()

        latencies_s: list[float] = []
        misses = 0
        for ref in references:
            best_idx: int | None = None
            best_dt = float("inf")
            for i, onset in enumerate(onset_events):
                if i in used_onsets:
                    continue
                dt = onset.timestamp_s - ref.timestamp_s
                if dt < 0:
                    # Onset happened before the partner finished talking.
                    # Not a response to this reference.
                    continue
                if dt > self.max_latency_s:
                    # Further onsets are even later; stop searching.
                    break
                if dt < best_dt:
                    best_dt = dt
                    best_idx = i
            if best_idx is None:
                misses += 1
            else:
                used_onsets.add(best_idx)
                latencies_s.append(best_dt)

        num_refs = len(references)
        num_onsets = len(onset_events)
        spurious = max(0, num_onsets - len(used_onsets))

        if latencies_s:
            mean_s = statistics.fmean(latencies_s)
            median_s = statistics.median(latencies_s)
            p95_s = _percentile(latencies_s, 0.95)
        else:
            mean_s = 0.0
            median_s = 0.0
            p95_s = 0.0

        return TaskResult(
            score=mean_s,
            details={
                "mean_latency_s": mean_s,
                "median_latency_s": median_s,
                "p95_latency_s": p95_s,
                "num_matched": len(latencies_s),
                "num_misses": misses,
                "num_references": num_refs,
                "num_onset_predictions": num_onsets,
                "num_spurious_onsets": spurious,
                "max_latency_s": self.max_latency_s,
            },
        )


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile.

    Keeps the dependency surface to the stdlib. For single-element
    inputs returns the element; for empty inputs the caller guards
    against this.
    """
    if not values:
        raise ValueError("percentile of empty list is undefined")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(q * (len(ordered) - 1))))
    return ordered[k]
