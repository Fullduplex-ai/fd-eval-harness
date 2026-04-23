"""Voice Activity Detection — observer-mode reference task (v0.1).

Scoring: event-detection F1 with a symmetric time-tolerance window. A
predicted onset/offset event matches a reference event if and only if
they share the same channel, the same ``is_speech`` polarity, and their
timestamps fall within ``tolerance_s`` of each other. Each reference
event may match at most one prediction and vice versa, computed via a
greedy nearest-first pairing within the tolerance band.

This is deliberately simpler than segmentation-IoU scoring. It scores
what the harness cares about at v0.1 — whether the adapter places state
changes at approximately the right moment — and leaves segmentation-area
metrics for later.

See ``docs/TASKS.md`` for the observer/participant split and
``_internal/DECISIONS.md`` D008 / D009.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from fd_eval.core import AudioSession, PredictionStream, Task

from ._types import TaskResult, VADPredictionEvent


@dataclass(frozen=True)
class VADReferenceEvent:
    """Ground-truth VAD state change. Same shape as the prediction."""

    timestamp_s: float
    channel: int
    is_speech: bool


class VoiceActivityDetection(Task):
    """Observer-mode VAD scored by event-detection F1.

    Parameters
    ----------
    tolerance_s:
        Symmetric matching window in seconds. A predicted event matches
        a reference event if the absolute time difference is at most
        ``tolerance_s``. Default 0.2 s (200 ms), which matches the
        common benchmark convention for frame-level VAD evaluated as
        event boundaries.
    """

    name = "voice_activity_detection"
    version = "0.1.0"
    mode = "observer"
    scoring_method = "algorithmic"

    def __init__(self, tolerance_s: float = 0.2) -> None:
        if tolerance_s < 0:
            raise ValueError(f"tolerance_s must be non-negative, got {tolerance_s}")
        self.tolerance_s = tolerance_s

    def parse_references(self, raw_labels: list[dict]) -> Sequence[VADReferenceEvent]:
        refs = []
        for d in raw_labels:
            if "timestamp_s" in d and "channel" in d and "is_speech" in d:
                refs.append(
                    VADReferenceEvent(
                        timestamp_s=float(d["timestamp_s"]),
                        channel=int(d["channel"]),
                        is_speech=bool(d["is_speech"]),
                    )
                )
        return refs

    def evaluate(
        self,
        session: AudioSession,
        predictions: PredictionStream,
        references: Sequence[VADReferenceEvent],
    ) -> TaskResult:
        # Materialize the stream — VAD scoring is not streaming, it is
        # a set-matching problem over the whole session.
        pred_events: list[VADPredictionEvent] = []
        for ev in predictions:
            if not isinstance(ev, VADPredictionEvent):
                raise TypeError(
                    f"VoiceActivityDetection expects VADPredictionEvent, got {type(ev).__name__}"
                )
            pred_events.append(ev)

        matched_pred, matched_ref = _match_events_greedy(
            predictions=pred_events,
            references=list(references),
            tolerance_s=self.tolerance_s,
        )

        tp = len(matched_pred)
        fp = len(pred_events) - tp
        fn = len(references) - len(matched_ref)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return TaskResult(
            score=f1,
            details={
                "precision": precision,
                "recall": recall,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "tolerance_s": self.tolerance_s,
                "num_predictions": len(pred_events),
                "num_references": len(references),
            },
        )


def _match_events_greedy(
    *,
    predictions: list[VADPredictionEvent],
    references: list[VADReferenceEvent],
    tolerance_s: float,
) -> tuple[set[int], set[int]]:
    """Greedy nearest-first event matching within tolerance.

    Returns the indices of matched predictions and matched references.
    A prediction and a reference are eligible to match only if they
    share the same channel and same ``is_speech`` polarity, and if the
    absolute time difference does not exceed ``tolerance_s``. Each side
    is matched at most once.
    """
    candidates: list[tuple[float, int, int]] = []
    for pi, p in enumerate(predictions):
        for ri, r in enumerate(references):
            if p.channel != r.channel or p.is_speech != r.is_speech:
                continue
            dt = abs(p.timestamp_s - r.timestamp_s)
            if dt <= tolerance_s:
                candidates.append((dt, pi, ri))

    # Sort by distance so the closest pairs are locked in first.
    candidates.sort(key=lambda c: c[0])

    matched_pred: set[int] = set()
    matched_ref: set[int] = set()
    for _, pi, ri in candidates:
        if pi in matched_pred or ri in matched_ref:
            continue
        matched_pred.add(pi)
        matched_ref.add(ri)

    return matched_pred, matched_ref
