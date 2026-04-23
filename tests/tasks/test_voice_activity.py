"""Tests for ``fd_eval.tasks.voice_activity`` (Slice 4).

Synthetic-fixture-only tests. No adapter runs here; we feed the
``evaluate`` method handcrafted prediction/reference event lists and
assert the scoring behaviour.
"""

from __future__ import annotations

import pytest

from fd_eval.core import AudioSession
from fd_eval.tasks import (
    TaskResult,
    VADPredictionEvent,
    VADReferenceEvent,
    VoiceActivityDetection,
)
from tests.fixtures import make_two_channel_alternating

SR = 24_000


def _session() -> AudioSession:
    audio = make_two_channel_alternating(
        segments=[("a", 0.2), ("silence", 0.2), ("b", 0.2)],
        sample_rate=SR,
    )
    return AudioSession(
        audio=audio,
        sample_rate=SR,
        input_channel_indices=[0],
        target_channel_indices=[1],
    )


def test_task_declares_declarative_attributes():
    assert VoiceActivityDetection.mode == "observer"
    assert VoiceActivityDetection.scoring_method == "algorithmic"


def test_rejects_negative_tolerance():
    with pytest.raises(ValueError):
        VoiceActivityDetection(tolerance_s=-0.1)


def test_perfect_match_yields_f1_one():
    task = VoiceActivityDetection(tolerance_s=0.05)
    preds = iter(
        [
            VADPredictionEvent(timestamp_s=0.10, channel=0, is_speech=True),
            VADPredictionEvent(timestamp_s=0.40, channel=0, is_speech=False),
        ]
    )
    refs = [
        VADReferenceEvent(timestamp_s=0.10, channel=0, is_speech=True),
        VADReferenceEvent(timestamp_s=0.40, channel=0, is_speech=False),
    ]
    result = task.evaluate(_session(), preds, refs)
    assert isinstance(result, TaskResult)
    assert result.score == pytest.approx(1.0)
    assert result.details["precision"] == pytest.approx(1.0)
    assert result.details["recall"] == pytest.approx(1.0)
    assert result.details["true_positives"] == 2
    assert result.details["false_positives"] == 0
    assert result.details["false_negatives"] == 0


def test_off_by_tolerance_still_matches():
    task = VoiceActivityDetection(tolerance_s=0.2)
    preds = iter([VADPredictionEvent(timestamp_s=0.30, channel=0, is_speech=True)])
    refs = [VADReferenceEvent(timestamp_s=0.20, channel=0, is_speech=True)]
    result = task.evaluate(_session(), preds, refs)
    assert result.score == pytest.approx(1.0)


def test_channel_mismatch_does_not_match():
    task = VoiceActivityDetection(tolerance_s=0.2)
    preds = iter([VADPredictionEvent(timestamp_s=0.10, channel=1, is_speech=True)])
    refs = [VADReferenceEvent(timestamp_s=0.10, channel=0, is_speech=True)]
    result = task.evaluate(_session(), preds, refs)
    assert result.score == pytest.approx(0.0)
    assert result.details["false_positives"] == 1
    assert result.details["false_negatives"] == 1


def test_polarity_mismatch_does_not_match():
    task = VoiceActivityDetection(tolerance_s=0.2)
    preds = iter([VADPredictionEvent(timestamp_s=0.10, channel=0, is_speech=False)])
    refs = [VADReferenceEvent(timestamp_s=0.10, channel=0, is_speech=True)]
    result = task.evaluate(_session(), preds, refs)
    assert result.score == pytest.approx(0.0)


def test_greedy_matching_locks_closest_first():
    # Two predictions compete for one reference; the closer prediction
    # should take the reference and leave the other as a false positive.
    task = VoiceActivityDetection(tolerance_s=0.5)
    preds = iter(
        [
            VADPredictionEvent(timestamp_s=0.10, channel=0, is_speech=True),
            VADPredictionEvent(timestamp_s=0.45, channel=0, is_speech=True),
        ]
    )
    refs = [VADReferenceEvent(timestamp_s=0.50, channel=0, is_speech=True)]
    result = task.evaluate(_session(), preds, refs)
    # 1 tp, 1 fp, 0 fn.
    assert result.details["true_positives"] == 1
    assert result.details["false_positives"] == 1
    assert result.details["false_negatives"] == 0
    # Precision 0.5, recall 1.0 -> F1 = 2 * 0.5 * 1 / 1.5 = 0.666...
    assert result.score == pytest.approx(2 / 3)


def test_empty_streams_give_zero_f1_not_nan():
    task = VoiceActivityDetection(tolerance_s=0.2)
    result = task.evaluate(_session(), iter([]), [])
    assert result.score == 0.0
    assert result.details["num_predictions"] == 0
    assert result.details["num_references"] == 0


def test_wrong_event_type_raises():
    from fd_eval.tasks import TurnTakingPredictionEvent

    task = VoiceActivityDetection()
    preds = iter([TurnTakingPredictionEvent(timestamp_s=0.1, channel=0, event_kind="onset")])
    with pytest.raises(TypeError):
        task.evaluate(_session(), preds, [])
