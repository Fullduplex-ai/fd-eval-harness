"""Tests for ``fd_eval.tasks.turn_taking_latency`` (Slice 4).

Synthetic-fixture-only tests. Validates onset pairing against partner
offset references, window handling, and misses vs spurious accounting.
"""

from __future__ import annotations

import pytest

from fd_eval.core import AudioSession
from fd_eval.tasks import (
    PartnerOffsetReference,
    TaskResult,
    TurnTakingLatency,
    TurnTakingPredictionEvent,
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
    assert TurnTakingLatency.mode == "participant"
    assert TurnTakingLatency.scoring_method == "algorithmic"


def test_rejects_non_positive_max_latency():
    with pytest.raises(ValueError):
        TurnTakingLatency(max_latency_s=0.0)
    with pytest.raises(ValueError):
        TurnTakingLatency(max_latency_s=-1.0)


def test_single_reference_single_onset_gives_exact_latency():
    task = TurnTakingLatency(max_latency_s=3.0)
    preds = iter([TurnTakingPredictionEvent(timestamp_s=0.45, channel=1, event_kind="onset")])
    refs = [PartnerOffsetReference(timestamp_s=0.20, channel=0)]
    result = task.evaluate(_session(), preds, refs)
    assert isinstance(result, TaskResult)
    assert result.score == pytest.approx(0.25)
    assert result.details["mean_latency_s"] == pytest.approx(0.25)
    assert result.details["median_latency_s"] == pytest.approx(0.25)
    assert result.details["p95_latency_s"] == pytest.approx(0.25)
    assert result.details["num_matched"] == 1
    assert result.details["num_misses"] == 0
    assert result.details["num_spurious_onsets"] == 0


def test_onset_before_reference_is_ignored():
    task = TurnTakingLatency(max_latency_s=3.0)
    # Onset fires before the partner has even stopped talking.
    preds = iter([TurnTakingPredictionEvent(timestamp_s=0.10, channel=1, event_kind="onset")])
    refs = [PartnerOffsetReference(timestamp_s=0.20, channel=0)]
    result = task.evaluate(_session(), preds, refs)
    assert result.details["num_matched"] == 0
    assert result.details["num_misses"] == 1
    assert result.details["num_spurious_onsets"] == 1


def test_onset_beyond_window_counts_as_miss():
    task = TurnTakingLatency(max_latency_s=0.5)
    preds = iter([TurnTakingPredictionEvent(timestamp_s=1.00, channel=1, event_kind="onset")])
    refs = [PartnerOffsetReference(timestamp_s=0.20, channel=0)]
    result = task.evaluate(_session(), preds, refs)
    assert result.details["num_matched"] == 0
    assert result.details["num_misses"] == 1
    # The onset was not consumed, so it still contributes to spurious.
    assert result.details["num_spurious_onsets"] == 1


def test_onset_on_non_target_channel_is_ignored():
    # Target channel in the session is [1]. An onset on channel 0
    # should never match, even if its timing is perfect.
    task = TurnTakingLatency(max_latency_s=3.0)
    preds = iter([TurnTakingPredictionEvent(timestamp_s=0.30, channel=0, event_kind="onset")])
    refs = [PartnerOffsetReference(timestamp_s=0.20, channel=0)]
    result = task.evaluate(_session(), preds, refs)
    assert result.details["num_matched"] == 0
    assert result.details["num_misses"] == 1
    # Filtered out before pairing, so it is not counted as a spurious
    # response onset on the target channel.
    assert result.details["num_spurious_onsets"] == 0
    assert result.details["num_onset_predictions"] == 0


def test_offset_events_are_ignored_for_v0_1():
    task = TurnTakingLatency(max_latency_s=3.0)
    preds = iter(
        [
            TurnTakingPredictionEvent(timestamp_s=0.30, channel=1, event_kind="offset"),
            TurnTakingPredictionEvent(timestamp_s=0.50, channel=1, event_kind="onset"),
        ]
    )
    refs = [PartnerOffsetReference(timestamp_s=0.20, channel=0)]
    result = task.evaluate(_session(), preds, refs)
    # Only the onset is eligible.
    assert result.details["num_matched"] == 1
    assert result.score == pytest.approx(0.3)


def test_closest_onset_is_matched_per_reference():
    # Two references, two onsets. Greedy forward-walk should pair each
    # reference with its nearest later onset.
    task = TurnTakingLatency(max_latency_s=2.0)
    preds = iter(
        [
            TurnTakingPredictionEvent(timestamp_s=0.50, channel=1, event_kind="onset"),
            TurnTakingPredictionEvent(timestamp_s=1.10, channel=1, event_kind="onset"),
        ]
    )
    refs = [
        PartnerOffsetReference(timestamp_s=0.30, channel=0),
        PartnerOffsetReference(timestamp_s=1.00, channel=0),
    ]
    result = task.evaluate(_session(), preds, refs)
    assert result.details["num_matched"] == 2
    # Latencies: 0.20, 0.10 -> mean 0.15.
    assert result.details["mean_latency_s"] == pytest.approx(0.15)
    assert result.details["num_spurious_onsets"] == 0


def test_no_references_gives_zero_score_and_no_misses():
    task = TurnTakingLatency(max_latency_s=3.0)
    preds = iter([TurnTakingPredictionEvent(timestamp_s=0.50, channel=1, event_kind="onset")])
    result = task.evaluate(_session(), preds, [])
    assert result.score == 0.0
    assert result.details["num_matched"] == 0
    assert result.details["num_misses"] == 0
    assert result.details["num_spurious_onsets"] == 1


def test_wrong_event_type_raises():
    from fd_eval.tasks import VADPredictionEvent

    task = TurnTakingLatency()
    preds = iter([VADPredictionEvent(timestamp_s=0.1, channel=1, is_speech=True)])
    with pytest.raises(TypeError):
        task.evaluate(_session(), preds, [])
