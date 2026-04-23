"""Round-trip tests: synthetic fixtures ↔ AudioSession contract.

These tests exercise the tests/fixtures/ helpers against the real
``AudioSession`` + ``stream()`` + ``collect_all()`` code path. They are a
slice-2 smoke test: they do not introduce any new core functionality,
they just confirm the fixtures plug into the adapter-facing contract.
"""

from __future__ import annotations

import numpy as np
import pytest
from tests.fixtures import (
    make_silence,
    make_sine,
    make_two_channel_alternating,
    make_two_channel_sine,
)

from fd_eval.core import AudioSession, collect_all

SR = 24_000


def test_silence_has_expected_length_and_is_zero():
    audio = make_silence(0.25, sample_rate=SR)
    assert audio.shape == (int(0.25 * SR),)
    assert np.all(audio == 0.0)


def test_sine_has_expected_amplitude_envelope():
    audio = make_sine(0.1, freq_hz=440.0, sample_rate=SR, amplitude=0.5)
    # Peak amplitude bounded by requested amplitude (exact equality not
    # guaranteed for short signals; allow tiny float headroom).
    assert np.max(np.abs(audio)) <= 0.5 + 1e-6
    assert np.max(np.abs(audio)) > 0.4  # non-trivially energetic


def test_two_channel_sine_preserves_channel_independence():
    audio = make_two_channel_sine(0.1, sample_rate=SR)
    assert audio.shape == (int(0.1 * SR), 2)
    # The two channels carry different frequencies, so the per-sample
    # difference must be non-zero somewhere.
    assert not np.allclose(audio[:, 0], audio[:, 1])


def test_alternating_fixture_isolates_active_channel_per_segment():
    audio = make_two_channel_alternating(
        segments=[("a", 0.05), ("silence", 0.05), ("b", 0.05)],
        sample_rate=SR,
    )
    assert audio.shape == (int(0.15 * SR), 2)

    n = int(0.05 * SR)
    # Segment 1: only channel 0 active.
    assert np.any(audio[:n, 0] != 0.0)
    assert np.all(audio[:n, 1] == 0.0)
    # Segment 2: both channels silent.
    assert np.all(audio[n : 2 * n, :] == 0.0)
    # Segment 3: only channel 1 active.
    assert np.all(audio[2 * n :, 0] == 0.0)
    assert np.any(audio[2 * n :, 1] != 0.0)


def test_alternating_fixture_rejects_unknown_segment_label():
    with pytest.raises(ValueError):
        make_two_channel_alternating(
            segments=[("c", 0.05)],
            sample_rate=SR,
        )


def test_fixture_flows_through_audio_session_stream_and_collect_all():
    audio = make_two_channel_alternating(
        segments=[("a", 0.04), ("b", 0.04)],
        sample_rate=SR,
    )
    session = AudioSession(
        audio=audio,
        sample_rate=SR,
        input_channel_indices=[0],
        target_channel_indices=[1],
    )

    reassembled = collect_all(session)
    assert reassembled.shape == audio.shape
    np.testing.assert_array_equal(reassembled, audio)

    # Streaming at 20ms emits the expected number of chunks (last chunk
    # may be short; total samples must match).
    chunks = list(session.stream(chunk_ms=20))
    total = sum(c.shape[0] for c in chunks)
    assert total == audio.shape[0]
    # All chunks are 2-channel (D007 round-trip check).
    assert all(c.shape[1] == 2 for c in chunks)


def test_audio_session_rejects_overlapping_input_and_target_channels():
    audio = make_two_channel_sine(0.05, sample_rate=SR)
    with pytest.raises(ValueError):
        AudioSession(
            audio=audio,
            sample_rate=SR,
            input_channel_indices=[0],
            target_channel_indices=[0],
        )
