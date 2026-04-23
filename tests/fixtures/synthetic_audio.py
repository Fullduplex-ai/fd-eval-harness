"""Synthetic audio generators for fd-eval-harness tests.

These helpers produce small deterministic audio arrays that can be wrapped
in an ``AudioSession`` without touching the filesystem. They exist so that
core tests (streaming contract, channel-role invariants, adapter
round-trips) do not depend on any real conversational recordings.

No real audio or reference labels are distributed with this repository.
Real data lives in benchmark plugin packages (see ``docs/DESIGN.md`` §12
item 3).
"""

from __future__ import annotations

import numpy as np

DEFAULT_SAMPLE_RATE = 24_000


def make_silence(duration_s: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Return a 1-D silence array of the given duration in seconds."""
    if duration_s < 0:
        raise ValueError(f"duration_s must be non-negative, got {duration_s}")
    n = round(duration_s * sample_rate)
    return np.zeros(n, dtype=np.float32)


def make_sine(
    duration_s: float,
    freq_hz: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Return a 1-D sine wave of the given frequency, amplitude, and duration.

    Amplitude is in the same units as the output array (peak value);
    0.5 is a reasonable default that avoids clipping at unit scale.
    """
    if duration_s < 0:
        raise ValueError(f"duration_s must be non-negative, got {duration_s}")
    if freq_hz <= 0:
        raise ValueError(f"freq_hz must be positive, got {freq_hz}")

    n = round(duration_s * sample_rate)
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def make_two_channel_sine(
    duration_s: float,
    freq_left_hz: float = 220.0,
    freq_right_hz: float = 440.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Return a (N, 2) array with independent sine tones per channel.

    Useful for speaker-separation-style tests where the two channels
    should carry distinct content at all times.
    """
    left = make_sine(duration_s, freq_left_hz, sample_rate, amplitude)
    right = make_sine(duration_s, freq_right_hz, sample_rate, amplitude)
    return np.stack([left, right], axis=-1)


def make_two_channel_alternating(
    segments: list[tuple[str, float]],
    freq_a_hz: float = 220.0,
    freq_b_hz: float = 440.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Return a (N, 2) array for turn-taking-style fixtures.

    ``segments`` is a list of ``(who, duration_s)`` pairs where ``who`` is
    one of ``"a"``, ``"b"``, or ``"silence"``. Channel 0 carries speaker
    A's content; channel 1 carries speaker B's content; the non-speaking
    channel is silent during that segment. This gives a clean 2-channel
    fixture that exercises both VAD-style and turn-taking-latency-style
    tasks without needing real audio.

    The total length is the sum of the per-segment durations rounded to
    the nearest integer sample count.
    """
    blocks: list[np.ndarray] = []
    for who, dur in segments:
        if dur < 0:
            raise ValueError(f"segment duration must be non-negative, got {dur}")
        n = round(dur * sample_rate)
        block = np.zeros((n, 2), dtype=np.float32)
        if who == "a":
            block[:, 0] = make_sine(dur, freq_a_hz, sample_rate, amplitude)[:n]
        elif who == "b":
            block[:, 1] = make_sine(dur, freq_b_hz, sample_rate, amplitude)[:n]
        elif who == "silence":
            pass
        else:
            raise ValueError(f"segment 'who' must be one of 'a', 'b', 'silence'; got {who!r}")
        blocks.append(block)

    if not blocks:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(blocks, axis=0)
