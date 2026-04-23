import numpy as np
import pytest

from fd_eval.core import AudioSession


def test_audio_session_valid():
    # 2 channels
    audio = np.zeros((100, 2))
    session = AudioSession(
        audio=audio,
        sample_rate=16000,
        input_channel_indices=[0],
        target_channel_indices=[1],
    )
    assert session.audio.shape == (100, 2)


def test_audio_session_1d_reshaped():
    # 1D array should be reshaped to (N, 1)
    audio = np.zeros(100)
    session = AudioSession(
        audio=audio,
        sample_rate=16000,
        input_channel_indices=[0],
        target_channel_indices=[],
    )
    assert session.audio.shape == (100, 1)


def test_audio_session_overlap_raises():
    audio = np.zeros((100, 2))
    with pytest.raises(ValueError, match="overlap"):
        AudioSession(
            audio=audio,
            sample_rate=16000,
            input_channel_indices=[0, 1],
            target_channel_indices=[1],
        )


def test_audio_session_out_of_bounds_raises():
    audio = np.zeros((100, 2))
    with pytest.raises(ValueError, match="out of bounds"):
        AudioSession(
            audio=audio,
            sample_rate=16000,
            input_channel_indices=[0],
            target_channel_indices=[2],
        )


def test_audio_session_stream():
    # 1 second of audio at 16000 Hz = 16000 samples
    audio = np.zeros((16000, 2))
    session = AudioSession(
        audio=audio,
        sample_rate=16000,
        input_channel_indices=[0],
        target_channel_indices=[1],
    )

    # 20ms chunks = 320 samples per chunk
    chunks = list(session.stream(chunk_ms=20))
    assert len(chunks) == 16000 / 320  # 50 chunks
    assert chunks[0].shape == (320, 2)
