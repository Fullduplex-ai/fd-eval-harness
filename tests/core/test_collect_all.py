import numpy as np

from fd_eval.core import AudioSession, collect_all


def test_collect_all():
    audio = np.random.rand(16000, 2)
    session = AudioSession(
        audio=audio,
        sample_rate=16000,
        input_channel_indices=[0],
        target_channel_indices=[1],
    )

    collected = collect_all(session)
    assert collected.shape == audio.shape
    np.testing.assert_array_equal(collected, audio)


def test_collect_all_empty():
    audio = np.zeros((0, 2))
    session = AudioSession(
        audio=audio,
        sample_rate=16000,
        input_channel_indices=[0],
        target_channel_indices=[1],
    )

    collected = collect_all(session)
    assert collected.shape == (0, 2)
