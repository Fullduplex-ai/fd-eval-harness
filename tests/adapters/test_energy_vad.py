import numpy as np

from fd_eval.adapters.energy_vad import EnergyVADAdapter
from fd_eval.core import AudioSession
from fd_eval.tasks._types import VADPredictionEvent


def test_energy_vad_adapter():
    adapter = EnergyVADAdapter(threshold=0.01)

    # Create a 60ms audio at 24kHz (1440 samples)
    # 20ms chunk = 480 samples
    audio = np.zeros((1440, 2))

    # Chunk 0 (0-20ms): Silence on 0, Loud on 1
    # Chunk 1 (20-40ms): Loud on 0, Silence on 1
    # Chunk 2 (40-60ms): Silence on 0, Silence on 1

    # Channel 0
    audio[480:960, 0] = 1.0

    # Channel 1
    audio[0:480, 1] = 1.0

    session = AudioSession(
        audio=audio,
        sample_rate=24000,
        input_channel_indices=[0, 1],
        target_channel_indices=[],
    )

    stream = adapter.process(session)
    events = list(stream)

    assert len(events) == 4

    # Chunk 0
    assert isinstance(events[0], VADPredictionEvent)
    assert events[0].timestamp_s == 0.0
    assert events[0].channel == 1
    assert events[0].is_speech is True

    # Chunk 1
    assert isinstance(events[1], VADPredictionEvent)
    assert events[1].timestamp_s == 0.02
    assert events[1].channel == 0
    assert events[1].is_speech is True

    assert isinstance(events[2], VADPredictionEvent)
    assert events[2].timestamp_s == 0.02
    assert events[2].channel == 1
    assert events[2].is_speech is False

    # Chunk 2
    assert isinstance(events[3], VADPredictionEvent)
    assert events[3].timestamp_s == 0.04
    assert events[3].channel == 0
    assert events[3].is_speech is False
