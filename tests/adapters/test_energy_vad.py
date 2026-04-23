import numpy as np

from fd_eval.adapters.energy_vad import EnergyVADAdapter
from fd_eval.core import AudioSession
from fd_eval.tasks._types import VADPredictionEvent


def _silent_session(duration_s: float = 0.1, num_channels: int = 2) -> AudioSession:
    sr = 24_000
    audio = np.zeros((int(sr * duration_s), num_channels))
    return AudioSession(
        audio=audio,
        sample_rate=sr,
        input_channel_indices=list(range(num_channels)),
        target_channel_indices=[],
    )


def test_energy_vad_adapter():
    adapter = EnergyVADAdapter(threshold=0.01)

    # 60ms audio at 24kHz (1440 samples), 20ms chunks = 480 samples.
    # Chunk 0 (0-20ms): silence on 0, loud on 1.
    # Chunk 1 (20-40ms): loud on 0, silence on 1.
    # Chunk 2 (40-60ms): silence on 0, silence on 1.
    audio = np.zeros((1440, 2))
    audio[480:960, 0] = 1.0
    audio[0:480, 1] = 1.0

    session = AudioSession(
        audio=audio,
        sample_rate=24000,
        input_channel_indices=[0, 1],
        target_channel_indices=[],
    )

    events = list(adapter.process(session))

    assert len(events) == 4

    assert isinstance(events[0], VADPredictionEvent)
    assert events[0].timestamp_s == 0.0
    assert events[0].channel == 1
    assert events[0].is_speech is True

    assert isinstance(events[1], VADPredictionEvent)
    assert events[1].timestamp_s == 0.02
    assert events[1].channel == 0
    assert events[1].is_speech is True

    assert isinstance(events[2], VADPredictionEvent)
    assert events[2].timestamp_s == 0.02
    assert events[2].channel == 1
    assert events[2].is_speech is False

    assert isinstance(events[3], VADPredictionEvent)
    assert events[3].timestamp_s == 0.04
    assert events[3].channel == 0
    assert events[3].is_speech is False


def test_energy_vad_emits_no_events_for_pure_silence():
    """Silence across the whole session must not fire any state changes."""
    adapter = EnergyVADAdapter(threshold=0.01)
    events = list(adapter.process(_silent_session()))
    assert events == []


def test_energy_vad_emits_only_state_changes_not_per_chunk():
    """Sustained speech emits exactly one onset, not one per chunk."""
    sr = 24_000
    # 100 ms of sustained loud audio on channel 0 = 5 x 20ms chunks.
    audio = np.zeros((int(sr * 0.1), 2))
    audio[:, 0] = 1.0
    session = AudioSession(
        audio=audio,
        sample_rate=sr,
        input_channel_indices=[0, 1],
        target_channel_indices=[],
    )

    events = list(EnergyVADAdapter(threshold=0.01).process(session))

    onsets_ch0 = [e for e in events if e.channel == 0 and e.is_speech]
    assert len(onsets_ch0) == 1
    assert onsets_ch0[0].timestamp_s == 0.0


def test_energy_vad_ignores_non_input_channels():
    """Channels not in input_channel_indices must not contribute events."""
    sr = 24_000
    audio = np.zeros((int(sr * 0.04), 2))
    audio[:, 1] = 1.0  # loud on channel 1
    session = AudioSession(
        audio=audio,
        sample_rate=sr,
        input_channel_indices=[0],  # only channel 0 is observed
        target_channel_indices=[1],
    )

    events = list(EnergyVADAdapter(threshold=0.01).process(session))

    assert all(e.channel == 0 for e in events)
    # Channel 0 is silent throughout, so no state change fires.
    assert events == []
