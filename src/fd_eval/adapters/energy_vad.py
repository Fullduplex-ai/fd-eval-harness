import numpy as np

from fd_eval.core import AudioSession, FDModelAdapter, PredictionStream
from fd_eval.tasks._types import VADPredictionEvent


class EnergyVADAdapter(FDModelAdapter):
    """
    A simple energy-based Voice Activity Detection adapter.
    Computes RMS energy of the audio chunks and yields VADPredictionEvent
    when the energy crosses the configured threshold.
    Serves as a lightweight reference adapter for Observer-mode tasks.
    """

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def process(self, session: AudioSession) -> PredictionStream:
        state = {ch: False for ch in session.input_channel_indices}

        chunk_ms = 20
        frame_size = int((chunk_ms / 1000) * session.sample_rate)

        for i, chunk in enumerate(session.stream(chunk_ms=chunk_ms)):
            timestamp_s = (i * frame_size) / session.sample_rate

            for ch in session.input_channel_indices:
                channel_audio = chunk[:, ch]
                if len(channel_audio) == 0:
                    continue

                rms = np.sqrt(np.mean(channel_audio**2))
                is_speech = bool(rms > self.threshold)

                if is_speech and not state[ch]:
                    state[ch] = True
                    yield VADPredictionEvent(timestamp_s=timestamp_s, channel=ch, is_speech=True)
                elif not is_speech and state[ch]:
                    state[ch] = False
                    yield VADPredictionEvent(timestamp_s=timestamp_s, channel=ch, is_speech=False)
