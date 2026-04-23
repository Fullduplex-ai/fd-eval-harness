from abc import ABC, abstractmethod

from .audio_session import AudioSession
from .prediction_stream import PredictionStream


class FDModelAdapter(ABC):
    @abstractmethod
    def process(self, session: AudioSession) -> PredictionStream:
        """Process an AudioSession and yield prediction events."""
        pass
