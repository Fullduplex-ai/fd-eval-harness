"""Stub adapter for the OpenAI Realtime API.

This is a placeholder for the actual OpenAI Realtime API integration
planned for v0.2. It currently does not make any network requests.
"""

from __future__ import annotations

from typing import Literal

from fd_eval.core import AudioSession, FDModelAdapter, PredictionStream

class OpenAIRealtimeAdapter(FDModelAdapter):
    """Adapter for openai-realtime-api (Stub).
    
    This adapter will be implemented in v0.2 to handle real network streaming
    via the OpenAI WebSocket API.
    """

    def __init__(self, model: str = "gpt-4o-realtime-preview", voice: Literal["alloy", "echo", "shimmer"] = "alloy"):
        self.model = model
        self.voice = voice
        # In the future, this would initialize an API client or WebSocket connection.

    def process(self, session: AudioSession) -> PredictionStream:
        """Process the audio session.
        
        Currently a stub that yields nothing.
        """
        # Yield nothing for now
        yield from []
