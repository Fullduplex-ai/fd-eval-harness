from unittest.mock import patch

import numpy as np

from fd_eval.adapters.openai_realtime import OpenAIRealtimeAdapter
from fd_eval.core import AudioSession
from fd_eval.tasks._types import TranscriptPredictionEvent


def test_openai_realtime_adapter(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")
    adapter = OpenAIRealtimeAdapter(model="gpt-4o", voice="alloy")
    assert adapter.model == "gpt-4o"
    assert adapter.voice == "alloy"

    session = AudioSession(
        audio=np.zeros((100, 2), dtype=np.float32),
        sample_rate=16000,
        input_channel_indices=[0],
        target_channel_indices=[1],
    )

    async def mock_run_websocket(self_obj, sess, q, stop_event):
        q.put(TranscriptPredictionEvent(timestamp_s=1.0, text="hello"))
        stop_event.set()

    with patch.object(OpenAIRealtimeAdapter, "_run_websocket", new=mock_run_websocket):
        predictions = list(adapter.process(session))
        assert len(predictions) == 1
        assert predictions[0].text == "hello"
