import numpy as np

from fd_eval.adapters.openai_realtime import OpenAIRealtimeAdapter
from fd_eval.core import AudioSession


def test_openai_realtime_adapter_stub():
    adapter = OpenAIRealtimeAdapter(model="gpt-4o", voice="alloy")
    assert adapter.model == "gpt-4o"
    assert adapter.voice == "alloy"

    session = AudioSession(
        audio=np.zeros((100, 2), dtype=np.float32),
        sample_rate=16000,
        input_channel_indices=[0],
        target_channel_indices=[1],
    )

    predictions = list(adapter.process(session))
    assert len(predictions) == 0
