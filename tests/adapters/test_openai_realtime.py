from unittest.mock import patch

import numpy as np
import pytest

from fd_eval.adapters.openai_realtime import OpenAIRealtimeAdapter
from fd_eval.core import AudioSession
from fd_eval.tasks._types import (
    ToolCallPredictionEvent,
    TranscriptPredictionEvent,
    TurnTakingPredictionEvent,
)


def test_openai_realtime_adapter(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")
    adapter = OpenAIRealtimeAdapter(model="gpt-4o", voice="alloy")
    assert adapter.model == "gpt-4o"
    assert adapter.voice == "alloy"

    session = AudioSession(
        audio=np.zeros((100, 2), dtype=np.float32),
        sample_rate=24000,
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


def test_sample_rate_rejection(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")
    adapter = OpenAIRealtimeAdapter()

    session = AudioSession(
        audio=np.zeros((100, 2), dtype=np.float32),
        sample_rate=16000,  # Invalid sample rate
        input_channel_indices=[0],
        target_channel_indices=[1],
    )

    with pytest.raises(ValueError, match="requires 24000 Hz"):
        list(adapter.process(session))


def test_float32_to_int16_quantization(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")
    adapter = OpenAIRealtimeAdapter()

    # Create float32 audio from -1.0 to 1.0
    audio_data = np.array([[1.0, 0.0], [-1.0, 0.0], [0.5, 0.0]], dtype=np.float32)
    session = AudioSession(
        audio=audio_data,
        sample_rate=24000,
        input_channel_indices=[0],
        target_channel_indices=[1],
    )

    class MockWS:
        def __init__(self):
            self.sent_messages = []

        async def send(self, msg):
            self.sent_messages.append(msg)

    ws = MockWS()

    async def run_test():
        task = asyncio.create_task(adapter._send_audio(ws, session))
        await asyncio.sleep(0.1)  # allow to yield
        task.cancel()

    import asyncio

    asyncio.run(run_test())

    # Check messages
    assert len(ws.sent_messages) > 0
    # First message should be audio buffer
    import base64
    import json

    msg = json.loads(ws.sent_messages[0])
    assert msg["type"] == "input_audio_buffer.append"

    raw_bytes = base64.b64decode(msg["audio"])
    decoded_int16 = np.frombuffer(raw_bytes, dtype=np.int16)

    # 1.0 -> 32767, -1.0 -> -32767, 0.5 -> 16383
    assert decoded_int16[0] == 32767
    assert decoded_int16[1] == -32767
    assert decoded_int16[2] == 16383


def test_event_type_mapping(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")
    adapter = OpenAIRealtimeAdapter()

    class MockWS:
        def __init__(self, messages):
            self.messages = messages
            self.idx = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.idx < len(self.messages):
                msg = self.messages[self.idx]
                self.idx += 1
                return msg
            raise StopAsyncIteration

    messages = [
        '{"type": "response.audio_transcript.done", "transcript": "test"}',
        '{"type": "conversation.item.truncated"}',
        '{"type": "response.function_call_arguments.done", "name": "foo", "arguments": "{\\"x\\": 1}"}',  # noqa: E501
    ]
    ws = MockWS(messages)

    import asyncio
    import queue

    q = queue.Queue()

    asyncio.run(adapter._receive_events(ws, q, start_time=0.0))

    assert q.qsize() == 3

    event1 = q.get()
    assert isinstance(event1, TranscriptPredictionEvent)
    assert event1.text == "test"

    event2 = q.get()
    assert isinstance(event2, TurnTakingPredictionEvent)

    event3 = q.get()
    assert isinstance(event3, ToolCallPredictionEvent)
    assert event3.tool_name == "foo"
    assert event3.arguments == {"x": 1}


def test_multi_input_channel_drops_extras(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")
    adapter = OpenAIRealtimeAdapter()

    audio_data = np.zeros((100, 3), dtype=np.float32)
    session = AudioSession(
        audio=audio_data,
        sample_rate=24000,
        input_channel_indices=[0, 1],  # 2 input channels
        target_channel_indices=[2],
    )

    class MockWS:
        def __init__(self):
            self.sent_messages = []

        async def send(self, msg):
            self.sent_messages.append(msg)

    ws = MockWS()

    import asyncio

    async def run_test():
        task = asyncio.create_task(adapter._send_audio(ws, session))
        await asyncio.sleep(0.1)
        task.cancel()

    asyncio.run(run_test())

    # Make sure we didn't crash and we sent audio
    assert len(ws.sent_messages) > 0
