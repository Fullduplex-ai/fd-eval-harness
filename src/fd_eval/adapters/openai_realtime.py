"""Stub adapter for the OpenAI Realtime API.

This is a placeholder for the actual OpenAI Realtime API integration
planned for v0.2. It currently does not make any network requests.
"""

import asyncio
import base64
import json
import logging
import os
import queue
import threading

import numpy as np
import websockets

from fd_eval.core import AudioSession, FDModelAdapter, PredictionStream
from fd_eval.tasks._types import (
    ToolCallPredictionEvent,
    TranscriptPredictionEvent,
    TurnTakingPredictionEvent,
)

logger = logging.getLogger(__name__)


class OpenAIRealtimeAdapter(FDModelAdapter):
    """Adapter for openai-realtime-api.

    This adapter handles real network streaming via the OpenAI WebSocket API.
    Per D016, it relies on server-side VAD and streams audio in real-time (1x pacing).
    """

    def __init__(
        self,
        model: str = "gpt-4o-realtime-preview-2024-10-01",
        voice: str = "alloy",
        timeout_s: float = 10.0,
    ):
        self.model = model
        self.voice = voice
        self.timeout_s = timeout_s
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required.")

    def process(self, session: AudioSession) -> PredictionStream:
        """Process the audio session by streaming to OpenAI Realtime API."""
        if session.sample_rate != 24000:
            raise ValueError(
                f"OpenAIRealtimeAdapter requires 24000 Hz audio, got {session.sample_rate}"
            )

        q = queue.Queue()
        stop_event = threading.Event()

        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._run_websocket(session, q, stop_event))
            finally:
                loop.close()

        t = threading.Thread(target=run_loop)
        t.start()

        while not stop_event.is_set() or not q.empty():
            try:
                event = q.get(timeout=0.1)
                yield event
            except queue.Empty:
                continue

        t.join()

    async def _run_websocket(
        self, session: AudioSession, q: queue.Queue, stop_event: threading.Event
    ):
        url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        start_time = asyncio.get_event_loop().time()
        try:
            async with websockets.connect(url, extra_headers=headers) as ws:
                # 1. Update session to use server_vad
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {
                                "modalities": ["text", "audio"],
                                "voice": self.voice,
                                "turn_detection": {"type": "server_vad"},
                            },
                        }
                    )
                )

                # 2. Start concurrent sender and receiver tasks
                sender_task = asyncio.create_task(self._send_audio(ws, session))
                receiver_task = asyncio.create_task(self._receive_events(ws, q, start_time))

                await asyncio.gather(sender_task, receiver_task)
        except Exception as e:
            # Per D016 Edit 1: Network errors currently swallow exceptions
            # and return an empty stream.
            # A run-level adapter_errors counter should be added in v0.3.
            logger.error(f"WebSocket error in Realtime Adapter: {e}")
        finally:
            stop_event.set()

    async def _send_audio(self, ws: websockets.WebSocketClientProtocol, session: AudioSession):
        chunk_ms = 20

        for chunk in session.stream(chunk_ms=chunk_ms):
            input_audio = chunk[:, session.input_channel_indices]

            if input_audio.dtype == np.float32 or input_audio.dtype == np.float64:
                input_audio = np.clip(input_audio, -1.0, 1.0)
                input_audio = (input_audio * 32767).astype(np.int16)

            mono_chunk = input_audio[:, 0]
            b64_chunk = base64.b64encode(mono_chunk.tobytes()).decode("utf-8")

            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": b64_chunk,
                    }
                )
            )

            # D016 Real-time pacing
            await asyncio.sleep(chunk_ms / 1000.0)

        # Commit to signal end of stream
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # Wait a bit to ensure the model's final response is received before closing
        await asyncio.sleep(self.timeout_s)
        # End sender task, which will also complete the gather and eventually close the websocket.

    async def _receive_events(
        self, ws: websockets.WebSocketClientProtocol, q: queue.Queue, start_time: float
    ):
        async for message in ws:
            event = json.loads(message)
            event_type = event.get("type")
            current_time = asyncio.get_event_loop().time() - start_time

            if event_type == "response.audio_transcript.done":
                q.put(
                    TranscriptPredictionEvent(
                        timestamp_s=current_time, text=event.get("transcript", "")
                    )
                )
            elif event_type == "conversation.item.truncated":
                q.put(TurnTakingPredictionEvent(timestamp_s=current_time))
            elif event_type == "response.function_call_arguments.done":
                args_str = event.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}
                q.put(
                    ToolCallPredictionEvent(
                        timestamp_s=current_time, tool_name=event.get("name", ""), arguments=args
                    )
                )
            elif event_type == "error":
                logger.error(f"OpenAI API Error: {event}")
