"""Stub adapter for testing ToolUseUnderDisfluency.

Since Moshi natively cannot emit tool calls, this adapter exists purely to
test the task evaluation logic. It emits a hardcoded ToolCallPredictionEvent
when its process() method is called.
"""

from __future__ import annotations

import json
from pathlib import Path

from fd_eval.core import AudioSession, FDModelAdapter, PredictionStream
from fd_eval.tasks._types import ToolCallPredictionEvent


class ToolUseStubAdapter(FDModelAdapter):
    """Emits predefined tool calls based on a stub file if provided, else a hardcoded one."""

    def __init__(self, stub_file: str | Path | None = None):
        self.stub_file = stub_file

    def process(self, session: AudioSession) -> PredictionStream:
        # If no target channels, we don't have anywhere to "emit" the tool call,
        # but we'll default to 0 if empty for testing purposes.
        channel = session.target_channel_indices[0] if session.target_channel_indices else 0

        if self.stub_file:
            path = Path(self.stub_file)
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                    for item in data:
                        yield ToolCallPredictionEvent(
                            timestamp_s=item.get("timestamp_s", 0.0),
                            channel=channel,
                            tool_name=item.get("tool_name", "unknown"),
                            arguments=item.get("arguments", {}),
                        )
                return

        # Default behavior: yield a single generic tool call
        yield ToolCallPredictionEvent(
            timestamp_s=1.0,
            channel=channel,
            tool_name="weather",
            arguments={"location": "Tokyo"},
        )
