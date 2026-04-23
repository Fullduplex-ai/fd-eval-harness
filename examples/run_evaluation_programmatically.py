"""Example: Running fd-eval-harness programmatically without the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fd_eval.core.audio_session import AudioSession
from fd_eval.core.registry import get_adapter, get_task


def main():
    print("1. Creating dummy 2-channel audio (5 seconds, 24kHz)...")
    sample_rate = 24000
    audio = np.zeros((sample_rate * 5, 2), dtype=np.float32)

    # Simulate speaker A (channel 0) speaking from 1.0s to 2.0s
    # Using a 440Hz sine wave as "speech" for the EnergyVAD adapter to detect
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio[sample_rate : sample_rate * 2, 0] = 0.5 * np.sin(2 * np.pi * 440 * t)

    print("2. Defining ground-truth reference labels...")
    raw_labels = [
        {"timestamp_s": 1.0, "channel": 0, "is_speech": True},
        {"timestamp_s": 2.0, "channel": 0, "is_speech": False},
    ]

    print("3. Constructing AudioSession (Observer Mode)...")
    session = AudioSession(
        audio=audio,
        sample_rate=sample_rate,
        # Observer mode: both channels are inputs, no targets
        input_channel_indices=[0, 1],
        target_channel_indices=[],
    )

    print("4. Running the Model Adapter (EnergyVAD)...")
    AdapterCls = get_adapter("energy_vad")
    adapter = AdapterCls(threshold=0.05)

    # Process returns an iterator (PredictionStream), we materialize it to a list
    predictions = list(adapter.process(session))
    print(f"   -> Adapter emitted {len(predictions)} events.")

    print("5. Evaluating the Task (VoiceActivityDetection)...")
    TaskCls = get_task("voice_activity_detection")
    task = TaskCls(tolerance_s=0.2)

    references = task.parse_references(raw_labels)
    result = task.evaluate(session, predictions, references)

    print("\n=== Evaluation Results ===")
    print(f"Task: {TaskCls.name} (v{TaskCls.version})")
    print(f"Score (F1): {result.score:.3f}")
    print("Details:")
    print(json.dumps(result.details, indent=2))


if __name__ == "__main__":
    main()
