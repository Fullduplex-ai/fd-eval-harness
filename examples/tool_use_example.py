#!/usr/bin/env python3
"""Example script demonstrating the tool_use_under_disfluency task in programmatic mode."""

import json
import tempfile
from pathlib import Path
import numpy as np

from fd_eval.core import AudioSession
from fd_eval.tasks.tool_use_under_disfluency import ToolUseUnderDisfluency
from fd_eval.adapters.tool_use_stub import ToolUseStubAdapter

def run_example():
    print("=== fd-eval-harness: Tool Use Example ===")
    
    # 1. Prepare synthetic session
    # We use a dummy audio array since the stub adapter doesn't actually process it.
    session = AudioSession(
        audio=np.zeros((24000, 2), dtype=np.float32),
        sample_rate=24000,
        input_channel_indices=[0],
        target_channel_indices=[1],
    )
    
    # 2. Setup the adapter
    # For this example, we provide a stub file so the adapter emits specific tool calls.
    stub_data = [
        {
            "timestamp_s": 2.5,
            "tool_name": "weather",
            "arguments": {"location": "Tokyo", "unit": "C"}
        }
    ]
    
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(stub_data, f)
        stub_path = f.name
        
    try:
        adapter = ToolUseStubAdapter(stub_file=stub_path)
        
        # 3. Setup the task and references
        task = ToolUseUnderDisfluency()
        
        # We expect the model to emit a tool call for "weather" with location "Tokyo".
        raw_labels = [
            {
                "tool_name": "weather",
                "arguments": {"location": "Tokyo"}
            }
        ]
        references = task.parse_references(raw_labels)
        
        # 4. Evaluate
        print("Running adapter process...")
        predictions = list(adapter.process(session))
        
        print("\nEvaluating predictions against references...")
        result = task.evaluate(session, predictions, references)
        
        print("\n=== Result ===")
        print(f"Task: {task.name} (v{task.version})")
        print(f"F1 Score: {result.score:.2f}")
        print(f"Details: {result.details}")
        
    finally:
        Path(stub_path).unlink()

if __name__ == "__main__":
    run_example()
