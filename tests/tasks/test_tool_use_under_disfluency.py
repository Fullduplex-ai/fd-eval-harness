import numpy as np
import pytest

from fd_eval.core import AudioSession
from fd_eval.tasks._types import ToolCallPredictionEvent
from fd_eval.tasks.tool_use_under_disfluency import ToolUseReference, ToolUseUnderDisfluency


@pytest.fixture
def session():
    return AudioSession(
        audio=np.zeros((24000, 2), dtype=np.float32),
        sample_rate=24000,
        input_channel_indices=[0],
        target_channel_indices=[1],
    )


def test_tool_use_subset_matching(session):
    task = ToolUseUnderDisfluency()

    # Reference: expects location="Tokyo" and unit="C"
    references = [
        ToolUseReference(tool_name="weather", arguments={"location": "Tokyo", "unit": "C"})
    ]

    # Prediction: has location="Tokyo", unit="C", plus an extra argument "date"
    # This should be accepted because of subset matching.
    predictions = [
        ToolCallPredictionEvent(
            timestamp_s=1.0,
            channel=1,
            tool_name="weather",
            arguments={"location": "Tokyo", "unit": "C", "date": "today"},
        )
    ]

    result = task.evaluate(session, predictions, references)
    assert result.score == 1.0
    assert result.details["true_positives"] == 1


def test_tool_use_subset_matching_failure(session):
    task = ToolUseUnderDisfluency()

    references = [
        ToolUseReference(tool_name="weather", arguments={"location": "Tokyo", "unit": "C"})
    ]

    # Prediction is missing "unit", so it is NOT a superset of the reference.
    predictions = [
        ToolCallPredictionEvent(
            timestamp_s=1.0, channel=1, tool_name="weather", arguments={"location": "Tokyo"}
        )
    ]

    result = task.evaluate(session, predictions, references)
    assert result.score == 0.0
    assert result.details["true_positives"] == 0


def test_tool_use_wrong_tool_name(session):
    task = ToolUseUnderDisfluency()

    references = [ToolUseReference(tool_name="weather", arguments={"location": "Tokyo"})]

    predictions = [
        ToolCallPredictionEvent(
            timestamp_s=1.0, channel=1, tool_name="search", arguments={"location": "Tokyo"}
        )
    ]

    result = task.evaluate(session, predictions, references)
    assert result.score == 0.0


def test_parse_references():
    task = ToolUseUnderDisfluency()
    raw_labels = [{"tool_name": "weather", "arguments": {"location": "Tokyo"}}]

    refs = task.parse_references(raw_labels)
    assert len(refs) == 1
    assert refs[0].tool_name == "weather"
    assert refs[0].arguments == {"location": "Tokyo"}
