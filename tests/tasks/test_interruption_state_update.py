import json
from unittest.mock import MagicMock, patch

import pytest

from fd_eval.core import AudioSession
from fd_eval.tasks._types import TranscriptPredictionEvent
from fd_eval.tasks.interruption_state_update import InterruptionReference, InterruptionStateUpdate


@pytest.fixture
def session():
    import numpy as np

    return AudioSession(
        audio=np.zeros((100, 2), dtype=np.float32),
        sample_rate=16000,
        input_channel_indices=[0],
        target_channel_indices=[1],
    )


def test_interruption_state_update_success(session):
    # Mock openai.Client to return a successful score
    with patch("fd_eval.tasks.interruption_state_update.openai") as mock_openai:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(
            {"score": 1.0, "reasoning": "The agent followed the new instruction."}
        )
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 20
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai.Client.return_value = mock_client

        task = InterruptionStateUpdate(judge_model="gpt-4o-2024-05-13")

        references = [
            InterruptionReference(
                initial_instruction="Tell me a joke.",
                interrupted_at_s=1.5,
                final_instruction="Actually, sing a song instead.",
            )
        ]

        predictions = [
            TranscriptPredictionEvent(timestamp_s=0.0, channel=1, text="Sure, here is a song.")
        ]

        result = task.evaluate(session, predictions, references)

        assert result.score == 1.0
        assert result.details["prompt_tokens"] == 50
        assert result.details["completion_tokens"] == 20
        assert result.details["num_judgments"] == 1

        # Verify prompt construction
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "gpt-4o-2024-05-13"
        prompt = call_args["messages"][0]["content"]
        assert "Tell me a joke." in prompt
        assert "Actually, sing a song instead." in prompt
        assert "Sure, here is a song." in prompt


def test_interruption_state_update_failure(session):
    with patch("fd_eval.tasks.interruption_state_update.openai") as mock_openai:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(
            {"score": 0.0, "reasoning": "The agent kept telling a joke."}
        )
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 20
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai.Client.return_value = mock_client

        task = InterruptionStateUpdate()

        references = [
            InterruptionReference(
                initial_instruction="Tell me a joke.",
                interrupted_at_s=1.5,
                final_instruction="Actually, sing a song instead.",
            )
        ]

        predictions = [
            TranscriptPredictionEvent(
                timestamp_s=0.0, channel=1, text="Why did the chicken cross the road?"
            )
        ]

        result = task.evaluate(session, predictions, references)

        assert result.score == 0.0


def test_interruption_state_update_invalid_model():
    with pytest.raises(ValueError, match="Judge model must end with a date suffix"):
        InterruptionStateUpdate(judge_model="gpt-4o")


def test_parse_references():
    with patch("fd_eval.tasks.interruption_state_update.openai"):
        task = InterruptionStateUpdate()

        raw_labels = [
            {"initial_instruction": "A", "interrupted_at_s": 2.0, "final_instruction": "B"}
        ]

        refs = task.parse_references(raw_labels)
        assert len(refs) == 1
        assert refs[0].initial_instruction == "A"
        assert refs[0].interrupted_at_s == 2.0
        assert refs[0].final_instruction == "B"
