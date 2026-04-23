from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fd_eval.adapters import MoshiAdapter
from fd_eval.adapters.moshi import MoshiPredictionEvent
from fd_eval.core import AudioSession
from fd_eval.tasks._types import TurnTakingPredictionEvent, VADPredictionEvent


@pytest.fixture
def mock_moshi_dependencies():
    with (
        patch("huggingface_hub.hf_hub_download") as mock_hf,
        patch("moshi.models.loaders.get_mimi") as mock_get_mimi,
        patch("moshi.models.loaders.get_moshi_lm") as mock_get_moshi_lm,
        patch("moshi.models.LMGen") as mock_lmgen_cls,
    ):
        mock_hf.return_value = "dummy_path"

        # Mock mimi
        mock_mimi = MagicMock()
        mock_mimi.frame_size = 1920
        mock_mimi.encode.return_value = MagicMock()

        call_counter = 0
        def mock_decode(*args, **kwargs):
            nonlocal call_counter
            import torch
            res = torch.ones((1, 1, 1920)) if call_counter % 2 == 0 else torch.zeros((1, 1, 1920))
            call_counter += 1
            return res
        mock_mimi.decode.side_effect = mock_decode

        mock_get_mimi.return_value = mock_mimi

        # Mock moshi
        mock_moshi = MagicMock()
        mock_get_moshi_lm.return_value = mock_moshi

        # Mock LMGen step
        mock_lmgen_instance = MagicMock()
        dummy_tokens = MagicMock()

        # For tokens_out[0, 1, 0].item() -> 42
        item_mock = MagicMock()
        item_mock.item.return_value = 42

        # For tokens_out[0, 1:, 0].cpu().numpy() -> array
        cpu_mock = MagicMock()
        cpu_mock.cpu.return_value.numpy.return_value = np.zeros(8)

        def mock_getitem(args):
            if isinstance(args, tuple) and args[1] == 1:
                return item_mock
            return cpu_mock

        dummy_tokens.__getitem__.side_effect = mock_getitem
        mock_lmgen_instance.step.return_value = dummy_tokens
        mock_lmgen_cls.return_value = mock_lmgen_instance

        yield mock_mimi, mock_lmgen_instance


def test_moshi_adapter_participant_mode(mock_moshi_dependencies):
    mock_mimi, mock_lmgen = mock_moshi_dependencies

    adapter = MoshiAdapter(voice="moshika")

    # 2 channels, 160ms audio at 24kHz = 3840 samples
    audio = np.zeros((3840, 2))
    session = AudioSession(
        audio=audio,
        sample_rate=24000,
        input_channel_indices=[0],
        target_channel_indices=[1],
    )

    stream = adapter.process(session)
    events = list(stream)

    assert len(events) == 2  # 160ms / 80ms chunk size
    assert isinstance(events[0], MoshiPredictionEvent)
    assert events[0].text_token == 42
    assert events[0].timestamp_s == 0.0
    assert events[1].timestamp_s == 0.08

    assert mock_mimi.encode.call_count == 2
    assert mock_lmgen.step.call_count == 2


def test_moshi_adapter_emit_turn_taking(mock_moshi_dependencies):
    _mock_mimi, _mock_lmgen = mock_moshi_dependencies

    adapter = MoshiAdapter(voice="moshika", emit_as="turn_taking")
    audio = np.zeros((3840, 2))
    session = AudioSession(
        audio=audio,
        sample_rate=24000,
        input_channel_indices=[0],
        target_channel_indices=[1],
    )

    stream = adapter.process(session)
    events = list(stream)

    assert len(events) == 2
    assert isinstance(events[0], TurnTakingPredictionEvent)
    assert events[0].event_kind == "onset"
    assert events[0].timestamp_s == 0.0
    assert events[0].channel == 1

    assert isinstance(events[1], TurnTakingPredictionEvent)
    assert events[1].event_kind == "offset"
    assert events[1].timestamp_s == 0.08
    assert events[1].channel == 1


def test_moshi_adapter_emit_vad(mock_moshi_dependencies):
    _mock_mimi, _mock_lmgen = mock_moshi_dependencies

    adapter = MoshiAdapter(voice="moshiko", emit_as="vad")
    audio = np.zeros((3840, 2))
    session = AudioSession(
        audio=audio,
        sample_rate=24000,
        input_channel_indices=[0],
        target_channel_indices=[1],
    )

    stream = adapter.process(session)
    events = list(stream)

    assert len(events) == 2
    assert isinstance(events[0], VADPredictionEvent)
    assert events[0].is_speech is True
    assert events[0].timestamp_s == 0.0
    assert events[0].channel == 1

    assert isinstance(events[1], VADPredictionEvent)
    assert events[1].is_speech is False
    assert events[1].timestamp_s == 0.08
    assert events[1].channel == 1
