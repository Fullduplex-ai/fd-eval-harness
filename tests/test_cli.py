import sys
from unittest.mock import patch

import numpy as np

from fd_eval.cli import main


def test_cli_help(capsys):
    with patch.object(sys, "argv", ["fd-eval", "--help"]):
        try:
            main()
        except SystemExit as e:
            assert e.code == 0
    captured = capsys.readouterr()
    assert "fd-eval-harness" in captured.out
    assert "--tasks" in captured.out


def test_cli_missing_args(capsys):
    with patch.object(sys, "argv", ["fd-eval"]):
        try:
            main()
        except SystemExit as e:
            assert e.code == 2


def test_cli_invalid_task(capsys):
    code = main(["--tasks", "nonexistent", "--adapter", "energy_vad"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Task 'nonexistent' not found" in captured.err


def test_cli_invalid_adapter(capsys):
    code = main(["--tasks", "voice_activity_detection", "--adapter", "nonexistent"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Adapter 'nonexistent' not found" in captured.err


def test_cli_valid_args(capsys):
    code = main(
        [
            "--tasks",
            "voice_activity_detection",
            "--adapter",
            "energy_vad",
            "--adapter-args",
            '{"threshold": 0.05}',
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "CLI arguments parsed successfully" in captured.out


def test_cli_evaluation_loop(capsys, tmp_path):
    with (
        patch("fd_eval.cli.load_audio") as mock_audio,
        patch("fd_eval.cli.load_labels") as mock_labels,
    ):
        # 1 second of audio, 2 channels
        mock_audio.return_value = (np.zeros((16000, 2), dtype="float32"), 16000)
        mock_labels.return_value = [
            {"timestamp_s": 0.5, "channel": 0, "is_speech": True},
            {"timestamp_s": 0.8, "channel": 0, "is_speech": False},
        ]

        out_json = tmp_path / "out.json"

        code = main(
            [
                "--tasks",
                "voice_activity_detection",
                "--adapter",
                "energy_vad",
                "--in-channels",
                "0,1",
                "--tgt-channels",
                "",
                "--audio-path",
                "dummy.wav",
                "--labels-path",
                "dummy.json",
                "--output",
                str(out_json),
            ]
        )

        assert code == 0
        assert out_json.exists()

        import json

        with open(out_json) as f:
            data = json.load(f)

        assert "voice_activity_detection" in data
        assert "score" in data["voice_activity_detection"]
