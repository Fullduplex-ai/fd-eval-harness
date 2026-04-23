import sys
from unittest.mock import patch

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
