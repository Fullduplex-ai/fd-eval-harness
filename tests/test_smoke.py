"""Smoke tests — keep these trivial. Real tests live alongside each module."""

from __future__ import annotations

import fd_eval
from fd_eval import cli


def test_package_has_version() -> None:
    assert isinstance(fd_eval.__version__, str)
    assert fd_eval.__version__


def test_cli_main_returns_zero(capsys) -> None:
    rc = cli.main([])
    assert rc == 0
    captured = capsys.readouterr()
    assert "fd-eval-harness" in captured.out
