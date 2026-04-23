"""Test fixtures for fd-eval-harness.

Fixtures here are synthetic only: no real conversational audio or
reference labels live in the repository (see ``docs/DESIGN.md`` §12 item 3).
"""

from .synthetic_audio import (
    make_silence,
    make_sine,
    make_two_channel_alternating,
    make_two_channel_sine,
)

__all__ = [
    "make_silence",
    "make_sine",
    "make_two_channel_alternating",
    "make_two_channel_sine",
]
