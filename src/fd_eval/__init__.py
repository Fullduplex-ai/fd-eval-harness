"""fd-eval-harness — evaluation harness for full-duplex speech-to-speech models.

This package is the benchmark-agnostic executor. Task plugins and model
adapters are discovered via ``importlib.metadata`` entry points under the
``fd_eval.tasks`` and ``fd_eval.adapters`` groups.

See ``docs/DESIGN.md`` for the architecture, ``docs/ROADMAP.md`` for the
v0.1 scope, and ``CURSOR_PROMPT.md`` for implementer guidance.
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
