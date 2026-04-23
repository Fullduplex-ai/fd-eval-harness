"""Command-line entry point for fd-eval-harness."""

from __future__ import annotations

import argparse
import json
import sys

from fd_eval.core.registry import get_adapter, get_task


def main(argv: list[str] | None = None) -> int:
    """Entry point wired to the ``fd-eval`` console script."""
    parser = argparse.ArgumentParser(
        description="fd-eval-harness: Full-duplex speech-to-speech evaluation"
    )

    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Comma-separated list of tasks to run (e.g. 'voice_activity_detection')",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Model adapter name (e.g. 'moshi' or 'energy_vad')",
    )
    parser.add_argument(
        "--adapter-args",
        type=str,
        default="{}",
        help="JSON string of adapter kwargs (e.g. '{\"voice\": \"moshika\"}')",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        help="Path to audio file (for v0.1 single-file ad-hoc evaluation)",
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        help="Path to labels JSON (for v0.1 single-file ad-hoc evaluation)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to write run.json output",
    )

    args = parser.parse_args(argv)

    try:
        task_names = [t.strip() for t in args.tasks.split(",")]
        # Validate tasks
        for t in task_names:
            _ = get_task(t)

        # Validate adapter
        adapter_cls = get_adapter(args.adapter)

        # Parse kwargs
        adapter_kwargs = json.loads(args.adapter_args)

        # Instantiate adapter (just to fail fast if args are bad)
        _adapter_instance = adapter_cls(**adapter_kwargs)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print("fd-eval-harness: CLI arguments parsed successfully.")
    print("Full evaluation pipeline execution will be implemented in Slice 7.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
