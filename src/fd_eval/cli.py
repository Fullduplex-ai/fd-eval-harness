"""Command-line entry point for fd-eval-harness."""

from __future__ import annotations

import argparse
import json
import sys

from fd_eval.core.audio_session import AudioSession
from fd_eval.core.registry import get_adapter, get_task
from fd_eval.data import load_audio, load_labels


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
        help='JSON string of adapter kwargs (e.g. \'{"voice": "moshika"}\')',
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
        "--in-channels",
        type=str,
        default="0",
        help="Comma-separated input channel indices (e.g. '0')",
    )
    parser.add_argument(
        "--tgt-channels",
        type=str,
        default="1",
        help="Comma-separated target channel indices (e.g. '1')",
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

        in_channels = (
            [int(c.strip()) for c in args.in_channels.split(",")] if args.in_channels else []
        )
        tgt_channels = (
            [int(c.strip()) for c in args.tgt_channels.split(",")] if args.tgt_channels else []
        )

        # Instantiate adapter
        adapter_instance = adapter_cls(**adapter_kwargs)

        if args.audio_path and args.labels_path:
            audio, sr = load_audio(args.audio_path)
            raw_labels = load_labels(args.labels_path)

            session = AudioSession(
                audio=audio,
                sample_rate=sr,
                input_channel_indices=in_channels,
                target_channel_indices=tgt_channels,
            )

            prediction_stream = adapter_instance.process(session)
            # Materialize stream so multiple tasks can evaluate it
            predictions = list(prediction_stream)

            results = {}
            for t_name in task_names:
                task_cls = get_task(t_name)
                task_instance = task_cls()

                references = task_instance.parse_references(raw_labels)
                result = task_instance.evaluate(session, predictions, references)

                results[t_name] = {
                    "score": result.score,
                    "details": result.details,
                    "task_version": task_cls.version,
                    "scoring_method": task_cls.scoring_method,
                }

            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
            else:
                print(json.dumps(results, indent=2))
        else:
            print("fd-eval-harness: CLI arguments parsed successfully.")
            print("Please provide --audio-path and --labels-path to run the evaluation.")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
