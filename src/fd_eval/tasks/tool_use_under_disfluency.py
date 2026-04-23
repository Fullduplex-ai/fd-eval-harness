"""Tool Use Under Disfluency task implementation.

Evaluates whether the model correctly emits a tool-call payload
despite user hesitations, repetitions, or self-corrections mid-sentence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from fd_eval.core import AudioSession, PredictionStream, Task, TaskResult
from fd_eval.tasks._types import ToolCallPredictionEvent


@dataclass(frozen=True)
class ToolUseReference:
    """Expected tool call from the model."""

    tool_name: str
    arguments: dict[str, Any]


class ToolUseUnderDisfluency(Task):
    """Measures tool call accuracy against user disfluency.

    Mode: participant
    Scoring: algorithmic (subset JSON matching)
    """

    name = "tool_use_under_disfluency"
    version = "0.1.0"
    mode = "participant"
    scoring_method = "algorithmic"

    def parse_references(self, raw_labels: list[dict]) -> Sequence[ToolUseReference]:
        refs = []
        for label in raw_labels:
            if "tool_name" not in label or "arguments" not in label:
                raise ValueError(
                    f"Label missing required fields 'tool_name' or 'arguments': {label}"
                )
            refs.append(
                ToolUseReference(
                    tool_name=label["tool_name"],
                    arguments=label["arguments"],
                )
            )
        return refs

    def _is_subset(self, subset: dict, superset: dict) -> bool:
        """Check if `subset` is entirely contained within `superset`."""
        for key, val in subset.items():
            if key not in superset:
                return False
            # Basic equality for scalar values; complex nested checking omitted for simplicity.
            if superset[key] != val:
                return False
        return True

    def evaluate(
        self,
        session: AudioSession,
        predictions: PredictionStream,
        references: Sequence[ToolUseReference],
    ) -> TaskResult:

        # Filter for ToolCallPredictionEvent
        preds = [p for p in predictions if isinstance(p, ToolCallPredictionEvent)]

        matched_preds = set()
        matched_refs = set()

        true_positives = 0

        # Simple greedy matching
        for i, ref in enumerate(references):
            for j, pred in enumerate(preds):
                if j in matched_preds:
                    continue
                if pred.tool_name == ref.tool_name and self._is_subset(
                    ref.arguments, pred.arguments
                ):
                    true_positives += 1
                    matched_preds.add(j)
                    matched_refs.add(i)
                    break

        num_preds = len(preds)
        num_refs = len(references)

        precision = true_positives / num_preds if num_preds > 0 else 0.0
        recall = true_positives / num_refs if num_refs > 0 else 0.0

        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

        false_positives = num_preds - true_positives
        false_negatives = num_refs - true_positives

        return TaskResult(
            score=f1,
            details={
                "precision": precision,
                "recall": recall,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "num_predictions": num_preds,
                "num_references": num_refs,
            },
        )
