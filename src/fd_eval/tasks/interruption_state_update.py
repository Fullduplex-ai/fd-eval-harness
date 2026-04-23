"""Interruption State Update task implementation.

Evaluates whether the model properly abandons an old instruction and follows
a new one when the user interrupts it mid-turn.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from fd_eval.core import AudioSession, PredictionStream, Task, TaskResult
from fd_eval.tasks._types import TranscriptPredictionEvent

logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    openai = None  # type: ignore


@dataclass(frozen=True)
class InterruptionReference:
    """Expected instruction state change."""

    initial_instruction: str
    interrupted_at_s: float
    final_instruction: str


class InterruptionStateUpdate(Task):
    """LLM-as-judge task measuring response to interruptions.

    Mode: participant
    Scoring: llm-judge
    """

    name = "interruption_state_update"
    version = "0.1.0"
    mode = "participant"
    scoring_method = "llm-judge"

    def __init__(self, judge_model: str = "gpt-4o-2024-05-13"):
        """Initialize the task.

        Args:
            judge_model: The exact version-pinned model to use for the judge.
        """
        # Enforce version locking per D015
        if judge_model == "gpt-4o":
            raise ValueError(
                "Floating model alias 'gpt-4o' is prohibited for judge models. "
                "Must use a date-pinned version (e.g., 'gpt-4o-2024-05-13')."
            )
        self.judge_model = judge_model

        if openai is None:
            raise ImportError(
                "The 'openai' package is required for llm-judge tasks. "
                "Install it with `pip install fd-eval-harness[llm-judge]`."
            )
        self.client = openai.Client()

    def parse_references(self, raw_labels: list[dict]) -> Sequence[InterruptionReference]:
        refs = []
        for label in raw_labels:
            if not all(
                k in label for k in ("initial_instruction", "interrupted_at_s", "final_instruction")
            ):
                raise ValueError(f"Label missing required fields: {label}")
            refs.append(
                InterruptionReference(
                    initial_instruction=label["initial_instruction"],
                    interrupted_at_s=label["interrupted_at_s"],
                    final_instruction=label["final_instruction"],
                )
            )
        return refs

    def _call_judge(self, reference: InterruptionReference, transcript: str) -> dict[str, Any]:
        """Call the LLM judge and return the structured JSON output."""
        prompt = f"""You are an expert evaluator of conversational AI agents.

The user interacted with the AI agent.
First, the user gave this instruction:
"{reference.initial_instruction}"

Then, at {reference.interrupted_at_s}s, the user interrupted the agent and gave a new instruction:
"{reference.final_instruction}"

Here is the transcript of what the AI agent said after the interruption:
"{transcript}"

Did the AI agent successfully abandon the initial instruction and correctly follow
the final instruction?
Output a JSON object with two fields:
- "reasoning": A step-by-step chain of thought explaining your judgment.
- "score": 1.0 if the agent successfully followed the new instruction, 0.0 otherwise.
"""
        response = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Judge returned empty response.")

        usage = response.usage
        tokens = (
            {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}
            if usage
            else {}
        )

        result = json.loads(content)
        result["tokens"] = tokens
        return result

    def evaluate(
        self,
        session: AudioSession,
        predictions: PredictionStream,
        references: Sequence[InterruptionReference],
    ) -> TaskResult:
        """Evaluate the predictions against references.

        Note: This task currently assumes 1 session = 1 interruption reference.
        If multiple references are provided, they are evaluated independently
        against the full transcript and their scores are averaged.
        """
        if not references:
            return TaskResult(score=0.0, details={"error": "No references provided."})

        # Assemble the full transcript from TranscriptPredictionEvents
        transcript_parts = []
        for p in predictions:
            if isinstance(p, TranscriptPredictionEvent):
                transcript_parts.append(p.text)

        full_transcript = " ".join(transcript_parts).strip()

        scores = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # For simplicity, we assume one reference per session (common for this task type).
        # If there are multiple, we average them.
        for ref in references:
            try:
                judge_out = self._call_judge(ref, full_transcript)
                score = float(judge_out.get("score", 0.0))
                scores.append(score)

                tokens = judge_out.get("tokens", {})
                total_prompt_tokens += tokens.get("prompt_tokens", 0)
                total_completion_tokens += tokens.get("completion_tokens", 0)
            except Exception as e:
                logger.error(f"LLM Judge failed: {e}")
                scores.append(0.0)

        final_score = sum(scores) / len(scores) if scores else 0.0

        return TaskResult(
            score=final_score,
            details={
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "num_judgments": len(scores),
                "transcript_length": len(full_transcript),
            },
        )
