"""Interruption State Update task implementation.

Evaluates whether the model properly abandons an old instruction and follows
a new one when the user interrupts it mid-turn.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fd_eval.core import AudioSession, PredictionStream, Task, TaskResult
from fd_eval.tasks._types import TranscriptPredictionEvent

logger = logging.getLogger(__name__)

_DATE_SUFFIX_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")

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
        if not _DATE_SUFFIX_RE.search(judge_model):
            raise ValueError(
                f"Judge model must end with a date suffix like '-2024-05-13'. Got: {judge_model!r}"
            )
        self.judge_model = judge_model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if openai is None:
                raise ImportError(
                    "The 'openai' package is required for llm-judge tasks. "
                    "Install it with `pip install fd-eval-harness[llm-judge]`."
                )
            self._client = openai.Client()
        return self._client

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

    def _call_judge(
        self, reference: InterruptionReference, transcript: str, prompt_template: str
    ) -> dict[str, Any]:
        """Call the LLM judge and return the structured JSON output."""
        prompt = prompt_template.format(
            initial_instruction=reference.initial_instruction,
            interrupted_at_s=reference.interrupted_at_s,
            final_instruction=reference.final_instruction,
            transcript=transcript,
        )

        max_retries = 3
        base_wait = 1.0

        for attempt in range(max_retries):
            try:
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
                    {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                    }
                    if usage
                    else {}
                )

                result = json.loads(content)
                result["tokens"] = tokens
                return result
            except (openai.APITimeoutError, openai.APIConnectionError):
                if attempt == max_retries - 1:
                    raise
                time.sleep(base_wait * (2**attempt))
            except json.JSONDecodeError as e:
                raise ValueError(f"Judge returned malformed JSON: {e}") from e
            except Exception:
                raise

        raise RuntimeError("Unreachable")

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

        prompt_path = Path(__file__).parents[1] / "prompts" / "interruption_state_update.md"
        with open(prompt_path, encoding="utf-8") as f:
            prompt_template = f.read()
        prompt_hash = hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()

        scores = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        judge_errors = 0

        # For simplicity, we assume one reference per session (common for this task type).
        # If there are multiple, we average them.
        for ref in references:
            try:
                judge_out = self._call_judge(ref, full_transcript, prompt_template)
                score = float(judge_out.get("score", 0.0))
                scores.append(score)

                tokens = judge_out.get("tokens", {})
                total_prompt_tokens += tokens.get("prompt_tokens", 0)
                total_completion_tokens += tokens.get("completion_tokens", 0)
            except Exception as e:
                logger.error(f"LLM Judge failed: {e}")
                judge_errors += 1

        final_score = sum(scores) / len(scores) if scores else 0.0

        return TaskResult(
            score=final_score,
            details={
                "prompt_hash": prompt_hash,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "num_judgments": len(scores),
                "judge_errors": judge_errors,
                "transcript_length": len(full_transcript),
            },
        )
