from abc import ABC
from collections.abc import Sequence
from typing import Any, Literal, get_args

ScoringMethod = Literal["algorithmic", "llm-judge", "human-mos", "hybrid", "other"]


class Task(ABC):
    scoring_method: ScoringMethod
    scoring_method_detail: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "version") or not isinstance(cls.version, str):
            raise ValueError(
                f"Task subclass {cls.__name__} must define 'version' as a str "
                "(e.g., version = '0.1.0')."
            )

        if not hasattr(cls, "scoring_method"):
            raise ValueError(f"Task subclass {cls.__name__} must define 'scoring_method'.")

        allowed_methods = get_args(ScoringMethod)
        if cls.scoring_method not in allowed_methods:
            raise ValueError(
                f"Task subclass {cls.__name__} has invalid scoring_method '{cls.scoring_method}'. "
                f"Allowed values: {allowed_methods}"
            )

        if cls.scoring_method == "other" and not cls.scoring_method_detail:
            raise ValueError(
                f"Task subclass {cls.__name__} specified scoring_method='other' but "
                "failed to provide 'scoring_method_detail'."
            )

    def parse_references(self, raw_labels: list[dict]) -> Sequence[Any]:
        """Parse raw JSON dictionaries into the task's specific reference type.

        By default, returns the raw list. Subclasses should override this
        to validate and instantiate their specific reference dataclasses.
        """
        return raw_labels
