from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class PredictionEvent:
    timestamp_s: float


# A stream of events
PredictionStream = Iterator[PredictionEvent]
