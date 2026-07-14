from __future__ import annotations

from typing import Protocol, runtime_checkable

from rca.ingest.types import IncidentData
from rca.schema import AnomalySignal


@runtime_checkable
class Detector(Protocol):
    name: str
    modality: str

    def detect(self, data: IncidentData) -> list[AnomalySignal]: ...
