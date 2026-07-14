from __future__ import annotations

from typing import Optional

from rca.config import Settings, get_settings
from rca.detect.common import two_sample_shift_signals
from rca.ingest.types import IncidentData
from rca.schema import AnomalySignal, Detector


class EpsDiagnosisDetector:
    name = "eps_diagnosis"
    modality = "metric"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def detect(self, data: IncidentData) -> list[AnomalySignal]:
        return two_sample_shift_signals(
            data.metrics,
            data.incident_id,
            data.inject_time,
            Detector.eps_diag,
            self.settings,
            "{metric} distribution shift +{shift:.1f} sigma vs baseline",
        )
