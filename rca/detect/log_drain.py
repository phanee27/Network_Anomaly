from __future__ import annotations

from typing import Optional

from rca.config import Settings, get_settings
from rca.detect.common import two_sample_shift_signals
from rca.ingest.types import IncidentData
from rca.schema import AnomalySignal, Detector


class LogTemplateDetector:
    name = "drain3"
    modality = "log"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def detect(self, data: IncidentData) -> list[AnomalySignal]:
        return two_sample_shift_signals(
            data.log_templates,
            data.incident_id,
            data.inject_time,
            Detector.drain3,
            self.settings,
            "log template {metric} frequency spike +{shift:.1f} sigma vs baseline",
        )
