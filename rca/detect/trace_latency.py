from __future__ import annotations

from typing import Optional

from rca.config import Settings, get_settings
from rca.detect.common import two_sample_shift_signals
from rca.ingest.types import IncidentData
from rca.schema import AnomalySignal, Detector


class TraceLatencyDetector:
    name = "trace_latency"
    modality = "trace"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def detect(self, data: IncidentData) -> list[AnomalySignal]:
        return two_sample_shift_signals(
            data.trace_latency,
            data.incident_id,
            data.inject_time,
            Detector.trace,
            self.settings,
            "trace latency {metric} +{shift:.1f} sigma vs baseline",
        )
