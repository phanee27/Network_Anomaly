from __future__ import annotations

from typing import Optional

from rca.config import Settings, get_settings
from rca.detect.base import Detector as DetectorProtocol
from rca.detect.eps_diagnosis import EpsDiagnosisDetector
from rca.detect.iforest import IsolationForestDetector
from rca.detect.log_drain import LogTemplateDetector
from rca.detect.trace_latency import TraceLatencyDetector
from rca.ingest.types import IncidentData
from rca.schema import AnomalySignal


def build_detectors(settings: Optional[Settings] = None) -> list[DetectorProtocol]:
    settings = settings or get_settings()
    detectors: list[DetectorProtocol] = [
        EpsDiagnosisDetector(settings),
        IsolationForestDetector(settings),
        LogTemplateDetector(settings),
        TraceLatencyDetector(settings),
    ]
    detectors.extend(_optional_detectors(settings))
    return detectors


def _optional_detectors(settings: Settings) -> list[DetectorProtocol]:
    out: list[DetectorProtocol] = []
    from rca.detect.artifacts import load_artifact_detectors

    out.extend(load_artifact_detectors(settings))
    return out


def run_detectors(
    data: IncidentData, detectors: list[DetectorProtocol]
) -> tuple[list[AnomalySignal], list[dict]]:
    signals: list[AnomalySignal] = []
    errors: list[dict] = []
    for det in detectors:
        try:
            signals.extend(det.detect(data))
        except Exception as exc:
            errors.append({"detector": getattr(det, "name", type(det).__name__), "error": str(exc)})
    return signals, errors
