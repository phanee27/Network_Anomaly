from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from rca.config import Settings, get_settings
from rca.ingest.assemble import entities_from
from rca.ingest.logs import parse_logs_drain3
from rca.ingest.metrics import clean_rcaeval_latency, load_metric_frame
from rca.ingest.traces import latency_from_traces, topology_from_traces
from rca.ingest.types import IncidentData

_SYSTEM_CODE = {"RE1-OB": "ob", "RE2-OB": "ob", "RE3-OB": "ob",
                "RE1-SS": "ss", "RE2-SS": "ss", "RE3-SS": "ss",
                "RE1-TT": "tt", "RE2-TT": "tt", "RE3-TT": "tt"}


def _read_inject_time(path: Path) -> Optional[datetime]:
    if not path.exists():
        return None
    lines = path.read_text().strip().splitlines()
    if not lines:
        return None
    try:
        return datetime.fromtimestamp(int(float(lines[0].strip())), tz=timezone.utc)
    except ValueError:
        return None


def _load_metrics(case: Path) -> Optional[pd.DataFrame]:
    for name in ("simple_metrics.csv", "metrics.csv", "data.csv"):
        path = case / name
        if path.exists():
            return clean_rcaeval_latency(load_metric_frame(path, time_col="time", time_unit="s", split="first_underscore"))
    j = case / "metrics.json"
    if j.exists():
        raw = pd.read_json(j)
        if "time" in raw.columns:
            raw.to_csv(case / "_metrics_from_json.csv", index=False)
            return clean_rcaeval_latency(
                load_metric_frame(case / "_metrics_from_json.csv", time_col="time", time_unit="s", split="first_underscore")
            )
    return None


def _load_trace_latency(case: Path) -> Optional[pd.DataFrame]:
    tracets = case / "tracets_lat.csv"
    if tracets.exists():
        return load_metric_frame(tracets, time_col="time", time_unit="s", split="first_underscore")
    traces = case / "traces.csv"
    if traces.exists():
        frame = latency_from_traces(traces)
        return frame if not frame.empty else None
    return None


def _load_log_templates(case: Path, use_drain3: bool, settings: Settings) -> Optional[pd.DataFrame]:
    logts = case / "logts.csv"
    if logts.exists() and not use_drain3:
        return load_metric_frame(logts, time_col="time", time_unit="s", split="first_underscore")
    logs = case / "logs.csv"
    if logs.exists():
        frame, _ = parse_logs_drain3(
            logs, service_col="container_name", message_col="message", time_col="timestamp", time_unit="ns", settings=settings
        )
        return frame
    if logts.exists():
        return load_metric_frame(logts, time_col="time", time_unit="s", split="first_underscore")
    return None


def parse_case_id(case: Path) -> tuple[str, dict]:
    instance = case.name
    service_fault = case.parent.name
    system_dir = case.parent.parent.name
    system = _SYSTEM_CODE.get(system_dir, system_dir.lower())
    if "_" in service_fault:
        service, fault = service_fault.rsplit("_", 1)
    else:
        service, fault = service_fault, "unknown"
    incident_id = f"{system}.{service}.{fault}.{instance}"
    return incident_id, {"system": system, "service": service, "fault": fault, "instance": instance}


def load_rcaeval_case(
    case_dir: str | Path,
    incident_id: Optional[str] = None,
    use_drain3: bool = False,
    settings: Optional[Settings] = None,
) -> IncidentData:
    settings = settings or get_settings()
    case = Path(case_dir)
    derived_id, ground_truth = parse_case_id(case)
    metrics = _load_metrics(case)
    trace_latency = _load_trace_latency(case)
    log_templates = _load_log_templates(case, use_drain3, settings)
    traces = case / "traces.csv"
    edges = topology_from_traces(traces) if traces.exists() else []
    entities = entities_from(metrics, trace_latency, edges)
    logs_csv = case / "logs.csv"
    return IncidentData(
        incident_id=incident_id or derived_id,
        inject_time=_read_inject_time(case / "inject_time.txt"),
        metrics=metrics,
        trace_latency=trace_latency,
        log_templates=log_templates,
        entities=entities,
        edges=edges,
        ground_truth=ground_truth,
        log_source=str(logs_csv) if logs_csv.exists() else None,
    )
