from __future__ import annotations

from pathlib import Path
from typing import Optional

from rca.config import Settings, get_settings
from rca.contract import SourceRefs
from rca.ingest.assemble import entities_from
from rca.ingest.config_changes import load_config_changes
from rca.ingest.logs import parse_logs_drain3
from rca.ingest.metrics import load_metric_frame
from rca.ingest.rcaeval import load_rcaeval_case
from rca.ingest.traces import latency_from_traces, topology_from_traces
from rca.ingest.types import IncidentData


def load_incident(
    incident_id: str,
    sources: SourceRefs,
    settings: Optional[Settings] = None,
    split: str = "first_underscore",
) -> IncidentData:
    settings = settings or get_settings()

    if sources.metrics_uri and Path(sources.metrics_uri).is_dir():
        data = load_rcaeval_case(sources.metrics_uri, incident_id=incident_id, settings=settings)
        if sources.config_changes:
            data.config_changes = load_config_changes(incident_id, sources.config_changes)
        return data

    metrics = None
    if sources.metrics_uri:
        metrics = load_metric_frame(sources.metrics_uri, split=split)

    edges = []
    trace_latency = None
    if sources.traces_uri:
        edges = topology_from_traces(sources.traces_uri)
        frame = latency_from_traces(sources.traces_uri)
        trace_latency = frame if not frame.empty else None

    log_templates = None
    if sources.logs_uri:
        log_templates, _ = parse_logs_drain3(
            sources.logs_uri, service_col="service", message_col="message", time_col="time", settings=settings
        )

    entities = entities_from(metrics, trace_latency, edges)
    return IncidentData(
        incident_id=incident_id,
        metrics=metrics,
        trace_latency=trace_latency,
        log_templates=log_templates,
        entities=entities,
        edges=edges,
        config_changes=load_config_changes(incident_id, sources.config_changes),
        log_source=sources.logs_uri,
    )
