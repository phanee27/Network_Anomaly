from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from rca.schema import ConfigChange, Entity, Event, TopologyEdge


@dataclass
class IncidentData:
    incident_id: str
    inject_time: Optional[datetime] = None
    metrics: Any = None
    log_templates: Any = None
    trace_latency: Any = None
    traces: Any = None
    flows: Any = None
    log_source: Optional[str] = None
    log_events: list[Event] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    edges: list[TopologyEdge] = field(default_factory=list)
    config_changes: list[ConfigChange] = field(default_factory=list)
    ground_truth: Optional[dict] = None

    def entity_ids(self) -> list[str]:
        return [e.id for e in self.entities]
