from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from rca.contract import Modality


class EntityKind(str, Enum):
    service = "service"
    host = "host"
    link = "link"


class Detector(str, Enum):
    eps_diag = "eps_diag"
    iforest = "iforest"
    drain3 = "drain3"
    trace = "trace_latency"
    lightgbm = "lightgbm"
    autoencoder = "autoencoder"
    deeplog = "deeplog"


class RelType(str, Enum):
    depends_on = "DEPENDS_ON"
    routes_through = "ROUTES_THROUGH"
    hosts = "HOSTS"


class Entity(BaseModel):
    id: str
    kind: EntityKind
    name: str
    meta: dict[str, Any] = Field(default_factory=dict)


class Event(BaseModel):
    incident_id: str
    entity_id: str
    ts: datetime
    modality: Modality
    payload: dict[str, Any] = Field(default_factory=dict)
    severity: Optional[float] = None
    id: Optional[int] = None


class AnomalySignal(BaseModel):
    incident_id: str
    entity_id: str
    detector: Detector
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    score: float
    indicator: str
    id: Optional[int] = None


class ConfigChange(BaseModel):
    incident_id: str
    entity_id: str
    ts: Optional[datetime] = None
    change_type: str
    detail: dict[str, Any] = Field(default_factory=dict)
    id: Optional[int] = None


class Alert(BaseModel):
    incident_id: str
    entity_id: str
    ts: datetime
    rule: str
    severity: float
    id: Optional[int] = None


class TopologyEdge(BaseModel):
    src: str
    dst: str
    rel: RelType
    weight: float = 1.0


class MetricSeries(BaseModel):
    entity_id: str
    metric: str
    ts: list[datetime]
    values: list[float]
