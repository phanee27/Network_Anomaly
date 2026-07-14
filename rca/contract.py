from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Modality(str, Enum):
    metric = "metric"
    log = "log"
    trace = "trace"
    alert = "alert"
    config = "config"


class EvidenceBucket(str, Enum):
    confirmed = "confirmed"
    correlated = "correlated"
    missing = "missing"


class IncidentStatus(str, Enum):
    processing = "processing"
    resolved = "resolved"
    abstained = "abstained"
    failed = "failed"


class GroundingDecision(str, Enum):
    resolved = "resolved"
    abstain = "abstain"


class NarrationSource(str, Enum):
    llm = "llm"
    template = "template"


class EvidenceItem(BaseModel):
    bucket: EvidenceBucket
    ref_type: str
    ref_id: str
    detail: str
    entity_id: Optional[str] = None
    score: Optional[float] = None


class EvidenceLedger(BaseModel):
    confirmed: list[EvidenceItem] = Field(default_factory=list)
    correlated: list[EvidenceItem] = Field(default_factory=list)
    missing: list[EvidenceItem] = Field(default_factory=list)

    def all_ref_ids(self) -> set[str]:
        return {e.ref_id for e in (self.confirmed + self.correlated + self.missing)}


class TimelineEvent(BaseModel):
    t: datetime
    event: str
    entity_id: Optional[str] = None
    refs: list[str] = Field(default_factory=list)


class RankedHypothesis(BaseModel):
    rank: int
    component: str
    cause: str
    score: float
    impact_path: list[str] = Field(default_factory=list)
    evidence: EvidenceLedger = Field(default_factory=EvidenceLedger)
    demoted: bool = False
    note: Optional[str] = None


class RecommendedStep(BaseModel):
    action: str
    rationale: Optional[str] = None
    refs: list[str] = Field(default_factory=list)


class AuditEntry(BaseModel):
    seq: int
    ts: datetime
    actor: str
    action: str
    input_ref: Optional[str] = None
    output_ref: Optional[str] = None
    detail: dict[str, Any] = Field(default_factory=dict)


class Incident(BaseModel):
    incident_id: str
    status: IncidentStatus
    detected_at: Optional[datetime] = None
    summary: Optional[str] = None
    ranked_hypotheses: list[RankedHypothesis] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    recommended_next_steps: list[RecommendedStep] = Field(default_factory=list)
    narration: Optional[str] = None
    narration_source: Optional[NarrationSource] = None
    grounding_decision: Optional[GroundingDecision] = None
    audit_trail_id: Optional[str] = None


class SourceRefs(BaseModel):
    metrics_uri: Optional[str] = None
    logs_uri: Optional[str] = None
    traces_uri: Optional[str] = None
    topology_uri: Optional[str] = None
    config_changes: list[dict[str, Any]] = Field(default_factory=list)


class IngestRequest(BaseModel):
    incident_id: str
    sources: SourceRefs
    inject_time: Optional[datetime] = None


class IngestAccepted(BaseModel):
    incident_id: str
    status: IncidentStatus = IncidentStatus.processing


class ImpactNode(BaseModel):
    id: str
    kind: str
    role: Optional[str] = None


class ImpactEdge(BaseModel):
    src: str
    dst: str
    rel: str
    weight: Optional[float] = None


class ImpactGraph(BaseModel):
    incident_id: str
    nodes: list[ImpactNode] = Field(default_factory=list)
    edges: list[ImpactEdge] = Field(default_factory=list)
