from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException

from rca.contract import (
    AuditEntry,
    ImpactEdge,
    ImpactGraph,
    ImpactNode,
    Incident,
    IncidentStatus,
    IngestAccepted,
    IngestRequest,
    SourceRefs,
)
from rca.db.postgres import get_repo
from rca.ingest.loader import load_incident
from rca.pipeline import analyze_incident


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_repo().init_schema()
    except Exception:
        pass
    yield


app = FastAPI(title="Network Anomaly Root-Cause Assistant", version="0.1.0", lifespan=lifespan)


def _process(incident_id: str, sources: SourceRefs) -> None:
    repo = get_repo()
    try:
        data = load_incident(incident_id, sources)
        analyze_incident(data, persist=True, narrate=True)
    except Exception as exc:
        try:
            repo.append_audit(incident_id, "api", "error", detail={"error": str(exc)})
            repo.set_incident_status(incident_id, IncidentStatus.failed)
        except Exception:
            pass


@app.post("/ingest", status_code=202, response_model=IngestAccepted)
def ingest(req: IngestRequest, background: BackgroundTasks) -> IngestAccepted:
    get_repo().set_incident_status(req.incident_id, IncidentStatus.processing)
    background.add_task(_process, req.incident_id, req.sources)
    return IngestAccepted(incident_id=req.incident_id)


@app.get("/incidents/{incident_id}", response_model=Incident)
def get_incident(incident_id: str) -> Incident:
    incident = get_repo().get_incident(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="incident not found")
    return incident


@app.get("/incidents/{incident_id}/impact-path", response_model=ImpactGraph)
def impact_path(incident_id: str) -> ImpactGraph:
    repo = get_repo()
    incident = repo.get_incident(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="incident not found")
    return _build_impact_graph(incident_id, incident, repo.get_topology_edges())


@app.get("/audit/{audit_trail_id}", response_model=list[AuditEntry])
def audit(audit_trail_id: str) -> list[AuditEntry]:
    incident_id = audit_trail_id.split("aud:", 1)[-1]
    return get_repo().get_audit(incident_id)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _build_impact_graph(incident_id: str, incident: Incident, edges) -> ImpactGraph:
    roles: dict[str, str] = {}
    nodes_set: set[str] = set()
    if incident.ranked_hypotheses:
        top = incident.ranked_hypotheses[0]
        for i, node in enumerate(top.impact_path):
            role = "cause" if i == 0 else ("symptom" if i == len(top.impact_path) - 1 else "intermediate")
            roles[node] = role
            nodes_set.add(node)
        for h in incident.ranked_hypotheses:
            nodes_set.add(h.component)
            roles.setdefault(h.component, "symptom" if h.demoted else "candidate")
    impact_nodes = [ImpactNode(id=n, kind="service", role=roles.get(n)) for n in nodes_set]
    impact_edges = [
        ImpactEdge(src=e.src, dst=e.dst, rel=e.rel.value, weight=e.weight)
        for e in edges
        if e.src in nodes_set and e.dst in nodes_set
    ]
    return ImpactGraph(incident_id=incident_id, nodes=impact_nodes, edges=impact_edges)
