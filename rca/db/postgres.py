from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from rca.config import Settings, get_settings
from rca.contract import (
    AuditEntry,
    EvidenceBucket,
    EvidenceItem,
    EvidenceLedger,
    GroundingDecision,
    Incident,
    IncidentStatus,
    NarrationSource,
    RankedHypothesis,
    RecommendedStep,
    TimelineEvent,
)
from rca.schema import Alert, AnomalySignal, ConfigChange, Detector, Entity, Event, RelType, TopologyEdge

_DDL_PATH = Path(__file__).with_name("postgres_ddl.sql")

_AUDIT_FN_SQL = """CREATE OR REPLACE FUNCTION rca_audit_immutable() RETURNS trigger AS $func$
BEGIN
  RAISE EXCEPTION 'audit_log is append-only';
END;
$func$ LANGUAGE plpgsql"""

_AUDIT_TRIGGER_DROP = "DROP TRIGGER IF EXISTS trg_audit_immutable ON audit_log"

_AUDIT_TRIGGER_SQL = """CREATE TRIGGER trg_audit_immutable
BEFORE UPDATE OR DELETE ON audit_log
FOR EACH ROW EXECUTE FUNCTION rca_audit_immutable()"""


class PostgresRepo:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.pg_password:
            raise RuntimeError("RCA_PG_PASSWORD is not set")
        self.pool = ConnectionPool(
            conninfo=self.settings.pg_dsn,
            kwargs={"row_factory": dict_row},
            open=False,
        )
        self.pool.open()

    def close(self) -> None:
        self.pool.close()

    def init_schema(self) -> None:
        ddl = _DDL_PATH.read_text()
        statements = [s.strip() for s in ddl.split(";") if s.strip()]
        with self.pool.connection() as conn:
            for stmt in statements:
                conn.execute(stmt)
            conn.execute(_AUDIT_FN_SQL)
            conn.execute(_AUDIT_TRIGGER_DROP)
            conn.execute(_AUDIT_TRIGGER_SQL)

    def upsert_entities(self, entities: list[Entity]) -> None:
        if not entities:
            return
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO entities (id, kind, name, meta) VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET kind = EXCLUDED.kind, name = EXCLUDED.name, meta = EXCLUDED.meta",
                [(e.id, e.kind.value, e.name, Jsonb(e.meta)) for e in entities],
            )

    def insert_events(self, events: list[Event]) -> None:
        if not events:
            return
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO events (incident_id, entity_id, ts, modality, payload, severity) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                [(e.incident_id, e.entity_id, e.ts, e.modality.value, Jsonb(e.payload), e.severity) for e in events],
            )

    def insert_signals(self, signals: list[AnomalySignal]) -> None:
        if not signals:
            return
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO anomaly_signals (incident_id, entity_id, detector, window_start, window_end, score, indicator) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [
                    (s.incident_id, s.entity_id, s.detector.value, s.window_start, s.window_end, s.score, s.indicator)
                    for s in signals
                ],
            )

    def insert_config_changes(self, changes: list[ConfigChange]) -> None:
        if not changes:
            return
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO config_changes (incident_id, entity_id, ts, change_type, detail) VALUES (%s, %s, %s, %s, %s)",
                [(c.incident_id, c.entity_id, c.ts, c.change_type, Jsonb(c.detail)) for c in changes],
            )

    def insert_alerts(self, alerts: list[Alert]) -> None:
        if not alerts:
            return
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO alerts (incident_id, entity_id, ts, rule, severity) VALUES (%s, %s, %s, %s, %s)",
                [(a.incident_id, a.entity_id, a.ts, a.rule, a.severity) for a in alerts],
            )

    def upsert_incident(self, incident: Incident) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO incidents (id, detected_at, status, summary, narration, narration_source, "
                "grounding_decision, audit_trail_id, timeline, recommended_next_steps, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now()) "
                "ON CONFLICT (id) DO UPDATE SET detected_at = EXCLUDED.detected_at, status = EXCLUDED.status, "
                "summary = EXCLUDED.summary, narration = EXCLUDED.narration, narration_source = EXCLUDED.narration_source, "
                "grounding_decision = EXCLUDED.grounding_decision, audit_trail_id = EXCLUDED.audit_trail_id, "
                "timeline = EXCLUDED.timeline, recommended_next_steps = EXCLUDED.recommended_next_steps, updated_at = now()",
                (
                    incident.incident_id,
                    incident.detected_at,
                    incident.status.value,
                    incident.summary,
                    incident.narration,
                    incident.narration_source.value if incident.narration_source else None,
                    incident.grounding_decision.value if incident.grounding_decision else None,
                    incident.audit_trail_id,
                    Jsonb([t.model_dump(mode="json") for t in incident.timeline]),
                    Jsonb([r.model_dump(mode="json") for r in incident.recommended_next_steps]),
                ),
            )

    def set_incident_status(self, incident_id: str, status: IncidentStatus) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO incidents (id, status) VALUES (%s, %s) "
                "ON CONFLICT (id) DO UPDATE SET status = EXCLUDED.status, updated_at = now()",
                (incident_id, status.value),
            )

    def replace_hypotheses(self, incident_id: str, hyps: list[RankedHypothesis]) -> None:
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM hypotheses WHERE incident_id = %s", (incident_id,))
            for h in hyps:
                cur.execute(
                    "INSERT INTO hypotheses (incident_id, entity_id, rank, score, cause, impact_path, demoted, note) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                    (incident_id, h.component, h.rank, h.score, h.cause, Jsonb(h.impact_path), h.demoted, h.note),
                )
                hyp_id = cur.fetchone()["id"]
                rows = []
                for bucket in (EvidenceBucket.confirmed, EvidenceBucket.correlated, EvidenceBucket.missing):
                    for e in getattr(h.evidence, bucket.value):
                        rows.append((hyp_id, bucket.value, e.ref_type, e.ref_id, e.detail, e.entity_id, e.score))
                if rows:
                    cur.executemany(
                        "INSERT INTO evidence (hypothesis_id, bucket, ref_type, ref_id, detail, entity_id, score) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        rows,
                    )

    def append_audit(
        self,
        incident_id: str,
        actor: str,
        action: str,
        input_ref: Optional[str] = None,
        output_ref: Optional[str] = None,
        detail: Optional[dict[str, Any]] = None,
    ) -> int:
        with self.pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO audit_log (incident_id, seq, actor, action, input_ref, output_ref, detail) "
                "SELECT %s, COALESCE(MAX(seq), 0) + 1, %s, %s, %s, %s, %s FROM audit_log WHERE incident_id = %s "
                "RETURNING seq",
                (incident_id, actor, action, input_ref, output_ref, Jsonb(detail or {}), incident_id),
            ).fetchone()
            return int(row["seq"])

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        with self.pool.connection() as conn:
            inc = conn.execute("SELECT * FROM incidents WHERE id = %s", (incident_id,)).fetchone()
            if inc is None:
                return None
            hyp_rows = conn.execute(
                "SELECT * FROM hypotheses WHERE incident_id = %s ORDER BY rank", (incident_id,)
            ).fetchall()
            hyps: list[RankedHypothesis] = []
            for hr in hyp_rows:
                ev_rows = conn.execute(
                    "SELECT * FROM evidence WHERE hypothesis_id = %s", (hr["id"],)
                ).fetchall()
                ledger = EvidenceLedger()
                for er in ev_rows:
                    getattr(ledger, er["bucket"]).append(
                        EvidenceItem(
                            bucket=EvidenceBucket(er["bucket"]),
                            ref_type=er["ref_type"],
                            ref_id=er["ref_id"],
                            detail=er["detail"],
                            entity_id=er["entity_id"],
                            score=er["score"],
                        )
                    )
                hyps.append(
                    RankedHypothesis(
                        rank=hr["rank"],
                        component=hr["entity_id"] or "",
                        cause=hr["cause"] or "",
                        score=hr["score"],
                        impact_path=hr["impact_path"],
                        evidence=ledger,
                        demoted=hr["demoted"],
                        note=hr["note"],
                    )
                )
        return Incident(
            incident_id=inc["id"],
            status=IncidentStatus(inc["status"]),
            detected_at=inc["detected_at"],
            summary=inc["summary"],
            ranked_hypotheses=hyps,
            timeline=[TimelineEvent(**t) for t in inc["timeline"]],
            recommended_next_steps=[RecommendedStep(**r) for r in inc["recommended_next_steps"]],
            narration=inc["narration"],
            narration_source=NarrationSource(inc["narration_source"]) if inc["narration_source"] else None,
            grounding_decision=GroundingDecision(inc["grounding_decision"]) if inc["grounding_decision"] else None,
            audit_trail_id=inc["audit_trail_id"],
        )

    def get_audit(self, incident_id: str) -> list[AuditEntry]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE incident_id = %s ORDER BY seq", (incident_id,)
            ).fetchall()
        return [
            AuditEntry(
                seq=r["seq"],
                ts=r["ts"],
                actor=r["actor"],
                action=r["action"],
                input_ref=r["input_ref"],
                output_ref=r["output_ref"],
                detail=r["detail"],
            )
            for r in rows
        ]

    def get_config_changes(self, incident_id: str) -> list[ConfigChange]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM config_changes WHERE incident_id = %s ORDER BY ts", (incident_id,)
            ).fetchall()
        return [
            ConfigChange(
                id=r["id"],
                incident_id=r["incident_id"],
                entity_id=r["entity_id"],
                ts=r["ts"],
                change_type=r["change_type"],
                detail=r["detail"],
            )
            for r in rows
        ]

    def get_signals(self, incident_id: str) -> list[AnomalySignal]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM anomaly_signals WHERE incident_id = %s", (incident_id,)
            ).fetchall()
        return [
            AnomalySignal(
                id=r["id"],
                incident_id=r["incident_id"],
                entity_id=r["entity_id"],
                detector=Detector(r["detector"]),
                window_start=r["window_start"],
                window_end=r["window_end"],
                score=r["score"],
                indicator=r["indicator"],
            )
            for r in rows
        ]

    def upsert_topology_edges(self, edges: list[TopologyEdge]) -> None:
        if not edges:
            return
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO topology_edges (src, dst, rel, weight) VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (src, dst, rel) DO UPDATE SET weight = EXCLUDED.weight",
                [(e.src, e.dst, e.rel.value, e.weight) for e in edges],
            )

    def get_topology_edges(self) -> list[TopologyEdge]:
        with self.pool.connection() as conn:
            rows = conn.execute("SELECT src, dst, rel, weight FROM topology_edges").fetchall()
        out: list[TopologyEdge] = []
        for r in rows:
            try:
                rel = RelType(r["rel"])
            except ValueError:
                continue
            out.append(TopologyEdge(src=r["src"], dst=r["dst"], rel=rel, weight=r["weight"]))
        return out


@lru_cache
def get_repo() -> PostgresRepo:
    return PostgresRepo()
