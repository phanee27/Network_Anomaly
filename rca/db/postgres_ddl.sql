CREATE TABLE IF NOT EXISTS entities (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  name TEXT NOT NULL,
  meta JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS events (
  id BIGSERIAL PRIMARY KEY,
  incident_id TEXT NOT NULL,
  entity_id TEXT,
  ts TIMESTAMPTZ NOT NULL,
  modality TEXT NOT NULL,
  payload JSONB NOT NULL DEFAULT '{}'::jsonb,
  severity REAL
);

CREATE INDEX IF NOT EXISTS ix_events_inc_ts ON events (incident_id, ts);

CREATE TABLE IF NOT EXISTS anomaly_signals (
  id BIGSERIAL PRIMARY KEY,
  incident_id TEXT NOT NULL,
  entity_id TEXT,
  detector TEXT NOT NULL,
  window_start TIMESTAMPTZ,
  window_end TIMESTAMPTZ,
  score REAL,
  indicator TEXT
);

CREATE INDEX IF NOT EXISTS ix_sig_inc ON anomaly_signals (incident_id);

CREATE TABLE IF NOT EXISTS config_changes (
  id BIGSERIAL PRIMARY KEY,
  incident_id TEXT NOT NULL,
  entity_id TEXT,
  ts TIMESTAMPTZ,
  change_type TEXT,
  detail JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_cfg_inc ON config_changes (incident_id);

CREATE TABLE IF NOT EXISTS alerts (
  id BIGSERIAL PRIMARY KEY,
  incident_id TEXT NOT NULL,
  entity_id TEXT,
  ts TIMESTAMPTZ,
  rule TEXT,
  severity REAL
);

CREATE INDEX IF NOT EXISTS ix_alert_inc ON alerts (incident_id);

CREATE TABLE IF NOT EXISTS incidents (
  id TEXT PRIMARY KEY,
  detected_at TIMESTAMPTZ,
  status TEXT NOT NULL,
  summary TEXT,
  narration TEXT,
  narration_source TEXT,
  grounding_decision TEXT,
  audit_trail_id TEXT,
  timeline JSONB NOT NULL DEFAULT '[]'::jsonb,
  recommended_next_steps JSONB NOT NULL DEFAULT '[]'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS hypotheses (
  id BIGSERIAL PRIMARY KEY,
  incident_id TEXT NOT NULL,
  entity_id TEXT,
  rank INT NOT NULL,
  score REAL NOT NULL,
  cause TEXT,
  impact_path JSONB NOT NULL DEFAULT '[]'::jsonb,
  demoted BOOLEAN NOT NULL DEFAULT FALSE,
  note TEXT
);

CREATE INDEX IF NOT EXISTS ix_hyp_inc ON hypotheses (incident_id, rank);

CREATE TABLE IF NOT EXISTS evidence (
  id BIGSERIAL PRIMARY KEY,
  hypothesis_id BIGINT NOT NULL REFERENCES hypotheses (id) ON DELETE CASCADE,
  bucket TEXT NOT NULL CHECK (bucket IN ('confirmed', 'correlated', 'missing')),
  ref_type TEXT,
  ref_id TEXT,
  detail TEXT,
  entity_id TEXT,
  score REAL
);

CREATE INDEX IF NOT EXISTS ix_ev_hyp ON evidence (hypothesis_id);

CREATE TABLE IF NOT EXISTS audit_log (
  id BIGSERIAL PRIMARY KEY,
  incident_id TEXT NOT NULL,
  seq INT NOT NULL,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  actor TEXT,
  action TEXT,
  input_ref TEXT,
  output_ref TEXT,
  detail JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_audit_inc_seq ON audit_log (incident_id, seq);

CREATE TABLE IF NOT EXISTS topology_edges (
  id BIGSERIAL PRIMARY KEY,
  src TEXT NOT NULL,
  dst TEXT NOT NULL,
  rel TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 1.0,
  UNIQUE (src, dst, rel)
)
