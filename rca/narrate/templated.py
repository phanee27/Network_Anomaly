from __future__ import annotations

from rca.contract import Incident


def render_summary(incident: Incident) -> str:
    if not incident.ranked_hypotheses:
        return "No anomalies detected in the analysed window."
    top = incident.ranked_hypotheses[0]
    return f"{top.cause} on {top.component}"


def render_narration(incident: Incident) -> str:
    if not incident.ranked_hypotheses:
        return "No anomalies were detected in the analysed window."
    top = incident.ranked_hypotheses[0]
    lines = [f"Most likely root cause: {top.cause} on {top.component} (score {top.score:.2f})."]
    if top.impact_path and len(top.impact_path) > 1:
        lines.append("Impact path: " + " -> ".join(top.impact_path) + ".")
    confirmed = [e.detail for e in top.evidence.confirmed]
    correlated = [e.detail for e in top.evidence.correlated]
    missing = [e.detail for e in top.evidence.missing]
    if confirmed:
        lines.append("Confirmed evidence: " + "; ".join(confirmed) + ".")
    if correlated:
        lines.append("Correlated signals: " + "; ".join(correlated) + ".")
    if missing:
        lines.append("Missing / not yet verified: " + "; ".join(missing) + ".")
    for demoted in [h for h in incident.ranked_hypotheses if h.demoted][:2]:
        lines.append(f"Demoted: {demoted.component} — {demoted.note}.")
    if incident.recommended_next_steps:
        lines.append("Recommended: " + "; ".join(s.action for s in incident.recommended_next_steps) + ".")
    return "\n".join(lines)
