from __future__ import annotations

from rca.contract import Incident

SYSTEM_PROMPT = (
    "You are a network incident root-cause narrator for an AIOps assistant. "
    "You may ONLY state facts that are backed by the evidence refs provided by the user. "
    "Every statement must list the exact ref ids it is grounded in. "
    "Never invent services, metrics, timestamps, causes, or numbers that are not in the refs. "
    "Return a single JSON object of the form "
    '{"statements": [{"text": "<one sentence>", "refs": ["<ref id>", ...]}, ...]}. '
    "Output only the JSON object, no prose before or after."
)


def build_grounding(incident: Incident) -> dict[str, str]:
    refs: dict[str, str] = {}
    for h in incident.ranked_hypotheses:
        refs[f"hyp:{h.rank}:{h.component}"] = (
            f"ranked #{h.rank} {h.component}: {h.cause} (score {h.score})"
            + (f"; DEMOTED: {h.note}" if h.demoted and h.note else "")
        )
        for bucket in ("confirmed", "correlated", "missing"):
            for e in getattr(h.evidence, bucket):
                refs[e.ref_id] = f"[{bucket}] {e.detail}"
    return refs


def build_user_prompt(incident: Incident, refs: dict[str, str]) -> str:
    lines = [f"Incident id: {incident.incident_id}"]
    if incident.ranked_hypotheses:
        top = incident.ranked_hypotheses[0]
        lines.append(f"Top ranked root cause: {top.component} — {top.cause} (score {top.score}).")
        if top.impact_path and len(top.impact_path) > 1:
            lines.append("Impact path: " + " -> ".join(top.impact_path) + ".")
    lines.append("")
    lines.append("Allowed evidence refs (cite only these ids, verbatim):")
    for ref_id, detail in refs.items():
        lines.append(f"- {ref_id}: {detail}")
    lines.append("")
    lines.append(
        "Write a concise incident explanation (3-6 sentences) covering the most likely root cause, "
        "the impact path, what is confirmed versus only correlated, what evidence is missing, and any "
        "recent change that was correctly not blamed. Each statement must cite the ref ids it relies on."
    )
    return "\n".join(lines)
