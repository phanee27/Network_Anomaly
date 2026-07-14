from __future__ import annotations

import json
from operator import add
from typing import Annotated, Optional

from typing_extensions import TypedDict

from rca.config import Settings, get_settings
from rca.contract import GroundingDecision, Incident, NarrationSource
from rca.narrate.base import LLMClient
from rca.narrate.llm import OllamaLLM
from rca.narrate.prompts import SYSTEM_PROMPT, build_grounding, build_user_prompt
from rca.narrate.templated import render_narration
from rca.narrate.validator import assemble, validate_statements


def _parse_json(text: Optional[str]):
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t[:4].lower() == "json":
            t = t[4:]
    try:
        return json.loads(t)
    except Exception:
        start, end = t.find("{"), t.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(t[start : end + 1])
            except Exception:
                return None
    return None


def _narrate_once(incident: Incident, refs: dict[str, str], llm: LLMClient) -> tuple[Optional[str], list]:
    raw = llm.complete(SYSTEM_PROMPT, build_user_prompt(incident, refs), json_mode=True)
    data = _parse_json(raw)
    if not isinstance(data, dict):
        return None, []
    valid, rejected = validate_statements(data.get("statements"), set(refs.keys()))
    if not valid:
        return None, rejected
    return assemble(valid), rejected


def _finalize(incident: Incident, text: str, source: NarrationSource, decision: GroundingDecision) -> None:
    incident.narration = text
    incident.narration_source = source
    incident.grounding_decision = decision


class _State(TypedDict):
    narration: str
    attempts: Annotated[int, add]


def _run_langgraph(incident, refs, llm, settings, audit) -> Optional[str]:
    from langgraph.graph import END, START, StateGraph

    max_attempts = 1 + max(0, settings.llm_max_retries)

    def narrate_node(state: _State) -> dict:
        narration, rejected = _narrate_once(incident, refs, llm)
        if audit and rejected:
            audit.record("narrator", "reject_ungrounded", detail={"count": len(rejected)})
        return {"narration": narration or "", "attempts": 1}

    def route(state: _State) -> str:
        if state["narration"]:
            return "done"
        return "retry" if state["attempts"] < max_attempts else "done"

    builder = StateGraph(_State)
    builder.add_node("narrate", narrate_node)
    builder.add_edge(START, "narrate")
    builder.add_conditional_edges("narrate", route, {"retry": "narrate", "done": END})
    graph = builder.compile()
    result = graph.invoke({"narration": "", "attempts": 0}, config={"recursion_limit": 10})
    return result.get("narration") or None


def narrate_incident(
    incident: Incident,
    settings: Optional[Settings] = None,
    audit=None,
    llm: Optional[LLMClient] = None,
) -> Incident:
    settings = settings or get_settings()
    llm = llm or OllamaLLM(settings)
    template = render_narration(incident)
    refs = build_grounding(incident)

    if not settings.llm_enabled or not refs or not llm.available():
        _finalize(incident, template, NarrationSource.template, GroundingDecision.resolved)
        if audit:
            audit.record("narrator", "template", detail={"reason": "llm disabled/unavailable or no evidence refs"})
        return incident

    try:
        narration = _run_langgraph(incident, refs, llm, settings, audit)
    except Exception:
        narration, _ = _narrate_once(incident, refs, llm)

    if narration:
        _finalize(incident, narration, NarrationSource.llm, GroundingDecision.resolved)
        if audit:
            audit.record("validator", "accept", detail={"chars": len(narration)})
    else:
        _finalize(incident, template, NarrationSource.template, GroundingDecision.abstain)
        if audit:
            audit.record("validator", "abstain_to_template", detail={})
    return incident
