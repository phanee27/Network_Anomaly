from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class RootCauseCandidate:
    entity_id: str
    score: float
    onset: Optional[datetime] = None
    severity: float = 0.0
    impact_path: list[str] = field(default_factory=list)
    demoted: bool = False
    note: Optional[str] = None
    reasons: list[str] = field(default_factory=list)
    fault_hint: Optional[str] = None


@dataclass
class CausalResult:
    candidates: list[RootCauseCandidate]
    frz: list[str] = field(default_factory=list)
    edges_used: list[tuple[str, str]] = field(default_factory=list)
    pc_applied: bool = False
    gnn_applied: bool = False
