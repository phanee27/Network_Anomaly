from __future__ import annotations

from typing import Optional

import pandas as pd

from rca.detect.common import parse_metric_col
from rca.schema import Entity, EntityKind, TopologyEdge


def entities_from(
    metrics: Optional[pd.DataFrame],
    trace_latency: Optional[pd.DataFrame],
    edges: list[TopologyEdge],
) -> list[Entity]:
    ids: list[str] = []

    def add(name: str) -> None:
        if name and name not in ids:
            ids.append(name)

    for frame in (metrics, trace_latency):
        if frame is not None and len(frame.columns):
            for col in frame.columns:
                add(parse_metric_col(str(col))[0])
    for edge in edges:
        add(edge.src)
        add(edge.dst)
    return [Entity(id=i, kind=EntityKind.service, name=i) for i in ids]
