from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from rca.schema import RelType, TopologyEdge


def topology_from_traces(
    path: Union[str, Path],
    span_col: str = "spanID",
    parent_col: str = "parentSpanID",
    service_col: str = "serviceName",
    min_calls: int = 1,
) -> list[TopologyEdge]:
    want = {span_col, parent_col, service_col}
    df = pd.read_csv(path, usecols=lambda c: c in want, dtype=str)
    if not want.issubset(df.columns):
        return []
    span_service = dict(zip(df[span_col], df[service_col]))
    df = df.assign(
        parent_service=df[parent_col].map(span_service),
        child_service=df[service_col],
    )
    df = df.dropna(subset=["parent_service"])
    df = df[df["parent_service"] != df["child_service"]]
    if df.empty:
        return []
    grouped = df.groupby(["parent_service", "child_service"]).size()
    edges: list[TopologyEdge] = []
    for (parent, child), count in grouped.items():
        if int(count) < min_calls:
            continue
        edges.append(TopologyEdge(src=str(parent), dst=str(child), rel=RelType.depends_on, weight=float(count)))
    return edges


def latency_from_traces(
    path: Union[str, Path],
    service_col: str = "serviceName",
    duration_col: str = "duration",
    time_col: str = "startTimeMillis",
    bucket: str = "15s",
) -> pd.DataFrame:
    want = {service_col, duration_col, time_col}
    df = pd.read_csv(path, usecols=lambda c: c in want)
    if not want.issubset(df.columns):
        return pd.DataFrame()
    index = pd.to_datetime(pd.to_numeric(df[time_col], errors="coerce"), unit="ms", utc=True)
    frame = pd.DataFrame(
        {"service": df[service_col].astype(str), "duration": pd.to_numeric(df[duration_col], errors="coerce")}
    )
    frame.index = index
    frame = frame.dropna()
    series = frame.groupby([pd.Grouper(freq=bucket), "service"])["duration"].mean().unstack()
    series.columns = [f"{c}::latency" for c in series.columns]
    return series.sort_index()
