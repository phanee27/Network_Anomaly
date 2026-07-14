from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from rca.detect.common import METRIC_SEP, metric_col


def _entity_metric(col: str, split: str) -> str:
    if split == "first_underscore" and "_" in col:
        entity, metric = col.split("_", 1)
        return metric_col(entity, metric)
    if split == "last_underscore" and "_" in col:
        entity, metric = col.rsplit("_", 1)
        return metric_col(entity, metric)
    return metric_col(col, "value")


def load_metric_frame(
    path: Union[str, Path],
    time_col: str = "time",
    time_unit: Optional[str] = "s",
    split: str = "first_underscore",
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if time_col not in df.columns:
        raise ValueError(f"metric file '{path}' is missing time column '{time_col}'")
    raw = df[time_col]
    if time_unit and pd.api.types.is_numeric_dtype(raw):
        index = pd.to_datetime(raw, unit=time_unit, utc=True)
    else:
        index = pd.to_datetime(raw, utc=True)
    body = df.drop(columns=[time_col]).apply(pd.to_numeric, errors="coerce")
    body.index = index
    body.columns = [_entity_metric(str(c), split) for c in body.columns]
    body = body[~body.index.duplicated(keep="first")].sort_index()
    return body


def clean_rcaeval_latency(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if str(c).endswith(f"{METRIC_SEP}latency-50")]
    out = df.drop(columns=drop)
    rename = {c: str(c).replace("latency-90", "latency") for c in out.columns if "latency-90" in str(c)}
    return out.rename(columns=rename)
