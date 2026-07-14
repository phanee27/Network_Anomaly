from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from rca.config import Settings, get_settings
from rca.detect.common import METRIC_SEP


def default_masking() -> list:
    from drain3.masking import MaskingInstruction

    return [
        MaskingInstruction(r"((?<=[^A-Za-z0-9])|^)(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})((?=[^A-Za-z0-9])|$)", "IP"),
        MaskingInstruction(r"((?<=[^A-Za-z0-9])|^)(0x[0-9a-fA-F]+)((?=[^A-Za-z0-9])|$)", "HEX"),
        MaskingInstruction(r"((?<=[^A-Za-z0-9])|^)([\-\+]?\d+)((?=[^A-Za-z0-9])|$)", "NUM"),
    ]


def build_miner(settings: Optional[Settings] = None):
    from drain3.template_miner import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig

    settings = settings or get_settings()
    config = TemplateMinerConfig()
    config.drain_sim_th = settings.drain_sim_th
    config.drain_depth = settings.drain_depth
    config.drain_max_children = settings.drain_max_children
    config.masking_instructions = default_masking()
    return TemplateMiner(config=config)


def parse_logs_drain3(
    path: Union[str, Path],
    service_col: str,
    message_col: str,
    time_col: str,
    time_unit: Optional[str] = None,
    bucket: str = "15s",
    settings: Optional[Settings] = None,
) -> tuple[pd.DataFrame, dict[int, str]]:
    settings = settings or get_settings()
    miner = build_miner(settings)
    df = pd.read_csv(path)
    for col in (service_col, message_col, time_col):
        if col not in df.columns:
            raise ValueError(f"log file '{path}' is missing column '{col}'")
    if time_unit:
        index = pd.to_datetime(pd.to_numeric(df[time_col], errors="coerce"), unit=time_unit, utc=True)
    else:
        index = pd.to_datetime(df[time_col], utc=True)
    services = df[service_col].astype(str)
    cluster_ids = [miner.add_log_message(str(m))["cluster_id"] for m in df[message_col]]
    long = pd.DataFrame({"service": services.to_numpy(), "cid": cluster_ids})
    long["col"] = long["service"] + METRIC_SEP + "tmpl" + long["cid"].astype(str)
    long.index = index
    long = long.dropna(subset=["col"])
    counts = long.groupby([pd.Grouper(freq=bucket), "col"]).size().unstack(fill_value=0)
    templates = {c.cluster_id: c.get_template() for c in miner.drain.clusters}
    return counts.sort_index(), templates
