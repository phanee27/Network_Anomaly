from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from rca.config import Settings, get_settings
from rca.ingest.types import IncidentData
from rca.schema import AnomalySignal, Detector


class DeepLogDetector:
    name = "deeplog"
    modality = "log"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._model = None
        self._miner = None
        self._meta = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from drain3.file_persistence import FilePersistence
        from drain3.template_miner import TemplateMiner
        from drain3.template_miner_config import TemplateMinerConfig

        from rca.ingest.logs import default_masking
        from rca.ml.models import DeepLog

        root = self.settings.artifacts_root
        self._meta = json.loads((root / "deeplog.json").read_text())
        config = TemplateMinerConfig()
        config.drain_sim_th = self.settings.drain_sim_th
        config.drain_depth = self.settings.drain_depth
        config.drain_max_children = self.settings.drain_max_children
        config.masking_instructions = default_masking()
        self._miner = TemplateMiner(persistence_handler=FilePersistence(str(root / "drain3_state.bin")), config=config)
        model = DeepLog(self._meta["num_keys"], self._meta.get("hidden", 64), self._meta.get("layers", 2))
        model.load_state_dict(torch.load(root / "deeplog.pt", map_location="cpu"))
        model.eval()
        self._model = model

    def detect(self, data: IncidentData) -> list[AnomalySignal]:
        if not data.log_source:
            return []
        self._ensure_loaded()
        import torch

        meta = self._meta
        service_col = meta.get("service_col", "container_name")
        message_col = meta.get("message_col", "message")
        time_col = meta.get("time_col", "timestamp")
        h, g, num_keys = meta.get("window", 10), meta.get("topg", 9), meta["num_keys"]
        max_lines = meta.get("max_lines")

        want = {service_col, message_col, time_col}
        try:
            df = pd.read_csv(data.log_source, usecols=lambda c: c in want, nrows=max_lines)
        except Exception:
            return []
        if service_col not in df.columns or message_col not in df.columns:
            return []
        df = df.dropna(subset=[message_col])
        if time_col in df.columns:
            df = df.sort_values(time_col)

        signals: list[AnomalySignal] = []
        for service, group in df.groupby(service_col):
            seq = []
            for msg in group[message_col].astype(str):
                cid = self._miner.add_log_message(msg)["cluster_id"]
                seq.append(cid - 1 if cid > 0 else 0)
            if len(seq) <= h:
                continue
            anomalies = windows = 0
            for i in range(len(seq) - h):
                nxt = seq[i + h]
                windows += 1
                if nxt >= num_keys:
                    anomalies += 1
                    continue
                x = torch.tensor(seq[i : i + h], dtype=torch.float).view(1, h, 1)
                with torch.no_grad():
                    top = torch.argsort(self._model(x), dim=1)[0][-g:].tolist()
                if nxt not in top:
                    anomalies += 1
            if windows == 0:
                continue
            frac = anomalies / windows
            if frac > meta.get("anom_frac", 0.1):
                signals.append(
                    AnomalySignal(
                        incident_id=data.incident_id,
                        entity_id=str(service),
                        detector=Detector.deeplog,
                        window_start=data.inject_time,
                        window_end=data.inject_time,
                        score=float(frac),
                        indicator=f"DeepLog: {frac:.0%} of log windows deviate from learned normal sequence",
                    )
                )
        return signals
