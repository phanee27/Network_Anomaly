from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from rca.schema import ConfigChange


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def load_config_changes(incident_id: str, items: list[dict[str, Any]]) -> list[ConfigChange]:
    out: list[ConfigChange] = []
    for item in items:
        entity = item.get("entity_id") or item.get("component") or item.get("service")
        if not entity:
            continue
        out.append(
            ConfigChange(
                incident_id=incident_id,
                entity_id=str(entity),
                ts=_parse_ts(item.get("ts") or item.get("time")),
                change_type=str(item.get("change_type") or item.get("type") or "change"),
                detail=item.get("detail") if isinstance(item.get("detail"), dict) else dict(item),
            )
        )
    return out


def synth_config_feed(
    incident_id: str,
    inject_time: Optional[datetime],
    on_path: Optional[str] = None,
    off_path: Optional[str] = None,
    lead_seconds: int = 15,
) -> list[ConfigChange]:
    ts = inject_time - timedelta(seconds=lead_seconds) if inject_time else None
    changes: list[ConfigChange] = []
    if off_path:
        changes.append(
            ConfigChange(
                incident_id=incident_id,
                entity_id=off_path,
                ts=ts,
                change_type="deploy",
                detail={"summary": f"version rollout on {off_path}", "source": "synthetic"},
            )
        )
    if on_path:
        changes.append(
            ConfigChange(
                incident_id=incident_id,
                entity_id=on_path,
                ts=ts,
                change_type="config_edit",
                detail={"summary": f"config edit on {on_path}", "source": "synthetic"},
            )
        )
    return changes
