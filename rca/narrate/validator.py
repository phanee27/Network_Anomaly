from __future__ import annotations

from typing import Any


def validate_statements(statements: Any, allowed: set[str]) -> tuple[list[dict], list[Any]]:
    valid: list[dict] = []
    rejected: list[Any] = []
    if not isinstance(statements, list):
        return valid, rejected
    for st in statements:
        if not isinstance(st, dict):
            rejected.append(st)
            continue
        text = str(st.get("text", "")).strip()
        refs = st.get("refs") or []
        if isinstance(refs, str):
            refs = [refs]
        if text and refs and all(r in allowed for r in refs):
            valid.append({"text": text, "refs": list(refs)})
        else:
            rejected.append(st)
    return valid, rejected


def assemble(valid: list[dict]) -> str:
    return " ".join(st["text"] for st in valid).strip()
