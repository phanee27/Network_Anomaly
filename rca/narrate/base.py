from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, system: str, user: str, json_mode: bool = False) -> Optional[str]: ...

    def available(self) -> bool: ...
