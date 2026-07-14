from __future__ import annotations

from typing import Optional

from rca.config import Settings, get_settings


class OllamaLLM:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._client = None

    def _client_or_none(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except Exception:
                return None
            self._client = OpenAI(
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                timeout=self.settings.llm_timeout_s,
                max_retries=self.settings.llm_max_retries,
            )
        return self._client

    def available(self) -> bool:
        return bool(self.settings.llm_enabled)

    def complete(self, system: str, user: str, json_mode: bool = False) -> Optional[str]:
        if not self.settings.llm_enabled:
            return None
        client = self._client_or_none()
        if client is None:
            return None
        kwargs = {
            "model": self.settings.llm_model,
            "temperature": self.settings.llm_temperature,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception:
            return None
