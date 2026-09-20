"""Optional OpenAI chat-model adapter used by prompt-aware subagents."""

from __future__ import annotations

import os
from typing import Any


class OpenAILLM:
    """Small adapter that keeps OpenAI configuration outside agent logic."""

    def __init__(self, model: Any) -> None:
        self._model = model

    @classmethod
    def from_environment(cls) -> OpenAILLM | None:
        if os.getenv("SUPPORT_PLATFORM_LLM_ENABLED", "false").lower() != "true":
            return None
        if not os.getenv("OPENAI_API_KEY"):
            return None

        from langchain_openai import ChatOpenAI

        return cls(
            ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0,
            )
        )

    def complete(self, system_prompt: str, user_input: str) -> str:
        response = self._model.invoke(
            [
                ("system", system_prompt),
                ("human", user_input),
            ]
        )
        content = response.content
        return content if isinstance(content, str) else str(content)
