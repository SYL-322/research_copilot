"""Minimal OpenAI Chat Completions wrapper (configurable model via Settings / env)."""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from core.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OpenAIClient:
    """Chat completions with optional JSON / structured parsing."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        key = (settings.openai_api_key or "").strip()
        if not key:
            logger.warning("OPENAI_API_KEY is empty; API calls will fail until set.")
        self._client = OpenAI(api_key=key or None)

    @property
    def default_model(self) -> str:
        return self._settings.openai_model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """
        Run a chat completion and return assistant message content (plain text).

        Raises if the API returns an error or empty content when JSON was expected.
        """
        use_model = model or self._settings.openai_model
        kwargs: dict[str, Any] = {
            "model": use_model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        logger.debug("OpenAI chat model=%s messages=%d", use_model, len(messages))
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        content = choice.message.content
        if content is None:
            return ""
        return content

    def chat_json_object(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Request a JSON object response (``response_format`` json_object)."""
        return self.chat(
            messages,
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
        )

    def chat_parse(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        *,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> T:
        """
        Return a validated Pydantic instance.

        Tries ``beta.chat.completions.parse`` when available; otherwise
        ``chat_json_object`` + ``model_validate_json``.
        """
        use_model = model or self._settings.openai_model
        parse_fn = getattr(self._client.beta.chat.completions, "parse", None)
        if parse_fn is not None:
            try:
                resp = parse_fn(
                    model=use_model,
                    messages=messages,
                    response_format=response_model,
                    temperature=temperature,
                )
                msg = resp.choices[0].message
                parsed = getattr(msg, "parsed", None)
                if parsed is not None:
                    return parsed
                raw = msg.content or ""
                if raw.strip():
                    return response_model.model_validate_json(strip_json_fences(raw))
            except Exception as e:
                logger.warning("Structured parse failed (%s); falling back to JSON object", e)

        raw = self.chat_json_object(messages, model=model, temperature=temperature)
        return response_model.model_validate_json(strip_json_fences(raw))


def strip_json_fences(text: str) -> str:
    """Remove optional markdown code fences from model output."""
    s = text.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if len(lines) >= 2 and lines[0].startswith("```"):
            inner = "\n".join(lines[1:])
            if "```" in inner:
                inner = inner.rsplit("```", 1)[0]
            return inner.strip()
    return s
