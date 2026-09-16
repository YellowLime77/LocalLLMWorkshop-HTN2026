"""
llm.py: call a cloud LLM, and fall back to local Ollama when the cloud stops working.

    pip install openai
    from llm import chat
    print(chat([{"role": "user", "content": "Write a haiku about Hack the North"}]))

Env vars (all optional):
    CLOUD_API_KEY   leave unset to always use local
    CLOUD_BASE_URL  any OpenAI-compatible API (default: OpenAI)
    CLOUD_MODEL     default gpt-5-mini
    LOCAL_BASE_URL  default http://localhost:11434/v1
    LOCAL_MODEL     default gemma4:e2b
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIStatusError, OpenAI, Timeout

COOLDOWN_SECONDS = 60
FALLBACK_STATUS_CODES = {401, 402, 403, 404, 408, 429}  # plus every 5xx
LOCAL_OPTIONS: dict[str, Any] = {"reasoning_effort": "none"}  # Gemma 4: skip thinking

log = logging.getLogger("llm")
_clients: tuple[OpenAI | None, OpenAI] | None = None
_cloud_retry_at = 0.0


@dataclass
class LLMResult:
    text: str
    provider: str  # "cloud" or "local"
    model: str
    response: Any  # the full ChatCompletion, for tool calls, usage, etc.


def _get_clients() -> tuple[OpenAI | None, OpenAI]:
    global _clients
    if _clients is None:
        cloud = None
        if os.getenv("CLOUD_API_KEY"):
            cloud = OpenAI(
                base_url=os.getenv("CLOUD_BASE_URL") or "https://api.openai.com/v1",
                api_key=os.environ["CLOUD_API_KEY"],
                timeout=Timeout(60.0, connect=5.0),  # give up fast when offline, allow long answers
                max_retries=0,  # falling back is faster than retrying
            )
        local = OpenAI(base_url=os.getenv("LOCAL_BASE_URL") or "http://localhost:11434/v1", api_key="ollama")
        _clients = (cloud, local)
    return _clients


def _cloud_is_down(err: Exception) -> bool:
    if isinstance(err, APIConnectionError):  # offline, DNS failure, or timeout
        return True
    return isinstance(err, APIStatusError) and (err.status_code in FALLBACK_STATUS_CODES or err.status_code >= 500)


def complete(messages: list[dict[str, Any]], **kwargs: Any) -> LLMResult:
    """Same arguments as client.chat.completions.create(), minus model. No streaming."""
    global _cloud_retry_at
    cloud, local = _get_clients()

    if cloud and time.monotonic() >= _cloud_retry_at:
        model = os.getenv("CLOUD_MODEL") or "gpt-5-mini"
        try:
            r = cloud.chat.completions.create(model=model, messages=messages, **kwargs)
            return LLMResult(r.choices[0].message.content or "", "cloud", model, r)
        except (APIConnectionError, APIStatusError) as err:
            if not _cloud_is_down(err):
                raise  # e.g. 400: the request is wrong, local would fail too
            log.warning("Cloud failed (%s). Using local Ollama for %ss.", type(err).__name__, COOLDOWN_SECONDS)
            _cloud_retry_at = time.monotonic() + COOLDOWN_SECONDS

    model = os.getenv("LOCAL_MODEL") or "gemma4:e2b"
    try:
        r = local.chat.completions.create(model=model, messages=messages, **{**LOCAL_OPTIONS, **kwargs})
    except APIConnectionError as err:
        raise RuntimeError("Cloud is unavailable and local Ollama isn't reachable. Is Ollama running?") from err
    return LLMResult(r.choices[0].message.content or "", "local", model, r)


def chat(messages: list[dict[str, Any]], **kwargs: Any) -> str:
    return complete(messages, **kwargs).text