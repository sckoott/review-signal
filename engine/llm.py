"""Minimal LLM client: Claude via the Anthropic API, or any model via OpenRouter. Optional.

If no key is configured, `is_available()` is False and callers fall back to extractive answers.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import requests

from engine.config import Settings

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_S = 45


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int


def is_available(settings: Settings) -> bool:
    return settings.llm_provider in ("anthropic", "openrouter")


def describe(settings: Settings) -> str:
    if settings.llm_provider == "anthropic":
        return f"Claude via Anthropic API · {settings.anthropic_model}"
    if settings.llm_provider == "openrouter":
        return f"OpenRouter · {settings.openrouter_model}"
    return "No LLM key: extractive mode"


def _cost(settings: Settings, tin: int, tout: int) -> float:
    return tin / 1e6 * settings.price_input_per_mtok + tout / 1e6 * settings.price_output_per_mtok


def complete(settings: Settings, system: str, user: str, max_tokens: int = 900) -> LLMResult:
    if not is_available(settings):
        raise LLMError("No LLM provider configured")
    start = time.perf_counter()
    if settings.llm_provider == "anthropic":
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMError("The 'anthropic' package is not installed") from exc
        try:
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=TIMEOUT_S, max_retries=1)
            msg = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:
            raise LLMError(f"Anthropic API error: {type(exc).__name__}: {exc}") from exc
        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
        tin = int(getattr(msg.usage, "input_tokens", 0) or 0)
        tout = int(getattr(msg.usage, "output_tokens", 0) or 0)
        model = settings.anthropic_model
    else:
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {settings.openrouter_api_key}",
                         "Content-Type": "application/json",
                         "X-Title": "Review Signal"},
                json={"model": settings.openrouter_model, "max_tokens": max_tokens,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}]},
                timeout=TIMEOUT_S,
            )
        except requests.RequestException as exc:
            raise LLMError(f"OpenRouter connection error: {exc}") from exc
        if resp.status_code != 200:
            raise LLMError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected OpenRouter response: {resp.text[:300]}") from exc
        usage = data.get("usage") or {}
        tin = int(usage.get("prompt_tokens", 0) or 0)
        tout = int(usage.get("completion_tokens", 0) or 0)
        model = settings.openrouter_model
    latency = int((time.perf_counter() - start) * 1000)
    return LLMResult(text=text, provider=settings.llm_provider, model=model, input_tokens=tin,
                     output_tokens=tout, cost_usd=_cost(settings, tin, tout), latency_ms=latency)


def parse_json(text: str) -> dict:
    """Extract the first JSON object from a model reply (tolerates ```json fences)."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise LLMError("Model reply contained no JSON object")
    try:
        obj = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"Model reply was not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise LLMError("Model reply JSON is not an object")
    return obj
