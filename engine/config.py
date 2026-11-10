"""Settings. Every value comes from an environment variable first, then from .streamlit/secrets.toml.

Where to put the API key (pick ONE):
  * Local:            .streamlit/secrets.toml  ->  ANTHROPIC_API_KEY = "sk-ant-..."
                      or  export ANTHROPIC_API_KEY=sk-ant-...
  * Databricks Apps:  store it in a Databricks secret, add it to the app as a resource,
                      and map it to the ANTHROPIC_API_KEY env var in app.yaml (see README).
  * OpenRouter instead of Anthropic: set OPENROUTER_API_KEY the same way.

With no key at all the app still works: answers are extractive (exact quotes + IDs) and
the SQL page uses predefined queries.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"


def _get(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is not None and value.strip() != "":
        return value.strip()
    try:  # secrets.toml is optional; any error simply means "not set"
        import streamlit as st

        if name in st.secrets:
            value = str(st.secrets[name]).strip()
            return value or default
    except Exception:
        pass
    return default


def _float(name: str, default: float) -> float:
    raw = _get(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    llm_provider: str  # "anthropic", "openrouter" or "none"
    anthropic_api_key: str | None
    anthropic_model: str
    openrouter_api_key: str | None
    openrouter_model: str
    price_input_per_mtok: float
    price_output_per_mtok: float
    genie_space_id: str | None
    genie_use_user_token: bool
    freshness_cutoff: str
    exec_emails: tuple[str, ...]


def load_settings() -> Settings:
    anthropic_key = _get("ANTHROPIC_API_KEY")
    openrouter_key = _get("OPENROUTER_API_KEY")
    requested = (_get("LLM_PROVIDER", "auto") or "auto").lower()
    if requested == "anthropic":
        provider = "anthropic" if anthropic_key else "none"
    elif requested == "openrouter":
        provider = "openrouter" if openrouter_key else "none"
    elif requested == "none":
        provider = "none"
    else:  # auto: Anthropic first, then OpenRouter
        provider = "anthropic" if anthropic_key else ("openrouter" if openrouter_key else "none")
    exec_raw = _get("EXEC_EMAILS", "") or ""
    return Settings(
        llm_provider=provider,
        anthropic_api_key=anthropic_key,
        anthropic_model=_get("ANTHROPIC_MODEL", "claude-sonnet-5-5") or "claude-sonnet-5-5",
        openrouter_api_key=openrouter_key,
        openrouter_model=_get("OPENROUTER_MODEL", "anthropic/claude-sonnet-5.5") or "anthropic/claude-sonnet-5.5",
        # Claude Sonnet 5.5 list price, USD per million tokens. Override if you change model.
        price_input_per_mtok=_float("LLM_PRICE_INPUT_PER_MTOK", 2.0),
        price_output_per_mtok=_float("LLM_PRICE_OUTPUT_PER_MTOK", 10.0),
        genie_space_id=_get("GENIE_SPACE_ID"),
        genie_use_user_token=(_get("GENIE_USE_USER_TOKEN", "true") or "true").lower() == "true",
        freshness_cutoff=_get("FRESHNESS_CUTOFF", "2026-01-01") or "2026-01-01",
        exec_emails=tuple(e.strip().lower() for e in exec_raw.split(",") if e.strip()),
    )
