"""Session state, cached data access, identity and telemetry shared by every page."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from engine.config import Settings, load_settings
from engine.data import build_corpus, build_segments, load_tables

ROLES = ("Product team", "Executive", "Solutions Architect")

PAGES = {
    "ask": "views/ask.py",
    "issue_map": "views/issue_map.py",
    "data_qa": "views/data_qa.py",
    "recommendations": "views/recommendations.py",
    "exec_brief": "views/exec_brief.py",
    "trust_center": "views/trust_center.py",
}


def init_state() -> None:
    defaults = {
        "role": None,
        "revoked": [],
        "deleted": [],
        "history": [],          # Ask answers
        "sql_history": [],      # Data Q&A results
        "telemetry": [],
        "rec_log": [],
        "exec_pins": [],
        "genie_conversation_id": None,
        "use_llm": True,
        "weights": {},
        "rec_draft": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def settings() -> Settings:
    return load_settings()


@st.cache_data(show_spinner=False)
def tables() -> dict[str, pd.DataFrame]:
    return load_tables()


@st.cache_data(show_spinner=False)
def _corpus(cutoff: str, revoked: tuple[str, ...], deleted: tuple[str, ...]) -> pd.DataFrame:
    return build_corpus(tables()["reviews"], cutoff, frozenset(revoked), frozenset(deleted))


def corpus() -> pd.DataFrame:
    """Current corpus, including any revocations/deletions simulated in the Trust Center."""
    return _corpus(settings().freshness_cutoff, tuple(sorted(st.session_state.get("revoked", []))),
                   tuple(sorted(st.session_state.get("deleted", []))))


def base_corpus() -> pd.DataFrame:
    return _corpus(settings().freshness_cutoff, (), ())


def segments() -> pd.DataFrame:
    return build_segments(corpus())


def products() -> list[str]:
    return sorted(tables()["reviews"]["product_id"].unique())


def topics() -> list[str]:
    return sorted(tables()["reviews"]["topic"].unique())


def simulation_active() -> bool:
    return bool(st.session_state.get("revoked") or st.session_state.get("deleted"))


# ---- identity (Databricks Apps forwards these headers; absent when running locally) ----------
def _header(name: str) -> str | None:
    try:
        return st.context.headers.get(name)
    except Exception:
        return None


def user_email() -> str:
    return _header("X-Forwarded-Email") or "local-user"


def user_token() -> str | None:
    """Read per request, never cached (Databricks guidance for on-behalf-of-user auth)."""
    return _header("X-Forwarded-Access-Token")


def default_role() -> str:
    return "Executive" if user_email().lower() in settings().exec_emails else "Product team"


# ---- telemetry ---------------------------------------------------------------------------------
def log_event(page: str, question: str, route: str, mode: str, latency_ms: int, cost_usd: float,
              answer_id: str) -> None:
    st.session_state["telemetry"].append({
        "time": dt.datetime.now().strftime("%H:%M:%S"), "user": user_email(), "role": st.session_state.get("role"),
        "page": page, "question": question, "route": route, "mode": mode, "latency_ms": int(latency_ms),
        "cost_usd": round(float(cost_usd), 6), "feedback": None, "answer_id": answer_id,
    })


def set_feedback(answer_id: str, value: str | None) -> None:
    for event in st.session_state["telemetry"]:
        if event["answer_id"] == answer_id:
            event["feedback"] = value


def telemetry_df() -> pd.DataFrame:
    cols = ["time", "user", "role", "page", "question", "route", "mode", "latency_ms", "cost_usd", "feedback",
            "answer_id"]
    return pd.DataFrame(st.session_state.get("telemetry", []), columns=cols)


def run_sql(question: str):
    from engine.sql_qa import ask_sql

    res = ask_sql(question, corpus(), settings(), use_llm=st.session_state.get("use_llm", True),
                  conversation_id=st.session_state.get("genie_conversation_id"), user_token=user_token())
    if res.conversation_id:
        st.session_state["genie_conversation_id"] = res.conversation_id
    return res


@st.cache_data(show_spinner=False)
def _scorecard(cutoff: str) -> pd.DataFrame:
    from engine.retrieval import scorecard

    return scorecard(tables(), build_corpus(tables()["reviews"], cutoff))


def scorecard() -> pd.DataFrame:
    """Evaluation on the unmodified corpus (simulations never change the benchmark)."""
    return _scorecard(settings().freshness_cutoff)


@st.cache_data(show_spinner=False)
def _test_matrix(cutoff: str) -> pd.DataFrame:
    from engine.governance import run_test_matrix

    return run_test_matrix(tables(), cutoff)


def test_matrix() -> pd.DataFrame:
    return _test_matrix(settings().freshness_cutoff)
