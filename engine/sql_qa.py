"""Counting / ranking questions: Databricks AI/BI Genie when configured, otherwise a local SQL engine.

Genie:    set GENIE_SPACE_ID (the app must run on Databricks Apps or have DATABRICKS_HOST + auth).
Fallback: DuckDB in memory over two permitted-only views. With an LLM key, the model writes the SQL
          (SELECT only, validated, sandboxed: no file or network access). Without a key, predefined
          queries are matched to the question.
Neither path can see restricted or outdated reviews: they are not in the views.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from engine import llm
from engine.config import Settings
from engine.data import build_segments


@dataclass
class SQLResult:
    question: str
    engine: str
    ok: bool
    sql: str = ""
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    answer_text: str = ""
    message: str = ""
    cost_usd: float = 0.0
    latency_ms: int = 0
    conversation_id: str | None = None


# ---- views exposed to SQL -----------------------------------------------------------------------
def sql_views(corpus: pd.DataFrame) -> dict[str, pd.DataFrame]:
    perm = corpus[corpus["permitted"]].copy()
    reviews_enriched = perm[["review_id", "product_id", "topic", "review_text", "review_date", "sentiment",
                             "is_evidence"]].copy()
    reviews_enriched["review_date"] = pd.to_datetime(reviews_enriched["review_date"])
    segments = build_segments(corpus).drop(columns=["evidence_ids", "deleted"])
    return {"reviews_enriched": reviews_enriched.reset_index(drop=True),
            "review_segments": segments.reset_index(drop=True)}


SCHEMA_DOC = """Tables (DuckDB SQL). Only permitted, current reviews exist here.
reviews_enriched(review_id VARCHAR, product_id VARCHAR, topic VARCHAR, review_text VARCHAR,
                 review_date TIMESTAMP, sentiment VARCHAR, is_evidence BOOLEAN)
  - product_id in ('jacket_a','jacket_b','shoe_a','shoe_b'); topic in ('comfort','fit','durability')
  - sentiment in ('positive','neutral','negative','unclassified'); is_evidence = false means the text
    carries no product evidence (e.g. 'I cannot assess...') and must be excluded from sentiment counts.
review_segments(product_id, topic, total, revoked, restricted, outdated, no_evidence, evidence,
                positive, neutral, negative, negative_share DOUBLE, first_review_date, last_review_date)
  - one row per product x topic; negative_share = negative / evidence."""

_FORBIDDEN = re.compile(
    r"\b(attach|detach|copy|pragma|install|load|create|insert|update|delete|drop|alter|export|import|call|set|"
    r"reset|checkpoint|vacuum|read_csv|read_csv_auto|read_parquet|read_json|read_json_auto|read_text|read_blob|"
    r"glob|httpfs|getenv|query_table|query)\b", re.IGNORECASE)


def validate_sql(sql: str) -> str:
    s = sql.strip().rstrip(";").strip()
    if not s:
        raise ValueError("Empty SQL")
    if ";" in s:
        raise ValueError("Only one statement is allowed")
    if not re.match(r"^(select|with)\b", s, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed")
    if _FORBIDDEN.search(s):
        raise ValueError(f"Forbidden keyword: {_FORBIDDEN.search(s).group(0)}")
    if re.search(r"\b(from|join)\s*['\"(]?\s*['\"]", s, re.IGNORECASE):
        raise ValueError("File paths are not allowed as tables")
    return s


def run_local_sql(sql: str, views: dict[str, pd.DataFrame], limit: int = 500) -> pd.DataFrame:
    import duckdb

    safe = validate_sql(sql)
    con = duckdb.connect(config={"enable_external_access": False})
    try:
        for name, frame in views.items():
            con.register(name, frame)
        return con.execute(f"SELECT * FROM ({safe}) AS q LIMIT {int(limit)}").df()
    finally:
        con.close()


# ---- predefined queries (no LLM needed) -------------------------------------------------------
def _filters(question: str, products: list[str], topics: list[str]) -> tuple[list[str], list[str]]:
    from engine.router import resolve_entities

    found_p, _, found_t, _ = resolve_entities(question, products)
    return [p for p in found_p if p in products], [t for t in found_t if t in topics]


def _where(prod: list[str], top: list[str], prefix: str = "WHERE") -> str:
    clauses = []
    if prod:
        clauses.append("product_id IN (" + ", ".join(f"'{p}'" for p in prod) + ")")
    if top:
        clauses.append("topic IN (" + ", ".join(f"'{t}'" for t in top) + ")")
    return f" {prefix} " + " AND ".join(clauses) if clauses else ""


PRESETS = [
    ("Negative share by product and issue", ("negative", "worst", "complain", "share", "highest", "most",
                                             "rank", "ranking", "top")),
    ("Reviews per product", ("how many", "count", "number of", "per product", "volume")),
    ("Sentiment breakdown", ("breakdown", "distribution", "sentiment", "positive", "neutral", "mix")),
    ("Excluded reviews by reason", ("excluded", "exclusion", "why", "no evidence", "restricted", "outdated", "quality")),
]


def preset_sql(name: str, question: str, products: list[str], topics: list[str]) -> str:
    prod, top = _filters(question, products, topics)
    if name == "Negative share by product and issue":
        return ("SELECT product_id, topic, evidence, negative, ROUND(negative_share * 100, 1) AS negative_pct "
                f"FROM review_segments{_where(prod, top)} ORDER BY negative_share DESC, evidence DESC, product_id, topic")
    if name == "Reviews per product":
        return ("SELECT product_id, COUNT(*) AS permitted_reviews, "
                "COUNT(*) FILTER (WHERE is_evidence) AS evidence_reviews "
                f"FROM reviews_enriched{_where(prod, top)} GROUP BY product_id ORDER BY product_id")
    if name == "Sentiment breakdown":
        return ("SELECT product_id, sentiment, COUNT(*) AS reviews FROM reviews_enriched WHERE is_evidence"
                f"{_where(prod, top, 'AND')} GROUP BY product_id, sentiment ORDER BY product_id, sentiment")
    return ("SELECT product_id, topic, total, restricted, outdated, no_evidence, evidence "
            f"FROM review_segments{_where(prod, top)} ORDER BY product_id, topic")


def match_preset(question: str) -> str:
    q = question.lower()
    best, best_hits = PRESETS[0][0], 0
    for name, words in PRESETS:
        hits = sum(1 for w in words if w in q)
        if hits > best_hits:
            best, best_hits = name, hits
    return best


def describe_result(preset: str, df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows matched."
    if preset == "Negative share by product and issue" and "negative_pct" in df:
        top = df["negative_pct"].max()
        leaders = df[df["negative_pct"] == top]
        if len(leaders) == len(df) and len(df) > 1:
            return f"All {len(df)} product × issue segments tie at {top:g}% negative reviews."
        names = ", ".join(f"{r.product_id} · {r.topic}" for r in leaders.itertuples())
        return f"Highest negative share: {names} at {top:g}% ({len(leaders)} of {len(df)} segments)."
    if preset == "Sentiment breakdown" and {"sentiment", "reviews"} <= set(df.columns):
        totals = df.groupby("sentiment")["reviews"].sum()
        n = int(totals.sum())
        mix = ", ".join(f"{int(totals.get(s, 0))} {s}" for s in ("positive", "neutral", "negative"))
        return f"Across {n} evidence reviews: {mix}. Every product shows the same mix in this dataset." \
            if df.groupby("product_id")["reviews"].apply(tuple).nunique() == 1 else f"Across {n} evidence reviews: {mix}."
    if preset == "Excluded reviews by reason" and "no_evidence" in df.columns:
        return (f"{int(df['restricted'].sum())} restricted, {int(df['outdated'].sum())} outdated and "
                f"{int(df['no_evidence'].sum())} no-evidence reviews are excluded; {int(df['evidence'].sum())} "
                "are used as evidence.")
    if preset == "Reviews per product":
        return "Permitted reviews per product: " + ", ".join(
            f"{r.product_id} {r.permitted_reviews} ({r.evidence_reviews} with product evidence)" for r in df.itertuples())
    return f"Returned {len(df)} rows."


# ---- LLM-written SQL ----------------------------------------------------------------------------
SQL_SYSTEM = f"""You translate business questions about customer reviews into ONE DuckDB SELECT query.
{SCHEMA_DOC}
Rules: SELECT or WITH only; only the two tables above; no file paths; prefer review_segments for shares
and counts by product/topic; round percentages to 1 decimal.
If the question cannot be answered from these tables (price, roadmap, warranty, sales...), return sql = "".
Reply with JSON only: {{"sql": "...", "explanation": "one sentence"}}"""

ANSWER_SYSTEM = """Answer the question in at most two sentences using ONLY the query result rows provided.
State ties honestly. Do not invent numbers. Reply with plain text."""


def _local(question: str, views, settings: Settings, products, topics, use_llm: bool) -> SQLResult:
    start = time.perf_counter()
    if use_llm and llm.is_available(settings):
        cost = 0.0
        try:
            r1 = llm.complete(settings, SQL_SYSTEM, f"Question: {question}", max_tokens=600)
            cost += r1.cost_usd
            obj = llm.parse_json(r1.text)
            sql = str(obj.get("sql") or "").strip()
            if not sql:
                return SQLResult(question, "Local SQL (LLM)", False, message="The model judged this question "
                                 "unanswerable from the review tables.", cost_usd=cost,
                                 latency_ms=int((time.perf_counter() - start) * 1000))
            try:
                df = run_local_sql(sql, views)
            except Exception as exc:  # one repair attempt
                r2 = llm.complete(settings, SQL_SYSTEM, f"Question: {question}\nYour previous SQL:\n{sql}\n"
                                  f"failed with: {exc}\nReturn corrected JSON.", max_tokens=600)
                cost += r2.cost_usd
                sql = str(llm.parse_json(r2.text).get("sql") or "").strip()
                df = run_local_sql(sql, views)
            r3 = llm.complete(settings, ANSWER_SYSTEM, f"Question: {question}\nRows:\n"
                              f"{df.head(30).to_csv(index=False)}", max_tokens=200)
            cost += r3.cost_usd
            return SQLResult(question, "Local SQL (LLM-written)", True, sql=validate_sql(sql), df=df,
                             answer_text=r3.text.strip(), message=str(obj.get("explanation") or ""),
                             cost_usd=cost, latency_ms=int((time.perf_counter() - start) * 1000))
        except Exception as exc:
            fallback = _preset(question, views, products, topics, start)
            fallback.message = f"LLM SQL failed ({type(exc).__name__}: {exc}); used a predefined query. " + \
                fallback.message
            fallback.cost_usd = cost
            return fallback
    return _preset(question, views, products, topics, start)


def _preset(question, views, products, topics, start) -> SQLResult:
    name = match_preset(question)
    sql = preset_sql(name, question, products, topics)
    df = run_local_sql(sql, views)
    return SQLResult(question, "Predefined query", True, sql=sql, df=df, answer_text=describe_result(name, df),
                     message=f"Matched preset: {name}.", latency_ms=int((time.perf_counter() - start) * 1000))


# ---- Genie ------------------------------------------------------------------------------------
def _genie_client(settings: Settings, user_token: str | None):
    from databricks.sdk import WorkspaceClient

    host = os.environ.get("DATABRICKS_HOST")
    if user_token and settings.genie_use_user_token and host:
        return WorkspaceClient(host=host, token=user_token, auth_type="pat"), "user"
    return WorkspaceClient(), "app"


def _statement_to_df(statement) -> pd.DataFrame:
    if statement is None or statement.manifest is None or statement.manifest.schema is None:
        return pd.DataFrame()
    cols = [c.name for c in (statement.manifest.schema.columns or [])]
    rows = (statement.result.data_array if statement.result is not None else None) or []
    df = pd.DataFrame(rows, columns=cols)
    for c in df.columns:
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().sum() == df[c].notna().sum():
            df[c] = converted
    return df


def genie_ask(settings: Settings, question: str, conversation_id: str | None = None,
              user_token: str | None = None, client=None) -> SQLResult:
    start = time.perf_counter()
    space = settings.genie_space_id
    try:
        w, identity = (client, "injected") if client is not None else _genie_client(settings, user_token)
        if conversation_id:
            msg = w.genie.create_message_and_wait(space, conversation_id, question, timeout=timedelta(seconds=120))
        else:
            msg = w.genie.start_conversation_and_wait(space, question, timeout=timedelta(seconds=120))
        status = getattr(msg.status, "value", msg.status)
        conv = msg.conversation_id
        msg_id = msg.message_id or msg.id
        if status != "COMPLETED":
            err = getattr(getattr(msg, "error", None), "error", None)
            return SQLResult(question, "Genie", False, message=f"Genie status {status}. {err or ''}".strip(),
                             conversation_id=conv, latency_ms=int((time.perf_counter() - start) * 1000))
        texts, sql, df, desc = [], "", pd.DataFrame(), ""
        for att in msg.attachments or []:
            if att.text is not None and att.text.content:
                texts.append(att.text.content)
            if att.query is not None:
                sql = att.query.query or sql
                desc = att.query.description or desc
                res = w.genie.get_message_attachment_query_result(space, conv, msg_id, att.attachment_id)
                df = _statement_to_df(res.statement_response)
        answer = " ".join(texts).strip() or desc or ("Query returned results." if sql else "")
        return SQLResult(question, "Genie", True, sql=sql, df=df, answer_text=answer,
                         message=f"Answered by Genie (identity: {identity}).", conversation_id=conv,
                         latency_ms=int((time.perf_counter() - start) * 1000))
    except Exception as exc:
        return SQLResult(question, "Genie", False, message=f"Genie error: {type(exc).__name__}: {exc}",
                         conversation_id=conversation_id, latency_ms=int((time.perf_counter() - start) * 1000))


def genie_configured(settings: Settings) -> bool:
    return bool(settings.genie_space_id)


def ask_sql(question: str, corpus: pd.DataFrame, settings: Settings, use_llm: bool = True,
            conversation_id: str | None = None, user_token: str | None = None) -> SQLResult:
    """Genie first if configured; on Genie failure, fall back to local SQL and say so."""
    products = sorted(corpus["product_id"].unique())
    topics = sorted(corpus["topic"].unique())
    if genie_configured(settings):
        res = genie_ask(settings, question, conversation_id, user_token)
        if res.ok:
            return res
        local = _local(question, sql_views(corpus), settings, products, topics, use_llm)
        local.message = f"{res.message} → fell back to local SQL. " + local.message
        local.conversation_id = res.conversation_id
        return local
    return _local(question, sql_views(corpus), settings, products, topics, use_llm)
