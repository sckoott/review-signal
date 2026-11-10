"""Data layer: load tables, apply the data contract, enrich reviews, build product x issue segments.

Mirrors the notebook:
  * W1 contract: source_allowed = true AND review_date >= cutoff  (96 of 120 rows)
  * W1 "permission != relevance": 36 permitted rows carry no product evidence
  * W3 simulated source changes: revoked and deleted review IDs
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from engine.config import DATA_DIR

NO_EVIDENCE_PHRASES = ("cannot assess", "does not describe", "no evidence")

# Status precedence: a row gets the first status that applies.
STATUS_ORDER = ["deleted", "revoked", "restricted", "outdated", "no_evidence", "evidence"]
STATUS_LABEL = {
    "deleted": "Deleted at source",
    "revoked": "Permission revoked",
    "restricted": "Restricted (source_allowed = false)",
    "outdated": "Outdated (before cutoff)",
    "no_evidence": "No product evidence in text",
    "evidence": "Used as evidence",
}

# Rule-based sentiment for this templated synthetic corpus. On real reviews, replace with
# Databricks SQL `ai_analyze_sentiment(review_text)` in the enrichment job.
_NEGATIVE = ("needs improvement", "disappointing", "poor", "bad", "terrible", "worse", "broke", "uncomfortable")
_POSITIVE = ("excellent", "reliable", "good", "great", "love", "perfect")
_NEUTRAL = ("average", "okay", "fine", "as expected")


def _has(text: str, phrase: str) -> bool:
    return re.search(r"\b" + re.escape(phrase) + r"\b", text) is not None


def classify_sentiment(text: str) -> str:
    t = str(text).lower()
    if any(_has(t, p) for p in _NEGATIVE):
        return "negative"
    if any(_has(t, p) for p in _POSITIVE):
        return "positive"
    if any(_has(t, p) for p in _NEUTRAL):
        return "neutral"
    return "unclassified"


def is_no_evidence(text: str) -> bool:
    t = str(text).lower()
    return any(p in t for p in NO_EVIDENCE_PHRASES)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False})


def load_tables(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load the three raw tables plus the notebook's exported records (if present)."""
    reviews = _read_csv(data_dir / "reviews.csv")
    reviews["source_allowed"] = _to_bool(reviews["source_allowed"])
    if reviews["source_allowed"].isna().any():
        raise ValueError("reviews.source_allowed contains values other than True/False")
    reviews["review_date"] = pd.to_datetime(reviews["review_date"], format="%Y-%m-%d").dt.date.astype(str)

    questions = _read_csv(data_dir / "questions.csv")
    questions["answerable"] = _to_bool(questions["answerable"])

    labels = _read_csv(data_dir / "relevance_labels.csv")
    labels["relevant"] = _to_bool(labels["relevant"])

    tables = {"reviews": reviews, "questions": questions, "relevance_labels": labels}
    for optional in ("app_review_provenance", "app_retrieval_metrics", "app_retrieval_per_question",
                     "app_extractive_answers", "app_reviews_permitted"):
        path = data_dir / f"{optional}.csv"
        if path.exists():
            tables[optional] = _read_csv(path)
    return tables


def build_corpus(reviews: pd.DataFrame, cutoff: str, revoked: frozenset[str] = frozenset(),
                 deleted: frozenset[str] = frozenset()) -> pd.DataFrame:
    """Every review with its contract status, evidence flag and sentiment. Never drops rows: the
    status column says whether a row may be used, so the app can explain every exclusion."""
    df = reviews.copy()
    df["sentiment"] = df["review_text"].map(classify_sentiment)
    df["no_evidence"] = df["review_text"].map(is_no_evidence)

    def status(row) -> str:
        if row["review_id"] in deleted:
            return "deleted"
        if row["review_id"] in revoked:
            return "revoked"
        if not row["source_allowed"]:
            return "restricted"
        if str(row["review_date"]) < cutoff:
            return "outdated"
        if row["no_evidence"]:
            return "no_evidence"
        return "evidence"

    df["status"] = df.apply(status, axis=1)
    df["permitted"] = df["status"].isin(["no_evidence", "evidence"])
    df["is_evidence"] = df["status"] == "evidence"
    return df


def permitted_records(corpus: pd.DataFrame) -> list[dict]:
    """The notebook's `reviews_permitted` corpus (contract only, no-evidence rows still included)."""
    cols = ["review_id", "product_id", "topic", "review_text", "review_date", "source_allowed"]
    return corpus.loc[corpus["permitted"], cols].to_dict("records")


def build_segments(corpus: pd.DataFrame) -> pd.DataFrame:
    """One row per product x issue: funnel counts, sentiment mix, evidence IDs."""
    rows = []
    for (product, topic), g in corpus.groupby(["product_id", "topic"], sort=True):
        ev = g[g["is_evidence"]]
        counts = g["status"].value_counts()
        n = len(ev)
        pos, neu, neg = (int((ev["sentiment"] == s).sum()) for s in ("positive", "neutral", "negative"))
        rows.append({
            "product_id": product,
            "topic": topic,
            "total": len(g),
            "deleted": int(counts.get("deleted", 0)),
            "revoked": int(counts.get("revoked", 0)),
            "restricted": int(counts.get("restricted", 0)),
            "outdated": int(counts.get("outdated", 0)),
            "no_evidence": int(counts.get("no_evidence", 0)),
            "evidence": n,
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "negative_share": round(neg / n, 4) if n else None,
            "first_review_date": ev["review_date"].min() if n else None,
            "last_review_date": ev["review_date"].max() if n else None,
            "evidence_ids": ", ".join(sorted(ev["review_id"])),
        })
    return pd.DataFrame(rows)


def overall_label(pos: int, neu: int, neg: int) -> str:
    n = pos + neu + neg
    if n == 0:
        return "No evidence"
    if pos and neg:
        if pos > neg + neu:
            return "Mostly positive, with some complaints"
        if neg > pos + neu:
            return "Mostly negative, with some praise"
        return "Mixed"
    if pos:
        return "Positive"
    if neg:
        return "Negative"
    return "Neutral"
