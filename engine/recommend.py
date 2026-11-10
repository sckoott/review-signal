"""Recommendations layer: prioritise segments, draft a recommendation card, log human decisions.

Kept separate from Q&A on purpose: answers are factual and cited; recommendations are judgement,
labelled as such, and only reach the Exec Brief after a person accepts them.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from engine import llm
from engine.config import Settings

TOPIC_PLAYBOOK = {
    "fit": ("Product Development – fit & sizing",
            "Check the size chart and grading against the negative fit reviews; add fit guidance on the product page.",
            "Share of negative fit reviews and size-related returns for this product"),
    "comfort": ("Product Development – materials & comfort",
                "Run a wear test focused on long-duration use; review cushioning / lining against the complaints.",
                "Share of negative comfort reviews after the next production run"),
    "durability": ("Quality Assurance",
                   "Inspect returned units and run abrasion / wear tests; compare materials spec with the complaints.",
                   "Share of negative durability reviews and durability-related returns"),
}


def priority_table(segments: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """priority = evidence volume x negative share x business weight.
    Trend is not included: the data covers a single period (Sept 2026), so no trend can be computed."""
    df = segments[segments["evidence"] > 0].copy()
    df["business_weight"] = df["product_id"].map(lambda p: float(weights.get(p, 1.0)))
    df["priority"] = (df["evidence"] * df["negative_share"].fillna(0) * df["business_weight"]).round(3)
    df = df.sort_values(["priority", "negative", "product_id", "topic"], ascending=[False, False, True, True])
    df["rank"] = range(1, len(df) + 1)
    return df[["rank", "product_id", "topic", "evidence", "negative", "negative_share", "business_weight",
               "priority"]].reset_index(drop=True)


def confidence(n: int, pos: int, neg: int) -> str:
    if n < 10:
        return f"Low — only {n} evidence reviews"
    if neg and pos and abs(neg - pos) / n < 0.2:
        return f"Medium — {n} reviews, mixed signal"
    return f"Medium — {n} reviews" if n < 50 else f"High — {n} reviews, consistent signal"


def draft_template(product: str, topic: str, evidence: pd.DataFrame) -> dict:
    owner, action, metric = TOPIC_PLAYBOOK.get(topic, ("Product team", "Investigate the complaints.",
                                                       "Share of negative reviews"))
    n = len(evidence)
    neg = evidence[evidence["sentiment"] == "negative"]
    pos = evidence[evidence["sentiment"] == "positive"]
    quotes = pd.concat([neg, evidence[evidence["sentiment"] == "neutral"], pos]).head(3)
    if len(neg):
        hyp = (f"Hypothesis (unverified): {topic} falls short of expectations for part of the customers, "
               f"e.g. “{neg.iloc[0]['review_text']}” [{neg.iloc[0]['review_id']}]. Root cause not yet known.")
    else:
        hyp = f"Hypothesis (unverified): no negative {topic} signal in the current evidence."
    return {
        "product_id": product, "topic": topic,
        "issue": f"{topic.capitalize()} of {product}: {len(neg)} of {n} evidence reviews are negative",
        "evidence": [{"review_id": r.review_id, "text": r.review_text, "sentiment": r.sentiment}
                     for r in quotes.itertuples()],
        "hypothesis": hyp,
        "action": action,
        "owner": owner,
        "confidence": confidence(n, len(pos), len(neg)),
        "change_mind": (f"Positive reviews ({len(pos)} of {n}) suggest the issue is not universal: a larger sample "
                        f"or return data showing no {topic} pattern would downgrade this."),
        "metric": metric,
        "source": "template",
    }


REC_SYSTEM = """You draft ONE product recommendation from customer review evidence for a product team.
Use ONLY the reviews provided. Label the cause as a hypothesis. Cite review IDs that exist in the list.
Reply with JSON only:
{"issue": "...", "hypothesis": "Hypothesis (unverified): ... [R001]", "action": "...",
 "owner": "team name", "change_mind": "what evidence would change this", "metric": "metric to watch",
 "evidence_ids": ["R001", "R002"]}"""


def draft_recommendation(settings: Settings, product: str, topic: str, evidence: pd.DataFrame,
                         use_llm: bool = True) -> dict:
    card = draft_template(product, topic, evidence)
    if not (use_llm and llm.is_available(settings)) or evidence.empty:
        return card
    lines = "\n".join(f"[{r.review_id}] ({r.sentiment}) {r.review_text}" for r in evidence.itertuples())
    try:
        res = llm.complete(settings, REC_SYSTEM, f"Segment: {product} · {topic}\nReviews:\n{lines}", max_tokens=700)
        obj = llm.parse_json(res.text)
        allowed = set(evidence["review_id"])
        ids = [i for i in (obj.get("evidence_ids") or []) if isinstance(i, str) and i in allowed]
        if not ids:
            card["llm_note"] = "Model draft cited no valid review IDs; template draft shown."
            return card
        by_id = evidence.set_index("review_id")
        card.update({k: str(obj[k]).strip() for k in ("issue", "hypothesis", "action", "owner", "change_mind",
                                                        "metric") if isinstance(obj.get(k), str) and obj[k].strip()})
        card["evidence"] = [{"review_id": i, "text": by_id.loc[i, "review_text"], "sentiment": by_id.loc[i, "sentiment"]}
                            for i in ids[:4]]
        card["source"] = f"LLM ({res.model})"
        card["cost_usd"] = res.cost_usd
    except llm.LLMError as exc:
        card["llm_note"] = f"LLM unavailable ({exc}); template draft shown."
    return card


def log_entry(card: dict, decision: str, reason: str, user: str) -> dict:
    entry = dict(card)
    entry.update({"decision": decision, "reason": reason, "decided_by": user,
                  "decided_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M")})
    return entry
