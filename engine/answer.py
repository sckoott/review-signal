"""Answer pipeline: route -> resolve -> segment context -> answer (LLM or extractive) -> citation check."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import pandas as pd

from engine import llm
from engine.config import Settings
from engine.data import STATUS_LABEL, overall_label
from engine.retrieval import segment_evidence
from engine.router import RouteDecision, route_question

SYSTEM_PROMPT = """You are Review Signal, an assistant that summarises customer reviews for product teams.
Rules:
- Use ONLY the reviews provided. Never add facts, numbers, or opinions that are not in them.
- Every point must cite the review IDs it relies on, using only IDs from the list.
- Report the sentiment mix honestly; do not smooth over complaints.
- Be concise and specific. No marketing language.
Reply with a single JSON object and nothing else:
{"verdict": "one sentence answering the question",
 "points": [{"text": "one sentence", "ids": ["R001"]}],
 "gaps": ["what these reviews do not tell us (max 2 items)"]}"""


@dataclass
class Answer:
    question: str
    decision: RouteDecision
    route: str
    verdict: str = ""
    points: list[dict] = field(default_factory=list)  # {"text": str, "ids": [str]}
    gaps: list[str] = field(default_factory=list)
    evidence: pd.DataFrame = field(default_factory=pd.DataFrame)
    excluded: pd.DataFrame = field(default_factory=pd.DataFrame)
    funnel: pd.DataFrame = field(default_factory=pd.DataFrame)
    sentiment: dict = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)
    mode: str = "extractive"  # extractive | llm | abstain | sql
    llm_note: str | None = None
    llm_result: object | None = None
    sql_result: object | None = None
    latency_ms: int = 0
    cost_usd: float = 0.0
    answer_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])

    @property
    def cited_ids(self) -> list[str]:
        return list(self.evidence["review_id"]) if not self.evidence.empty else []


class _Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.last = self.t0
        self.trace: list[dict] = []

    def step(self, name: str, detail: str):
        now = time.perf_counter()
        self.trace.append({"step": name, "detail": detail, "ms": round((now - self.last) * 1000, 1)})
        self.last = now

    @property
    def total_ms(self) -> int:
        return int((time.perf_counter() - self.t0) * 1000)


def _fmt_list(items: list[str]) -> str:
    return ", ".join(items) if items else "none"


def extractive_points(evidence: pd.DataFrame) -> list[dict]:
    """Grounded by construction: one point per segment and sentiment, quoting the reviews."""
    points = []
    for (product, topic), g in evidence.groupby(["product_id", "topic"], sort=True):
        n = len(g)
        for sentiment in ("negative", "neutral", "positive", "unclassified"):
            s = g[g["sentiment"] == sentiment]
            if s.empty:
                continue
            quotes = "; ".join(f"“{t}”" for t in s["review_text"].head(2))
            points.append({"text": f"{product} · {topic}: {len(s)} of {n} reviews are {sentiment}, e.g. {quotes}",
                           "ids": list(s["review_id"])})
    return points


def deterministic_verdict(evidence: pd.DataFrame) -> str:
    parts = []
    for (product, topic), g in evidence.groupby(["product_id", "topic"], sort=True):
        pos, neu, neg = (int((g["sentiment"] == s).sum()) for s in ("positive", "neutral", "negative"))
        parts.append(f"{product} {topic}: {overall_label(pos, neu, neg).lower()} "
                     f"({pos} positive, {neu} neutral, {neg} negative of {len(g)})")
    return ("; ".join(parts) + ".") if parts else ""


def deterministic_gaps(evidence: pd.DataFrame, funnel: pd.DataFrame) -> list[str]:
    gaps = []
    n_min = int(funnel["evidence"].min()) if not funnel.empty else 0
    if n_min < 10:
        gaps.append(f"Small sample: as few as {n_min} evidence reviews per segment. Treat as directional.")
    excluded = int(funnel[["restricted", "outdated", "no_evidence", "revoked", "deleted"]].sum().sum()) \
        if not funnel.empty else 0
    if excluded:
        gaps.append(f"{excluded} in-scope reviews were excluded (restricted, outdated, revoked or no product "
                    "evidence). See the context inspector.")
    if not evidence.empty and evidence["review_date"].nunique() and \
            evidence["review_date"].min()[:7] == evidence["review_date"].max()[:7]:
        gaps.append(f"All evidence comes from one month ({evidence['review_date'].min()[:7]}); no trend can be "
                    "computed yet.")
    return gaps


def _llm_answer(settings: Settings, question: str, evidence: pd.DataFrame, funnel: pd.DataFrame):
    lines = [f"[{r.review_id}] ({r.product_id} · {r.topic} · {r.review_date} · {r.sentiment}) {r.review_text}"
             for r in evidence.itertuples()]
    stats = [f"- {r.product_id} · {r.topic}: {r.evidence} evidence reviews "
             f"({r.positive} positive, {r.neutral} neutral, {r.negative} negative)"
             for r in funnel.itertuples()]
    user = (f"Question: {question}\n\nSegment statistics:\n" + "\n".join(stats) +
            "\n\nReviews (the only allowed evidence):\n" + "\n".join(lines))
    result = llm.complete(settings, SYSTEM_PROMPT, user, max_tokens=900)
    obj = llm.parse_json(result.text)
    allowed = set(evidence["review_id"])
    points, dropped = [], 0
    for p in obj.get("points") or []:
        if not isinstance(p, dict) or not isinstance(p.get("text"), str) or not p["text"].strip():
            dropped += 1
            continue
        ids = [i for i in (p.get("ids") or []) if isinstance(i, str) and i in allowed]
        if not ids:
            dropped += 1
            continue
        points.append({"text": p["text"].strip(), "ids": ids})
    verdict = obj.get("verdict") if isinstance(obj.get("verdict"), str) else ""
    gaps = [g.strip() for g in (obj.get("gaps") or []) if isinstance(g, str) and g.strip()][:2]
    return result, verdict.strip(), points, gaps, dropped


def answer_question(question: str, corpus: pd.DataFrame, settings: Settings, use_llm: bool = True,
                    sql_runner=None) -> Answer:
    timer = _Timer()
    products = sorted(corpus["product_id"].unique())
    topics = sorted(corpus["topic"].unique())
    decision = route_question(question, products, topics)
    matched = "; ".join(f"{k}: {', '.join(v)}" for k, v in decision.matched.items()) or "no product/issue terms"
    timer.step("Route", f"{decision.route.upper()} — {decision.reason} (matched {matched})")
    ans = Answer(question=question, decision=decision, route=decision.route)

    if decision.route == "abstain":
        ans.mode = "abstain"
        ans.verdict = "I can't answer this from customer reviews."
        ans.trace, ans.latency_ms = timer.trace, timer.total_ms
        return ans

    if decision.route == "sql":
        if sql_runner is None:
            ans.mode = "abstain"
            ans.verdict = "Counting questions are answered on the Data Q&A page."
        else:
            res = sql_runner(question)
            timer.step("SQL", f"{res.engine}: {'ok' if res.ok else 'failed'} — {res.message}")
            ans.mode, ans.sql_result = "sql", res
            ans.verdict = res.answer_text or ("Query returned results." if res.ok else res.message)
            ans.cost_usd = res.cost_usd
        ans.trace, ans.latency_ms = timer.trace, timer.total_ms
        return ans

    # Evidence route
    in_scope = corpus[corpus["product_id"].isin(decision.products) & corpus["topic"].isin(decision.topics)]
    from engine.data import build_segments
    funnel = build_segments(in_scope)
    timer.step("Resolve", f"products: {_fmt_list(decision.products)} · issues: {_fmt_list(decision.topics)} "
               f"→ {len(funnel)} segment(s), {len(in_scope)} reviews in scope")
    evidence = segment_evidence(corpus, decision.products, decision.topics)
    excluded = in_scope[~in_scope["is_evidence"]].copy()
    excluded["reason"] = excluded["status"].map(STATUS_LABEL)
    reasons = ", ".join(f"{n} {STATUS_LABEL[s].split(' (')[0].lower()}"
                        for s, n in excluded["status"].value_counts().items()) if len(excluded) else "none"
    timer.step("Context", f"{len(evidence)} evidence reviews used; {len(excluded)} excluded ({reasons})")
    ans.funnel, ans.evidence, ans.excluded = funnel, evidence, excluded
    ans.sentiment = {s: int((evidence["sentiment"] == s).sum()) for s in ("positive", "neutral", "negative")}

    if evidence.empty:
        ans.route = ans.mode = "abstain"
        decision.abstain_kind = "no_evidence"
        decision.reason = ("No permitted, current review with product evidence remains for this scope "
                           "(it may have been revoked, deleted or aged out). The assistant does not substitute "
                           "other reviews.")
        ans.verdict = "Not enough evidence to answer."
        timer.step("Answer", "Abstained: no evidence left in scope")
        ans.trace, ans.latency_ms = timer.trace, timer.total_ms
        return ans

    ans.gaps = deterministic_gaps(evidence, funnel)
    if use_llm and llm.is_available(settings):
        try:
            result, verdict, points, gaps, dropped = _llm_answer(settings, question, evidence, funnel)
            ans.llm_result, ans.cost_usd = result, result.cost_usd
            if points and verdict:
                ans.mode, ans.verdict, ans.points = "llm", verdict, points
                ans.gaps = gaps + ans.gaps
                note = f"{len(points)} points, all citing allowed review IDs"
                if dropped:
                    note += f"; {dropped} point(s) dropped by the citation check"
                ans.llm_note = note
                timer.step("Answer", f"LLM synthesis ({result.model}, {result.input_tokens}+{result.output_tokens} "
                           f"tokens, ${result.cost_usd:.4f}); citation check: {note}")
            else:
                ans.llm_note = "Model output failed the citation check; showing the extractive answer instead."
                timer.step("Answer", ans.llm_note)
        except llm.LLMError as exc:
            ans.llm_note = f"LLM unavailable ({exc}); showing the extractive answer."
            timer.step("Answer", ans.llm_note)
    if ans.mode != "llm":
        ans.mode = "extractive"
        ans.verdict = deterministic_verdict(evidence)
        ans.points = extractive_points(evidence)
        if not timer.trace or timer.trace[-1]["step"] != "Answer":
            timer.step("Answer", "Extractive answer: exact quotes with review IDs")
    ans.trace, ans.latency_ms = timer.trace, timer.total_ms
    return ans
