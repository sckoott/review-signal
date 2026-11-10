"""Engine tests. Run from the app folder:  python -m pytest -q"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import llm  # noqa: E402
from engine.answer import answer_question  # noqa: E402
from engine.config import Settings  # noqa: E402
from engine.data import build_corpus, build_segments, load_tables  # noqa: E402
from engine.governance import run_test_matrix  # noqa: E402
from engine.recommend import draft_recommendation, priority_table  # noqa: E402
from engine.retrieval import scorecard, segment_evidence  # noqa: E402
from engine.router import route_question  # noqa: E402
from engine.sql_qa import ask_sql, genie_ask, run_local_sql, sql_views, validate_sql  # noqa: E402

CUTOFF = "2026-01-01"


def settings(**kw) -> Settings:
    base = dict(llm_provider="none", anthropic_api_key=None, anthropic_model="claude-sonnet-5-5",
                openrouter_api_key=None, openrouter_model="anthropic/claude-sonnet-5.5", price_input_per_mtok=2.0,
                price_output_per_mtok=10.0, genie_space_id=None, genie_use_user_token=True,
                freshness_cutoff=CUTOFF, exec_emails=())
    base.update(kw)
    return Settings(**base)


@pytest.fixture(scope="module")
def tables():
    return load_tables()


@pytest.fixture(scope="module")
def corpus(tables):
    return build_corpus(tables["reviews"], CUTOFF)


P = ["jacket_a", "jacket_b", "shoe_a", "shoe_b"]
T = ["comfort", "durability", "fit"]


# ---- data contract (notebook W1) ----------------------------------------------------------------
def test_contract_counts(corpus):
    assert len(corpus) == 120
    assert int(corpus["permitted"].sum()) == 96
    counts = corpus["status"].value_counts().to_dict()
    assert counts == {"evidence": 60, "no_evidence": 36, "restricted": 12, "outdated": 12}


def test_contract_matches_notebook_export(tables, corpus):
    assert set(corpus.loc[corpus["permitted"], "review_id"]) == set(tables["app_review_provenance"]["review_id"])
    assert set(corpus.loc[corpus["permitted"], "review_id"]) == set(tables["app_reviews_permitted"]["review_id"])


def test_segments_shape(corpus):
    seg = build_segments(corpus)
    assert len(seg) == 12
    assert (seg["evidence"] == 5).all() and (seg["negative"] == 2).all() and (seg["positive"] == 2).all()
    # evidence reviews are exactly the gold relevant sets
    assert int(seg["evidence"].sum()) == 60


def test_evidence_equals_gold_labels(tables, corpus):
    ev = set(corpus.loc[corpus["is_evidence"], "review_id"])
    gold = set(tables["relevance_labels"].loc[tables["relevance_labels"]["relevant"], "review_id"])
    assert gold <= ev  # every labelled review is used as evidence (80 labels over 16 questions)


# ---- router ------------------------------------------------------------------------------------
@pytest.mark.parametrize("q,route,products,topics", [
    ("How durable is shoe_a according to its reviews?", "evidence", ["shoe_a"], ["durability"]),
    ("What do users say about the sizing and fit of jacket_b?", "evidence", ["jacket_b"], ["fit"]),
    ("Summarize the comfort feedback for shoe_b.", "evidence", ["shoe_b"], ["comfort"]),
    ("Shoe A comfort", "evidence", ["shoe_a"], ["comfort"]),
    ("a jacket a bit tight on fit", "evidence", ["jacket_a", "jacket_b"], ["fit"]),
    ("tell me about shoes fit", "evidence", ["shoe_a", "shoe_b"], ["fit"]),
    ("What is the confidential competitor product roadmap?", "abstain", None, None),
    ("What is the current wholesale price of shoe_a?", "abstain", None, None),
    ("How comfortable is the unreleased shoe_z?", "abstain", None, None),
    ("Does shoe_a have a lifetime durability guarantee?", "abstain", None, None),
    ("what do customers think?", "abstain", None, None),
    ("Which product has the highest share of negative fit reviews?", "sql", None, None),
    ("How many reviews per product?", "sql", None, None),
])
def test_router(q, route, products, topics):
    d = route_question(q, P, T)
    assert d.route == route, d.reason
    if products is not None:
        assert d.products == products and d.topics == topics


def test_unknown_product_never_substituted():
    d = route_question("How comfortable is the unreleased shoe_z?", P, T)
    assert d.abstain_kind == "unknown_product"


# ---- evaluation (notebook W2 / W4) ------------------------------------------------------------
def test_scorecard_matches_notebook(tables, corpus):
    sc = scorecard(tables, corpus)
    nb = tables["app_retrieval_metrics"]
    for r in nb.itertuples():
        mine = sc[(sc["split"] == r.split) & (sc["method"] == r.method)].iloc[0]
        assert mine["precision"] == pytest.approx(float(r.mean_precision_at_k_answerable))
        assert mine["recall"] == pytest.approx(float(r.mean_recall_at_k_answerable))
        assert mine["abstention_accuracy"] == pytest.approx(float(r.abstention_accuracy))
        assert mine["unsupported_answer_rate"] == pytest.approx(float(r.unsupported_answer_rate))


def test_segment_method_perfect_on_both_splits(tables, corpus):
    sc = scorecard(tables, corpus)
    seg = sc[sc["method"] == "segment"]
    assert (seg["precision"] == 1.0).all() and (seg["recall"] == 1.0).all()
    assert (seg["abstention_accuracy"] == 1.0).all() and (seg["unsupported_answer_rate"] == 0.0).all()


# ---- governance (notebook W3) -----------------------------------------------------------------
def test_governance_matrix(tables):
    tm = run_test_matrix(tables, CUTOFF).set_index("test")
    results = dict(zip(tm.index + " | " + tm["system"], tm["result"]))
    assert all(v == "Pass" for k, v in results.items() if k.startswith(("T1", "T2", "T3", "T4a", "T4d", "T4e", "T5")))
    assert [v for k, v in results.items() if k.startswith(("T4b", "T4c"))] == ["Fail", "Fail"]  # old approach
    assert [v for k, v in results.items() if k.startswith("T6")] == ["Unknown"]


def test_revoked_review_never_used(tables):
    c = build_corpus(tables["reviews"], CUTOFF, revoked=frozenset({"R005"}))
    a = answer_question("How comfortable is shoe_a according to users?", c, settings())
    assert "R005" not in a.cited_ids and len(a.cited_ids) == 4


def test_deleted_segment_abstains(tables, corpus):
    gone = frozenset(corpus[(corpus.product_id == "shoe_a") & (corpus.topic == "comfort")]["review_id"])
    c = build_corpus(tables["reviews"], CUTOFF, deleted=gone)
    a = answer_question("How comfortable is shoe_a according to users?", c, settings())
    assert a.route == "abstain" and a.decision.abstain_kind == "no_evidence" and not a.cited_ids


def test_restricted_and_outdated_never_cited(tables, corpus):
    bad = set(corpus.loc[~corpus["permitted"], "review_id"])
    for q in tables["questions"]["question_text"]:
        a = answer_question(q, corpus, settings())
        assert not set(a.cited_ids) & bad
        for p in a.points:
            assert not set(p["ids"]) & bad


# ---- answers -------------------------------------------------------------------------------------
def test_extractive_answer(corpus):
    a = answer_question("How durable is shoe_a according to its reviews?", corpus, settings())
    assert a.route == "evidence" and a.mode == "extractive"
    assert sorted(a.cited_ids) == ["R023", "R024", "R025", "R026", "R027"]
    assert a.sentiment == {"positive": 2, "neutral": 1, "negative": 2}
    assert all(set(p["ids"]) <= set(a.cited_ids) for p in a.points)
    assert len(a.excluded) == 5


def _fake_llm(monkeypatch, payload):
    def fake_complete(s, system, user, max_tokens=900):
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return llm.LLMResult(text=text, provider="anthropic", model="fake", input_tokens=100, output_tokens=50,
                             cost_usd=0.0007, latency_ms=5)
    monkeypatch.setattr(llm, "complete", fake_complete)


def test_llm_answer_citation_check(monkeypatch, corpus):
    _fake_llm(monkeypatch, {"verdict": "Mixed durability.", "gaps": ["No return data."],
                            "points": [{"text": "Some say it lasts.", "ids": ["R025", "R999"]},
                                       {"text": "Invented claim.", "ids": ["R001"]},  # restricted -> dropped
                                       {"text": "No ids at all.", "ids": []}]})
    a = answer_question("How durable is shoe_a according to its reviews?", corpus,
                        settings(llm_provider="anthropic", anthropic_api_key="x"))
    assert a.mode == "llm"
    assert a.points == [{"text": "Some say it lasts.", "ids": ["R025"]}]
    assert "2 point(s) dropped" in a.llm_note
    assert a.cost_usd == pytest.approx(0.0007)


def test_llm_garbage_falls_back_to_extractive(monkeypatch, corpus):
    _fake_llm(monkeypatch, "Sorry, I can't produce JSON.")
    a = answer_question("How durable is shoe_a according to its reviews?", corpus,
                        settings(llm_provider="anthropic", anthropic_api_key="x"))
    assert a.mode == "extractive" and a.points and "LLM unavailable" in a.llm_note


def test_llm_error_falls_back(monkeypatch, corpus):
    def boom(*args, **kwargs):
        raise llm.LLMError("network down")
    monkeypatch.setattr(llm, "complete", boom)
    a = answer_question("How durable is shoe_a according to its reviews?", corpus,
                        settings(llm_provider="anthropic", anthropic_api_key="x"))
    assert a.mode == "extractive" and "network down" in a.llm_note


def test_parse_json_fenced():
    assert llm.parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_openrouter_request_shape(monkeypatch):
    captured = {}

    class Resp:
        status_code = 200
        text = "{}"

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    def fake_post(url, headers, json, timeout):
        captured.update(url=url, headers=headers, json=json)
        return Resp()
    monkeypatch.setattr(llm.requests, "post", fake_post)
    r = llm.complete(settings(llm_provider="openrouter", openrouter_api_key="k"), "sys", "user", 50)
    assert r.text == "hi" and r.input_tokens == 10 and r.output_tokens == 5
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer k"
    assert captured["json"]["model"] == "anthropic/claude-sonnet-5.5"
    assert r.cost_usd == pytest.approx(10 / 1e6 * 2 + 5 / 1e6 * 10)


def test_anthropic_request_shape(monkeypatch):
    import anthropic
    captured = {}

    class FakeMessages:
        def create(self, **kw):
            captured.update(kw)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")],
                                   usage=SimpleNamespace(input_tokens=7, output_tokens=3))

    class FakeClient:
        def __init__(self, **kw):
            captured["client"] = kw
            self.messages = FakeMessages()
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    r = llm.complete(settings(llm_provider="anthropic", anthropic_api_key="sk"), "sys", "hello", 77)
    assert r.text == "ok" and r.input_tokens == 7
    assert captured["model"] == "claude-sonnet-5-5" and captured["max_tokens"] == 77 and captured["system"] == "sys"
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["client"]["api_key"] == "sk"


# ---- SQL ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["DROP TABLE x", "select 1; select 2", "select * from read_csv('/etc/passwd')",
                                 "select * from '/etc/passwd'", "ATTACH 'x.db'", "COPY t TO 'x.csv'",
                                 "update review_segments set evidence = 0"])
def test_sql_validation_rejects(bad):
    with pytest.raises(ValueError):
        validate_sql(bad)


def test_sql_views_hide_restricted(corpus):
    views = sql_views(corpus)
    df = run_local_sql("SELECT review_id FROM reviews_enriched", views)
    assert len(df) == 96
    assert not set(df["review_id"]) & set(corpus.loc[~corpus["permitted"], "review_id"])


def test_preset_queries(corpus):
    r = ask_sql("Which product has the highest share of negative fit reviews?", corpus, settings())
    assert r.ok and r.engine == "Predefined query" and len(r.df) == 4 and "tie at 40%" in r.answer_text
    r = ask_sql("How many reviews per product?", corpus, settings())
    assert r.ok and list(r.df["permitted_reviews"]) == [24, 24, 24, 24]
    assert list(r.df["evidence_reviews"]) == [15, 15, 15, 15]
    r = ask_sql("How many reviews were excluded and why?", corpus, settings())
    assert r.ok and "no_evidence" in r.df.columns
    r = ask_sql("Sentiment breakdown by product", corpus, settings())
    assert r.ok and set(r.df["sentiment"]) == {"positive", "neutral", "negative"}


def test_llm_sql_path(monkeypatch, corpus):
    calls = []

    def fake_complete(s, system, user, max_tokens=900):
        calls.append(system[:20])
        if len(calls) == 1:
            text = json.dumps({"sql": "SELECT product_id, SUM(negative) AS neg FROM review_segments "
                                      "WHERE topic = 'durability' GROUP BY product_id ORDER BY neg DESC, product_id",
                               "explanation": "sum negatives"})
        else:
            text = "All four products have 2 negative durability reviews."
        return llm.LLMResult(text, "anthropic", "fake", 10, 10, 0.0001, 1)
    monkeypatch.setattr(llm, "complete", fake_complete)
    r = ask_sql("Rank products by negative durability reviews", corpus,
                settings(llm_provider="anthropic", anthropic_api_key="x"))
    assert r.ok and r.engine == "Local SQL (LLM-written)" and len(r.df) == 4 and r.answer_text.startswith("All four")


def test_llm_sql_injection_falls_back(monkeypatch, corpus):
    def fake_complete(s, system, user, max_tokens=900):
        return llm.LLMResult(json.dumps({"sql": "SELECT * FROM read_csv('/etc/passwd')"}), "anthropic", "f",
                             1, 1, 0.0, 1)
    monkeypatch.setattr(llm, "complete", fake_complete)
    r = ask_sql("How many reviews per product?", corpus, settings(llm_provider="anthropic", anthropic_api_key="x"))
    assert r.ok and r.engine == "Predefined query" and "LLM SQL failed" in r.message


def test_genie_parsing():
    from databricks.sdk.service import dashboards as d
    from databricks.sdk.service import sql as s

    msg = d.GenieMessage(
        space_id="sp", conversation_id="c1", content="q", message_id="m1", status=d.MessageStatus.COMPLETED,
        attachments=[d.GenieAttachment(attachment_id="a0", text=d.TextAttachment(content="shoe_a leads.")),
                     d.GenieAttachment(attachment_id="a1", query=d.GenieQueryAttachment(
                         query="SELECT 1", description="desc"))])
    stmt = s.StatementResponse(
        manifest=s.ResultManifest(schema=s.ResultSchema(columns=[s.ColumnInfo(name="product_id"),
                                                                 s.ColumnInfo(name="n")])),
        result=s.ResultData(data_array=[["shoe_a", "3"], ["shoe_b", "2"]]))

    class FakeGenie:
        def start_conversation_and_wait(self, space, content, timeout):
            assert space == "sp"
            return msg

        def get_message_attachment_query_result(self, space, conv, mid, att):
            assert (conv, mid, att) == ("c1", "m1", "a1")
            return d.GenieGetMessageQueryResultResponse(statement_response=stmt)

    client = SimpleNamespace(genie=FakeGenie())
    r = genie_ask(settings(genie_space_id="sp"), "q", client=client)
    assert r.ok and r.engine == "Genie" and r.sql == "SELECT 1" and r.answer_text == "shoe_a leads."
    assert list(r.df["n"]) == [3, 2] and r.conversation_id == "c1"


def test_genie_failure_falls_back_to_local(monkeypatch, corpus):
    import engine.sql_qa as sq

    monkeypatch.setattr(sq, "_genie_client", lambda s, t: (_ for _ in ()).throw(RuntimeError("no workspace")))
    r = ask_sql("How many reviews per product?", corpus, settings(genie_space_id="sp"))
    assert r.ok and r.engine == "Predefined query" and "Genie error" in r.message


# ---- recommendations -------------------------------------------------------------------------
def test_priority_and_template(corpus):
    seg = build_segments(corpus)
    pt = priority_table(seg, {"shoe_b": 2.0})
    assert list(pt.head(3)["product_id"]) == ["shoe_b"] * 3
    card = draft_recommendation(settings(), "shoe_b", "fit", segment_evidence(corpus, ["shoe_b"], ["fit"]))
    assert card["source"] == "template" and card["evidence"][0]["sentiment"] == "negative"
    assert "Hypothesis (unverified)" in card["hypothesis"] and card["confidence"].startswith("Low")


def test_recommendation_llm_citation_check(monkeypatch, corpus):
    _fake_llm(monkeypatch, {"issue": "Fit issue", "hypothesis": "Hypothesis (unverified): x", "action": "Fix",
                            "owner": "PD", "change_mind": "more data", "metric": "neg share",
                            "evidence_ids": ["R001", "R999"]})
    card = draft_recommendation(settings(llm_provider="anthropic", anthropic_api_key="x"), "shoe_a", "fit",
                                segment_evidence(corpus, ["shoe_a"], ["fit"]))
    assert card["source"] == "template" and "no valid review IDs" in card["llm_note"]
