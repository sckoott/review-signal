"""Source authorization, freshness and removal tests (notebook W3), run live against the app engine."""
from __future__ import annotations

import pandas as pd

from engine.data import build_corpus, permitted_records
from engine.retrieval import retrieve, segment_retrieve


def revalidate(cached_ids, current_corpus_records):
    """Notebook version: availability check only."""
    valid = {r['review_id'] for r in current_corpus_records}
    kept = [i for i in cached_ids if i in valid]
    return {'kept': kept, 'dropped': [i for i in cached_ids if i not in valid],
            'action': 'reuse' if kept == list(cached_ids) else ('revise' if kept else 'abstain')}


def revalidate_scoped(cached_ids, corpus: pd.DataFrame, products, topics) -> dict:
    """Review Signal version: a cited review must still be permitted evidence AND still on-topic."""
    ok = corpus[corpus["is_evidence"] & corpus["product_id"].isin(products) & corpus["topic"].isin(topics)]
    valid = set(ok["review_id"])
    kept = [i for i in cached_ids if i in valid]
    dropped = [i for i in cached_ids if i not in valid]
    action = "reuse" if not dropped else ("revise" if kept else "abstain")
    return {"kept": kept, "dropped": dropped, "action": action}


def _ids(hits):
    return [r['review_id'] for r in hits]


def run_test_matrix(tables: dict, cutoff: str) -> pd.DataFrame:
    raw = tables["reviews"]
    base = build_corpus(raw, cutoff)
    perm = permitted_records(base)
    products = sorted(base["product_id"].unique())
    topics = sorted(base["topic"].unique())
    restricted = set(base.loc[~base["source_allowed"], "review_id"])
    outdated = set(base.loc[base["review_date"].astype(str) < cutoff, "review_id"])
    qs = tables["questions"]
    dev = qs[qs["split"] == "development"].to_dict("records")
    q01 = next(q for q in dev if q["question_id"] == "Q01")
    rows = []

    def add(test, system, expected, observed, passed):
        rows.append({"test": test, "system": system, "expected": expected, "observed": observed,
                     "result": "Unknown" if passed is None else ("Pass" if passed else "Fail")})

    # T1 contract parity
    app_ids = {r["review_id"] for r in perm}
    if "app_review_provenance" in tables:
        nb_ids = set(tables["app_review_provenance"]["review_id"])
        add("T1 Contract parity with notebook review_provenance", "Data layer", "identical ID sets",
            f"{len(app_ids)} vs {len(nb_ids)}", app_ids == nb_ids)
    else:
        add("T1 Contract parity with notebook review_provenance", "Data layer", "identical ID sets",
            "notebook export not found", None)

    # T2 no denied / stale ID retrievable
    leaked = set()
    for q in dev:
        for m in ("lexical", "review_rules"):
            leaked |= set(_ids(retrieve(q["question_text"], perm, m))) & (restricted | outdated)
        leaked |= set(segment_retrieve(q["question_text"], base, products, topics)) & (restricted | outdated)
    add("T2 Denied/stale IDs never retrieved (12 dev questions × 3 methods)", "All", "0 leaked IDs",
        f"{len(leaked)} leaked", not leaked)

    # T3 revoke the top cited source of Q01
    cited = _ids(retrieve(q01["question_text"], perm, "lexical"))
    revoked = frozenset(cited[:1])
    after = build_corpus(raw, cutoff, revoked=revoked)
    rv = revalidate(cited, permitted_records(after))
    fresh = _ids(retrieve(q01["question_text"], permitted_records(after), "lexical"))
    add(f"T3 Revoke cited source {sorted(revoked)} (Q01)", "Keyword top-3", "revoked ID not reused; answer revised",
        f"cache {cited} → {rv['action']} (kept {rv['kept']}); fresh {fresh}",
        rv["action"] == "revise" and not set(fresh) & revoked)
    seg_before = segment_retrieve(q01["question_text"], base, products, topics)
    seg_after = segment_retrieve(q01["question_text"], after, products, topics)
    rvs = revalidate_scoped(seg_before, after, ["shoe_a"], ["comfort"])
    add(f"T3 Revoke cited source {sorted(revoked)} (Q01)", "Review Signal", "revoked ID not reused; answer revised",
        f"cache {len(seg_before)} IDs → {rvs['action']} (dropped {rvs['dropped']}); fresh {len(seg_after)} IDs",
        rvs["action"] == "revise" and not set(seg_after) & revoked)

    # T4 all evidence for Q01's product/topic deleted
    gone = frozenset(r["review_id"] for r in perm if r["product_id"] == q01["product_id"] and r["topic"] == q01["topic"])
    after = build_corpus(raw, cutoff, deleted=gone)
    perm_after = permitted_records(after)
    rules = _ids(retrieve(q01["question_text"], perm_after, "review_rules"))
    lex = _ids(retrieve(q01["question_text"], perm_after, "lexical"))
    rv = revalidate(cited, perm_after)
    seg = segment_retrieve(q01["question_text"], after, products, topics)
    rvs = revalidate_scoped(seg_before, after, ["shoe_a"], ["comfort"])
    add(f"T4a Delete all {len(gone)} shoe_a/comfort reviews (Q01)", "Notebook review_rules", "abstain",
        f"returns {rules or '[] → abstains'}", not rules)
    add(f"T4b Delete all {len(gone)} shoe_a/comfort reviews (Q01)", "Keyword top-3", "abstain",
        f"substitutes {lex}" if lex else "abstains", not lex)
    add("T4c Revalidate cached keyword answer after deletion", "Keyword top-3", "abstain",
        f"{rv['action']} (kept {rv['kept']})", rv["action"] == "abstain")
    add(f"T4d Delete all {len(gone)} shoe_a/comfort reviews (Q01)", "Review Signal", "abstain",
        f"returns {seg or '[] → abstains'}", not seg)
    add("T4e Revalidate cached Review Signal answer after deletion", "Review Signal", "abstain",
        f"{rvs['action']} (kept {rvs['kept']})", rvs["action"] == "abstain")

    # T5 freshness cutoff moved
    new_cutoff = "2026-09-06"
    after = build_corpus(raw, new_cutoff)
    newly_stale = set(base.loc[base["permitted"], "review_id"]) - set(after.loc[after["permitted"], "review_id"])
    returned = set()
    for q in dev:
        for m in ("lexical", "review_rules"):
            returned |= set(_ids(retrieve(q["question_text"], permitted_records(after), m))) & newly_stale
        returned |= set(segment_retrieve(q["question_text"], after, products, topics)) & newly_stale
    rvs = revalidate_scoped(seg_before, after, ["shoe_a"], ["comfort"])
    add(f"T5 Freshness cutoff moved to {new_cutoff}", "All", "newly stale IDs never returned",
        f"{len(newly_stale)} age out, {len(returned)} returned; Q01 Review Signal cache → {rvs['action']} "
        f"(dropped {rvs['dropped']})", not returned)

    # T6 cannot run here
    add("T6 Denied read under a second, non-authorized identity", "Platform (Unity Catalog)",
        "read refused", "Not executable in this app: needs a row filter + a second principal", None)
    return pd.DataFrame(rows)
