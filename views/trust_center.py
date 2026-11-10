import pandas as pd
import streamlit as st

from engine.answer import answer_question
from engine.data import permitted_records
from engine.governance import revalidate_scoped
from engine.retrieval import retrieve, segment_retrieve
from ui import state
from ui.components import esc, id_chips, page_header

page_header("Trust Center", "How we know the answers can be trusted: the same frozen questions scored for the old "
            "and new approach, governance tests run live, and the cost and speed of every answer.")

tabs = st.tabs(["Scorecard", "Arena", "Governance tests", "Live revocation demo", "Telemetry"])
tables = state.tables()
questions = tables["questions"]
gold = tables["relevance_labels"]
gold = gold[gold["relevant"]].groupby("question_id")["review_id"].apply(set).to_dict()

# ---- Scorecard ----------------------------------------------------------------------------------
with tabs[0]:
    sc = state.scorecard()
    view = sc.drop(columns=["per_question", "method"]).rename(columns={
        "label": "Method", "split": "Split", "n": "Questions", "precision": "Precision", "recall": "Recall",
        "abstention_accuracy": "Abstention accuracy", "unsupported_answer_rate": "Unsupported answered"})
    pct = st.column_config.NumberColumn(format="percent")
    st.dataframe(view[["Split", "Method", "Questions", "Precision", "Recall", "Abstention accuracy",
                       "Unsupported answered"]], hide_index=True,
                 column_config={c: pct for c in ("Precision", "Recall", "Abstention accuracy", "Unsupported answered")})
    st.caption("Precision / recall over answerable questions; ‘Unsupported answered’ = share of the out-of-scope "
               "questions that got an answer anyway (lower is better). Keyword and review_rules use the notebook's "
               "definition (top-3, precision denominator 3). Review Signal returns the whole segment "
               "(precision denominator = reviews returned).")
    if "app_retrieval_metrics" in tables:
        nb = tables["app_retrieval_metrics"]
        ok = True
        for r in nb.itertuples():
            mine = sc[(sc["split"] == r.split) & (sc["method"] == r.method)]
            if mine.empty or abs(float(mine.iloc[0]["precision"]) - float(r.mean_precision_at_k_answerable)) > 1e-9 \
                    or abs(float(mine.iloc[0]["recall"]) - float(r.mean_recall_at_k_answerable)) > 1e-9 \
                    or abs(float(mine.iloc[0]["unsupported_answer_rate"]) - float(r.unsupported_answer_rate)) > 1e-9:
                ok = False
        (st.success if ok else st.error)(
            "Keyword and review_rules numbers match the notebook export exactly." if ok else
            "Mismatch with the notebook export (app_retrieval_metrics.csv) — check the data version.")
    st.warning("Limits: 20 templated synthetic questions; segments are clean by construction, so 100% recall is "
               "expected here and will not hold on real reviews. Groundedness of LLM synthesis is enforced by a "
               "citation check but has not been human-rated.")
    with st.expander("Per-question results"):
        rows = []
        for r in sc.itertuples():
            for pq in r.per_question:
                rows.append({"split": r.split, "method": r.method, "question_id": pq["question_id"],
                             "answerable": pq["answerable"], "returned": ", ".join(pq["evidence_ids"]),
                             "precision": pq["precision_at_k"], "recall": pq["recall_at_k"],
                             "correct_abstention": pq["correct_abstention"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True)

# ---- Arena --------------------------------------------------------------------------------------
with tabs[1]:
    st.markdown("Same question, two approaches. ✓ = labelled relevant, ✗ = not relevant.")
    labels = {r.question_id: f"{r.question_id} · {r.question_text}" for r in questions.itertuples()}
    qid = st.selectbox("Benchmark question", list(labels), format_func=lambda k: labels[k], index=2)
    qrow = questions[questions["question_id"] == qid].iloc[0]
    base = state.base_corpus()
    perm = permitted_records(base)
    by_id = base.set_index("review_id")
    relevant = gold.get(qid, set())

    def show(ids, title, note):
        st.markdown(f"**{title}**")
        st.caption(note)
        if not ids:
            st.markdown("<div class='rs-abstain'>Abstains — no answer given.</div>", unsafe_allow_html=True)
        for i in ids:
            r = by_id.loc[i]
            mark = "✓" if i in relevant else "✗"
            st.markdown(f"<div class='rs-quote'>{mark} “{esc(r['review_text'])}”{id_chips([i])}"
                        f"<div class='rs-meta'>{esc(r['product_id'])} · {esc(r['topic'])}</div></div>",
                        unsafe_allow_html=True)
        found = len(set(ids) & relevant)
        if relevant:
            st.metric("Recall", f"{found / len(relevant):.0%}", help=f"{found} of {len(relevant)} relevant reviews")
        else:
            st.metric("Correct behaviour", "Yes" if not ids else "No — answered an unanswerable question")

    left, right = st.columns(2)
    with left:
        show([r["review_id"] for r in retrieve(qrow["question_text"], perm, "lexical")],
             "Keyword top-3 (proxy for the current tool)", "Ranks reviews by word overlap; never abstains.")
    with right:
        show(segment_retrieve(qrow["question_text"], base, state.products(), state.topics()),
             "Review Signal", "Routes the question, then uses the whole product × issue segment.")
    st.caption(f"Question answerable according to labels: {'yes' if qrow['answerable'] else 'no'}. "
               "Labels are used for scoring only, never for retrieval.")

# ---- Governance tests --------------------------------------------------------------------------
with tabs[2]:
    tm = state.test_matrix()
    st.dataframe(tm, hide_index=True, column_config={"result": st.column_config.TextColumn("Result")})
    st.caption("T4b and T4c fail by design: they show the old keyword approach silently substituting off-topic "
               "reviews when evidence disappears. Review Signal abstains instead (T4d, T4e). T6 needs a Unity Catalog "
               "row filter and a second identity — it cannot be run inside this app, so it stays Unknown.")
    st.info("Permission is enforced by the app's data contract, not by the platform. Before production: add a Unity "
            "Catalog row filter on source_allowed and the freshness cutoff, and run T6.")

# ---- Live revocation demo ---------------------------------------------------------------------
with tabs[3]:
    st.markdown("Change the source and watch a cached answer revalidate itself. Affects the whole app until reset.")
    base = state.base_corpus()
    permitted_ids = sorted(base.loc[base["permitted"], "review_id"])
    seg_options = [(p, t) for p in state.products() for t in state.topics()]
    c1, c2 = st.columns(2)
    with c1:
        revoke = st.multiselect("Revoke permission on review IDs", permitted_ids,
                                default=[i for i in st.session_state["revoked"] if i in permitted_ids])
    with c2:
        current_deleted = st.session_state["deleted"]
        default_seg = []
        for p, t in seg_options:
            seg_ids = set(base[(base["product_id"] == p) & (base["topic"] == t) & base["permitted"]]["review_id"])
            if seg_ids and seg_ids <= set(current_deleted):
                default_seg.append((p, t))
        delete = st.multiselect("Delete all reviews of a segment at source", seg_options, default=default_seg,
                                format_func=lambda o: f"{o[0]} · {o[1]}")
    b1, b2 = st.columns([1, 1])
    if b1.button("Apply source change", type="primary"):
        st.session_state["revoked"] = list(revoke)
        chosen = set(delete)
        mask = [(p, t) in chosen for p, t in zip(base["product_id"], base["topic"])]
        st.session_state["deleted"] = sorted(set(base.loc[mask, "review_id"]))
        st.rerun()
    if b2.button("Reset to original data"):
        st.session_state["revoked"], st.session_state["deleted"] = [], []
        st.rerun()

    st.markdown("**Cached answer check** — the answer was produced on the original data; before reuse it is "
                "revalidated against the current source.")
    demo_q = st.selectbox("Cached question", [labels[q] for q in labels if questions.set_index("question_id")
                                              .loc[q, "answerable"]], index=0)
    demo_text = demo_q.split(" · ", 1)[1]
    cached = answer_question(demo_text, base, state.settings(), use_llm=False)
    now = state.corpus()
    rv = revalidate_scoped(cached.cited_ids, now, cached.decision.products, cached.decision.topics)
    colour = {"reuse": "success", "revise": "warning", "abstain": "error"}[rv["action"]]
    getattr(st, colour)(f"Action: **{rv['action'].upper()}** — kept {len(rv['kept'])}, dropped "
                        f"{len(rv['dropped'])} {rv['dropped'] if rv['dropped'] else ''}")
    fresh = answer_question(demo_text, now, state.settings(), use_llm=False)
    l, r = st.columns(2)
    with l:
        st.markdown("**Cached (original data)**")
        st.markdown(esc(cached.verdict))
        st.markdown(id_chips(cached.cited_ids), unsafe_allow_html=True)
    with r:
        st.markdown("**Fresh answer (current data)**")
        st.markdown(esc(fresh.verdict))
        if fresh.route == "abstain":
            st.caption(fresh.decision.reason)
        st.markdown(id_chips(fresh.cited_ids), unsafe_allow_html=True)

# ---- Telemetry ---------------------------------------------------------------------------------
with tabs[4]:
    tel = state.telemetry_df()
    if tel.empty:
        st.caption("No questions asked yet in this session.")
    else:
        c = st.columns(4)
        c[0].metric("Questions", len(tel))
        c[1].metric("Median latency", f"{tel['latency_ms'].median() / 1000:.2f} s")
        c[2].metric("p95 latency", f"{tel['latency_ms'].quantile(0.95) / 1000:.2f} s")
        c[3].metric("Total LLM cost", f"US$ {tel['cost_usd'].sum():.4f}")
        st.dataframe(tel.drop(columns=["answer_id"]), hide_index=True)
        st.download_button("Download telemetry (CSV)", tel.to_csv(index=False), "review_signal_telemetry.csv",
                           "text/csv")
    st.caption("Session-only telemetry for the demo. In production, write these events to a Delta table to measure "
               "weekly active users, repeat use and cost over time.")
