import pandas as pd
import streamlit as st

from engine.recommend import draft_recommendation, log_entry, priority_table
from engine.retrieval import segment_evidence
from ui import state
from ui.components import esc, id_chips, page_header, sentiment_bar

page_header("Recommendations", "A separate, human-approved step: rank segments, draft an action from the evidence, "
            "then accept, edit or reject it. Only accepted actions reach the Exec Brief.")

segs = state.segments()
products = state.products()

st.markdown("#### 1 · Prioritise")
st.caption("priority = evidence reviews × negative share × business weight. Trend is left out: the data covers a "
           "single month, so no trend can be computed yet.")
weights = {p: float(st.session_state["weights"].get(p, 1.0)) for p in products}
edited = st.data_editor(
    pd.DataFrame({"product_id": products, "business_weight": [weights[p] for p in products]}),
    hide_index=True, disabled=["product_id"], key="weights_editor",
    column_config={"business_weight": st.column_config.NumberColumn(
        "Business weight", min_value=0.0, max_value=10.0, step=0.5,
        help="e.g. revenue share or strategic importance; 1.0 = neutral")})
st.session_state["weights"] = {r.product_id: float(r.business_weight if pd.notna(r.business_weight) else 1.0)
                               for r in edited.itertuples()}
ranking = priority_table(segs, st.session_state["weights"])
st.dataframe(ranking, hide_index=True, column_config={
    "negative_share": st.column_config.NumberColumn("Negative share", format="percent")})
if ranking["priority"].nunique() == 1:
    st.info("All segments tie: the synthetic data gives every segment the same sentiment mix. "
            "Change a business weight to see the ranking respond.")

st.markdown("#### 2 · Draft")
options = [(r.product_id, r.topic) for r in ranking.itertuples()]
target = st.session_state.pop("rec_target", None)
if target in options:
    st.session_state["rec_choice"] = target
if not options:
    st.warning("No segment has evidence left.")
    st.stop()
choice = st.selectbox("Segment", options, key="rec_choice", format_func=lambda o: f"{o[0]} · {o[1]}")
evidence = segment_evidence(state.corpus(), [choice[0]], [choice[1]])
if st.button("Draft recommendation", type="primary"):
    with st.spinner("Drafting from the segment's evidence…"):
        st.session_state["rec_draft"] = draft_recommendation(state.settings(), choice[0], choice[1], evidence,
                                                             use_llm=st.session_state.get("use_llm", True))

card = st.session_state.get("rec_draft")
if card:
    with st.container(border=True):
        st.markdown('<div class="rs-kicker">Draft · ' + esc(card.get("source", "")) + "</div>", unsafe_allow_html=True)
        st.subheader(card["issue"])
        seg = segs[(segs["product_id"] == card["product_id"]) & (segs["topic"] == card["topic"])]
        if not seg.empty:
            r = seg.iloc[0]
            sentiment_bar(int(r["positive"]), int(r["neutral"]), int(r["negative"]))
        st.markdown("**Evidence**")
        st.markdown("".join(f"<div class='rs-quote'>“{esc(e['text'])}” {id_chips([e['review_id']])} "
                            f"<span class='rs-meta'>{esc(e['sentiment'])}</span></div>" for e in card["evidence"]),
                    unsafe_allow_html=True)
        if card.get("llm_note"):
            st.caption(card["llm_note"])
        with st.form("rec_form"):
            hypothesis = st.text_area("Hypothesis (labelled, not a fact)", card["hypothesis"])
            action = st.text_area("Recommended action", card["action"])
            c1, c2 = st.columns(2)
            owner = c1.text_input("Owner team", card["owner"])
            metric = c2.text_input("Metric to watch", card["metric"])
            st.text_input("Confidence (from evidence count)", card["confidence"], disabled=True)
            change_mind = st.text_area("What would change our mind", card["change_mind"])
            reason = st.text_input("Reason (required to reject)")
            b1, b2 = st.columns(2)
            accept = b1.form_submit_button("Accept", type="primary", width="stretch")
            reject = b2.form_submit_button("Reject", width="stretch")
        if accept or reject:
            if reject and not reason.strip():
                st.error("Add a reason before rejecting.")
            else:
                final = dict(card, hypothesis=hypothesis, action=action, owner=owner, metric=metric,
                             change_mind=change_mind)
                edited_flag = any(final[k] != card[k] for k in ("hypothesis", "action", "owner", "metric",
                                                                  "change_mind"))
                decision = ("Accepted (edited)" if edited_flag else "Accepted") if accept else "Rejected"
                st.session_state["rec_log"].append(log_entry(final, decision, reason.strip(), state.user_email()))
                st.session_state["rec_draft"] = None
                st.toast(f"{decision}: {card['product_id']} · {card['topic']}")
                st.rerun()

st.markdown("#### 3 · Decision log")
log = st.session_state["rec_log"]
if not log:
    st.caption("No decisions yet. Accepted recommendations appear in the Exec Brief.")
else:
    df = pd.DataFrame([{k: e.get(k) for k in ("decided_at", "product_id", "topic", "decision", "action", "owner",
                                               "confidence", "reason", "decided_by")} for e in log])
    st.dataframe(df, hide_index=True)
    accepted = sum(e["decision"].startswith("Accepted") for e in log)
    st.caption(f"Acceptance rate: {accepted}/{len(log)} — a usefulness signal for the tool, not proof of impact.")
