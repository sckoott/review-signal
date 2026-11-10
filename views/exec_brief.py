import datetime as dt

import pandas as pd
import streamlit as st

from engine.recommend import priority_table
from ui import state
from ui.components import esc, heatmap, page_header

segs = state.segments()
corpus = state.corpus()
last_review = corpus.loc[corpus["is_evidence"], "review_date"].max() if corpus["is_evidence"].any() else "n/a"
page_header("Exec Brief", f"One screen for decisions. Data: permitted reviews up to {last_review} · "
            f"generated {dt.date.today():%d %b %Y} · synthetic teaching data.", kicker="Executive view")

# ---- Band 1: decisions needed ---------------------------------------------------------------
st.markdown("### Decisions needed")
accepted = [e for e in st.session_state["rec_log"] if e["decision"].startswith("Accepted")]
if accepted:
    for e in accepted[-3:][::-1]:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**{esc(e['issue'])}**")
            c1.markdown(f"Action: {esc(e['action'])}")
            c1.caption(f"Evidence: {', '.join(x['review_id'] for x in e['evidence'])} · {e['confidence']}")
            c2.markdown(f'<div class="rs-kicker">Owner</div><b>{esc(e["owner"])}</b>', unsafe_allow_html=True)
            c2.caption(f"Accepted {e['decided_at']} by {e['decided_by']}")
else:
    top = priority_table(segs, st.session_state["weights"]).head(3)
    st.caption("No recommendation accepted yet. Highest-priority segments awaiting a recommendation:")
    st.dataframe(top[["rank", "product_id", "topic", "evidence", "negative", "priority"]], hide_index=True)
    if st.button("Open Recommendations"):
        st.switch_page(state.PAGES["recommendations"])

# ---- Band 2: product health -----------------------------------------------------------------
st.markdown("### Product health")
c1, c2 = st.columns([3, 2])
with c1:
    heatmap(segs, key="exec_heatmap", height=220)
with c2:
    st.metric("Evidence reviews", int(segs["evidence"].sum()))
    neg = int(segs["negative"].sum())
    st.metric("Negative share (all segments)", f"{neg / max(int(segs['evidence'].sum()), 1):.0%}")
    st.caption("Change vs. last period: not available — the data covers one month only.")

# ---- Band 3: tool health (for the CDO) ------------------------------------------------------
st.markdown("### Tool health")
sc = state.scorecard()
test = sc[sc["split"] == "test"].set_index("method")
tel = state.telemetry_df()
asks = tel[tel["page"].isin(["Ask", "Data Q&A"])]
c = st.columns(4)
c[0].metric("Relevance (test set)", f"{test.loc['segment', 'recall']:.0%} recall",
            delta=f"{(test.loc['segment', 'recall'] - test.loc['lexical', 'recall']) * 100:+.0f} pts vs keyword",
            help="Share of labelled relevant reviews put in context, held-out test questions.")
c[1].metric("Out-of-scope answered", f"{test.loc['segment', 'unsupported_answer_rate']:.0%}",
            delta=f"{(test.loc['segment', 'unsupported_answer_rate'] - test.loc['lexical', 'unsupported_answer_rate']) * 100:+.0f} pts vs keyword",
            delta_color="inverse")
if len(asks):
    p95 = float(asks["latency_ms"].quantile(0.95)) / 1000
    c[2].metric("p95 latency (this session)", f"{p95:.2f} s", help=f"{len(asks)} questions this session")
    answered = asks[asks["route"] != "abstain"]
    cpa = float(asks["cost_usd"].sum()) / max(len(answered), 1)
    c[3].metric("Cost per answered question", f"US$ {cpa:.4f}")
else:
    c[2].metric("p95 latency (this session)", "—")
    c[3].metric("Cost per answered question", "—")
    cpa = 0.0
with st.expander("Adoption & cost projection"):
    volume = st.number_input("Questions per month (assumption)", min_value=0, value=2000, step=100)
    st.write(f"Projected LLM cost: **US$ {cpa * volume:,.2f} / month** at this session's cost per answer. "
             "SQL warehouse and app compute are not included.")
    if len(asks):
        st.write(f"This session: {len(asks)} questions, "
                 f"{(asks['route'] == 'abstain').mean():.0%} abstained, "
                 f"{(asks['feedback'] == 'useful').sum()} marked useful.")
    st.caption("Weekly active product-team users needs a persistent telemetry table (not part of this demo): Unknown.")

# ---- pinned insights + follow-up ------------------------------------------------------------
if st.session_state["exec_pins"]:
    st.markdown("### Pinned insights")
    for pin in st.session_state["exec_pins"]:
        with st.container(border=True):
            st.markdown(f"**{esc(pin['question'])}** — {esc(pin['answer'])}")
            st.caption(f"Source: {pin['engine']}")
            if pin["rows"]:
                st.dataframe(pd.DataFrame(pin["rows"]), hide_index=True)

st.markdown("### Ask a follow-up")
with st.form("exec_followup", clear_on_submit=True):
    q = st.text_input("Numbers only — answered with SQL, never free text",
                      placeholder="e.g. Which product has the most negative durability reviews?")
    go = st.form_submit_button("Ask")
if go and q.strip():
    res = state.run_sql(q.strip())
    state.log_event("Exec Brief", q.strip(), "sql", res.engine, res.latency_ms, res.cost_usd,
                    f"exec{len(st.session_state['telemetry'])}")
    if res.ok:
        st.markdown(f"**{esc(res.answer_text)}**")
        st.dataframe(res.df, hide_index=True)
        with st.expander("SQL"):
            st.code(res.sql, language="sql")
    else:
        st.error(res.message)


# ---- export -----------------------------------------------------------------------------------
def brief_markdown() -> str:
    lines = [f"# Review Signal — Exec Brief ({dt.date.today():%d %b %Y})", "",
             f"Data: permitted reviews up to {last_review}. Synthetic teaching data.", "", "## Decisions needed"]
    if accepted:
        for e in accepted:
            lines += [f"- **{e['issue']}** — {e['action']} (owner: {e['owner']}; {e['confidence']}; "
                      f"evidence: {', '.join(x['review_id'] for x in e['evidence'])})"]
    else:
        lines += ["- No recommendation accepted yet."]
    lines += ["", "## Product health", "", "| Product | Issue | Evidence | Negative |", "|---|---|---|---|"]
    lines += [f"| {r.product_id} | {r.topic} | {r.evidence} | {r.negative} |" for r in segs.itertuples()]
    lines += ["", "## Tool health",
              f"- Test-set recall: {test.loc['segment', 'recall']:.0%} (keyword baseline "
              f"{test.loc['lexical', 'recall']:.0%})",
              f"- Unsupported questions answered: {test.loc['segment', 'unsupported_answer_rate']:.0%} (keyword "
              f"baseline {test.loc['lexical', 'unsupported_answer_rate']:.0%})"]
    for pin in st.session_state["exec_pins"]:
        lines += ["", f"**Pinned:** {pin['question']} — {pin['answer']}"]
    return "\n".join(lines) + "\n"


st.download_button("Download brief (Markdown)", brief_markdown(), file_name="review_signal_exec_brief.md",
                   mime="text/markdown")
