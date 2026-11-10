import streamlit as st

from engine.answer import answer_question
from ui import state
from ui.components import (auto_chart, chip, esc, funnel_table, id_chips, money, page_header, quotes_block,
                           safe_excluded, sentiment_bar)

page_header("Ask", "Ask what customers say about comfort, fit or durability. Every answer shows the reviews "
            "behind it — or says plainly when reviews can't answer.")

SUGGESTIONS = [
    "How durable is shoe_a according to its reviews?",
    "What do users say about the sizing and fit of jacket_b?",
    "Summarize the comfort feedback for shoe_b.",
    "Which product has the highest share of negative fit reviews?",
    "What is the current wholesale price of shoe_a?",
    "How comfortable is the unreleased shoe_z?",
]


def _queue(q: str):
    st.session_state["pending_question"] = q


with st.container(horizontal=True, gap="small"):
    for i, q in enumerate(SUGGESTIONS):
        st.button(q, key=f"sugg_{i}", on_click=_queue, args=(q,))


def render_answer(a) -> None:
    route_label = {"evidence": "Evidence", "sql": "SQL · " + (a.sql_result.engine if a.sql_result else ""),
                   "abstain": "Abstained"}.get(a.route, a.route)
    mode_label = {"llm": "LLM synthesis, citation-checked", "extractive": "Extractive quotes",
                  "abstain": "No answer given", "sql": "Query result"}.get(a.mode, a.mode)
    st.markdown(chip(route_label, f"route-{a.route}") + chip(mode_label) + chip(f"{a.latency_ms / 1000:.2f} s")
                + f'<span class="rs-chip">{money(a.cost_usd)}</span>', unsafe_allow_html=True)

    if a.route == "abstain":
        d = a.decision
        owner = f"<br><b>Who can answer:</b> {esc(d.owner)}" if d.owner else ""
        st.markdown(f'<div class="rs-abstain"><div class="rs-verdict">{esc(a.verdict)}</div>'
                    f'{esc(d.reason)}{owner}</div>', unsafe_allow_html=True)
        if d.abstain_kind == "no_scope":
            st.caption("Try: “What do users say about the fit of shoe_b?”")
    elif a.route == "sql":
        res = a.sql_result
        st.markdown(f'<div class="rs-verdict">{esc(a.verdict)}</div>', unsafe_allow_html=True)
        if res is not None and res.ok and not res.df.empty:
            auto_chart(res.df)
            st.dataframe(res.df, hide_index=True)
        if res is not None:
            st.caption(res.message)
            if res.sql:
                with st.expander("SQL"):
                    st.code(res.sql, language="sql")
    else:
        st.markdown(f'<div class="rs-verdict">{esc(a.verdict)}</div>', unsafe_allow_html=True)
        sentiment_bar(a.sentiment.get("positive", 0), a.sentiment.get("neutral", 0), a.sentiment.get("negative", 0))
        if a.points:
            st.markdown("**Key points**")
            st.markdown("".join(f"<div style='margin:4px 0'>• {esc(p['text'])} {id_chips(p['ids'])}</div>"
                                for p in a.points), unsafe_allow_html=True)
        if a.gaps:
            st.markdown("**What these reviews don't tell us**")
            st.markdown("".join(f"<div class='rs-note'>– {esc(g)}</div>" for g in a.gaps), unsafe_allow_html=True)
        if a.llm_note:
            st.caption(a.llm_note)
        with st.expander(f"All {len(a.evidence)} evidence reviews used"):
            quotes_block(a.evidence)

    with st.expander("Why this answer? · context inspector"):
        st.markdown("**Pipeline trace**")
        st.markdown("".join(f"<div style='margin:3px 0'><b>{esc(t['step'])}</b> "
                            f"<span class='rs-meta'>{t['ms']} ms</span><br>{esc(t['detail'])}</div>"
                            for t in a.trace), unsafe_allow_html=True)
        if a.route != "abstain" or not a.funnel.empty:
            if not a.funnel.empty:
                st.markdown("**Segment funnel** — every review in scope and what happened to it")
                st.dataframe(funnel_table(a.funnel), hide_index=True)
            if not a.excluded.empty:
                st.markdown("**Excluded reviews**")
                st.dataframe(safe_excluded(a.excluded), hide_index=True)
            if not a.evidence.empty:
                st.markdown(f"**Passed to the {'model' if a.mode == 'llm' else 'answer'}**: "
                            + id_chips(a.cited_ids), unsafe_allow_html=True)

    fb = st.feedback("thumbs", key=f"fb_{a.answer_id}")
    if fb is not None:
        state.set_feedback(a.answer_id, "useful" if fb == 1 else "not useful")


history = st.session_state["history"]
for a in history:
    with st.chat_message("user"):
        st.markdown(esc(a.question))
    with st.chat_message("assistant"):
        render_answer(a)

prompt = st.chat_input("Ask about comfort, fit or durability of shoe_a, shoe_b, jacket_a, jacket_b…")
question = prompt or st.session_state.pop("pending_question", None)

if question:
    with st.chat_message("user"):
        st.markdown(esc(question))
    with st.chat_message("assistant"):
        with st.status("Routing the question…", expanded=False) as status:
            a = answer_question(question, state.corpus(), state.settings(),
                                use_llm=st.session_state.get("use_llm", True), sql_runner=state.run_sql)
            for t in a.trace:
                st.write(f"**{t['step']}** · {t['detail']} ({t['ms']} ms)")
            status.update(label=f"Done in {a.latency_ms / 1000:.2f} s", state="complete")
        render_answer(a)
    history.append(a)
    state.log_event("Ask", question, a.route, a.mode, a.latency_ms, a.cost_usd, a.answer_id)

if history:
    if st.button("Clear conversation", type="tertiary"):
        st.session_state["history"] = []
        st.rerun()
