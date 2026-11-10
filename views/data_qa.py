import streamlit as st

from engine import llm
from engine.sql_qa import SCHEMA_DOC, genie_configured
from ui import state
from ui.components import auto_chart, chip, esc, money, page_header

page_header("Data Q&A", "Counts, rankings and shares over the permitted review tables — answered by Databricks "
            "Genie when it is configured, with the SQL always shown.", kicker="Numbers · Genie")

s = state.settings()
if genie_configured(s):
    st.markdown(chip("Engine: Databricks AI/BI Genie", "route-sql"), unsafe_allow_html=True)
elif llm.is_available(s) and st.session_state.get("use_llm", True):
    st.markdown(chip("Engine: local SQL written by the LLM (Genie not configured)", "route-sql"),
                unsafe_allow_html=True)
else:
    st.markdown(chip("Engine: predefined SQL queries (no Genie, no LLM key)", "route-sql"), unsafe_allow_html=True)

QUICK = [
    "Which product has the highest share of negative fit reviews?",
    "How many reviews per product?",
    "Sentiment breakdown by product",
    "How many reviews were excluded and why?",
]


def _queue(q: str):
    st.session_state["pending_sql"] = q


with st.container(horizontal=True):
    for i, q in enumerate(QUICK):
        st.button(q, key=f"quick_{i}", on_click=_queue, args=(q,))

with st.form("sql_form", clear_on_submit=True):
    typed = st.text_input("Your question", placeholder="e.g. Rank products by number of negative durability reviews")
    submitted = st.form_submit_button("Ask", type="primary")

question = (typed.strip() if submitted and typed and typed.strip() else None) or st.session_state.pop("pending_sql", None)
if question:
    with st.spinner("Querying…"):
        res = state.run_sql(question)
    st.session_state["sql_history"].insert(0, res)
    state.log_event("Data Q&A", question, "sql", res.engine, res.latency_ms, res.cost_usd,
                    f"sql{len(st.session_state['telemetry'])}")

cols = st.columns([1, 1, 4])
if cols[0].button("New conversation", help="Starts a fresh Genie thread (recommended per session)."):
    st.session_state["genie_conversation_id"] = None
    st.session_state["sql_history"] = []
    st.rerun()
with cols[1].popover("Tables & schema"):
    st.code(SCHEMA_DOC, language="text")

for i, res in enumerate(st.session_state["sql_history"]):
    with st.container(border=True):
        st.markdown(f"**{esc(res.question)}**")
        st.markdown(chip(res.engine, "route-sql") + chip(f"{res.latency_ms / 1000:.2f} s")
                    + f'<span class="rs-chip">{money(res.cost_usd)}</span>', unsafe_allow_html=True)
        if not res.ok:
            st.error(res.message)
            continue
        if res.answer_text:
            st.markdown(f'<div class="rs-verdict">{esc(res.answer_text)}</div>', unsafe_allow_html=True)
        if not res.df.empty:
            auto_chart(res.df)
            st.dataframe(res.df, hide_index=True)
        st.caption(res.message)
        if res.sql:
            with st.expander("SQL"):
                st.code(res.sql, language="sql")
        b = st.columns([1, 1, 2])
        if b[0].button("Pin to Exec Brief", key=f"pin_{i}"):
            st.session_state["exec_pins"].append({"question": res.question, "answer": res.answer_text,
                                                  "engine": res.engine, "sql": res.sql,
                                                  "rows": res.df.head(20).to_dict("records")})
            st.toast("Pinned to the Exec Brief")
        if {"product_id", "topic"} <= set(res.df.columns) and not res.df.empty:
            first = res.df.iloc[0]
            if b[1].button(f"See the reviews behind {first['product_id']} · {first['topic']}", key=f"see_{i}"):
                st.session_state["nav_product"], st.session_state["nav_topic"] = first["product_id"], first["topic"]
                st.switch_page(state.PAGES["issue_map"])
