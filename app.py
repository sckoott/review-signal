"""Review Signal — evidence-first customer review insights (Streamlit / Databricks Apps).

Run locally:   streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Review Signal", page_icon=":material/insights:", layout="wide")

from engine import llm  # noqa: E402
from engine.sql_qa import genie_configured  # noqa: E402
from ui import state  # noqa: E402
from ui.components import inject_css  # noqa: E402

inject_css()
state.init_state()
if st.session_state["role"] is None:
    st.session_state["role"] = state.default_role()

ROLE_HOME = {"Product team": "ask", "Executive": "exec_brief", "Solutions Architect": "trust_center"}
PAGE_META = {
    "ask": ("Ask", ":material/forum:"),
    "issue_map": ("Issue Map", ":material/grid_view:"),
    "data_qa": ("Data Q&A (Genie)", ":material/query_stats:"),
    "recommendations": ("Recommendations", ":material/task_alt:"),
    "exec_brief": ("Exec Brief", ":material/summarize:"),
    "trust_center": ("Trust Center", ":material/verified_user:"),
}


with st.sidebar:
    st.markdown('<div class="rs-wordmark">Review<span>/</span><br>Signal</div>'
                '<div class="rs-tagline">Evidence-first review intelligence</div>', unsafe_allow_html=True)
    st.radio("View as", state.ROLES, key="role",
             help="On Databricks Apps the default comes from the signed-in user (EXEC_EMAILS). "
                  "The switch is kept for the demo.")

role = st.session_state["role"]
home = ROLE_HOME.get(role, "ask")
order = [home] + [k for k in PAGE_META if k != home]
pages = {k: st.Page(state.PAGES[k], title=PAGE_META[k][0], icon=PAGE_META[k][1], default=(k == home))
         for k in order}
nav = st.navigation({"Insights": [pages[k] for k in order if k in ("ask", "issue_map", "data_qa", "exec_brief")
                                   ],
                     "Act & govern": [pages[k] for k in order if k in ("recommendations", "trust_center")]})

if nav is not pages[home]:
    with st.sidebar:
        st.page_link(pages[home], label=f"Go to {PAGE_META[home][0]}", icon=":material/home:",
                     help=f"Home page for the {role} view")

with st.sidebar:
    st.divider()
    s = state.settings()
    st.markdown('<div class="rs-tagline" style="color:#C9F31D">Engine status</div>', unsafe_allow_html=True)
    st.caption(f"LLM: {llm.describe(s)}")
    if llm.is_available(s):
        st.toggle("Use LLM synthesis", key="use_llm",
                  help="Off = extractive answers only (exact quotes, no model call, no cost).")
    st.caption("SQL: " + (f"Genie space {s.genie_space_id}" if genie_configured(s)
                          else "Local SQL (Genie not configured)"))
    c = state.corpus()
    st.caption(f"Data: {int(c['permitted'].sum())} permitted of {len(c)} reviews · cutoff {s.freshness_cutoff}")
    if state.simulation_active():
        st.warning(f"Simulation on: {len(st.session_state['revoked'])} revoked, "
                   f"{len(st.session_state['deleted'])} deleted review(s). Reset in Trust Center.")
    st.caption(f"Signed in as {state.user_email()}")
    st.caption("Synthetic teaching data · not real customer reviews")

nav.run()
