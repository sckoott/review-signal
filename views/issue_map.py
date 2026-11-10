import streamlit as st

from engine.retrieval import segment_evidence
from ui import state
from ui.components import heatmap, page_header, segment_panel

page_header("Issue Map", "Every product × issue at a glance. Colour = share of negative evidence reviews; "
            "click a cell to see the reviews behind it.")

segs = state.segments()
corpus = state.corpus()
products, topics = state.products(), state.topics()

k = st.columns(4)
k[0].metric("Segments", len(segs))
k[1].metric("Evidence reviews", int(segs["evidence"].sum()))
k[2].metric("Excluded (no evidence)", int(segs["no_evidence"].sum()),
            help="Permitted and current, but the text carries no product signal.")
k[3].metric("Excluded (restricted / outdated / revoked)",
            int(segs[["restricted", "outdated", "revoked", "deleted"]].sum().sum()))

# Navigation requests from other pages, then deep links (?product=..&issue=..)
nav_p, nav_t = st.session_state.pop("nav_product", None), st.session_state.pop("nav_topic", None)
if nav_p in products and nav_t in topics:
    st.session_state["seg_product"], st.session_state["seg_topic"] = nav_p, nav_t
elif "seg_product" not in st.session_state:
    qp_p, qp_t = st.query_params.get("product"), st.query_params.get("issue")
    st.session_state["seg_product"] = qp_p if qp_p in products else products[0]
    st.session_state["seg_topic"] = qp_t if qp_t in topics else topics[0]

event = heatmap(segs, selectable=True, key="issue_heatmap", height=300)
try:
    picked = (event.selection.get("cell") or []) if event is not None else []
except AttributeError:
    picked = []
if picked:
    p, t = picked[0].get("product_id"), picked[0].get("topic")
    if p in products and t in topics and st.session_state.get("_last_pick") != (p, t):
        st.session_state["_last_pick"] = (p, t)
        st.session_state["seg_product"], st.session_state["seg_topic"] = p, t

st.caption("Every segment shows 40% negative because the synthetic dataset gives each product × issue the same "
           "mix (2 positive, 1 neutral, 2 negative). With real reviews this map is where differences show up.")

c1, c2 = st.columns(2)
c1.selectbox("Product", products, key="seg_product")
c2.selectbox("Issue", topics, key="seg_topic")
product, topic = st.session_state["seg_product"], st.session_state["seg_topic"]
st.query_params["product"], st.query_params["issue"] = product, topic

row = segs[(segs["product_id"] == product) & (segs["topic"] == topic)]
if row.empty:
    st.info("No reviews for this product and issue.")
else:
    with st.container(border=True):
        evidence = segment_evidence(corpus, [product], [topic])
        in_scope = corpus[(corpus["product_id"] == product) & (corpus["topic"] == topic)]
        excluded = in_scope[~in_scope["is_evidence"]]
        action = segment_panel(row.iloc[0], evidence, excluded, key_prefix="map")
    st.caption(f"Share this view: add ?product={product}&issue={topic} to the app URL.")
    if action == "ask":
        st.session_state["pending_question"] = f"What do users say about the {topic} of {product}?"
        st.switch_page(state.PAGES["ask"])
    elif action == "recommend":
        st.session_state["rec_target"] = (product, topic)
        st.switch_page(state.PAGES["recommendations"])
