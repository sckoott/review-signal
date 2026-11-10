"""Reusable visual components. All review text is HTML-escaped before rendering."""
from __future__ import annotations

import html

import altair as alt
import pandas as pd
import streamlit as st

from engine.data import STATUS_LABEL, overall_label

SENTIMENT_COLORS = {"negative": "#FF4D00", "neutral": "#B4B4AE", "positive": "#1F4BFF", "unclassified": "#DADAD5"}
ACCENT = "#1F4BFF"
INK = "#0A0A0A"
VOLT = "#C9F31D"

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}

/* ---------- hero band (keyed container) ---------- */
.st-key-hero {background:#0A0A0A; color:#fff; padding:22px 190px 18px 28px; position:relative; overflow:hidden;
              margin-bottom:0.6rem;}
.st-key-hero::after {content:""; position:absolute; top:0; right:-40px; width:150px; height:100%;
                     background:#C9F31D; transform:skewX(-18deg);}
.st-key-hero h1 {color:#fff !important; font-style:italic; text-transform:uppercase; letter-spacing:0.01em;
                 font-size:3.1rem !important; line-height:1 !important; padding:0 !important; margin:0 !important;}
.st-key-hero h1 a {display:none !important;}
.rs-hero-kicker {font-family:"JetBrains Mono", monospace; font-size:0.72rem; letter-spacing:0.18em; color:#C9F31D;
                 text-transform:uppercase; margin-bottom:6px;}
.rs-hero-sub {color:#CFCFCA; font-size:0.98rem; max-width:860px; margin-top:6px;}
.rs-hero-stats {margin-top:12px; position:relative; z-index:1;}
.rs-stat {display:inline-block; font-family:"JetBrains Mono", monospace; font-size:0.72rem; color:#fff;
          border:1px solid #3A3A3A; padding:3px 9px; margin:0 6px 4px 0; letter-spacing:0.06em; text-transform:uppercase;}
.rs-stat b {color:#C9F31D; font-weight:600;}

/* ---------- headings ---------- */
h2, h3 {text-transform:uppercase; font-style:italic; letter-spacing:0.01em;}
h4 {text-transform:uppercase; letter-spacing:0.04em;}

/* ---------- metrics ---------- */
[data-testid="stMetric"] {border-left:4px solid #0A0A0A; padding-left:12px;}
[data-testid="stMetricLabel"] p {text-transform:uppercase; letter-spacing:0.08em; font-size:0.72rem !important;
                                 font-weight:600; color:#55554F;}
[data-testid="stMetricValue"] {font-family:"Barlow Condensed", Inter, sans-serif; font-weight:800; font-style:italic;}

/* ---------- buttons & tabs ---------- */
[data-testid="stBaseButton-primary"] {text-transform:uppercase; letter-spacing:0.06em; font-weight:700;}
[data-testid="stBaseButton-primary"]:hover {background:#1F4BFF !important; border-color:#1F4BFF !important;}
[data-testid="stBaseButton-secondary"]:hover {border-color:#0A0A0A !important; color:#0A0A0A !important;}
[data-testid="stTab"] p {text-transform:uppercase; letter-spacing:0.06em; font-weight:600;}

/* ---------- sidebar ---------- */
.rs-wordmark {font-family:"Barlow Condensed", Inter, sans-serif; font-weight:800; font-style:italic; font-size:2rem;
              line-height:0.95; text-transform:uppercase; color:#fff; letter-spacing:0.01em;}
.rs-wordmark span {color:#C9F31D;}
.rs-tagline {font-family:"JetBrains Mono", monospace; font-size:0.66rem; letter-spacing:0.16em; color:#9C9C96;
             text-transform:uppercase; margin-top:4px; margin-bottom:14px;}

/* ---------- content components ---------- */
.rs-sub {color:#55554F; font-size:0.98rem; margin-top:-0.6rem; margin-bottom:1.2rem;}
.rs-chip {display:inline-block; padding:2px 9px; margin:0 6px 4px 0; font-size:0.7rem; border:1px solid #0A0A0A;
          background:#fff; color:#0A0A0A; white-space:nowrap; text-transform:uppercase; letter-spacing:0.08em;
          font-family:"JetBrains Mono", monospace;}
.rs-chip.route-evidence {background:#1F4BFF; border-color:#1F4BFF; color:#fff;}
.rs-chip.route-sql {background:#0A0A0A; color:#C9F31D;}
.rs-chip.route-abstain {background:#FF4D00; border-color:#FF4D00; color:#fff;}
.rs-id {font-family:"JetBrains Mono", monospace; font-size:0.72rem; background:#0A0A0A; color:#fff;
        padding:1px 6px; margin-left:4px; letter-spacing:0.04em;}
.rs-verdict {font-size:1.2rem; font-weight:700; line-height:1.4; margin:0.3rem 0 0.6rem 0;}
.rs-bar {display:flex; height:10px; overflow:hidden; margin:4px 0 4px 0; background:#eee;}
.rs-legend {font-size:0.78rem; color:#55554F; margin-bottom:0.6rem; text-transform:uppercase; letter-spacing:0.05em;}
.rs-dot {display:inline-block; width:9px; height:9px; margin-right:6px;}
.rs-quote {border-left:4px solid #ddd; padding:5px 0 5px 12px; margin:7px 0; font-size:0.95rem; background:#FAFAF8;}
.rs-meta {color:#77776F; font-size:0.74rem; text-transform:uppercase; letter-spacing:0.04em;}
.rs-abstain {border-left:4px solid #FF4D00; background:#FFF3EC; padding:14px 16px;}
.rs-note {color:#55554F; font-size:0.86rem;}
.rs-kicker {font-family:"JetBrains Mono", monospace; text-transform:uppercase; letter-spacing:0.16em; font-size:0.68rem;
            color:#1F4BFF; margin-bottom:2px;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(text) -> str:
    return html.escape(str(text))


def money(value: float) -> str:
    return f"&#36;{value:.4f}"  # HTML entity avoids Markdown math mode on "$"


def chip(text: str, cls: str = "") -> str:
    return f'<span class="rs-chip {cls}">{esc(text)}</span>'


def id_chips(ids) -> str:
    return "".join(f'<span class="rs-id">{esc(i)}</span>' for i in ids)


def page_header(title: str, subtitle: str, kicker: str = "Performance insights") -> None:
    """Black hero band with condensed italic title and live data stats."""
    from ui import state  # local import: state does not depend on components

    corpus = state.corpus()
    stats = [("Permitted reviews", int(corpus["permitted"].sum())),
             ("Evidence", int(corpus["is_evidence"].sum())),
             ("Segments", int(corpus.groupby(["product_id", "topic"]).ngroups)),
             ("Engine", "LLM" if state.settings().llm_provider != "none" and st.session_state.get("use_llm", True)
              else "Extractive")]
    with st.container(key="hero"):
        st.markdown(f'<div class="rs-hero-kicker">Review Signal &nbsp;/&nbsp; {esc(kicker)}</div>',
                    unsafe_allow_html=True)
        st.title(title, anchor=False)
        st.markdown(f'<div class="rs-hero-sub">{esc(subtitle)}</div><div class="rs-hero-stats">'
                    + "".join(f'<span class="rs-stat">{esc(k)} <b>{esc(v)}</b></span>' for k, v in stats)
                    + "</div>", unsafe_allow_html=True)


def sentiment_bar(pos: int, neu: int, neg: int) -> None:
    n = pos + neu + neg
    if n == 0:
        st.caption("No evidence reviews.")
        return
    segs = "".join(f'<div style="width:{c / n * 100:.2f}%; background:{SENTIMENT_COLORS[s]}"></div>'
                   for s, c in (("positive", pos), ("neutral", neu), ("negative", neg)) if c)
    legend = " &nbsp; ".join(
        f'<span class="rs-dot" style="background:{SENTIMENT_COLORS[s]}"></span>{c} {s}'
        for s, c in (("positive", pos), ("neutral", neu), ("negative", neg)))
    st.markdown(f'<div class="rs-bar">{segs}</div><div class="rs-legend">{legend} &nbsp;·&nbsp; n = {n} · '
                f'{esc(overall_label(pos, neu, neg))}</div>', unsafe_allow_html=True)


def quote(row) -> str:
    color = SENTIMENT_COLORS.get(row["sentiment"], "#ccc")
    return (f'<div class="rs-quote" style="border-left-color:{color}">“{esc(row["review_text"])}”'
            f'<div class="rs-meta">{esc(row["product_id"])} · {esc(row["topic"])} · {esc(row["review_date"])} · '
            f'{esc(row["sentiment"])}{id_chips([row["review_id"]])}</div></div>')


def quotes_block(df: pd.DataFrame, limit: int | None = None) -> None:
    rows = df if limit is None else df.head(limit)
    st.markdown("".join(quote(r) for _, r in rows.iterrows()), unsafe_allow_html=True)


def heatmap(segments: pd.DataFrame, selectable: bool = False, key: str = "heatmap", height: int = 260):
    df = segments.copy()
    df["negative_pct"] = (df["negative_share"].fillna(0) * 100).round(0)
    df["label"] = df.apply(lambda r: f"{int(r['negative_pct'])}% neg · n={int(r['evidence'])}"
                           if r["evidence"] else "no evidence", axis=1)
    base = alt.Chart(df).encode(
        x=alt.X("topic:N", title="Issue", axis=alt.Axis(labelAngle=0, labelFontSize=12)),
        y=alt.Y("product_id:N", title="Product", axis=alt.Axis(labelFontSize=12)),
    )
    tooltip = [alt.Tooltip("product_id:N", title="Product"), alt.Tooltip("topic:N", title="Issue"),
               alt.Tooltip("evidence:Q", title="Evidence reviews"), alt.Tooltip("negative:Q", title="Negative"),
               alt.Tooltip("negative_pct:Q", title="Negative %"), alt.Tooltip("no_evidence:Q", title="No-evidence"),
               alt.Tooltip("restricted:Q", title="Restricted"), alt.Tooltip("outdated:Q", title="Outdated")]
    color = alt.Color("negative_pct:Q", title="Negative %",
                      scale=alt.Scale(domain=[0, 100], range=["#F2F2F0", "#FF4D00"], interpolate="rgb"))
    if selectable:
        sel = alt.selection_point(name="cell", fields=["product_id", "topic"], on="click", clear=False)
        rect = base.mark_rect(cornerRadius=0, stroke="#ffffff", strokeWidth=4).encode(
            color=color, tooltip=tooltip,
            opacity=alt.condition(sel, alt.value(1.0), alt.value(0.55))).add_params(sel)
    else:
        rect = base.mark_rect(cornerRadius=0, stroke="#ffffff", strokeWidth=4).encode(color=color, tooltip=tooltip)
    text = base.mark_text(fontSize=13, fontWeight="bold", font="JetBrains Mono").encode(
        text="label:N",
        color=alt.condition("datum.negative_pct > 60", alt.value("#ffffff"), alt.value("#222222")))
    chart = (rect + text).properties(height=height)
    if selectable:
        return st.altair_chart(chart, key=key, on_select="rerun", selection_mode=["cell"])
    st.altair_chart(chart, key=key)
    return None


def auto_chart(df: pd.DataFrame) -> None:
    if df is None or df.empty or len(df) > 60:
        return
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if not num or not cat:
        return
    y = num[-1]
    if "product_id" in df.columns and "topic" in df.columns:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("product_id:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset="topic:N",
            y=alt.Y(f"{y}:Q"), color=alt.Color("topic:N", scale=alt.Scale(range=[INK, ACCENT, "#FF4D00"])),
            tooltip=list(df.columns))
    elif "sentiment" in df.columns and "product_id" in df.columns:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("product_id:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y(f"{y}:Q", stack="zero"),
            color=alt.Color("sentiment:N", scale=alt.Scale(
                domain=[s for s in SENTIMENT_COLORS if s in set(df["sentiment"])],
                range=[c for s, c in SENTIMENT_COLORS.items() if s in set(df["sentiment"])])),
            tooltip=list(df.columns))
    else:
        chart = alt.Chart(df).mark_bar(color=INK).encode(
            x=alt.X(f"{cat[0]}:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y(f"{y}:Q"),
            tooltip=list(df.columns))
    st.altair_chart(chart.properties(height=240))


def safe_excluded(excluded: pd.DataFrame) -> pd.DataFrame:
    """Excluded rows with their reason. Text is shown only for permitted rows (no-evidence ones);
    restricted, outdated, revoked or deleted text is never displayed."""
    show = excluded[["review_id", "product_id", "topic", "review_date", "status", "review_text"]].copy()
    hide = ~show["status"].eq("no_evidence")
    show["review_text"] = show["review_text"].where(~hide, "— hidden: not permitted for use —")
    show["status"] = show["status"].map(STATUS_LABEL)
    return show.rename(columns={"status": "reason"})


def funnel_table(funnel: pd.DataFrame) -> pd.DataFrame:
    cols = ["product_id", "topic", "total", "restricted", "outdated", "revoked", "deleted", "no_evidence", "evidence"]
    return funnel[cols].rename(columns={"product_id": "Product", "topic": "Issue", "total": "In segment",
                                        "restricted": "Restricted", "outdated": "Outdated", "revoked": "Revoked",
                                        "deleted": "Deleted", "no_evidence": "No evidence", "evidence": "Used"})


def segment_panel(seg_row: pd.Series, evidence: pd.DataFrame, excluded: pd.DataFrame, key_prefix: str) -> str | None:
    """Renders a segment; returns 'ask' or 'recommend' if a button was pressed."""
    product, topic = seg_row["product_id"], seg_row["topic"]
    st.markdown(f'<div class="rs-kicker">Segment</div>', unsafe_allow_html=True)
    st.subheader(f"{product} · {topic}")
    sentiment_bar(int(seg_row["positive"]), int(seg_row["neutral"]), int(seg_row["negative"]))
    c = st.columns(5)
    c[0].metric("In segment", int(seg_row["total"]))
    c[1].metric("Used as evidence", int(seg_row["evidence"]))
    c[2].metric("No evidence", int(seg_row["no_evidence"]))
    c[3].metric("Restricted", int(seg_row["restricted"]))
    c[4].metric("Outdated / revoked", int(seg_row["outdated"]) + int(seg_row["revoked"]) + int(seg_row["deleted"]))
    if evidence.empty:
        st.info("No permitted evidence left in this segment.")
    else:
        quotes_block(evidence)
    with st.expander(f"Excluded reviews ({len(excluded)}) and why"):
        if excluded.empty:
            st.caption("None.")
        else:
            st.dataframe(safe_excluded(excluded), hide_index=True)
    b1, b2 = st.columns(2)
    action = None
    if b1.button("Ask about this segment", key=f"{key_prefix}_ask", width="stretch"):
        action = "ask"
    if b2.button("Draft a recommendation", key=f"{key_prefix}_rec", type="primary", width="stretch"):
        action = "recommend"
    return action
