# Review Signal

Evidence-first customer review insights. A Streamlit app (runs locally or as a Databricks App) that replaces a slow, generic review chatbot with one that **picks the right reviews, shows them, and refuses what reviews can't answer**.

Built on the team_10 Adidas notebook (synthetic teaching data): 120 reviews, 4 products × 3 issues, 20 benchmark questions.

| Method (held-out test set) | Precision | Recall | Out-of-scope questions answered |
|---|---|---|---|
| Keyword top-3 (proxy for current tool) | 28% | 17% | 100% |
| Notebook `review_rules` top-3 | 100% | 60% | 0% |
| **Review Signal: whole product × issue segment** | **100%** | **100%** | **0%** |

The first two rows reproduce `app_retrieval_metrics.csv` from the notebook exactly (checked by a test and live in the Trust Center).

---

## 1. Run it locally (2 minutes)

```bash
cd review_signal
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

It works **with no API key**: answers are then extractive (exact quotes + review IDs), and the Data Q&A page uses predefined SQL queries.

## 2. Where to put the API key

Use **one** of these (Anthropic first; OpenRouter is the alternative):

**Option A — secrets file (local).** Copy the template and fill it in:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

```toml
# .streamlit/secrets.toml
ANTHROPIC_API_KEY = "sk-ant-..."
# or
# OPENROUTER_API_KEY = "sk-or-..."
```

**Option B — environment variable.**

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # or: export OPENROUTER_API_KEY=sk-or-...
streamlit run app.py
```

Restart the app after adding a key. The sidebar shows which engine is active (e.g. "Claude via Anthropic API · claude-sonnet-5-5").

Defaults: model `claude-sonnet-5-5` (Anthropic) or `anthropic/claude-sonnet-5.5` (OpenRouter); cost estimate uses US$2 / US$10 per million input / output tokens. Override with `ANTHROPIC_MODEL`, `OPENROUTER_MODEL`, `LLM_PRICE_INPUT_PER_MTOK`, `LLM_PRICE_OUTPUT_PER_MTOK`.

## 3. Deploy on Databricks Apps

1. Upload the `review_signal` folder to your workspace (or a Git folder) and create a **Custom** app pointing to it. `app.yaml` already contains the start command.
2. **API key:** store it in a Databricks secret → in the app's **Resources**, add that secret with resource key `anthropic-api-key` → uncomment the `ANTHROPIC_API_KEY` lines in `app.yaml`.
3. **Genie (optional):** create a Genie space over two tables built from this data (see §5), add it as an app resource with key `genie-space`, and uncomment the `GENIE_SPACE_ID` lines. To query Genie as the signed-in user, enable user authorization with the Genie scope; otherwise the app's service principal is used (set `GENIE_USE_USER_TOKEN=false` to force that).
4. **Exec landing page:** put executives' emails in `EXEC_EMAILS` (comma-separated). They land on the Exec Brief.
5. Deploy.

## 4. What's in the app

| Page | For | What it does |
|---|---|---|
| **Ask** | Product teams | Chat. Routes each question to *evidence*, *SQL* or *abstain*; answers with a sentiment bar, cited key points and "what the reviews don't tell us". **Context inspector** shows the pipeline trace, the segment funnel and every excluded review with its reason. Thumbs feedback. |
| **Issue Map** | Product teams | Product × issue heatmap (click a cell), segment drawer with quotes, deep links (`?product=shoe_a&issue=fit`), buttons to ask or draft a recommendation. |
| **Data Q&A (Genie)** | Analysts, PMs | Counting/ranking questions. Genie if configured; otherwise LLM-written SQL (validated, sandboxed) or predefined queries. SQL always shown; pin results to the Exec Brief; jump to the reviews behind a number. |
| **Recommendations** | Product leads | Separate, human-approved step: editable priority weights → draft card (hypothesis labelled, evidence cited, owner, confidence, what would change our mind) → Accept / Edit / Reject with reason → decision log. |
| **Exec Brief** | CDO, sponsor | Decisions needed, product health, tool health (test-set relevance, unsupported-answer rate, p95 latency, cost per answer, monthly projection), pinned insights, numbers-only follow-up, Markdown export. |
| **Trust Center** | Solutions Architect | Scorecard (3 methods × dev/test), Arena (old vs new side by side with ✓/✗ against labels), governance test matrix T1–T6, live revocation/deletion demo, session telemetry. |

The **View as** switch in the sidebar changes the home page and menu order (demo convenience; in production the role comes from the signed-in user).

## 5. How the relevance engine works

1. **Contract gate:** `source_allowed = true AND review_date >= 2026-01-01` (96 of 120).
2. **Evidence gate:** rows saying "cannot assess / does not describe / no evidence" are excluded from context but counted (36).
3. **Enrichment:** sentiment per review (rule-based for this templated corpus; on real data use `ai_analyze_sentiment` in Databricks SQL).
4. **Segments:** one per product × issue, with the funnel counts and evidence IDs.
5. **Router:** out-of-scope (price, roadmap, warranty, stock…) and unknown products → abstain with the team to ask; counting questions → SQL; review questions → evidence.
6. **Context:** *all* evidence reviews of the matched segment(s) (cap 30 per segment, stratified by sentiment).
7. **Answer:** with an LLM, the model must return JSON whose every point cites allowed review IDs; points citing anything else are dropped; if nothing survives, the extractive answer is shown. Without an LLM, the answer is extractive.
8. **Revalidation:** a cached answer is rechecked against the current source (still permitted *and* still on-topic) → reuse / revise / abstain.

Tables exposed to SQL/Genie (permitted rows only): `reviews_enriched(review_id, product_id, topic, review_text, review_date, sentiment, is_evidence)` and `review_segments(product_id, topic, total, revoked, restricted, outdated, no_evidence, evidence, positive, neutral, negative, negative_share, first_review_date, last_review_date)`.

## 6. Tests

```bash
pip install pytest
python -m pytest -q          # 69 tests: engine (metrics vs notebook, router, governance, LLM/Genie mocks, SQL sandbox) + every page via Streamlit AppTest
```

## 7. Known limits (say them in the demo)

- Synthetic, templated data: every segment has the same 2 positive / 1 neutral / 2 negative mix, so the heatmap is uniform and priorities tie (use the business weights).
- 100% recall is by construction here; real reviews mix issues and will not segment this cleanly.
- Permission is enforced by the app's data contract, not the platform. Before production: Unity Catalog row filter + a denied-read test under a second identity (T6).
- LLM groundedness is enforced by the citation check but not human-rated.
- Telemetry, recommendation log and pins live in the browser session only; persist them to Delta tables for real adoption metrics.
- The Genie integration is written against `databricks-sdk` 0.164 and tested with mocks; it needs a real Genie space to run end to end.

## Theme

"Performance tech" look: black and white, condensed italic headings, signal-orange for negative and electric-blue for positive (colour-blind safe), lime accent only on black. Fonts (Barlow Condensed, Inter, JetBrains Mono) are bundled in `static/fonts` under the SIL Open Font License, so nothing loads from the internet. Change colours in `.streamlit/config.toml` and `ui/components.py`. No third-party logos or brand marks are used.

## Files

```
app.py                  entry point, navigation, sidebar
views/                  one file per page
ui/state.py             session state, cached data, identity, telemetry
ui/components.py        theme CSS, answer/segment components, charts
engine/config.py        settings (keys, models, Genie, cutoff)
engine/data.py          load tables, contract, enrichment, segments
engine/router.py        question routing + entity resolution
engine/retrieval.py     notebook retrieval helpers + segment method + scorecard
engine/answer.py        answer pipeline with citation check
engine/llm.py           Anthropic / OpenRouter client
engine/sql_qa.py        Genie client + sandboxed local SQL
engine/recommend.py     prioritisation and recommendation drafts
engine/governance.py    T1–T6 test matrix, revalidation
data/                   team_10 tables exported from the notebook
tests/                  pytest suites
app.yaml                Databricks Apps config
```
