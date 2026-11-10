"""UI smoke tests with Streamlit's AppTest. Run from the app folder:  python -m pytest -q"""
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["views/ask.py", "views/issue_map.py", "views/data_qa.py", "views/recommendations.py",
         "views/exec_brief.py", "views/trust_center.py"]


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "GENIE_SPACE_ID"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "none")
    os.chdir(Path(APP).parent)


def fresh(page=None):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    if page:
        at.switch_page(page).run()
        assert not at.exception, at.exception
    return at


def all_text(at) -> str:
    return " ".join(m.value for m in at.markdown) + " " + " ".join(c.value for c in at.caption)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    fresh(page)


def test_role_switch_offers_home_link():
    at = fresh()
    assert "Ask" in [t.value for t in at.title]
    at.sidebar.radio[0].set_value("Executive").run()
    assert not at.exception, at.exception
    assert at.session_state["role"] == "Executive"
    assert any(getattr(el, "proto", None) is not None and "Exec Brief" in str(el.proto)
               for el in at.sidebar.get("page_link"))


def test_exec_user_lands_on_exec_brief(monkeypatch):
    import streamlit as st
    monkeypatch.setenv("EXEC_EMAILS", "cdo@corp.com, local-user")
    st.cache_resource.clear()
    try:
        at = fresh()
        assert "Exec Brief" in [t.value for t in at.title]
    finally:
        monkeypatch.delenv("EXEC_EMAILS")
        st.cache_resource.clear()


@pytest.mark.parametrize("question,expect", [
    ("How durable is shoe_a according to its reviews?", "R023"),
    ("What is the current wholesale price of shoe_a?", "Pricing"),
    ("How comfortable is the unreleased shoe_z?", "No reviews exist for shoe_z"),
    ("Which product has the highest share of negative fit reviews?", "tie at 40%"),
    ("what do customers think?", "does not name a product"),
])
def test_ask_flow(question, expect):
    at = fresh("views/ask.py")
    at.chat_input[0].set_value(question).run()
    assert not at.exception, at.exception
    text = all_text(at) + " " + " ".join(str(e.value) for e in at.get("expandable"))
    assert expect in text or any(expect in str(df.value.to_string()) for df in at.dataframe), text[:2000]
    # second question keeps history and renders without key collisions
    at.chat_input[0].set_value("Summarize the comfort feedback for shoe_b.").run()
    assert not at.exception, at.exception
    assert len(at.session_state["history"]) == 2
    assert len(at.session_state["telemetry"]) == 2


def test_suggestion_button():
    at = fresh("views/ask.py")
    at.button(key="sugg_0").click().run()
    assert not at.exception, at.exception
    assert len(at.session_state["history"]) == 1


def test_data_qa_quick_questions_and_pin():
    at = fresh("views/data_qa.py")
    for i in range(4):
        at.button(key=f"quick_{i}").click().run()
        assert not at.exception, at.exception
    assert len(at.session_state["sql_history"]) == 4
    assert all(r.ok for r in at.session_state["sql_history"])
    at.button(key="pin_0").click().run()
    assert not at.exception and len(at.session_state["exec_pins"]) == 1
    at.switch_page("views/exec_brief.py").run()
    assert not at.exception, at.exception
    assert "Pinned insights" in all_text(at)


def test_issue_map_to_ask_and_recommend():
    at = fresh("views/issue_map.py")
    at.selectbox(key="seg_product").set_value("jacket_b").run()
    at.selectbox(key="seg_topic").set_value("fit").run()
    assert not at.exception, at.exception
    at.button(key="map_rec").click().run()
    assert not at.exception, at.exception
    assert "Recommendations" in [t.value for t in at.title]
    assert at.selectbox(key="rec_choice").value == ("jacket_b", "fit")


def test_issue_map_ask_button():
    at = fresh("views/issue_map.py")
    at.button(key="map_ask").click().run()
    assert not at.exception, at.exception
    assert "Ask" in [t.value for t in at.title]
    assert len(at.session_state["history"]) == 1
    assert at.session_state["history"][0].route == "evidence"


def test_recommendation_accept_reaches_exec_brief():
    at = fresh("views/recommendations.py")
    btn = [b for b in at.button if b.label == "Draft recommendation"][0]
    btn.click().run()
    assert not at.exception, at.exception
    assert at.session_state["rec_draft"] is not None
    accept = [b for b in at.button if b.label == "Accept"][0]
    accept.click().run()
    assert not at.exception, at.exception
    assert at.session_state["rec_log"][0]["decision"] == "Accepted"
    at.switch_page("views/exec_brief.py").run()
    assert not at.exception, at.exception
    assert "Action:" in all_text(at)


def test_reject_requires_reason():
    at = fresh("views/recommendations.py")
    [b for b in at.button if b.label == "Draft recommendation"][0].click().run()
    [b for b in at.button if b.label == "Reject"][0].click().run()
    assert not at.exception, at.exception
    assert at.session_state["rec_log"] == []
    assert any("reason" in e.value for e in at.error)


def test_business_weights_change_ranking():
    at = fresh("views/recommendations.py")
    at.session_state["weights"] = {"shoe_b": 3.0}
    at.run()
    assert not at.exception, at.exception
    ranking = at.dataframe[1].value
    assert list(ranking.head(3)["product_id"]) == ["shoe_b"] * 3


def test_trust_center_simulation_and_reset():
    at = fresh("views/trust_center.py")
    assert any("match the notebook export exactly" in s.value for s in at.success)
    at.multiselect[0].set_value(["R005"]).run()
    [b for b in at.button if b.label == "Apply source change"][0].click().run()
    assert not at.exception, at.exception
    assert at.session_state["revoked"] == ["R005"]
    assert any("REVISE" in w.value for w in at.warning)
    # the whole app now ignores R005
    at.switch_page("views/ask.py").run()
    at.chat_input[0].set_value("How comfortable is shoe_a according to users?").run()
    assert "R005" not in at.session_state["history"][-1].cited_ids
    at.switch_page("views/trust_center.py").run()
    at.multiselect[1].set_value([("shoe_a", "comfort")]).run()
    [b for b in at.button if b.label == "Apply source change"][0].click().run()
    assert not at.exception, at.exception
    assert any("ABSTAIN" in e.value for e in at.error)
    [b for b in at.button if b.label == "Reset to original data"][0].click().run()
    assert at.session_state["revoked"] == [] and at.session_state["deleted"] == []


def test_arena_unanswerable_question():
    at = fresh("views/trust_center.py")
    at.selectbox[0].set_value("Q18").run()
    assert not at.exception, at.exception
    vals = [m.value for m in at.metric]
    assert "No — answered an unanswerable question" in vals and "Yes" in vals


def test_exec_followup_sql():
    at = fresh("views/exec_brief.py")
    at.text_input[0].input("Which product has the most negative durability reviews?")
    [b for b in at.button if b.label == "Ask"][0].click().run()
    assert not at.exception, at.exception
    assert any("tie" in m.value for m in at.markdown)
