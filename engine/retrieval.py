"""Retrieval methods and evaluation.

`tokens`, `retrieve` and `evaluate_retrieval` are copied from the notebook's shared helpers so the
app reproduces the notebook numbers exactly (lexical and review_rules, k = 3, precision denominator k).

`segment_retrieve` is the Review Signal method: route the question, then return EVERY evidence review
of the matched product x issue segment(s). Its precision uses the number of returned reviews as the
denominator (k = what was returned), recall uses the labelled relevant set, exactly like the notebook.
"""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

from engine.data import NO_EVIDENCE_PHRASES
from engine.router import route_question

SEGMENT_CAP = 30  # max evidence reviews per segment passed to the model


# ---- notebook helpers (verbatim logic) ------------------------------------------------------
def tokens(text):
    return set(re.findall(r'[a-z0-9_]+', text.lower()))


def retrieve(question_text, corpus, method='lexical', k=3):
    """No answerability/relevance labels enter retrieval. A bounded teaching baseline."""
    query = tokens(question_text)
    candidates = corpus
    if method == 'review_rules':
        lowered = question_text.lower()
        if not any(phrase in lowered for phrase in ['according to', 'what do users', 'summarize']):
            return []
        products = query & {r['product_id'] for r in corpus}
        topic = next((t for t, words in {'comfort': {'comfort', 'comfortable'},
                      'fit': {'fit', 'sizing'}, 'durability': {'durability', 'durable'}}.items() if query & words), None)
        if len(products) != 1 or topic is None:
            return []
        product = next(iter(products))
        candidates = [r for r in corpus if r['product_id'] == product and r['topic'] == topic
                      and not any(phrase in r['review_text'].lower() for phrase in NO_EVIDENCE_PHRASES)]
    elif method != 'lexical':
        raise ValueError('Unknown retrieval method')
    ranked = []
    for row in candidates:
        terms = tokens(row['review_text'])
        overlap = len(query & terms) / len(query | terms)
        if overlap > 0:
            ranked.append((overlap, row['review_id'], row))
    return [r for _, _, r in sorted(ranked, key=lambda x: (-x[0], x[1]))[:k]]


def evaluate_retrieval(questions, corpus, judgements, method, k=3):
    gold = defaultdict(set)
    for label in judgements:
        if label['relevant']:
            gold[label['question_id']].add(label['review_id'])
    per_question = []
    for question in questions:
        hits = retrieve(question['question_text'], corpus, method, k)
        ids = [r['review_id'] for r in hits]
        relevant = gold[question['question_id']]
        found = len(set(ids) & relevant)
        per_question.append(dict(question_id=question['question_id'], answerable=question['answerable'],
                                 evidence_ids=ids, precision_at_k=found/k,
                                 recall_at_k=found/len(relevant) if relevant else None,
                                 correct_abstention=(not ids) == (not question['answerable'])))
    return _summarise(per_question)


# ---- Review Signal method ---------------------------------------------------------------------
def segment_evidence(corpus: pd.DataFrame, products: list[str], topics: list[str],
                     cap: int = SEGMENT_CAP) -> pd.DataFrame:
    """All evidence reviews of the matched segments, newest first. Above `cap` per segment, keep a
    sample stratified by sentiment (true counts stay available in the segment table)."""
    ev = corpus[corpus["is_evidence"] & corpus["product_id"].isin(products) & corpus["topic"].isin(topics)]
    parts = []
    for _, g in ev.groupby(["product_id", "topic"], sort=True):
        g = g.sort_values(["review_date", "review_id"], ascending=[False, True])
        if len(g) > cap:
            keep = []
            for sentiment in sorted(g["sentiment"].unique()):
                s = g[g["sentiment"] == sentiment]
                keep.append(s.head(max(1, round(cap * len(s) / len(g)))))
            g = pd.concat(keep)
        parts.append(g)
    if not parts:
        return ev.iloc[0:0]
    return pd.concat(parts).sort_values(["product_id", "topic", "review_date", "review_id"],
                                        ascending=[True, True, False, True])


def segment_retrieve(question_text: str, corpus: pd.DataFrame, known_products: list[str],
                     known_topics: list[str]) -> list[str]:
    decision = route_question(question_text, known_products, known_topics)
    if decision.route != "evidence":
        return []
    return list(segment_evidence(corpus, decision.products, decision.topics)["review_id"])


def evaluate_segment(questions, corpus: pd.DataFrame, judgements, known_products, known_topics):
    gold = defaultdict(set)
    for label in judgements:
        if label['relevant']:
            gold[label['question_id']].add(label['review_id'])
    per_question = []
    for question in questions:
        ids = segment_retrieve(question['question_text'], corpus, known_products, known_topics)
        relevant = gold[question['question_id']]
        found = len(set(ids) & relevant)
        per_question.append(dict(question_id=question['question_id'], answerable=question['answerable'],
                                 evidence_ids=ids, precision_at_k=found/len(ids) if ids else 0.0,
                                 recall_at_k=found/len(relevant) if relevant else None,
                                 correct_abstention=(not ids) == (not question['answerable'])))
    return _summarise(per_question)


def _summarise(per_question):
    supported = [r for r in per_question if r['answerable']]
    unsupported = [r for r in per_question if not r['answerable']]
    return dict(n=len(per_question), answerable_n=len(supported), unsupported_n=len(unsupported),
                mean_precision_at_k_answerable=(sum(r['precision_at_k'] for r in supported)/len(supported)
                                                if supported else None),
                mean_recall_at_k_answerable=(sum(r['recall_at_k'] for r in supported)/len(supported)
                                             if supported else None),
                abstention_accuracy=sum(r['correct_abstention'] for r in per_question)/len(per_question),
                unsupported_answer_rate=(sum(bool(r['evidence_ids']) for r in unsupported)/len(unsupported)
                                         if unsupported else None),
                per_question=per_question)


def scorecard(tables: dict, corpus: pd.DataFrame) -> pd.DataFrame:
    """Metrics for the three methods on development and test splits."""
    from engine.data import permitted_records

    perm = permitted_records(corpus)
    questions = tables["questions"].to_dict("records")
    judgements = tables["relevance_labels"].to_dict("records")
    products = sorted(corpus["product_id"].unique())
    topics = sorted(corpus["topic"].unique())
    rows = []
    for split in ("development", "test"):
        qs = [q for q in questions if q["split"] == split]
        for method, label in (("lexical", "Keyword top-3 (proxy for current tool)"),
                              ("review_rules", "Notebook review_rules top-3"),
                              ("segment", "Review Signal: whole segment")):
            if method == "segment":
                res = evaluate_segment(qs, corpus, judgements, products, topics)
            else:
                res = evaluate_retrieval(qs, perm, judgements, method, k=3)
            rows.append({"split": split, "method": method, "label": label, "n": res["n"],
                         "precision": res["mean_precision_at_k_answerable"],
                         "recall": res["mean_recall_at_k_answerable"],
                         "abstention_accuracy": res["abstention_accuracy"],
                         "unsupported_answer_rate": res["unsupported_answer_rate"],
                         "per_question": res["per_question"]})
    return pd.DataFrame(rows)
