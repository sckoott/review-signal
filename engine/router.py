"""Question router and entity resolution.

Routes:
  evidence  - what customers say (review summaries)        -> segment context
  sql       - counts, rankings, shares                      -> Genie (or local SQL fallback)
  abstain   - out of scope, unknown product, or no scope    -> honest refusal + owner to ask

Deterministic on purpose: fast, testable, and explainable in the context inspector. Uses no
gold labels (answerable / relevance) - those only enter evaluation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TOPIC_SYNONYMS: dict[str, tuple[str, ...]] = {
    "comfort": ("comfort", "comfortable", "uncomfortable", "comfy", "cushioning", "cushion"),
    "fit": ("fit", "fits", "fitting", "sizing", "size", "sizes", "runs small", "runs large", "true to size"),
    "durability": ("durability", "durable", "longevity", "lasts", "lasting", "holds up", "falls apart",
                   "wear and tear"),
}

# Requests that customer reviews cannot answer, with the team that owns the answer.
OUT_OF_SCOPE: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("pricing", ("price", "prices", "pricing", "cost", "costs", "wholesale", "margin", "margins", "msrp",
                 "discount", "discounts"), "Pricing & Commercial team"),
    ("strategy", ("roadmap", "competitor", "competitors", "competitive", "confidential", "unreleased",
                  "launch date", "upcoming release"), "Product Strategy / Roadmap owners"),
    ("warranty", ("warranty", "warranties", "guarantee", "guaranteed", "return policy", "refund", "refunds"),
     "Customer Service & Legal (warranty terms)"),
    ("commercial data", ("stock", "inventory", "revenue", "sales figures", "units sold", "sell-through"),
     "Commercial Analytics / Supply Chain"),
)

QUANT_PHRASES = ("how many", "number of", "count", "which product", "which products", "which item", "rank",
                 "ranking", "most negative", "most positive", "least", "highest", "lowest", "share",
                 "percentage", "percent", "proportion", "breakdown", "distribution", "top 3", "top three",
                 "worst", "best performing", "per product", "by product", "by topic", "by issue")
REVIEW_PHRASES = ("what do users", "what do customers", "according to", "summarize", "summarise", "feedback",
                  "say about", "reviews say", "opinion", "complain", "complaints", "how comfortable",
                  "how durable", "how does", "how do", "tell me about")
GENERIC_REVIEW_WORDS = ("review", "reviews", "users", "customers", "feedback", "people")

# "shoe_a", "shoe-a", "Shoe_A" always count. With a space, only "Shoe A" (capital letter) or a lone
# letter at the end of a clause ("... of shoe a?") counts, so "a jacket a bit tight" is not jacket_a.
PRODUCT_PATTERN = re.compile(r"\b(?i:(shoe|jacket))(?:[_-]((?i:[a-z]))|\s([A-Z])|\s([a-z])(?=\s*(?:[?.!,;:)]|$)))(?!\w)")
FAMILY_PATTERN = re.compile(r"\b(shoes|jackets|shoe|jacket)\b")


@dataclass
class RouteDecision:
    route: str  # evidence | sql | abstain
    reason: str
    products: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    matched: dict[str, list[str]] = field(default_factory=dict)
    abstain_kind: str | None = None  # out_of_scope | unknown_product | no_scope | no_evidence
    owner: str | None = None


def _contains(text: str, phrase: str) -> bool:
    return re.search(r"(?<![a-z0-9_])" + re.escape(phrase) + r"(?![a-z0-9_])", text) is not None


def resolve_entities(question: str, known_products: list[str]) -> tuple[list[str], list[str], list[str],
                                                                         dict[str, list[str]]]:
    """Returns (known products, unknown product mentions, topics, matched terms)."""
    q = question.lower()
    matched: dict[str, list[str]] = {}
    mentioned = []
    for m in PRODUCT_PATTERN.finditer(question):  # original case: "Shoe A" vs "a shoe a bit"
        letter = next(g for g in m.groups()[1:] if g)
        pid = f"{m.group(1).lower()}_{letter.lower()}"
        if pid not in mentioned:
            mentioned.append(pid)
    known = [p for p in mentioned if p in known_products]
    unknown = [p for p in mentioned if p not in known_products]
    if not mentioned:  # family words without a letter: "shoes", "jackets"
        families = {m.group(1).rstrip("s") for m in FAMILY_PATTERN.finditer(q)}
        if families:
            known = [p for p in known_products if p.split("_")[0] in families]
            matched["product family"] = sorted(families)
    if known:
        matched["products"] = known
    topics = []
    for topic, words in TOPIC_SYNONYMS.items():
        hits = [w for w in words if _contains(q, w)]
        if hits:
            topics.append(topic)
            matched[f"topic:{topic}"] = hits
    return known, unknown, topics, matched


def route_question(question: str, known_products: list[str], known_topics: list[str]) -> RouteDecision:
    q = question.lower().strip()
    products, unknown, topics, matched = resolve_entities(question, known_products)
    topics = [t for t in topics if t in known_topics]

    if not q:
        return RouteDecision("abstain", "Empty question.", abstain_kind="no_scope")

    if unknown:
        return RouteDecision(
            "abstain",
            f"No reviews exist for {', '.join(unknown)}. Only {', '.join(known_products)} have permitted reviews; "
            "the assistant does not substitute the closest product.",
            products, topics, matched, abstain_kind="unknown_product",
            owner="Product team that owns the item (no customer reviews yet)")

    for label, words, owner in OUT_OF_SCOPE:
        hits = [w for w in words if _contains(q, w)]
        if hits:
            matched["out_of_scope"] = hits
            return RouteDecision(
                "abstain",
                f"This is a {label} question ('{hits[0]}'). Customer reviews are not a reliable source for it.",
                products, topics, matched, abstain_kind="out_of_scope", owner=owner)

    quant_hits = [p for p in QUANT_PHRASES if _contains(q, p)] + (["%"] if "%" in q else [])
    review_hits = [p for p in REVIEW_PHRASES if _contains(q, p)]
    if quant_hits and not review_hits:
        matched["quantitative"] = quant_hits
        return RouteDecision("sql", f"Counting / ranking question ('{quant_hits[0]}'): answered with SQL over the "
                             "permitted review tables.", products, topics, matched)

    if products or topics:
        products = products or list(known_products)
        topics = topics or list(known_topics)
        scope = f"{len(products)} product(s) x {len(topics)} issue(s)"
        return RouteDecision("evidence", f"Review-summary question. Scope: {scope}.", products, topics, matched)

    if any(_contains(q, w) for w in GENERIC_REVIEW_WORDS):
        return RouteDecision(
            "abstain", "The question does not name a product or an issue. Name one (e.g. 'fit of shoe_b') so the "
            "answer uses the right reviews.", abstain_kind="no_scope", owner=None, matched=matched)

    return RouteDecision(
        "abstain", "This does not look like a question about customer reviews of "
        f"{', '.join(known_products)} ({', '.join(known_topics)}).", abstain_kind="out_of_scope",
        owner=None, matched=matched)
