"""Routing policy for severity suggestions (the guardrail).

The study found that neither model is safe to trust on its own: the LLM labels more than half of S1 defects as
S3 or S4, the TF-IDF model finds only one S1 defect in four, and the LLM's stated confidence carries no
information (RQ4). The assistant therefore never sets the severity field itself. It posts a suggestion and
routes every defect to one of three queues:

  priority-review  either model suggests S1 or S2, or the report contains a risk signal (crash, regression,
                   data loss, security, hang ...). A triager looks at these first.
  review           the two models disagree. A triager decides with both suggestions in view.
  suggest          both models agree on S3 or S4 and there is no risk signal. The suggestion is pre-filled for
                   the triager to confirm in one click.

The rule uses only information available at filing time and does not use stated confidence.
"""
import re

RISK_TEXT = re.compile(
    r"\b(?:crash\w*|regress\w*|data ?loss|security|hangs?|hung|freez\w*|deadlock\w*|corrupt\w*|"
    r"won'?t (?:start|launch|open)|fails? to (?:start|load|launch|open)|unable to|cannot|can'?t|broken|"
    r"blank (?:page|screen)|leak\w*|assert\w*)\b", re.I)
RISK_KEYWORDS = re.compile(r"regression|crash|dataloss|hang|sec-|csectype", re.I)
HIGH = {"S1", "S2"}


def risk_signals(summary: str, keywords: str = "") -> list[str]:
    found = sorted({m.group(0).lower() for m in RISK_TEXT.finditer(summary or "")})
    found += sorted({m.group(0).lower() for m in RISK_KEYWORDS.finditer(keywords or "")})
    return found


def route(tfidf: str | None, llm: str | None, summary: str, keywords: str = "") -> dict:
    """Return the queue, the suggested severity and the reasons, for one defect."""
    reasons = []
    signals = risk_signals(summary, keywords)
    if tfidf in HIGH:
        reasons.append(f"TF-IDF model suggests {tfidf}")
    if llm in HIGH:
        reasons.append(f"LLM suggests {llm}")
    if signals:
        reasons.append("risk signal in report: " + ", ".join(signals))
    if reasons:
        queue = "priority-review"
    elif llm is not None and tfidf != llm:
        queue, reasons = "review", [f"models disagree (TF-IDF {tfidf}, LLM {llm})"]
    else:
        queue = "suggest"
        reasons = ["both models agree on a low severity and no risk signal" if llm else
                   "TF-IDF suggests a low severity and no risk signal (LLM not available)"]
    # the more severe of the two suggestions is shown, so the suggestion never under-rates relative to either model
    order = {"S1": 4, "S2": 3, "S3": 2, "S4": 1}
    cands = [s for s in (tfidf, llm) if s in order]
    suggestion = max(cands, key=order.get) if cands else None
    return {"queue": queue, "suggested_severity": suggestion, "reasons": reasons}
