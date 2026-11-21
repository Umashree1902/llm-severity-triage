"""Model loading, prediction and audit logging for the triage assistant."""
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_llm import SYSTEM, call, parse  # noqa: E402  (same prompts and parsing as the study)

from .policy import route  # noqa: E402

MODEL_PATH = Path(os.environ.get("TFIDF_MODEL", ROOT / "models" / "tfidf_lr.joblib"))
LOG_DB = Path(os.environ.get("TRIAGE_LOG_DB", ROOT / "data" / "triage_log.db"))
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")          # ollama | openai | anthropic | none
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.1:8b")
P1 = (ROOT / "prompts" / "P1.txt").read_text()
MAX_DESC = 3000


class Triage:
    def __init__(self):
        bundle = joblib.load(MODEL_PATH)
        self.pipe = bundle["pipeline"]
        self.model_info = {k: v for k, v in bundle.items() if k != "pipeline"}
        LOG_DB.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(LOG_DB) as con:
            con.execute("""CREATE TABLE IF NOT EXISTS triage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, received_at TEXT, source TEXT, bug_id INTEGER,
                summary TEXT, tfidf TEXT, tfidf_probs TEXT, llm TEXT, llm_model TEXT, llm_seconds REAL,
                queue TEXT, suggested TEXT, reasons TEXT, final_severity TEXT)""")

    def tfidf(self, summary, description):
        text = f"{summary} {description[:MAX_DESC]}".lower()
        probs = self.pipe.predict_proba([text])[0]
        classes = list(self.pipe.classes_)
        return classes[probs.argmax()], {c: round(float(p), 3) for c, p in zip(classes, probs)}

    def llm(self, d):
        if LLM_PROVIDER == "none":
            return None, None, "LLM disabled"
        prompt = P1.format(summary=d["summary"], description=d["description"][:MAX_DESC],
                           product=d.get("product") or "", component=d.get("component") or "",
                           platform=d.get("platform") or "", op_sys=d.get("op_sys") or "",
                           keywords=(d.get("keywords") or "none").replace(";", ", "))
        try:
            text, secs = call(LLM_PROVIDER, LLM_MODEL, prompt, None)
        except Exception as e:  # noqa: BLE001  - the assistant must still answer if the LLM is down
            return None, None, f"LLM unavailable: {type(e).__name__}"
        sev, _ = parse(text)
        return sev, secs, None

    def triage(self, d: dict, source="api"):
        t0 = time.time()
        tf, probs = self.tfidf(d["summary"], d.get("description", ""))
        llm, secs, llm_note = self.llm(d)
        decision = route(tf, llm, d["summary"], d.get("keywords") or "")
        if llm_note:
            decision["reasons"].append(llm_note)
        out = {"bug_id": d.get("bug_id"),
               "tfidf": {"severity": tf, "probabilities": probs},
               "llm": {"severity": llm, "model": None if LLM_PROVIDER == "none" else LLM_MODEL, "seconds": secs},
               **decision,
               "note": "Suggestion only. A triager confirms or changes the severity; the assistant never sets it.",
               "latency_seconds": round(time.time() - t0, 3)}
        with sqlite3.connect(LOG_DB) as con:
            cur = con.execute("""INSERT INTO triage_log (received_at, source, bug_id, summary, tfidf, tfidf_probs, llm,
                           llm_model, llm_seconds, queue, suggested, reasons) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (datetime.now(timezone.utc).isoformat(timespec="seconds"), source, d.get("bug_id"),
                         d["summary"], tf, json.dumps(probs), llm, out["llm"]["model"], secs, decision["queue"],
                         decision["suggested_severity"], json.dumps(decision["reasons"])))
            out["log_id"] = cur.lastrowid
        return out

    def record_decision(self, log_id: int, final_severity: str):
        with sqlite3.connect(LOG_DB) as con:
            con.execute("UPDATE triage_log SET final_severity=? WHERE id=?", (final_severity, log_id))

    def recent(self, n=20):
        with sqlite3.connect(LOG_DB) as con:
            con.row_factory = sqlite3.Row
            return [dict(r) for r in con.execute("SELECT * FROM triage_log ORDER BY id DESC LIMIT ?", (n,))]


def bugzilla_comment(result: dict) -> str:
    """Text the assistant would post on the Bugzilla bug (it never changes the severity field)."""
    lines = [f"[Triage assistant] Suggested severity: {result['suggested_severity']} "
             f"(queue: {result['queue']})",
             f"- TF-IDF model: {result['tfidf']['severity']}",
             f"- LLM ({result['llm']['model']}): {result['llm']['severity']}" if result["llm"]["model"] else
             "- LLM: not available",
             "- Reasons: " + "; ".join(result["reasons"]),
             "This is a suggestion for the triager to confirm; it does not change the Severity field."]
    return "\n".join(lines)


_ = SYSTEM  # imported so a missing prompt file fails at start-up, not on the first request
