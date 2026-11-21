"""Tests for the triage assistant. Run: LLM_PROVIDER=none pytest -q"""
import os
import tempfile

os.environ.setdefault("LLM_PROVIDER", "none")
os.environ["TRIAGE_LOG_DB"] = os.path.join(tempfile.mkdtemp(), "log.db")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.policy import route  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_triage_returns_suggestion_and_queue():
    r = client.post("/api/triage", json={"summary": "Typo in the about dialog", "description": "Says 'Firefx'.",
                                         "product": "Firefox", "component": "General"})
    d = r.json()
    assert r.status_code == 200
    assert d["tfidf"]["severity"] in {"S1", "S2", "S3", "S4"}
    assert abs(sum(d["tfidf"]["probabilities"].values()) - 1) < 0.01
    assert d["queue"] in {"priority-review", "review", "suggest"}
    assert d["log_id"] > 0


def test_risk_signal_forces_priority_review():
    d = client.post("/api/triage", json={"summary": "Browser crashes on startup after update"}).json()
    assert d["queue"] == "priority-review"
    assert any("crash" in r for r in d["reasons"])


def test_policy_rules():
    assert route("S3", "S1", "Button misaligned")["queue"] == "priority-review"
    assert route("S3", "S4", "Button misaligned")["queue"] == "review"
    assert route("S4", "S4", "Button misaligned")["queue"] == "suggest"
    assert route("S4", "S3", "Button misaligned")["suggested_severity"] == "S3"   # never the less severe one


def test_decision_and_log():
    d = client.post("/api/triage", json={"summary": "Scrollbar colour wrong in dark mode"}).json()
    assert client.post(f"/api/triage/{d['log_id']}/decision", json={"final_severity": "S4"}).status_code == 200
    assert client.post(f"/api/triage/{d['log_id']}/decision", json={"final_severity": "high"}).status_code == 422
    log = client.get("/api/log?n=5").json()
    assert any(row["id"] == d["log_id"] and row["final_severity"] == "S4" for row in log)


def test_bugzilla_webhook_dry_run():
    payload = {"event": {"action": "create"},
               "bug": {"id": 123, "summary": "Video playback hangs on YouTube", "product": "Core",
                       "component": "Audio/Video", "keywords": ["regression"]},
               "comment": {"body": "Since the last nightly, playback freezes after 5 seconds."}}
    r = client.post("/webhooks/bugzilla", json=payload).json()
    assert r["posted"] is False and "Suggested severity" in r["comment"]
    assert r["result"]["queue"] == "priority-review"


def test_bad_input_rejected():
    assert client.post("/api/triage", json={"summary": ""}).status_code == 422
    assert client.post("/webhooks/bugzilla", json={"bug": {}}).status_code == 422
