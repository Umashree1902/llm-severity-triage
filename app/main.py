"""Severity triage assistant: a small web service around the study's models and guardrail.

Run locally:
    pip install -r requirements.txt -r app/requirements.txt
    uvicorn app.main:app --port 8000            # then open http://localhost:8000

Environment:
    LLM_PROVIDER  ollama (default) | openai | anthropic | none
    LLM_MODEL     llama3.1:8b (default) or e.g. gpt-4o-mini-2024-07-18
    LLM_API_KEY / ANTHROPIC_API_KEY for API providers; OLLAMA_URL for a non-default Ollama server
    BUGZILLA_POST 1 to post suggestion comments back to Bugzilla (default: dry run, comment returned only)
    BUGZILLA_API_KEY, BUGZILLA_URL (default https://bugzilla.mozilla.org)
"""
import os
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .service import Triage, bugzilla_comment

app = FastAPI(title="Defect Severity Triage Assistant", version="1.0",
              description="Suggests a Mozilla-style severity (S1-S4) for a new defect and routes it to a review "
                          "queue. Suggestions only: a triager always confirms the severity.")
triager = Triage()
STATIC = Path(__file__).parent / "static"


class Defect(BaseModel):
    bug_id: int | None = None
    summary: str = Field(..., min_length=3, examples=["Firefox crashes when opening a PDF with forms"])
    description: str = Field("", examples=["Steps to reproduce: open the attached PDF ..."])
    product: str | None = Field(None, examples=["Firefox"])
    component: str | None = Field(None, examples=["PDF Viewer"])
    platform: str | None = None
    op_sys: str | None = None
    keywords: str | None = Field(None, examples=["regression"])


class Decision(BaseModel):
    final_severity: str = Field(..., pattern="^S[1-4]$")


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "tfidf_model": triager.model_info,
            "llm_provider": os.environ.get("LLM_PROVIDER", "ollama"),
            "llm_model": os.environ.get("LLM_MODEL", "llama3.1:8b")}


@app.post("/api/triage")
def triage(d: Defect):
    """Suggest a severity and a review queue for one defect."""
    return triager.triage(d.model_dump(), source="api")


@app.post("/api/triage/{log_id}/decision")
def decision(log_id: int, body: Decision):
    """Record the severity the triager finally chose, so agreement can be monitored after deployment."""
    triager.record_decision(log_id, body.final_severity)
    return {"ok": True}


@app.get("/api/log")
def log(n: int = 20):
    """Most recent suggestions (audit trail)."""
    return triager.recent(n)


@app.post("/webhooks/bugzilla")
def bugzilla_webhook(payload: dict):
    """Receives a Bugzilla (BMO) webhook for a new bug and returns, or posts, a suggestion comment.

    Expected payload (BMO webhook format): {"event": {"action": "create"}, "bug": {"id", "summary", "product",
    "component", "platform", "op_sys", "keywords": [...]}, "comment": {"body": "..."}} - "comment" is optional.
    """
    bug = payload.get("bug") or {}
    if not bug.get("summary"):
        raise HTTPException(422, "payload has no bug.summary")
    kw = bug.get("keywords") or []
    d = {"bug_id": bug.get("id"), "summary": bug["summary"],
         "description": (payload.get("comment") or {}).get("body", ""),
         "product": bug.get("product"), "component": bug.get("component"), "platform": bug.get("platform"),
         "op_sys": bug.get("op_sys"), "keywords": ";".join(kw) if isinstance(kw, list) else kw}
    result = triager.triage(d, source="bugzilla-webhook")
    comment = bugzilla_comment(result)
    posted = False
    if os.environ.get("BUGZILLA_POST") == "1" and d["bug_id"]:
        base = os.environ.get("BUGZILLA_URL", "https://bugzilla.mozilla.org")
        r = requests.post(f"{base}/rest/bug/{d['bug_id']}/comment", timeout=30,
                          headers={"X-BUGZILLA-API-KEY": os.environ["BUGZILLA_API_KEY"]},
                          json={"comment": comment, "is_private": False})
        posted = r.ok
    return {"result": result, "comment": comment, "posted": posted}


@app.post("/webhooks/jira")
def jira_webhook(payload: dict):
    """Same flow for a Jira 'issue_created' webhook (fields.summary / fields.description)."""
    issue = payload.get("issue") or {}
    f = issue.get("fields") or {}
    if not f.get("summary"):
        raise HTTPException(422, "payload has no issue.fields.summary")
    desc = f.get("description") or ""
    d = {"bug_id": None, "summary": f["summary"], "description": desc if isinstance(desc, str) else str(desc),
         "product": (f.get("project") or {}).get("name"),
         "component": ", ".join(c.get("name", "") for c in f.get("components") or []),
         "keywords": ";".join(f.get("labels") or [])}
    result = triager.triage(d, source="jira-webhook")
    return {"issue_key": issue.get("key"), "result": result, "comment": bugzilla_comment(result)}
