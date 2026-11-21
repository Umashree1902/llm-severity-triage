"""Load the study data into a single SQLite database.

Usage:  python src/build_db.py            -> data/severity_triage.db (rebuilt from scratch, ~5 seconds)

Why SQLite: the study is a single-analyst, read-mostly workload of about 13,000 rows of defects and model
responses. SQLite needs no server, ships with Python, stores everything in one file that can be versioned and
shared, and is queried with standard SQL, so the same queries move unchanged to PostgreSQL if the triage
assistant is later run as a shared service.

Tables
  severity_levels      lookup: code, ordinal, label, Mozilla definition
  defects              analysis sample (2,498 defects, 2025-01 to 2026-06), one row per Bugzilla defect
  defect_subsets       which analysis subsets each defect belongs to (validation, test, rq2, rq3, reliability)
  training_defects     2021-2024 defects used only to train the TF-IDF baseline
  population_counts    eligible defects per severity in each time window (sampling frame)
  duplicate_pairs      sampled duplicates and the severity of the report they duplicate
  baseline_predictions TF-IDF and majority-class predictions for every sampled defect
  llm_runs             every LLM response: model, prompt condition, run number, label, confidence, latency
Views
  v_llm_vs_human       each LLM response joined to the human label, with exact match and signed error
  v_agreement          exact agreement and mean signed error per model, condition and run
"""
import json
import sqlite3

import pandas as pd

from common import PROCESSED, RAW, RESULTS, ROOT, ORDINAL, LABEL, SEVERITIES

DB = ROOT / "data" / "severity_triage.db"

DEFINITIONS = {
    "S1": "Catastrophic: blocks development or testing, affects a large number of users, or risks data loss "
          "or security; no workaround",
    "S2": "Serious: major functionality or product severely impaired and no satisfactory workaround",
    "S3": "Normal: blocks non-critical functionality and a workaround exists",
    "S4": "Small/trivial: minor significance, cosmetic issue or low or no impact to users",
}

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE severity_levels (
    code        TEXT PRIMARY KEY CHECK (code IN ('S1','S2','S3','S4')),
    ordinal     INTEGER NOT NULL,          -- 4 = most severe
    label       TEXT NOT NULL,
    definition  TEXT NOT NULL
);
CREATE TABLE defects (
    bug_id            INTEGER PRIMARY KEY,
    creation_time     TEXT NOT NULL,        -- ISO 8601, UTC
    product           TEXT, component TEXT, platform TEXT, op_sys TEXT, keywords TEXT, priority TEXT,
    status            TEXT, resolution TEXT, dupe_of INTEGER,
    summary           TEXT NOT NULL,
    description       TEXT NOT NULL,        -- first comment, anonymised
    description_llm   TEXT NOT NULL,        -- truncated to 3,000 characters for the LLM prompts
    desc_len          INTEGER, desc_truncated INTEGER, comment_count INTEGER,
    final_severity    TEXT NOT NULL REFERENCES severity_levels(code),   -- Y: the human label
    first_severity    TEXT REFERENCES severity_levels(code),
    severity_changes  INTEGER, days_to_triage REAL, set_at_filing INTEGER,
    weight            REAL NOT NULL          -- population count / sample count of its severity
);
CREATE TABLE defect_subsets (
    bug_id  INTEGER NOT NULL REFERENCES defects(bug_id),
    subset  TEXT NOT NULL CHECK (subset IN ('main','validation','test','rq2','rq3','reliability')),
    PRIMARY KEY (bug_id, subset)
);
CREATE TABLE training_defects (
    bug_id          INTEGER PRIMARY KEY,
    creation_time   TEXT, product TEXT, component TEXT,
    summary         TEXT, description_llm TEXT,
    final_severity  TEXT NOT NULL REFERENCES severity_levels(code)
);
CREATE TABLE population_counts (
    time_window  TEXT NOT NULL CHECK (time_window IN ('primary','baseline')),
    severity     TEXT NOT NULL REFERENCES severity_levels(code),
    n            INTEGER NOT NULL,
    PRIMARY KEY (time_window, severity)
);
CREATE TABLE duplicate_pairs (
    bug_id           INTEGER PRIMARY KEY,     -- duplicates found among all cleaned defects, not only the sample
    dupe_of          INTEGER NOT NULL,
    final_severity   TEXT NOT NULL, target_severity TEXT NOT NULL
);
CREATE TABLE baseline_predictions (
    bug_id     INTEGER NOT NULL REFERENCES defects(bug_id),
    model      TEXT NOT NULL CHECK (model IN ('tfidf','majority')),
    predicted  TEXT NOT NULL REFERENCES severity_levels(code),
    PRIMARY KEY (bug_id, model)
);
CREATE TABLE llm_runs (
    run_pk          INTEGER PRIMARY KEY AUTOINCREMENT,
    bug_id          INTEGER NOT NULL REFERENCES defects(bug_id),
    provider        TEXT NOT NULL,           -- ollama | openai | anthropic
    model           TEXT NOT NULL,           -- exact model version
    condition       TEXT NOT NULL,           -- P0, P1, P1_reworded_1, P1_reworded_2, P2
    run_id          INTEGER NOT NULL,        -- 1-3 for repeated identical prompts
    llm_severity    TEXT,                    -- NULL = invalid output, counted as wrong
    llm_confidence  REAL,
    raw             TEXT, seconds REAL, temperature REAL, timestamp TEXT,
    UNIQUE (bug_id, model, condition, run_id)
);
CREATE INDEX ix_runs_cond ON llm_runs(model, condition, run_id);
CREATE INDEX ix_defects_sev ON defects(final_severity);
CREATE INDEX ix_subsets_subset ON defect_subsets(subset);

CREATE VIEW v_llm_vs_human AS
SELECT r.bug_id, r.model, r.condition, r.run_id, d.final_severity AS human, r.llm_severity AS llm,
       r.llm_confidence, d.weight, d.product,
       (r.llm_severity = d.final_severity) AS exact_match,
       (SELECT ordinal FROM severity_levels WHERE code = r.llm_severity)
     - (SELECT ordinal FROM severity_levels WHERE code = d.final_severity) AS signed_error   -- + = over-escalation
FROM llm_runs r JOIN defects d USING (bug_id);

CREATE VIEW v_agreement AS
SELECT model, condition, run_id, COUNT(*) AS n,
       ROUND(AVG(COALESCE(exact_match, 0)), 4) AS exact_agreement,
       ROUND(AVG(signed_error), 3) AS mean_signed_error,
       ROUND(AVG(signed_error > 0), 3) AS over_escalation,
       ROUND(AVG(signed_error < 0), 3) AS under_escalation
FROM v_llm_vs_human GROUP BY model, condition, run_id;
"""


def main():
    DB.unlink(missing_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)

    con.executemany("INSERT INTO severity_levels VALUES (?,?,?,?)",
                    [(s, ORDINAL[s], LABEL[s], DEFINITIONS[s]) for s in SEVERITIES])

    s = pd.read_csv(PROCESSED / "sample.csv")
    cols = ["bug_id", "creation_time", "product", "component", "platform", "op_sys", "keywords", "priority",
            "status", "resolution", "dupe_of", "summary", "description", "description_llm", "desc_len",
            "desc_truncated", "comment_count", "final_severity", "first_severity", "severity_changes",
            "days_to_triage", "set_at_filing", "weight"]
    d = s[cols].copy()
    d["dupe_of"] = d.dupe_of.astype("Int64")
    d[["desc_truncated", "set_at_filing"]] = d[["desc_truncated", "set_at_filing"]].astype(int)
    d.to_sql("defects", con, if_exists="append", index=False)

    rows = [(b, sp) for b, sp in zip(s.bug_id, s.sample_split)]
    for col, name in (("in_rq2", "rq2"), ("in_rq3", "rq3"), ("in_reliability", "reliability")):
        rows += [(b, name) for b in s.loc[s[col].astype(bool), "bug_id"]]
    con.executemany("INSERT INTO defect_subsets VALUES (?,?)", rows)

    t = pd.read_csv(PROCESSED / "baseline_train.csv")
    t[[c for c in ["bug_id", "creation_time", "product", "component", "summary", "description_llm",
                   "final_severity"] if c in t]].to_sql("training_defects", con, if_exists="append", index=False)

    pop = json.load(open(RAW / "population_counts.json"))
    con.executemany("INSERT INTO population_counts VALUES (?,?,?)",
                    [(w, sev, n) for w in ("primary", "baseline") for sev, n in pop[w].items()])

    dp = pd.read_csv(PROCESSED / "dupe_pairs.csv")
    dp["dupe_of"] = dp.dupe_of.astype(int)
    dp.to_sql("duplicate_pairs", con, if_exists="append", index=False)

    b = pd.read_csv(RESULTS / "predictions_baseline.csv")
    b = b[b.bug_id.isin(s.bug_id)]
    con.executemany("INSERT INTO baseline_predictions VALUES (?,?,?)",
                    [(i, "tfidf", p) for i, p in zip(b.bug_id, b.pred_tfidf)] +
                    [(i, "majority", p) for i, p in zip(b.bug_id, b.pred_majority)])

    r = pd.read_json(RESULTS / "llm_runs.jsonl", lines=True)
    r = r[r.bug_id.isin(s.bug_id)].drop_duplicates(["bug_id", "model", "condition", "run_id"], keep="last")
    r = r[["bug_id", "provider", "model", "condition", "run_id", "llm_severity", "llm_confidence", "raw", "seconds",
           "temperature", "timestamp"]]
    r["timestamp"] = r.timestamp.astype(str)
    r.to_sql("llm_runs", con, if_exists="append", index=False)
    con.commit()

    print(f"Built {DB.relative_to(ROOT)}")
    for tbl in ("severity_levels", "defects", "defect_subsets", "training_defects", "population_counts",
                "duplicate_pairs", "baseline_predictions", "llm_runs"):
        print(f"  {tbl:22s} {con.execute(f'SELECT COUNT(*) FROM {tbl}').fetchone()[0]:>6,} rows")
    print("\nAgreement by model and condition (run 1):")
    for row in con.execute("SELECT * FROM v_agreement WHERE run_id = 1 ORDER BY model, condition"):
        print("  ", row)
    con.close()


if __name__ == "__main__":
    main()
