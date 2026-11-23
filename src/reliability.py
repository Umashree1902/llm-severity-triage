"""Analyse the returned labelling sheets of the human label reliability check.

Usage:  put the two returned workbooks in data/reliability/returned/ (any file names containing
        'Reviewer_A' and 'Reviewer_B'), then:  python src/reliability.py
Output: results/reliability.json

Reports, with 95% confidence intervals:
  - exact agreement and quadratic weighted kappa between the two reviewers (the reliability of a human label)
  - each reviewer, and the reviewers' consensus, against the Bugzilla label
  - Fleiss' kappa across reviewer A, reviewer B and the Bugzilla label
  - for each LLM (P1, run 1): agreement with the Bugzilla label and with the reviewer consensus on the same
    152 defects (secondary RQ1 analysis)
"""
import json
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score

from common import ROOT, RESULTS, SEVERITIES, SEED

REL = ROOT / "data" / "reliability"
ORD = {"S1": 4, "S2": 3, "S3": 2, "S4": 1}


def read_returned(tag):
    files = sorted((REL / "returned").glob(f"*{tag}*.xlsx"))
    if not files:
        raise SystemExit(f"No returned sheet for {tag} in {REL / 'returned'}")
    df = pd.read_excel(files[-1], sheet_name="Labels", header=1, dtype=str)
    df = df.rename(columns={"Severity (S1-S4)": "sev", "Confidence (1-3)": "conf"})[["Item", "sev", "conf"]]
    df["sev"] = df.sev.str.strip().str.upper()
    return df.rename(columns={"Item": "item", "sev": f"sev_{tag[-1]}", "conf": f"conf_{tag[-1]}"})


def qwk(a, b):
    return cohen_kappa_score(a, b, weights="quadratic", labels=SEVERITIES)


def boot_ci(f, *cols, n=2000):
    rng = np.random.RandomState(SEED)
    arr = [np.asarray(c) for c in cols]
    vals = []
    for _ in range(n):
        i = rng.randint(0, len(arr[0]), len(arr[0]))
        vals.append(f(*[a[i] for a in arr]))
    return [round(float(x), 4) for x in np.percentile(vals, [2.5, 97.5])]


def pair(a, b):
    ok = np.asarray(a) == np.asarray(b)
    ci = binomtest(int(ok.sum()), len(ok)).proportion_ci()
    return {"n": int(len(ok)), "exact_agreement": round(float(ok.mean()), 4),
            "exact_agreement_ci95": [round(ci.low, 4), round(ci.high, 4)],
            "qwk": round(float(qwk(a, b)), 4), "qwk_ci95": boot_ci(qwk, a, b),
            "within_one_level": round(float(np.mean([abs(ORD[x] - ORD[y]) <= 1 for x, y in zip(a, b)])), 4)}


def fleiss(labels):
    counts = np.array([[sum(1 for c in row if c == k) for k in SEVERITIES] for row in labels], float)
    n = counts.sum(1)[0]
    p_j = counts.sum(0) / counts.sum()
    P_i = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    return float((P_i.mean() - (p_j ** 2).sum()) / (1 - (p_j ** 2).sum()))


def main():
    key = pd.read_csv(REL / "key.csv")
    d = key.merge(read_returned("Reviewer_A"), on="item").merge(read_returned("Reviewer_B"), on="item")
    valid = d.sev_A.isin(SEVERITIES) & d.sev_B.isin(SEVERITIES)
    out = {"n_items": int(len(d)), "n_complete": int(valid.sum()), "missing_or_invalid": int((~valid).sum())}
    d = d[valid]
    out["reviewer_A_vs_B"] = pair(d.sev_A, d.sev_B)
    out["reviewer_A_vs_bugzilla"] = pair(d.sev_A, d.final_severity)
    out["reviewer_B_vs_bugzilla"] = pair(d.sev_B, d.final_severity)
    agree = d[d.sev_A == d.sev_B]
    out["consensus_vs_bugzilla"] = pair(agree.sev_A, agree.final_severity) | {
        "note": "defects where both reviewers agree"}
    out["fleiss_kappa_A_B_bugzilla"] = round(fleiss(d[["sev_A", "sev_B", "final_severity"]].values), 4)
    out["by_bugzilla_severity_A_vs_B_exact"] = {
        s: round(float((g.sev_A == g.sev_B).mean()), 3) for s, g in d.groupby("final_severity")}

    con = sqlite3.connect(ROOT / "data" / "severity_triage.db")
    runs = pd.read_sql("SELECT bug_id, model, llm_severity FROM llm_runs WHERE condition='P1' AND run_id=1", con)
    out["llm"] = {}
    for m, g in runs.groupby("model"):
        x = d.merge(g, on="bug_id")
        x = x[x.llm_severity.isin(SEVERITIES)]
        xa = x[x.sev_A == x.sev_B]
        out["llm"][m] = {"vs_bugzilla": pair(x.llm_severity, x.final_severity),
                         "vs_reviewer_A": pair(x.llm_severity, x.sev_A),
                         "vs_reviewer_B": pair(x.llm_severity, x.sev_B),
                         "vs_consensus": pair(xa.llm_severity, xa.sev_A) if len(xa) > 10 else None}
    json.dump(out, open(RESULTS / "reliability.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
