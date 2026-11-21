"""Offline evaluation of the triage assistant's routing policy (app/policy.py) on the study data.

Usage:  python src/evaluate_policy.py [--model llama3.1:8b]
Reads data/severity_triage.db (build it first with build_db.py). Writes results/policy_eval_<model>.json and
results/figures/fig17_policy_queues_<model>.png.

For every sampled defect, the stored TF-IDF prediction and the stored LLM P1 prediction (run 1) are passed through
the same route() function the service uses. The policy was fixed before this evaluation; results are reported on
the held-out RQ4 test subset and, for workload, on the full sample weighted to the population mix.
"""
import argparse
import json
import sqlite3
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from common import ROOT, RESULTS, FIGURES

sys.path.insert(0, str(ROOT))
from app.policy import route  # noqa: E402

QUEUES = ["priority-review", "review", "suggest"]


def summarise(df):
    out = {}
    W = df.weight.sum()
    for q in QUEUES:
        g = df[df.queue == q]
        n_s1 = int((g.y == "S1").sum())
        out[q] = {"n": int(len(g)), "share_sample": round(len(g) / len(df), 3),
                  "share_population": round(g.weight.sum() / W, 3),
                  "suggestion_accuracy": round(float((g.suggested == g.y).mean()), 3) if len(g) else None,
                  "suggestion_accuracy_population": round(float(((g.suggested == g.y) * g.weight).sum() /
                                                                g.weight.sum()), 3) if len(g) else None,
                  "s1_defects": n_s1, "s2_defects": int((g.y == "S2").sum())}
    s1 = df[df.y == "S1"]
    k = int((s1.queue == "suggest").sum())
    out["s1_total"] = int(len(s1))
    out["s1_routed_to_suggest"] = k
    out["s1_routed_to_suggest_ci95"] = [round(x, 3) for x in binomtest(k, len(s1)).proportion_ci()] if len(s1) else None
    out["s1_or_s2_reaching_priority_review"] = round(float((df[df.y.isin(["S1", "S2"])].queue == "priority-review").mean()), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    args = ap.parse_args()
    con = sqlite3.connect(ROOT / "data" / "severity_triage.db")
    df = pd.read_sql("""
        SELECT d.bug_id, d.summary, d.keywords, d.final_severity AS y, d.weight, b.predicted AS tfidf,
               r.llm_severity AS llm,
               EXISTS (SELECT 1 FROM defect_subsets s WHERE s.bug_id = d.bug_id AND s.subset = 'test') AS in_test
        FROM defects d
        JOIN baseline_predictions b ON b.bug_id = d.bug_id AND b.model = 'tfidf'
        JOIN llm_runs r ON r.bug_id = d.bug_id AND r.condition = 'P1' AND r.run_id = 1 AND r.model = ?""",
                     con, params=(args.model,))
    df["keywords"] = df.keywords.fillna(""); df["summary"] = df.summary.fillna("")
    df["llm"] = df.llm.where(df.llm.notna(), None)
    dec = [route(t, l, s, k) for t, l, s, k in zip(df.tfidf, df.llm, df.summary, df.keywords)]
    df["queue"] = [d["queue"] for d in dec]
    df["suggested"] = [d["suggested_severity"] for d in dec]
    res = {"model": args.model, "test_subset": summarise(df[df.in_test == 1]), "full_sample": summarise(df)}
    slug = args.model.split(":")[0].replace(".", "")
    json.dump(res, open(RESULTS / f"policy_eval_{slug}.json", "w"), indent=2)
    print(json.dumps(res, indent=2))

    # stacked bars: what each queue holds, by human severity (full sample, population-weighted)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    colors = {"S1": "#b3261e", "S2": "#d9711c", "S3": "#2f5d9f", "S4": "#9aa3b2"}
    left = np.zeros(len(QUEUES))
    W = df.weight.sum()
    for sev in ["S1", "S2", "S3", "S4"]:
        vals = np.array([df[(df.queue == q) & (df.y == sev)].weight.sum() / W * 100 for q in QUEUES])
        ax.barh(QUEUES, vals, left=left, color=colors[sev], label=f"Human {sev}")
        left += vals
    for i, v in enumerate(left):
        ax.text(v + 0.8, i, f"{v:.0f}%", va="center", fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, max(left) + 10)
    ax.set_xlabel("Share of incoming defects (population-weighted, %)")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(FIGURES / f"fig17_policy_queues_{slug}.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
