"""Paired comparison of the two LLMs and the TF-IDF baseline on the same defects (prompt P1, run 1).

Usage:  python src/compare_models.py        (needs data/severity_triage.db from build_db.py)
Output: results/model_comparison.json
"""
import json
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score, f1_score

from common import ROOT, RESULTS, SEVERITIES, SEED

MODELS = {"gpt": "gpt-4o-mini-2024-07-18", "llama": "llama3.1:8b"}


def qwk(a, b):
    return cohen_kappa_score(a, b, weights="quadratic", labels=SEVERITIES)


def mcnemar(ok_a, ok_b):
    b, c = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    return {"a_only_correct": b, "b_only_correct": c, "exact_p": float(binomtest(b, b + c).pvalue) if b + c else 1.0}


def main():
    con = sqlite3.connect(ROOT / "data" / "severity_triage.db")
    y = pd.read_sql("SELECT bug_id, final_severity FROM defects", con).set_index("bug_id").final_severity
    get = lambda m, c="P1", k=1: pd.read_sql(
        "SELECT bug_id, llm_severity FROM llm_runs WHERE model=? AND condition=? AND run_id=?",
        con, params=(m, c, k)).set_index("bug_id").llm_severity
    preds = {k: get(m) for k, m in MODELS.items()}
    preds["tfidf"] = pd.read_sql("SELECT bug_id, predicted FROM baseline_predictions WHERE model='tfidf'",
                                 con).set_index("bug_id").predicted
    ids = y.index
    for p in preds.values():
        ids = ids.intersection(p.index)
    Y = y[ids].values
    P = {k: v[ids].fillna("INVALID").values for k, v in preds.items()}

    out = {"n": int(len(ids)), "per_model": {}}
    for k, p in P.items():
        out["per_model"][k] = {
            "qwk": round(float(qwk(Y, p)), 4), "accuracy": round(float((Y == p).mean()), 4),
            "macro_f1": round(float(f1_score(Y, p, labels=SEVERITIES, average="macro")), 4),
            "recall": {c: round(float((p[Y == c] == c).mean()), 3) for c in SEVERITIES},
            "s1_labelled_s3_or_s4": int(np.isin(p[Y == "S1"], ["S3", "S4"]).sum()),
            "prediction_share": {c: round(float((p == c).mean()), 3) for c in SEVERITIES}}

    rng = np.random.RandomState(SEED)
    boots = {"gpt_minus_llama": [], "gpt_minus_tfidf": [], "llama_minus_tfidf": []}
    for _ in range(2000):
        i = rng.randint(0, len(Y), len(Y))
        q = {k: qwk(Y[i], p[i]) for k, p in P.items()}
        boots["gpt_minus_llama"].append(q["gpt"] - q["llama"])
        boots["gpt_minus_tfidf"].append(q["gpt"] - q["tfidf"])
        boots["llama_minus_tfidf"].append(q["llama"] - q["tfidf"])
    out["qwk_difference_ci95"] = {k: [round(float(x), 4) for x in np.percentile(v, [2.5, 97.5])]
                                  for k, v in boots.items()}
    ok = {k: P[k] == Y for k in P}
    out["accuracy_mcnemar"] = {"gpt_vs_llama": mcnemar(ok["gpt"], ok["llama"]),
                               "tfidf_vs_gpt": mcnemar(ok["tfidf"], ok["gpt"]),
                               "tfidf_vs_llama": mcnemar(ok["tfidf"], ok["llama"])}

    rq2 = pd.read_sql("SELECT bug_id FROM defect_subsets WHERE subset='rq2'", con).bug_id
    flips = {}
    for k, m in MODELS.items():
        runs = pd.concat([get(m, "P1", r).reindex(rq2) for r in (1, 2, 3)], axis=1)
        flips[k] = (runs.nunique(axis=1) > 1).values
    out["flip_rate"] = {k: round(float(v.mean()), 4) for k, v in flips.items()}
    out["flip_mcnemar_gpt_vs_llama"] = mcnemar(flips["gpt"], flips["llama"])   # 'correct' = flipped here
    json.dump(out, open(RESULTS / "model_comparison.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
