"""Evaluate LLM runs against human labels and the baselines (preliminary results for the interim report).

Usage:  python src/evaluate.py
Inputs: results/llm_runs.jsonl, data/processed/sample.csv, results/predictions_baseline.csv
Outputs: results/llm_metrics.csv, results/llm_metrics.json,
         results/figures/fig10_*.png (confusion), fig11_signed_error.png, fig12_model_comparison.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import PROCESSED, RESULTS, FIGURES, SEVERITIES, ORDINAL
from metrics import all_metrics


def main():
    runs = pd.read_json(RESULTS / "llm_runs.jsonl", lines=True)
    runs = runs[runs.run_id == 1]                         # first run only for agreement and bias
    sample = pd.read_csv(PROCESSED / "sample.csv")[["bug_id", "final_severity", "weight"]]
    base = pd.read_csv(RESULTS / "predictions_baseline.csv")[["bug_id", "pred_tfidf", "pred_majority"]]
    df = runs.merge(sample, on="bug_id").merge(base, on="bug_id", how="left")
    pilot_ids = df.bug_id.unique()

    rows, out = [], {"n_defects": int(len(pilot_ids)), "by_config": {}}
    ref = sample[sample.bug_id.isin(pilot_ids)].merge(base, on="bug_id")
    for name, col in [("TF-IDF baseline", "pred_tfidf"), ("Majority class", "pred_majority")]:
        m = all_metrics(ref.final_severity, ref[col], n_boot=500)
        out["by_config"][name] = m
        rows.append({"model": name, "condition": "-", **{k: v for k, v in m.items() if not isinstance(v, (dict, list))}})
    for (model, cond), g in df.groupby(["model", "condition"]):
        m = all_metrics(g.final_severity, g.llm_severity, n_boot=500)
        m["mean_seconds_per_call"] = round(float(g.seconds.dropna().mean()), 2)
        m["mean_confidence"] = round(float(g.llm_confidence.dropna().mean()), 3) if g.llm_confidence.notna().any() else None
        out["by_config"][f"{model} | {cond}"] = m
        rows.append({"model": model, "condition": cond, **{k: v for k, v in m.items() if not isinstance(v, (dict, list))}})

        # confusion matrix
        cm = pd.crosstab(g.final_severity, g.llm_severity.fillna("Invalid")).reindex(
            index=SEVERITIES, columns=SEVERITIES + ["Invalid"], fill_value=0)
        cm = cm.loc[:, (cm.sum() > 0) | cm.columns.isin(SEVERITIES)]
        fig, ax = plt.subplots(figsize=(4.4, 3.6))
        im = ax.imshow(cm.values, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm.values[i, j], ha="center", va="center",
                        color="white" if cm.values[i, j] > cm.values.max() / 2 else "black")
        ax.set_xticks(range(cm.shape[1]), cm.columns); ax.set_yticks(range(4), SEVERITIES)
        ax.set_xlabel(f"Predicted ({model}, {cond})"); ax.set_ylabel("Human final severity")
        fig.colorbar(im, ax=ax, shrink=.8); fig.tight_layout()
        safe = model.replace(":", "-").replace("/", "-")
        fig.savefig(FIGURES / f"fig10_confusion_{safe}_{cond}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # RQ3 preview: signed error and the effect of context (P0 vs P1) on the same defects
    tests = {}
    for model, g in df.groupby("model"):
        piv = g.pivot_table(index="bug_id", columns="condition", values="llm_severity", aggfunc="first")
        truth = sample.set_index("bug_id").final_severity
        for c in piv.columns:
            piv[c + "_err"] = piv[c].map(ORDINAL) - truth.reindex(piv.index).map(ORDINAL)
        res = {}
        for c in [c for c in ("P0", "P1") if c in piv.columns]:
            e = piv[c + "_err"].dropna()
            if (e != 0).sum() > 0:
                st = wilcoxon(e[e != 0])
                res[f"signed_error_{c}"] = {"median": float(e.median()), "mean": round(float(e.mean()), 3),
                                            "wilcoxon_W": float(st.statistic), "p_value": float(st.pvalue), "n": int(len(e))}
        if {"P0_err", "P1_err"} <= set(piv.columns):
            pair = piv[["P0_err", "P1_err"]].dropna().abs()
            d = pair.P0_err - pair.P1_err
            if (d != 0).sum() > 0:
                st = wilcoxon(d[d != 0])
                res["abs_error_P0_vs_P1"] = {"mean_abs_P0": round(float(pair.P0_err.mean()), 3),
                                             "mean_abs_P1": round(float(pair.P1_err.mean()), 3),
                                             "wilcoxon_W": float(st.statistic), "p_value": float(st.pvalue),
                                             "n": int(len(pair))}
        tests[model] = res

        # signed-error distribution chart
        fig, ax = plt.subplots(figsize=(6, 3.2))
        vals = range(-3, 4)
        width = .38
        for k, c in enumerate([c for c in ("P0", "P1") if c + "_err" in piv.columns]):
            e = piv[c + "_err"].dropna()
            share = [(e == v).mean() * 100 for v in vals]
            ax.bar(np.array(list(vals)) + (k - .5) * width, share, width, label=c)
        ax.set_xticks(list(vals)); ax.set_xlabel("Signed error (LLM ordinal score − human ordinal score)")
        ax.set_ylabel("Share of defects (%)"); ax.axvline(0, color="grey", lw=.8)
        ax.legend(frameon=False, title=model); fig.tight_layout()
        safe = model.replace(":", "-").replace("/", "-")
        fig.savefig(FIGURES / f"fig11_signed_error_{safe}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    out["rq3_tests"] = tests

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "llm_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    short = {"TF-IDF baseline": "TF-IDF\nbaseline", "Majority class": "Majority\nclass"}
    labels = [short.get(r.model, r.model) if r.condition == "-" else
              f"{r.model.split(':')[0]}\n{r.condition.replace('P1_reworded_', 'P1 reword ')}" for r in table.itertuples()]
    x = np.arange(len(table))
    ax.bar(x - .2, table.macro_f1, .4, label="Macro-F1"); ax.bar(x + .2, table.qwk, .4, label="Weighted kappa")
    ax.set_xticks(x, labels, fontsize=8); ax.set_ylabel("Score"); ax.set_ylim(min(0, table.qwk.min() - .05), max(.6, table.macro_f1.max() + .1))
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(FIGURES / "fig12_model_comparison.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    json.dump(out, open(RESULTS / "llm_metrics.json", "w"), indent=2, default=str)
    print(table.to_string(index=False)); print(json.dumps(tests, indent=2))


if __name__ == "__main__":
    main()
