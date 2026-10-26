"""RQ1 hypothesis tests, RQ2 (repeatability and prompt sensitivity) and RQ4 (guardrails, calibration, deferral).

Usage:  python src/evaluate_rq2_rq4.py
Inputs: results/llm_runs.jsonl, data/processed/sample.csv, results/predictions_baseline.csv
Outputs: results/rq_tests.json, results/figures/fig13_rq2_stability.png,
         fig14_reliability_diagram.png, fig15_coverage_accuracy.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import cohen_kappa_score

from common import PROCESSED, RESULTS, FIGURES, SEVERITIES, ORDINAL, SEED

TARGET_ACC = 0.90          # τ rule: highest coverage with selective accuracy >= 90% and no S1 accepted below S1


def qwk(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isin(b, SEVERITIES)
    return cohen_kappa_score(a[m], b[m], weights="quadratic", labels=SEVERITIES)


def boot_diff(y, p1, p2, n=2000):
    rng = np.random.RandomState(SEED)
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        d.append(qwk(y[i], p1[i]) - qwk(y[i], p2[i]))
    return [round(float(x), 4) for x in np.percentile(d, [2.5, 97.5])]


def fleiss_kappa(labels):
    """labels: array (n_items, n_raters) of category strings."""
    cats = sorted({c for row in labels for c in row})
    counts = np.array([[sum(1 for c in row if c == k) for k in cats] for row in labels], float)
    n = counts.sum(1)[0]
    p_j = counts.sum(0) / counts.sum()
    P_i = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else 1.0


def mcnemar_exact(b, c):
    return float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    total, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            gap = abs(correct[m].mean() - conf[m].mean())
            total += m.mean() * gap
            rows.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return float(total), rows


def main():
    runs = pd.read_json(RESULTS / "llm_runs.jsonl", lines=True)
    sample = pd.read_csv(PROCESSED / "sample.csv")
    truth = sample.set_index("bug_id").final_severity
    base = pd.read_csv(RESULTS / "predictions_baseline.csv").set_index("bug_id")
    out = {}

    for model, R in runs.groupby("model"):
        res = {}
        lab = R.assign(lbl=R.llm_severity.fillna("INVALID"))
        get = lambda cond, run=1: lab[(lab.condition == cond) & (lab.run_id == run)].set_index("bug_id").lbl

        # ---------- RQ1: H1 and comparison with TF-IDF ----------
        rq1 = {}
        for c in ("P0", "P1"):
            p = get(c)
            ids = p.index.intersection(truth.index)
            if len(ids) < 30:
                continue
            y, yp = truth[ids].values, p[ids].values
            maj = base.loc[ids, "pred_majority"].values
            tf = base.loc[ids, "pred_tfidf"].values
            rq1[c] = {"n": int(len(ids)), "qwk": round(float(qwk(y, yp)), 4),
                      "qwk_minus_majority_ci95": boot_diff(y, yp, maj),
                      "qwk_minus_tfidf_ci95": boot_diff(y, yp, tf),
                      "H1_rejected": bool(boot_diff(y, yp, maj)[0] > 0)}
        res["rq1"] = rq1

        # ---------- RQ2: repeatability and prompt sensitivity ----------
        rq2_ids = sample[sample.in_rq2].bug_id
        r1, r2, r3 = get("P1", 1), get("P1", 2), get("P1", 3)
        w1, w2 = get("P1_reworded_1"), get("P1_reworded_2")
        ids = rq2_ids[rq2_ids.isin(r1.index) & rq2_ids.isin(r2.index) & rq2_ids.isin(r3.index)]
        if len(ids):
            rep = pd.DataFrame({"r1": r1[ids], "r2": r2[ids], "r3": r3[ids]})
            flip = (rep.nunique(axis=1) > 1)
            k, n = int(flip.sum()), int(len(flip))
            bt = binomtest(k, n, 0.05, alternative="greater")
            ci = binomtest(k, n).proportion_ci(confidence_level=0.95)
            rq2 = {"n": n, "flip_rate": round(k / n, 4), "flip_ci95": [round(ci.low, 4), round(ci.high, 4)],
                   "binomial_p_vs_5pct": float(bt.pvalue), "H2a_rejected": bool(bt.pvalue < 0.05),
                   "fleiss_kappa_runs": round(fleiss_kappa(rep.values), 4)}
            ids_w = ids[ids.isin(w1.index) & ids.isin(w2.index)]
            if len(ids_w):
                rep_chg = (r2[ids_w] != r1[ids_w]) | (r3[ids_w] != r1[ids_w])
                wrd_chg = (w1[ids_w] != r1[ids_w]) | (w2[ids_w] != r1[ids_w])
                b = int((wrd_chg & ~rep_chg).sum()); c = int((~wrd_chg & rep_chg).sum())
                p = mcnemar_exact(b, c)
                rq2.update({"n_reworded": int(len(ids_w)), "repeat_change_rate": round(float(rep_chg.mean()), 4),
                            "prompt_change_rate": round(float(wrd_chg.mean()), 4),
                            "discordant_reword_only": b, "discordant_repeat_only": c,
                            "mcnemar_exact_p": p, "H2b_rejected": bool(p < 0.05)})
            res["rq2"] = rq2
            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            vals = [rq2["flip_rate"] * 100, rq2.get("prompt_change_rate", np.nan) * 100]
            ax.bar(["Identical prompt\n(label changed in 3 runs)", "Reworded prompts\n(label differs from run 1)"], vals,
                   color=["#67a9cf", "#ef8a62"], width=.55)
            ax.axhline(5, ls="--", color="grey", lw=.8); ax.text(1.27, 5.6, "5% threshold", ha="right", fontsize=8)
            ax.set_ylabel("Share of defects (%)")
            for i, v in enumerate(vals):
                if not np.isnan(v):
                    ax.text(i, v + .6, f"{v:.1f}%", ha="center", fontsize=9)
            fig.tight_layout(); fig.savefig(FIGURES / "fig13_rq2_stability.png", dpi=150); plt.close(fig)

        # ---------- RQ3: direction of errors by human severity, weighted, effect of context ----------
        rq3, se_cls = {}, {}
        for c in ("P0", "P1"):
            p = get(c); ids = p.index.intersection(truth.index)
            y = truth[ids]; yp = p[ids]
            ok = yp.isin(SEVERITIES)
            se = (yp[ok].map(ORDINAL) - y[ok].map(ORDINAL)); w = sample.set_index("bug_id").weight[ids][ok]
            se_cls[c] = se.groupby(y[ok]).mean().reindex(SEVERITIES)
            rq3[c] = {"signed_error_by_class": {k: round(float(v), 3) for k, v in se_cls[c].items()},
                      "weighted_mean_signed_error": round(float((se * w).sum() / w.sum()), 3),
                      "weighted_over_escalation": round(float(((se > 0) * w).sum() / w.sum()), 3),
                      "weighted_under_escalation": round(float(((se < 0) * w).sum() / w.sum()), 3),
                      "s1_labelled_s3_or_s4": int(((y == "S1") & yp.isin(["S3", "S4"])).sum()),
                      "n_s1": int((y == "S1").sum()),
                      "prediction_share": {k: round(float(v), 3) for k, v in yp.value_counts(normalize=True).items()}}
        p0, p1 = get("P0"), get("P1")
        ids = p0.index.intersection(p1.index).intersection(truth.index)
        d = p1[ids].map(ORDINAL) - p0[ids].map(ORDINAL)
        d = d.dropna()
        rq3["P1_minus_P0"] = {"n": int(len(d)), "label_changed": round(float((d != 0).mean()), 3),
                              "mean_shift": round(float(d.mean()), 3),
                              "wilcoxon_p": float(wilcoxon(d[d != 0]).pvalue) if (d != 0).sum() else 1.0}
        res["rq3"] = rq3
        if se_cls:
            fig, ax = plt.subplots(figsize=(5.2, 3.2)); x = np.arange(4)
            for i, (c, v) in enumerate(se_cls.items()):
                ax.bar(x + (i - .5) * .38, v.values, .38, label=c, color=["#67a9cf", "#ef8a62"][i])
            ax.axhline(0, color="black", lw=.8); ax.set_xticks(x, SEVERITIES)
            ax.set_xlabel("Human final severity"); ax.set_ylabel("Mean signed error\n(+ over-escalation)")
            ax.legend(frameon=False); fig.tight_layout()
            fig.savefig(FIGURES / "fig16_signed_error_by_class.png", dpi=150); plt.close(fig)

        # ---------- RQ4: guardrails, calibration, deferral ----------
        p2 = lab[(lab.condition == "P2") & (lab.run_id == 1)].set_index("bug_id")
        if len(p2):
            val = sample[sample.sample_split == "validation"].bug_id
            tst = sample[sample.sample_split == "test"].bug_id
            p1 = get("P1")
            def frame(ids):
                ids = ids[ids.isin(p2.index) & ids.isin(p1.index)]
                f = pd.DataFrame({"y": truth[ids].values, "p1": p1[ids].values, "p2": p2.loc[ids, "lbl"].values,
                                  "conf": p2.loc[ids, "llm_confidence"].fillna(0).values}, index=ids)
                f["ok2"] = f.p2 == f.y; f["ok1"] = f.p1 == f.y
                f["crit_miss"] = (f.y == "S1") & (f.p2 != "S1")
                return f
            V, T = frame(val), frame(tst)
            # choose τ on validation only
            best = None
            for tau in np.round(np.arange(0, 1.0001, 0.01), 2):
                acc = V[V.conf >= tau]
                if len(acc) and acc.ok2.mean() >= TARGET_ACC and not acc.crit_miss.any():
                    best = float(tau); break
            tau = best if best is not None else 1.01
            acc_t = T[T.conf >= tau]
            b = int((T.ok2 & ~T.ok1).sum()); c = int((~T.ok2 & T.ok1).sum())
            e2, rows2 = ece(T.conf, T.ok2)
            rq4 = {"n_validation": int(len(V)), "n_test": int(len(T)), "tau": tau,
                   "tau_found": best is not None,
                   "test_accuracy_P1": round(float(T.ok1.mean()), 4), "test_accuracy_P2": round(float(T.ok2.mean()), 4),
                   "mcnemar_b_P2_only": b, "mcnemar_c_P1_only": c, "mcnemar_exact_p": mcnemar_exact(b, c),
                   "H4_rejected": bool(mcnemar_exact(b, c) < 0.05),
                   "ece_P2": round(e2, 4), "mean_confidence_P2": round(float(T.conf.mean()), 4),
                   "coverage": round(len(acc_t) / max(len(T), 1), 4),
                   "selective_accuracy": round(float(acc_t.ok2.mean()), 4) if len(acc_t) else None,
                   "critical_misses": acc_t[acc_t.crit_miss].index.tolist(),
                   "s1_in_test": int((T.y == "S1").sum())}
            both = pd.concat([V, T])
            s1 = both.p1.map(ORDINAL) - both.y.map(ORDINAL); s2 = both.p2.map(ORDINAL) - both.y.map(ORDINAL)
            dd = (s2 - s1).dropna()
            rq4.update({"n_val_accepted": int((V.conf >= tau).sum()),
                        "signed_error_P1_same_defects": round(float(s1.mean()), 3),
                        "signed_error_P2": round(float(s2.mean()), 3),
                        "wilcoxon_signed_P2_vs_P1_p": float(wilcoxon(dd[dd != 0]).pvalue),
                        "share_predicted_S1_P1": round(float((both.p1 == "S1").mean()), 3),
                        "share_predicted_S1_P2": round(float((both.p2 == "S1").mean()), 3),
                        "recall_S1_P2": round(float((both.p2[both.y == "S1"] == "S1").mean()), 3),
                        "confidence_values": {str(k): int(v) for k, v in both.conf.round(2).value_counts().items()},
                        "accepted_test": [[str(a), str(b)] for a, b in zip(acc_t.p2, acc_t.y)]})
            res["rq4"] = rq4
            # reliability diagram
            fig, ax = plt.subplots(figsize=(4, 3.6))
            ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=.8, label="Perfect calibration")
            if rows2:
                ax.plot([r[0] for r in rows2], [r[1] for r in rows2], marker="o", label=f"{model} P2 (ECE {e2:.2f})")
            ax.set_xlabel("Mean stated confidence"); ax.set_ylabel("Observed accuracy"); ax.legend(frameon=False, fontsize=8)
            fig.tight_layout(); fig.savefig(FIGURES / "fig14_reliability_diagram.png", dpi=150); plt.close(fig)
            # coverage-accuracy curve on test
            taus = np.round(np.arange(0, 1.0001, 0.02), 2)
            cov = [(T.conf >= t).mean() for t in taus]
            sacc = [T[T.conf >= t].ok2.mean() if (T.conf >= t).any() else np.nan for t in taus]
            fig, ax = plt.subplots(figsize=(5.2, 3.4))
            ax.plot(np.array(cov) * 100, np.array(sacc) * 100, marker=".")
            if best is not None:
                ax.scatter([rq4["coverage"] * 100], [(rq4["selective_accuracy"] or 0) * 100], color="#b2182b", zorder=3,
                           label=f"τ = {tau:.2f} (set on validation)")
                ax.legend(frameon=False, fontsize=8)
            ax.set_xlabel("Coverage: defects accepted automatically (%)"); ax.set_ylabel("Accuracy on accepted (%)")
            fig.tight_layout(); fig.savefig(FIGURES / "fig15_coverage_accuracy.png", dpi=150); plt.close(fig)
        out[model] = res

    json.dump(out, open(RESULTS / "rq_tests.json", "w"), indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
