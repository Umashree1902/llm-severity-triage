"""Evaluation metrics shared by the baseline and LLM evaluations."""
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, recall_score

from common import ORDINAL, SEVERITIES, SEED


def all_metrics(y_true, y_pred, weights=None, n_boot=1000):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ok = np.isin(y_pred, SEVERITIES)                      # malformed / missing outputs count as wrong
    yp = np.where(ok, y_pred, "INVALID")
    s_true = np.array([ORDINAL[v] for v in y_true])
    s_pred = np.array([ORDINAL.get(v, np.nan) for v in yp])
    signed = s_pred - s_true
    rng = np.random.RandomState(SEED)

    def qwk(a, b):
        m = np.isin(b, SEVERITIES)
        return cohen_kappa_score(a[m], b[m], weights="quadratic", labels=SEVERITIES) if m.sum() > 1 else np.nan

    boots = []
    for _ in range(n_boot):
        i = rng.randint(0, len(y_true), len(y_true))
        boots.append(qwk(y_true[i], yp[i]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    out = {
        "n": int(len(y_true)), "invalid_outputs": int((~ok).sum()),
        "accuracy": round(accuracy_score(y_true, yp), 4),
        "macro_f1": round(f1_score(y_true, yp, labels=SEVERITIES, average="macro", zero_division=0), 4),
        "qwk": round(float(qwk(y_true, yp)), 4), "qwk_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "recall_per_class": dict(zip(SEVERITIES, np.round(
            recall_score(y_true, yp, labels=SEVERITIES, average=None, zero_division=0), 4).tolist())),
        "mean_signed_error": round(float(np.nanmean(signed)), 4),
        "median_signed_error": float(np.nanmedian(signed)),
        "over_escalation_rate": round(float(np.nanmean(signed > 0)), 4),
        "under_escalation_rate": round(float(np.nanmean(signed < 0)), 4),
    }
    if weights is not None:
        w = np.asarray(weights, float)
        out["accuracy_weighted"] = round(float(np.sum(w * (yp == y_true)) / w.sum()), 4)
    return out
