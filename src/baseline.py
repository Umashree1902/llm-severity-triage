"""Supervised baseline: TF-IDF + multinomial logistic regression trained on 2021-2024 defects,
plus a majority-class baseline, both evaluated on the primary sample.

Usage:  python src/baseline.py
Outputs: results/baseline_metrics.json, results/predictions_baseline.csv,
         results/figures/fig9_baseline_confusion.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline

from common import RAW, PROCESSED, RESULTS, FIGURES, SEVERITIES, SEED
from metrics import all_metrics


def text(df):
    return (df["summary"].fillna("") + " " + df["description_llm"].fillna("")).str.lower()


def main():
    train = pd.read_csv(PROCESSED / "baseline_train.csv")
    test = pd.read_csv(PROCESSED / "sample.csv")
    # leakage guard: no defect may appear in both sets (date windows do not overlap)
    overlap = set(train.bug_id) & set(test.bug_id)
    assert not overlap, f"{len(overlap)} defects in both training and test sets"

    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    grid = GridSearchCV(pipe, {"logisticregression__C": [0.1, 0.3, 1, 3, 10]}, scoring="f1_macro",
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
    grid.fit(text(train), train.final_severity)

    test["pred_tfidf"] = grid.predict(text(test))
    # the training draw is balanced by design, so the majority class is taken from the training population
    pop = json.load(open(RAW / "population_counts.json"))["baseline"]
    majority = max(pop, key=pop.get)
    test["pred_majority"] = majority

    res = {"training_rows": len(train), "test_rows": len(test), "best_C": grid.best_params_,
           "cv_macro_f1": round(grid.best_score_, 4), "majority_class": majority,
           "tfidf": all_metrics(test.final_severity, test.pred_tfidf, weights=test.weight),
           "majority": all_metrics(test.final_severity, test.pred_majority, weights=test.weight)}

    # most informative terms per class (feature importance)
    vec = grid.best_estimator_.named_steps["tfidfvectorizer"]
    lr = grid.best_estimator_.named_steps["logisticregression"]
    terms = np.array(vec.get_feature_names_out())
    res["top_terms"] = {c: terms[np.argsort(lr.coef_[i])[-10:][::-1]].tolist() for i, c in enumerate(lr.classes_)}

    json.dump(res, open(RESULTS / "baseline_metrics.json", "w"), indent=2)
    test[["bug_id", "final_severity", "pred_tfidf", "pred_majority", "weight", "sample_split",
          "in_rq3"]].to_csv(RESULTS / "predictions_baseline.csv", index=False)

    cm = pd.crosstab(test.final_severity, test.pred_tfidf).reindex(index=SEVERITIES, columns=SEVERITIES, fill_value=0)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm.values, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm.values[i, j], ha="center", va="center",
                    color="white" if cm.values[i, j] > cm.values.max() / 2 else "black")
    ax.set_xticks(range(4), SEVERITIES); ax.set_yticks(range(4), SEVERITIES)
    ax.set_xlabel("Predicted (TF-IDF baseline)"); ax.set_ylabel("Human final severity")
    fig.colorbar(im, ax=ax, shrink=.8); fig.tight_layout()
    fig.savefig(FIGURES / "fig9_baseline_confusion.png", dpi=150, bbox_inches="tight")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
