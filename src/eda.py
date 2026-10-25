"""Exploratory data analysis: summary statistics and charts for the interim report.

Usage:  python src/eda.py
Inputs: data/processed/sample.csv, data/processed/dupe_pairs.csv, data/raw/population_counts.json
Outputs: results/eda_summary.json, results/eda_by_severity.csv, results/figures/*.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RAW, PROCESSED, RESULTS, FIGURES, SEVERITIES, LABEL

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
COLORS = {"S1": "#b2182b", "S2": "#ef8a62", "S3": "#67a9cf", "S4": "#2166ac"}
ORDER = SEVERITIES


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(PROCESSED / "sample.csv", parse_dates=["creation_time"])
    counts = json.load(open(RAW / "population_counts.json"))
    summary = {"n_sample": len(df), "population": counts["primary"],
               "sample": df.final_severity.value_counts().reindex(ORDER).to_dict()}

    # per-severity summary table
    g = df.groupby("final_severity")
    tab = pd.DataFrame({
        "n": g.size(),
        "desc_len_median": g.desc_len.median(),
        "desc_len_iqr": g.desc_len.quantile(.75) - g.desc_len.quantile(.25),
        "days_to_triage_median": g.days_to_triage.median(),
        "share_severity_changed": g.severity_changes.apply(lambda s: (s > 0).mean()),
        "share_first_differs_from_final": (df.first_severity.notna() & (df.first_severity != df.final_severity))
            .groupby(df.final_severity).mean(),
        "share_has_regression_kw": g.keywords.apply(lambda k: k.fillna("").str.contains("regression").mean()),
        "share_crash_in_summary": g.summary.apply(lambda t: t.fillna("").str.contains("crash", case=False).mean()),
        "top_product": g["product"].agg(lambda p: p.value_counts().index[0]),
    }).reindex(ORDER)
    tab.round(3).to_csv(RESULTS / "eda_by_severity.csv")
    summary["by_severity"] = tab.round(3).reset_index().to_dict(orient="records")

    # Fig 1: population vs sample distribution
    fig, ax = plt.subplots(figsize=(6, 3.2))
    pop = np.array([counts["primary"][s] for s in ORDER], float)
    smp = df.final_severity.value_counts().reindex(ORDER).fillna(0).values
    x = np.arange(4)
    ax.bar(x - .2, pop / pop.sum() * 100, .4, label="Population (2025–Jun 2026)", color="#999999")
    ax.bar(x + .2, smp / smp.sum() * 100, .4, label="Stratified sample", color="#2166ac")
    ax.set_xticks(x, [f"{s}\n{LABEL[s]}" for s in ORDER]); ax.set_ylabel("Share of defects (%)")
    ax.legend(frameon=False); save(fig, "fig1_severity_distribution.png")

    # Fig 2: description length by severity (log scale)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.boxplot([df[df.final_severity == s].desc_len.clip(lower=1) for s in ORDER], labels=ORDER,
               showfliers=False)
    ax.set_yscale("log"); ax.set_ylabel("Description length (characters, log)")
    ax.set_xlabel("Final severity"); save(fig, "fig2_description_length.png")

    # Fig 3: top products by severity (stacked shares)
    top = df["product"].value_counts().index[:8]
    ct = pd.crosstab(df[df["product"].isin(top)]["product"], df["final_severity"], normalize="index")
    ct = ct.reindex(columns=ORDER, fill_value=0).loc[top]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    left = np.zeros(len(ct))
    for s in ORDER:
        ax.barh(ct.index, ct[s] * 100, left=left, color=COLORS[s], label=s); left += ct[s].values * 100
    ax.invert_yaxis(); ax.set_xlabel("Share of sampled defects (%)")
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.0))
    save(fig, "fig3_products_by_severity.png")

    # Fig 4: days to triage by severity
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.boxplot([df[df.final_severity == s].days_to_triage.dropna().clip(lower=0) for s in ORDER],
               labels=ORDER, showfliers=False)
    ax.set_ylabel("Days from filing to first severity"); ax.set_xlabel("Final severity")
    save(fig, "fig4_days_to_triage.png")

    # Fig 5: label instability - share whose severity changed / first differs from final
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(x - .2, tab.share_severity_changed * 100, .4, label="Severity changed at least once", color="#ef8a62")
    ax.bar(x + .2, tab.share_first_differs_from_final * 100, .4, label="First severity ≠ final", color="#67a9cf")
    ax.set_xticks(x, ORDER); ax.set_ylabel("Share of defects (%)"); ax.legend(frameon=False)
    save(fig, "fig5_label_instability.png")

    # Fig 6: monthly filing volume by severity (sample)
    m = df.assign(month=df.creation_time.dt.to_period("M").dt.to_timestamp()).groupby(
        ["month", "final_severity"]).size().unstack(fill_value=0).reindex(columns=ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for s in ORDER:
        ax.plot(m.index, m[s], marker="o", ms=3, color=COLORS[s], label=s)
    ax.set_ylabel("Sampled defects filed"); ax.legend(ncol=4, frameon=False)
    save(fig, "fig6_monthly_volume.png")

    # Fig 7: first vs final severity transition matrix
    t = df[df.first_severity.isin(ORDER)]
    cm = pd.crosstab(t.first_severity, t.final_severity).reindex(index=ORDER, columns=ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm.values, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm.values[i, j], ha="center", va="center",
                    color="white" if cm.values[i, j] > cm.values.max() / 2 else "black")
    ax.set_xticks(range(4), ORDER); ax.set_yticks(range(4), ORDER)
    ax.set_xlabel("Final severity"); ax.set_ylabel("First assigned severity")
    fig.colorbar(im, ax=ax, shrink=.8); save(fig, "fig7_first_vs_final.png")
    summary["first_vs_final_agreement"] = float(np.trace(cm.values) / max(cm.values.sum(), 1))

    # Fig 8: duplicate-pair severity agreement (descriptive indicator)
    try:
        dp = pd.read_csv(PROCESSED / "dupe_pairs.csv")
        dcm = pd.crosstab(dp.final_severity, dp.target_severity).reindex(index=ORDER, columns=ORDER, fill_value=0)
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        im = ax.imshow(dcm.values, cmap="Oranges")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, dcm.values[i, j], ha="center", va="center")
        ax.set_xticks(range(4), ORDER); ax.set_yticks(range(4), ORDER)
        ax.set_xlabel("Severity of original report"); ax.set_ylabel("Severity of duplicate")
        save(fig, "fig8_duplicate_pairs.png")
        summary["duplicate_pairs"] = {"n": int(len(dp)),
                                      "exact_agreement": float((dp.final_severity == dp.target_severity).mean())}
    except FileNotFoundError:
        summary["duplicate_pairs"] = {"n": 0}

    # completeness
    summary["missing_values"] = pd.read_csv(RESULTS / "missing_values.csv").to_dict(orient="records")
    summary["outliers"] = json.load(open(RESULTS / "outliers.json"))
    json.dump(summary, open(RESULTS / "eda_summary.json", "w"), indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
