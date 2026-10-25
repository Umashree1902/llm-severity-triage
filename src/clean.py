"""Apply exclusions, derive variables, remove personal data, draw the stratified sample and subsets.

Usage:  python src/clean.py

Inputs:  data/raw/*.jsonl, data/raw/population_counts.json (from extract.py)
Outputs: data/processed/sample.csv          analysis sample with subset flags
         data/processed/baseline_train.csv  2021-2024 training set for the TF-IDF baseline
         data/processed/dupe_pairs.csv      duplicate report pairs with both severities
         results/cleaning_log.json          rows removed at each step (primary and baseline)
         results/missing_values.csv         share of missing / unspecified values per column
         results/outliers.json              IQR outlier counts for numeric variables
"""
import json, re

import numpy as np
import pandas as pd

from common import RAW, PROCESSED, RESULTS, SEVERITIES, ORDINAL, SEED, PER_STRATUM, SUBSETS

MAX_DESC_CHARS = 3000          # description sent to the LLM is truncated to bound prompt length
MIN_DESC_CHARS = 20            # shorter descriptions carry no usable information
STABILITY_DAYS = 30            # severity must be unchanged for 30 days before extraction

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
NICK = re.compile(r"\[:[^\]]+\]")                       # Bugzilla ":nick" handles
REPLY = re.compile(r"\(In reply to [^)]*\)")


def scrub(text):
    text = REPLY.sub("", text or "")
    text = EMAIL.sub("<email>", text)
    text = NICK.sub("<user>", text)
    return re.sub(r"\s+", " ", text).strip()


def load(name):
    return pd.read_json(RAW / name, lines=True)


def derive(df):
    df = df.copy()
    df["creation_time"] = pd.to_datetime(df["creation_time"], utc=True)
    df["keywords"] = df["keywords"].apply(lambda k: k if isinstance(k, list) else [])
    df["description"] = df["description"].fillna("").map(scrub)
    df["summary"] = df["summary"].fillna("").map(scrub)
    df["desc_len"] = df["description"].str.len()
    df["desc_truncated"] = df["desc_len"] > MAX_DESC_CHARS
    df["description_llm"] = df["description"].str.slice(0, MAX_DESC_CHARS)
    if "severity_history" in df:
        def hist(h):
            h = sorted(h if isinstance(h, list) else [], key=lambda e: e["when"])
            first = next((e for e in h if e.get("added") in SEVERITIES), None)
            changes_after = [e for e in h if e.get("removed") in SEVERITIES]
            return pd.Series({
                "first_severity": first["added"] if first else None,
                "first_severity_time": first["when"] if first else None,
                "severity_changes": len(changes_after),
                "last_severity_change": h[-1]["when"] if h else None})
        df = pd.concat([df, df["severity_history"].apply(hist)], axis=1)
        for c in ("first_severity_time", "last_severity_change"):
            df[c] = pd.to_datetime(df[c], utc=True)
        df["days_to_triage"] = (df["first_severity_time"] - df["creation_time"]).dt.total_seconds() / 86400
    df["final_severity"] = df["severity"]
    df["ordinal"] = df["final_severity"].map(ORDINAL)
    return df


def apply_exclusions(df, log, with_stability):
    steps = [
        ("duplicate bug_id rows", lambda d: d.drop_duplicates("id")),
        ("type is not defect", lambda d: d[d["type"] == "defect"]),
        ("severity not S1-S4 at extraction", lambda d: d[d["severity"].isin(SEVERITIES)]),
        ("security-restricted (non-empty groups)",
         lambda d: d[d["groups"].apply(lambda g: not (isinstance(g, list) and g))] if "groups" in d else d),
        ("keyword intermittent-failure", lambda d: d[~d["keywords"].apply(lambda k: "intermittent-failure" in k)]),
        (f"description shorter than {MIN_DESC_CHARS} characters", lambda d: d[d["desc_len"] >= MIN_DESC_CHARS]),
    ]
    if with_stability:
        def stable(d):
            extracted = pd.to_datetime(d["extracted_at"], utc=True)
            recent = d["last_severity_change"].notna() & \
                ((extracted - d["last_severity_change"]).dt.days < STABILITY_DAYS)
            return d[~recent]
        steps.append((f"severity changed within {STABILITY_DAYS} days of extraction", stable))
    for name, fn in steps:
        before = len(df)
        df = fn(df)
        log.append({"step": name, "removed": before - len(df), "remaining": len(df)})
    return df


def iqr_outliers(s):
    s = s.dropna()
    q1, q3 = s.quantile([.25, .75])
    hi, lo = q3 + 1.5 * (q3 - q1), q1 - 1.5 * (q3 - q1)
    return {"q1": float(q1), "q3": float(q3), "upper_fence": float(hi),
            "n_outliers": int(((s > hi) | (s < lo)).sum()), "n": int(len(s))}


def stratified_take(df, per_level, rng, exclude=None):
    pool = df if exclude is None else df[~df.index.isin(exclude)]
    parts = []
    for s, n in per_level.items():
        g = pool[pool["final_severity"] == s]
        parts.append(g.sample(min(n, len(g)), random_state=rng))
    return pd.concat(parts).index


def main():
    rng = np.random.RandomState(SEED)
    counts = json.load(open(RAW / "population_counts.json"))
    log = {"primary": [], "baseline": []}

    prim = derive(load("primary_bugs.jsonl"))
    log["primary"].append({"step": "extracted (all S1 + oversample of S2-S4)", "removed": 0, "remaining": len(prim)})
    prim = apply_exclusions(prim, log["primary"], with_stability=True)

    # stratified sample: all S1, PER_STRATUM each of S2-S4
    keep = [prim[prim.final_severity == "S1"]] + [
        prim[prim.final_severity == s].sample(min(PER_STRATUM, (prim.final_severity == s).sum()), random_state=rng)
        for s in SEVERITIES[1:]]
    sample = pd.concat(keep).reset_index(drop=True)
    log["primary"].append({"step": f"stratified draw (S1 census, {PER_STRATUM} each S2-S4)",
                           "removed": len(prim) - len(sample), "remaining": len(sample)})
    pop = counts["primary"]
    nsamp = sample.final_severity.value_counts()
    sample["weight"] = sample.final_severity.map(lambda s: pop[s] / nsamp[s])

    # subsets: validation and RQ4 test are disjoint; S1 is limited so it is allocated proportionally
    n_s1 = int(nsamp.get("S1", 0))
    val_alloc = {"S1": min(50, n_s1 // 4)}
    val_alloc.update({s: (SUBSETS["validation"] - val_alloc["S1"] + 2) // 3 for s in SEVERITIES[1:]})
    val_idx = stratified_take(sample, val_alloc, rng)
    test_s1 = min(n_s1 - val_alloc["S1"], round(SUBSETS["rq4_test"] / 4))
    test_alloc = {"S1": test_s1}
    test_alloc.update({s: (SUBSETS["rq4_test"] - test_s1 + 2) // 3 for s in SEVERITIES[1:]})
    test_idx = stratified_take(sample, test_alloc, rng, exclude=val_idx)
    sample["sample_split"] = "main"
    sample.loc[val_idx, "sample_split"] = "validation"
    sample.loc[test_idx, "sample_split"] = "test"
    sample["in_rq2"] = sample.index.isin(sample.sample(SUBSETS["rq2"], random_state=rng).index)
    sample["in_rq3"] = sample.index.isin(sample.sample(SUBSETS["rq3"], random_state=rng).index)
    rel = {s: SUBSETS["reliability_per_level"] for s in SEVERITIES}
    sample["in_reliability"] = sample.index.isin(stratified_take(sample, rel, rng))

    cols = ["id", "creation_time", "product", "component", "platform", "op_sys", "keywords", "priority",
            "status", "resolution", "dupe_of", "summary", "description", "description_llm", "desc_len",
            "desc_truncated", "comment_count", "final_severity", "ordinal", "first_severity", "severity_changes",
            "days_to_triage", "weight", "sample_split", "in_rq2", "in_rq3", "in_reliability"]
    out = sample[cols].rename(columns={"id": "bug_id"})
    out["keywords"] = out["keywords"].apply(lambda k: ";".join(k))
    out.to_csv(PROCESSED / "sample.csv", index=False)

    # duplicate pairs (descriptive indicator of label instability only)
    dupes = load("dupe_targets.jsonl") if (RAW / "dupe_targets.jsonl").stat().st_size else pd.DataFrame()
    if len(dupes):
        tgt = dupes.set_index("id")["severity"]
        pairs = prim[prim["dupe_of"].notna()][["id", "final_severity", "dupe_of"]].copy()
        pairs["target_severity"] = pairs["dupe_of"].astype(int).map(tgt)
        pairs = pairs[pairs["target_severity"].isin(SEVERITIES)]
        pairs.rename(columns={"id": "bug_id"}).to_csv(PROCESSED / "dupe_pairs.csv", index=False)

    # baseline training set
    base = derive(load("baseline_bugs.jsonl"))
    log["baseline"].append({"step": "extracted (2021-2024, up to 1,000 per severity)", "removed": 0, "remaining": len(base)})
    base = apply_exclusions(base, log["baseline"], with_stability=False)
    base[["id", "creation_time", "product", "component", "summary", "description_llm", "final_severity",
          "ordinal"]].rename(columns={"id": "bug_id"}).to_csv(PROCESSED / "baseline_train.csv", index=False)

    # data quality reports
    miss = []
    for c in ["product", "component", "platform", "op_sys", "priority", "summary", "description",
              "first_severity", "days_to_triage", "keywords"]:
        col = sample[c]
        unspecified = col.isna() | col.astype(str).str.strip().isin(["", "--", "Unspecified", "[]", "None", "nan"])
        miss.append({"variable": c, "missing_or_unspecified": int(unspecified.sum()),
                     "share": round(float(unspecified.mean()), 4)})
    pd.DataFrame(miss).to_csv(RESULTS / "missing_values.csv", index=False)
    json.dump({"desc_len": iqr_outliers(sample["desc_len"]),
               "days_to_triage": iqr_outliers(sample["days_to_triage"]),
               "severity_changes": iqr_outliers(sample["severity_changes"]),
               "descriptions_truncated": int(sample["desc_truncated"].sum())},
              open(RESULTS / "outliers.json", "w"), indent=2)
    log["subsets"] = {"validation": val_alloc, "rq4_test": test_alloc,
                      "rq2": SUBSETS["rq2"], "rq3": SUBSETS["rq3"], "reliability": rel,
                      "sample_by_severity": {k: int(v) for k, v in nsamp.items()}}
    json.dump(log, open(RESULTS / "cleaning_log.json", "w"), indent=2, default=str)
    print(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
