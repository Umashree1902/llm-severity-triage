"""Retrieve defects, first comments and severity history from the Bugzilla REST API into data/raw/.

Usage:
    export BUGZILLA_API_KEY=...        # optional but recommended (higher rate limits)
    python src/extract.py              # primary sample + baseline training set
    python src/extract.py --quick      # small run for testing (50 per stratum)

Outputs (JSON Lines, one bug per line):
    data/raw/population_counts.json    counts per severity for both windows
    data/raw/primary_bugs.jsonl        sampled primary-window bugs with comments and history
    data/raw/baseline_bugs.jsonl       sampled 2021-2024 bugs with first comment (baseline training)
    data/raw/dupe_targets.jsonl        severity of bugs that sampled duplicates point to
"""
import argparse, json, os, random, sys, time
from datetime import datetime, timezone

import requests

from common import RAW, SEVERITIES, SEED, PRIMARY_FROM, PRIMARY_TO, BASELINE_FROM, BASELINE_TO, \
    OVERSAMPLE, BASELINE_PER_STRATUM

BASE = "https://bugzilla.mozilla.org/rest"
FIELDS = ("id,creation_time,product,component,type,summary,platform,op_sys,keywords,"
          "priority,severity,status,resolution,dupe_of,groups,last_change_time")
session = requests.Session()
if os.environ.get("BUGZILLA_API_KEY"):
    session.headers["X-BUGZILLA-API-KEY"] = os.environ["BUGZILLA_API_KEY"]


def get(path, params, tries=5):
    for attempt in range(tries):
        try:
            r = session.get(f"{BASE}/{path}", params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            time.sleep(0.2)                      # be polite to the API
            return r.json()
        except Exception as e:                   # noqa: BLE001
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/{tries} after error: {e} (waiting {wait}s)", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"failed: {path} {params}")


def all_ids(severity, date_from, date_to):
    """All defect ids with this severity filed in [date_from, date_to), excluding intermittent-failure reports."""
    ids, offset, limit = [], 0, 1000
    while True:
        data = get("bug", {"type": "defect", "severity": severity, "creation_time": date_from,
                           "f1": "creation_ts", "o1": "lessthan", "v1": date_to,
                           "f2": "keywords", "o2": "nowords", "v2": "intermittent-failure",
                           "include_fields": "id", "order": "bug_id", "limit": limit, "offset": offset})
        batch = [b["id"] for b in data.get("bugs", [])]
        ids += batch
        if len(batch) < limit:
            return ids
        offset += limit


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def fetch_bugs(ids, with_history=True):
    out = {}
    for batch in chunks(ids, 100):
        for b in get("bug", {"id": ",".join(map(str, batch)), "include_fields": FIELDS}).get("bugs", []):
            out[b["id"]] = b
    def add_comments(data):
        for bid, v in data.get("bugs", {}).items():
            comments = v.get("comments", [])
            if int(bid) in out:
                out[int(bid)]["description"] = (comments[0] if comments else {}).get("text", "")
                out[int(bid)]["comment_count"] = len(comments)

    def add_history(data):
        for h in data.get("bugs", []):
            if h["id"] in out:
                # keep only severity changes (when/removed/added); no names or emails
                out[h["id"]]["severity_history"] = [
                    {"when": ev["when"], "removed": c.get("removed"), "added": c.get("added")}
                    for ev in h.get("history", []) for c in ev.get("changes", [])
                    if "severity" in str(c.get("field_name", ""))]

    done = 0
    for batch in chunks(list(out), 25):
        add_comments(get(f"bug/{batch[0]}/comment", {"ids": batch}))
        done += len(batch)
        print(f"  comments: {done}/{len(out)}", end="\r")
    for bid in [b for b in out if "description" not in out[b]]:   # fallback: one by one
        add_comments(get(f"bug/{bid}/comment", {}))
    print()
    if with_history:
        done = 0
        for batch in chunks(list(out), 25):
            add_history(get(f"bug/{batch[0]}/history", {"ids": batch}))
            done += len(batch)
            print(f"  history: {done}/{len(out)}", end="\r")
        for bid in [b for b in out if "severity_history" not in out[b]]:
            add_history(get(f"bug/{bid}/history", {}))
        print()
    return list(out.values())


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def refetch_history():
    """Re-fetch severity history for the already-sampled primary bugs (same sample, no new draw)."""
    rows = [json.loads(l) for l in open(RAW / "primary_bugs.jsonl")]
    by_id = {r["id"]: r for r in rows}
    ids = list(by_id)
    seen_fields = {}
    for i, batch in enumerate(chunks(ids, 25)):
        for h in get(f"bug/{batch[0]}/history", {"ids": batch}).get("bugs", []):
            for ev in h.get("history", []):
                for c in ev.get("changes", []):
                    seen_fields[c.get("field_name")] = seen_fields.get(c.get("field_name"), 0) + 1
            if h["id"] in by_id:
                by_id[h["id"]]["severity_history"] = [
                    {"when": ev["when"], "removed": c.get("removed"), "added": c.get("added")}
                    for ev in h.get("history", []) for c in ev.get("changes", [])
                    if "severity" in str(c.get("field_name", ""))]
        print(f"  history: {min((i + 1) * 25, len(ids))}/{len(ids)}", end="\r")
    print()
    write_jsonl(RAW / "primary_bugs.jsonl", rows)
    with_hist = sum(1 for r in rows if r.get("severity_history"))
    print(f"Bugs with at least one severity change in history: {with_hist}/{len(rows)}")
    print("Most common field names in history:", sorted(seen_fields.items(), key=lambda x: -x[1])[:12])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="50 per stratum, for a smoke test")
    ap.add_argument("--history-only", action="store_true", help="re-fetch severity history for the existing sample")
    args = ap.parse_args()
    if args.history_only:
        return refetch_history()
    per = 50 if args.quick else OVERSAMPLE
    per_base = 50 if args.quick else BASELINE_PER_STRATUM
    rng = random.Random(SEED)
    extracted_at = datetime.now(timezone.utc).isoformat()

    counts = {"extracted_at": extracted_at, "primary": {}, "baseline": {}}
    primary_ids, baseline_ids = [], []
    for s in SEVERITIES:
        print(f"Listing {s} ids ...")
        p = all_ids(s, PRIMARY_FROM, PRIMARY_TO)
        b = all_ids(s, BASELINE_FROM, BASELINE_TO)
        counts["primary"][s], counts["baseline"][s] = len(p), len(b)
        primary_ids += p if (s == "S1" and not args.quick) else rng.sample(p, min(per, len(p)))
        baseline_ids += rng.sample(b, min(per_base, len(b)))
        print(f"  {s}: primary {len(p)}, baseline {len(b)}")
    json.dump(counts, open(RAW / "population_counts.json", "w"), indent=2)

    print(f"Fetching {len(primary_ids)} primary bugs ...")
    primary = fetch_bugs(primary_ids, with_history=True)
    for r in primary:
        r["extracted_at"] = extracted_at
    write_jsonl(RAW / "primary_bugs.jsonl", primary)

    print(f"Fetching {len(baseline_ids)} baseline bugs ...")
    write_jsonl(RAW / "baseline_bugs.jsonl", fetch_bugs(baseline_ids, with_history=False))

    targets = sorted({r["dupe_of"] for r in primary if r.get("dupe_of")})
    print(f"Fetching {len(targets)} duplicate targets ...")
    dupes = []
    for batch in chunks(targets, 100):
        dupes += get("bug", {"id": ",".join(map(str, batch)), "include_fields": "id,severity"}).get("bugs", [])
    write_jsonl(RAW / "dupe_targets.jsonl", dupes)
    print("Done.")


if __name__ == "__main__":
    main()
