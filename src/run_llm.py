"""Run prompt conditions on both models, with identical repeats (RQ2a) and reworded prompts (RQ2b).

Every call is a fresh request (no conversation history). Results are appended to
results/llm_runs.jsonl; rows already present are skipped, so the script can be stopped and resumed.

Examples:
    # pilot for the interim report: 50 defects per severity, P0 and P1, local open-weight model
    python src/run_llm.py --provider ollama --model llama3.1:8b --conditions P0 P1 --pilot 50

    # commercial model through an OpenAI-compatible API (OpenAI, or Gemini via its OpenAI endpoint)
    export LLM_API_KEY=...  LLM_BASE_URL=https://api.openai.com/v1
    python src/run_llm.py --provider openai --model gpt-4o-mini --conditions P0 P1 --pilot 50

    # Anthropic
    export ANTHROPIC_API_KEY=...
    python src/run_llm.py --provider anthropic --model <model-id> --conditions P0 P1 --pilot 50
"""
import argparse, json, os, random, re, threading, time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from common import PROCESSED, RESULTS, PROMPTS, SEVERITIES, SEED

OUT = RESULTS / "llm_runs.jsonl"
SYSTEM = (PROMPTS / "system.txt").read_text()


def render(template, row):
    fields = {k: ("" if pd.isna(row.get(k)) else str(row.get(k)))
              for k in ["summary", "product", "component", "platform", "op_sys", "keywords"]}
    fields["keywords"] = fields["keywords"].replace(";", ", ") or "none"
    fields["description"] = "" if pd.isna(row.get("description_llm")) else str(row["description_llm"])
    return template.format(**fields)


def call(provider, model, prompt, temperature):
    t0 = time.time()
    if provider == "ollama":
        body = {"model": model, "stream": False, "format": "json",
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
        if temperature is not None:
            body["options"] = {"temperature": temperature}
        r = requests.post(os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/chat", json=body, timeout=300)
        r.raise_for_status()
        text = r.json()["message"]["content"]
    elif provider == "openai":
        body = {"model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}}
        if temperature is not None:
            body["temperature"] = temperature
        r = requests.post(os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1") + "/chat/completions",
                          headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"}, json=body, timeout=120)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
    elif provider == "anthropic":
        body = {"model": model, "max_tokens": 100, "system": SYSTEM,
                "messages": [{"role": "user", "content": prompt}]}
        if temperature is not None:
            body["temperature"] = temperature
        r = requests.post("https://api.anthropic.com/v1/messages", json=body, timeout=120,
                          headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})
        r.raise_for_status()
        text = "".join(b.get("text", "") for b in r.json()["content"])
    else:
        raise ValueError(provider)
    return text, round(time.time() - t0, 3)


def parse(text):
    sev, conf = None, None
    try:
        obj = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        sev = str(obj.get("severity", "")).strip().upper()
        conf = float(obj.get("confidence")) if obj.get("confidence") is not None else None
    except Exception:  # noqa: BLE001
        m = re.search(r"\bS[1-4]\b", text or "")
        sev = m.group(0) if m else None
    return (sev if sev in SEVERITIES else None), conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["ollama", "openai", "anthropic"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--conditions", nargs="+", default=["P0", "P1"])
    ap.add_argument("--pilot", type=int, default=0, help="defects per severity for a pilot (0 = use --subset)")
    ap.add_argument("--subset", default="all", choices=["all", "rq2", "rq3", "validation", "test"])
    ap.add_argument("--repeats", type=int, default=1, help="identical repeats per condition (RQ2a uses 3)")
    ap.add_argument("--temperature", type=float, default=None, help="omit to use the model's default")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", 1)),
                    help="parallel requests (set OLLAMA_NUM_PARALLEL to the same value)")
    args = ap.parse_args()

    df = pd.read_csv(PROCESSED / "sample.csv")
    if args.pilot:
        df = pd.concat([g.sample(min(args.pilot, len(g)), random_state=SEED)
                        for _, g in df.groupby("final_severity")])
    elif args.subset == "rq2":
        df = df[df.in_rq2]
    elif args.subset == "rq3":
        df = df[df.in_rq3]
    elif args.subset in ("validation", "test"):
        df = df[df.sample_split == args.subset]

    done = set()
    if OUT.exists():
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["bug_id"], r["model"], r["condition"], r["run_id"]))

    templates = {c: (PROMPTS / f"{c}.txt").read_text() for c in args.conditions}
    jobs = [(row, c, k) for _, row in df.iterrows() for c in args.conditions for k in range(1, args.repeats + 1)
            if (int(row.bug_id), args.model, c, k) not in done]
    random.Random(SEED).shuffle(jobs)   # random order, so a partly finished run is still a random subset
    print(f"{len(jobs)} calls to run ({len(df)} defects x {len(args.conditions)} conditions x {args.repeats} runs)"
          f" with {args.workers} worker(s)")
    lock, counter = threading.Lock(), [0]

    def work(job):
        row, c, k = job
        try:
            text, secs = call(args.provider, args.model, render(templates[c], row), args.temperature)
        except Exception as e:  # noqa: BLE001
            text, secs = f"ERROR: {e}", None
        sev, conf = parse(text)
        rec = json.dumps({"bug_id": int(row.bug_id), "model": args.model, "provider": args.provider,
                          "condition": c, "run_id": k, "llm_severity": sev, "llm_confidence": conf,
                          "raw": text[:500], "seconds": secs, "temperature": args.temperature,
                          "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n"
        with lock:
            f.write(rec); f.flush()
            counter[0] += 1
            i = counter[0]
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done (last call {secs}s)", flush=True)

    with open(OUT, "a") as f, ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, jobs))


if __name__ == "__main__":
    main()
