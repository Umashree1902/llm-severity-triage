# Trusting AI in Defect Triage: Agreement, Consistency and Bias in LLM Severity Labels

QM640 Data Analytics Capstone — Walsh College  
Author: Umashree Pandiarajan

## Purpose
This project evaluates how far QA teams can trust severity labels that large language models (LLMs) assign to software defects. It measures:

- **RQ1 – Agreement:** LLM severity vs. human triagers, interpreted against agreement between two independent reviewers who re-label a 152-defect subset
- **RQ2a – Repeatability:** label stability when the identical prompt is repeated
- **RQ2b – Prompt sensitivity:** label changes when only the wording of the prompt changes
- **RQ3 – Bias:** over- or under-escalation, and the effect of product context
- **RQ4 – Guardrails:** effect of severity definitions and confidence-based deferral on correctness and calibration; the threshold is set on a validation subset and evaluated once on untouched test defects, including missed critical (S1) defects

## Models
Two LLMs: one commercial model accessed through its API and one open-weight model run locally. Exact model versions are recorded here once data collection starts.

## Scope
- **Core:** RQ1 and RQ3 on the full stratified sample (P0, P1); RQ2 with 3 identical repeats and 2 reworded prompts on 469 defects; RQ4 with severity definitions and confidence-based deferral (469 test + 200 validation defects); human label reliability check on 152 defects
- **Optional (if time allows):** retrieval of similar past defects in P2; 5 repeats and temperature zero in RQ2

## Data
Public defect reports from Mozilla Bugzilla (https://bugzilla.mozilla.org), retrieved through the Bugzilla REST API.

- Primary sample: defects filed 1 January 2025 – 30 June 2026 with severity S1–S4
- Baseline set: defects filed 2021–2024, used to train the TF-IDF baseline and, if the optional retrieval extension runs, to supply similar past defects (no overlap with the primary sample)
- Stratified sample: all eligible S1 defects plus 800 each from S2–S4 (about 2,513), with subsets for RQ2 (469), RQ3 (825), RQ4 test (469), threshold validation (200) and the reliability check (152)
- Excluded: tasks, enhancements, defects without a description, security-restricted reports, automatically filed intermittent test-failure reports, and defects whose severity changed in the 30 days before extraction
- Reporter names, emails and user handles are removed during cleaning

See [`data/data_dictionary.md`](data/data_dictionary.md) for all variables.

## Repository structure
```
llm-severity-triage/
├── README.md
├── data/
│   ├── raw/                 # API responses (JSON)
│   ├── processed/           # analysis-ready CSV files
│   └── data_dictionary.md
├── src/
│   ├── extract.py           # Bugzilla REST API extraction
│   ├── clean.py             # filtering, mapping, anonymization
│   ├── run_llm.py           # prompt conditions P0–P2, repeats and reworded prompts
│   └── retrieve.py          # optional: similar-defect retrieval for P2
├── prompts/                 # versioned prompt templates
├── notebooks/               # RQ1–RQ4 analyses
└── results/                 # tables and figures
```

## How to run
Requires Python 3.10+ and, for the local model, [Ollama](https://ollama.com).

```bash
pip install -r requirements.txt
export BUGZILLA_API_KEY=...            # optional; raises Bugzilla rate limits
cd src
python extract.py --quick              # 2-minute smoke test (50 per severity)
python extract.py                      # full extraction (about 15-25 minutes)
python clean.py                        # exclusions, derived variables, stratified sample and subsets
python eda.py                          # summary statistics and figures -> results/
python baseline.py                     # TF-IDF + logistic regression and majority-class baselines
ollama pull llama3.1:8b
python run_llm.py --provider ollama --model llama3.1:8b --conditions P0 P1 --pilot 50
python evaluate.py                     # metrics, Wilcoxon tests and figures for the LLM runs
```
All random draws use seed 42. `run_llm.py` can be stopped and restarted; completed calls are skipped.

## Status
Synopsis stage (final synopsis submitted October 2026) — data extraction and analysis to follow.
