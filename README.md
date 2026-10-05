# Trusting AI in Defect Triage: Agreement, Consistency and Bias in LLM Severity Labels

QM640 Data Analytics Capstone — Walsh College  
Author: Umashree Pandiarajan

## Purpose
This project evaluates how far QA teams can trust severity labels that large language models (LLMs) assign to software defects. It measures:

- **RQ1 – Agreement:** LLM severity vs. human triagers, relative to human–human agreement
- **RQ2 – Consistency:** label stability across repeated runs and equivalent prompts
- **RQ3 – Bias:** over- or under-escalation, and the effect of product context
- **RQ4 – Guardrails:** effect of severity definitions, similar past defects and confidence-based deferral on correctness and calibration

## Data
Public defect reports from Mozilla Bugzilla (https://bugzilla.mozilla.org), retrieved through the Bugzilla REST API.

- Primary sample: defects filed 1 January 2025 – 30 June 2026 with severity S1–S4
- Baseline / retrieval set: defects filed 2021–2024 (no overlap with the primary sample)
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
│   ├── run_llm.py           # prompt conditions P0–P2, repeated runs
│   └── retrieve.py          # similar-defect retrieval for P2
├── prompts/                 # versioned prompt templates
├── notebooks/               # RQ1–RQ4 analyses
└── results/                 # tables and figures
```

## Status
Synopsis stage — data extraction and analysis to follow.
