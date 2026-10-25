"""Shared paths, constants and helpers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
PROMPTS = ROOT / "prompts"

for d in (RAW, PROCESSED, RESULTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

SEVERITIES = ["S1", "S2", "S3", "S4"]
ORDINAL = {"S1": 4, "S2": 3, "S3": 2, "S4": 1}        # higher = more severe
LABEL = {"S1": "Critical", "S2": "High", "S3": "Medium", "S4": "Low"}
SEED = 42

# Study windows
PRIMARY_FROM, PRIMARY_TO = "2025-01-01", "2026-07-01"   # filed 1 Jan 2025 - 30 Jun 2026
BASELINE_FROM, BASELINE_TO = "2021-01-01", "2025-01-01"  # filed 2021-2024 (training only)

# Stratified design (from the synopsis)
PER_STRATUM = 800        # S2-S4 in the main sample; S1 = census
OVERSAMPLE = 1100        # pulled per stratum before exclusions so 800 remain after cleaning
BASELINE_PER_STRATUM = 1000
SUBSETS = {"rq2": 469, "rq3": 825, "rq4_test": 469, "validation": 200, "reliability_per_level": 38}
