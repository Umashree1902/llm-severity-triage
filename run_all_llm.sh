#!/usr/bin/env bash
# Order: the RQ2 and RQ4 subsets first, so every research question has results even if the
# full-sample P0/P1 run is stopped early. Every step is resumable: rerun this script to continue.
set -e
cd "$(dirname "$0")/src"
if command -v python >/dev/null 2>&1; then PY=python; else PY=python3; fi
P="${PROVIDER:-ollama}"     # ollama | openai | anthropic
M="${MODEL:-llama3.1:8b}"
export WORKERS="${WORKERS:-2}"
"$PY" run_llm.py --provider "$P" --model "$M" --conditions P1 --subset rq2 --repeats 3
"$PY" run_llm.py --provider "$P" --model "$M" --conditions P1_reworded_1 P1_reworded_2 --subset rq2
"$PY" run_llm.py --provider "$P" --model "$M" --conditions P1 P2 --subset validation
"$PY" run_llm.py --provider "$P" --model "$M" --conditions P1 P2 --subset test
"$PY" run_llm.py --provider "$P" --model "$M" --conditions P0 P1 --subset rq3
"$PY" run_llm.py --provider "$P" --model "$M" --conditions P0 P1
"$PY" evaluate.py && "$PY" evaluate_rq2_rq4.py
