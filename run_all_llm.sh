#!/usr/bin/env bash
# Full local-model run for RQ1-RQ4 (about 11 hours on a 16 GB Mac). Safe to stop and restart:
# completed calls are skipped. Run with:  caffeinate -i bash run_all_llm.sh
set -e
cd "$(dirname "$0")/src"
M="${MODEL:-llama3.1:8b}"
echo "== RQ1/RQ3: P0 and P1 on the full sample";        python run_llm.py --provider ollama --model "$M" --conditions P0 P1
echo "== RQ2a: identical P1 repeats (runs 2-3)";        python run_llm.py --provider ollama --model "$M" --conditions P1 --subset rq2 --repeats 3
echo "== RQ2b: reworded P1 prompts";                    python run_llm.py --provider ollama --model "$M" --conditions P1_reworded_1 P1_reworded_2 --subset rq2
echo "== RQ4: P2 on validation and test subsets";      python run_llm.py --provider ollama --model "$M" --conditions P2 --subset validation
python run_llm.py --provider ollama --model "$M" --conditions P2 --subset test
echo "== Evaluation";                                   python evaluate.py && python evaluate_rq2_rq4.py
echo "All done."
