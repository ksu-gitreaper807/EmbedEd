#!/usr/bin/env bash
# Reproduces every number in report/measurements.md (all phases). On Colab run
# the notebooks instead - same commands, same order, with per-cell gates.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

# --- Phase 0: corpus verify + s' acquisition ---------------------------------
$PY -m embeded.data.prepare_data --hf --verify-spec          # gate G0
$PY -m scripts.fetch_sprime                                  # s' gate (zenodo)

# --- Phase 1: embeddings, mining, tests, gates --------------------------------
$PY -m embeded.mining.semantic_index                          # GPU
$PY -m embeded.negatives --strategies random,bm25,semantic,filtered_random,filtered_bm25,filtered_semantic
$PY -m pytest -q embeded/tests
$PY -m scripts.hardness_diagnostics                           # no verdict; scale + floors
$PY -m embeded.hardcheck                                      # gate G1 (rule D1)

# --- Phase 2: benchmark campaign (C0 + C1-C6, seeds 13-15) --------------------
$PY -m embeded.train --condition C1 --seed 13 --smoke         # harness check (~3 min)
for c in C1 C2 C3 C4 C5 C6; do for s in 13 14 15; do
  $PY -m embeded.train    --condition "$c" --seed "$s"
  $PY -m embeded.evaluate --condition "$c" --seed "$s"
done; done
$PY -m embeded.evaluate --condition C0                        # untuned, evaluated once
$PY -m embeded.evaluate --results-table                       # main table

# --- Phase 3: s' generalisation + figure + demo (PHASE3_PLAN) -----------------
$PY -m embeded.generalize --census --holdout-k 3              # D4 rule, before any F1
$PY -m embeded.generalize --mine --holdout-k 3 --k 20 --cap 16000
$PY -m embeded.generalize --hardness --holdout-k 3            # gate G1s
$PY -m embeded.generalize --condition C1 --seed 13 --smoke --holdout-k 3
for c in C1 C2 C3 C4 C5 C6; do for s in 13 14 15; do
  $PY -m embeded.generalize --condition "$c" --seed "$s" --holdout-k 3
done; done
for c in C1 C2 C3 C4 C5 C6; do for s in 13 14 15; do
  $PY -m embeded.generalize --condition "$c" --seed "$s" --holdout-k 3 --transfer
done; done
$PY -m embeded.generalize --condition C0 --seed 13 --holdout-k 3 --transfer
$PY -m embeded.generalize --results-table
$PY -m embeded.visualize --panels C0,C1,C2,C3,C4,C5,C6
$PY -m demo.app --check

# --- optional sync (off by default: writers only) ------------------------------
# $PY -m scripts.hf_artifacts push --include-report
