#!/usr/bin/env bash
# Reproduces every Phase 0/1 number (IMPLEMENTATION_PLAN §3). Later phases get
# wired in as they land. On Colab run the notebook — same commands, same order.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

# --- Phase 0: verify counts, overlap, token lengths (gate G0) --------------
# Colab: --hf. Offline dev on the mini fixture: --dir /tmp/fixture (after
# `python -m code.fixtures.make_fixture /tmp/fixture`).
$PY -m code.data.prepare_data --hf --verify-spec

# Phase 0.5 throughput (sets TRAIN_PAIRS_CAP/EPOCHS/MAX_LEN — run once, edit
# code/settings.py from its output):
#   $PY -m scripts.phase0_throughput --candidates 256:32 256:16
# Phase 0.7 s′ acquisition:
#   $PY -m scripts.fetch_sprime

# --- Phase 1: mine, test, gate ----------------------------------------------
$PY -m code.mining.semantic_index            # GPU here; all-fragment embeddings
$PY -m code.negatives --strategies random,bm25,semantic
$PY -m pytest -q code/tests                  # §7.2 adversarial test + pipeline
$PY -m code.hardcheck                         # gate G1: C1 < C2 <= C3, visible gap

# --- Phase 2+ (TODO, wired in later) ----------------------------------------
# $PY -m code.train --condition C1 --seed 13 ...   (18 runs)
# $PY -m code.evaluate ; $PY -m code.generalize
# $PY -m code.visualize ; tables + limitations first (FINAL_SPEC §12)
echo "Phase 0/1 complete — see report/measurements.md (commit it)."
