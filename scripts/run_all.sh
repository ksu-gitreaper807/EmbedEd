#!/usr/bin/env bash
# Reproduces every Phase 0/1 number (IMPLEMENTATION_PLAN §3). Later phases get
# wired in as they land. On Colab run the notebook — same commands, same order.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

# --- Phase 0: verify counts, overlap, token lengths (gate G0) --------------
# Colab: --hf. Offline dev on the mini fixture: --dir /tmp/fixture (after
# `python -m embeded.fixtures.make_fixture /tmp/fixture`).
$PY -m embeded.data.prepare_data --hf --verify-spec

# Phase 0.5 throughput (sets TRAIN_PAIRS_CAP/EPOCHS/MAX_LEN — run once, edit
# embeded/settings.py from its output):
#   $PY -m scripts.phase0_throughput --candidates 512:4 512:8 512:16 512:32
# Phase 0.7 s′ acquisition:
#   $PY -m scripts.fetch_sprime

# --- Phase 1: mine, test, gate ----------------------------------------------
$PY -m embeded.mining.semantic_index            # GPU here; all-fragment embeddings
$PY -m embeded.negatives --strategies random,bm25,semantic
$PY -m pytest -q embeded/tests                  # §7.2 adversarial test + pipeline
$PY -m scripts.hardness_diagnostics             # scale of the space + percentiles (no verdict)
$PY -m embeded.hardcheck                         # gate G1, rule D1: C1 < C2 <= C3 and d(C1->C2) >= 0.5

# --- Phase 2: main experiment (PHASE2_PLAN) ---------------------------------
# Entry criteria first: G1 passing, artifacts verified, and the blind 50+50
# audit scored. Then one run per cell — each is idempotent and resumes from its
# own checkpoint, so a disconnect costs one run, not the batch.
$PY -m embeded.train --condition C1 --seed 13 --smoke    # harness check, ~3 min
for c in C1 C2 C3; do for s in 13 14 15; do
  $PY -m embeded.train    --condition $c --seed $s
  $PY -m embeded.evaluate --condition $c --seed $s
done; done
$PY -m embeded.evaluate --condition C0                  # untuned baseline, evaluated once
$PY -m embeded.evaluate --results-table                 # rebuilds report/measurements.md

# --- Phase 3: generalisation on s′ + the figure (PHASE3_PLAN) ------------------
# generalize.py has landed: its lines are live up to the generalisation table.
# embeded/visualize.py and demo/app.py are still the remaining Phase 3 deliverables —
# their lines stay commented until those modules land. Same commands, same order as
# notebooks/Phase3_Colab.ipynb (the CLI contract is that notebook's second cell) — keep
# the two in step. The runs need s′ on this machine and a GPU.
# $PY -m scripts.fetch_sprime                               # s′ (skips size-verified files)
$PY -m embeded.generalize --census --holdout-k 3          # D4: measured holdout, before any F1
$PY -m embeded.generalize --mine --holdout-k 3 --k 20    # --cap defaults to settings.SPRIME_TRAIN_CAP (< main cap, enforced)
$PY -m embeded.generalize --hardness --holdout-k 3        # gate G1s: exit 1 = FAIL -> stop
for c in C1 C2 C3; do for s in 13 14 15; do
  $PY -m embeded.generalize --condition $c --seed $s --holdout-k 3
done; done
$PY -m embeded.generalize --results-table
# $PY -m embeded.visualize --panels C0,C1,C2,C3
# $PY -m demo.app --check
echo "done — see report/measurements.md (commit it)."
