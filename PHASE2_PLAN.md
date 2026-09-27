# Phase 2 — Main experiment and false-negative audit

**Status:** Planned; Phase 2 training has not started.
**Purpose:** Measure whether the negative-mining strategy (C1 random, C2 BM25, C3 semantic) affects clone-detection performance, while checking whether the mined negatives contain functional clones.

This file is the Phase 2 plan and running record. Mark work as complete only when its outputs exist and the corresponding measurements have been recorded in `report/measurements.md`.

## 1. Starting point and Phase 1 work already completed

Before Phase 2, the project has:

- Verified the CodeXGLUE/BigCloneBench dataset counts and fixed the corpus and train/validation/test split.
- Measured the Colab T4 training throughput and frozen the training budget at `BATCH = 8`, `TRAIN_PAIRS_CAP = 32,000`, `EPOCHS = 1`, `MAX_LEN = 512`, using fp16 autocast.
- Acquired and checked the BCB s′ data for the later generalisation phase.
- Implemented and tested C1/C2/C3 mining and produced hardness diagnostics. The original G1 check with an absolute cosine margin of 0.02 failed: the C1→C2 gap was about 0.015.
- Found evidence that the failure is likely due to the absolute threshold not reflecting the narrow cosine range: C1→C2 has `d ≈ 0.63`; C2 negatives have median corpus percentile ≈ 91.9, compared with ≈ 50.1 for C1. The original failure must remain in the history.
- Implemented blinded 50+50 false-negative audit tooling; the audit itself still needs to be conducted and scored.

The repository does not yet contain Phase 2 training/evaluation modules. Building and testing these is part of the plan below.

## 2. Entry criteria — do not train until all are satisfied

1. **Freeze the G1 rule.** Record the owner-approved, scale-aware interpretation of “visible gap” before training. Recommended rule from `PROGRESS.md`: retain `C1 < C2 ≤ C3`, require standardized C1→C2 gap `d ≥ 0.5`, and report the corpus-percentile diagnostics. Do not silently change `HARDNESS_MARGIN` or erase the original 0.02-margin FAIL. Update the gate implementation, tests, settings/version, and documentation together if this rule is adopted.
2. **Regenerate/retrieve the exact Phase 1 artifacts** for the frozen corpus/version: embeddings and all three conditions’ triples. Verify hashes/metadata and ensure all conditions use the same anchors, positives, `k`, and split-safe candidate corpus. If mining changes, re-mine all three conditions together and rerun tests, gate, and diagnostics.
3. **Pass the updated G1 and mining tests.** For uncertainty estimates, resample anchors (not individual negatives as if all 32,000 were independent), since each anchor contributes multiple negatives.
4. **Complete and record the blind audit.** Generate 50 C2 and 50 C3 pairs with `python -m scripts.audit_sample make`; label without opening the key; score with `python -m scripts.audit_sample score`. Record the rubric, rates, Wilson intervals, and unsure upper bounds. Decide how the observed false-negative rates affect interpretation before training.
5. **Run an end-to-end smoke test** on a tiny subset: one condition, one seed, forward/backward pass, checkpoint save/reload, validation evaluation, and test-metric output. Confirm the full-sized configuration fits the target GPU.
6. **Freeze the experiment configuration.** Record model/tokenizer revision, loss and loss margin, optimizer, scheduler, batch size, epochs, max length, precision, sampling order, seed list, threshold-selection policy, and artifact/version hashes. Use the same settings in C1/C2/C3 except for the negative triples and the planned random seeds.

If a precondition fails, stop and fix it before launching the nine full runs.

## 3. Main experiment

### 3.1 Experimental matrix

| Condition | Negative source | Seeds | Runs |
|---|---|---:|---:|
| C1 | Random negatives | 13, 14, 15 | 3 |
| C2 | BM25/lexical negatives | 13, 14, 15 | 3 |
| C3 | Semantic negatives | 13, 14, 15 | 3 |
| C0 | Untuned pretrained model; no training | Evaluate once | 1 evaluation |

Use the same train/validation/test definitions, positive pairs, training-pair cap, number of negatives (`k = 20`), model initialization, and all non-condition hyperparameters across conditions. A seed may affect initialization/training order as specified, but must not create a different data split for one condition. Log the exact seed and artifact hashes for every run.

### 3.2 Training objective and implementation

- Implement the Phase 2 training entry point, reusable loss/data-loading code, checkpointing, and deterministic seed setup under `embeded/`.
- **Recommended objective:** explicit-triplet `TripletLoss`/triplet-margin loss. The mined negative is an explicit member of each training example, so this objective directly tests the negative-selection intervention. Use one frozen loss and margin for all conditions; do not tune them separately by condition.
- Keep the measured resource recipe: `MAX_LEN = 512`, `BATCH = 8`, `TRAIN_PAIRS_CAP = 32,000`, `EPOCHS = 1`, fp16 autocast, and no gradient checkpointing, unless a documented smoke-test failure requires a new measured configuration. Any such change requires updating the throughput basis and rerunning all conditions consistently.
- Save resumable checkpoints and per-run configuration/metrics. Upload/checkpoint completed artifacts according to the repository’s existing artifact workflow; do not commit large model weights or datasets to Git.
- Run C0 with the same tokenizer, pooling, truncation, and scoring path as trained models, but without fine-tuning.

### 3.3 Evaluation protocol

- Use the held-out **test** split only for final metrics. Never use test outcomes to choose a threshold or hyperparameters.
- On the **validation** split, select and record the F1 decision threshold for each condition/model according to one frozen policy. Apply that threshold unchanged to its test predictions.
- Report primary **F1**, with **precision** and **recall** alongside it. Report threshold-free **MAP@R** as the secondary metric.
- Report per-seed results and aggregate mean plus spread across the three seeds; do not report only the best seed. Include C0 as a reference and clearly distinguish it from the three trained conditions.
- Keep prediction scores, selected validation thresholds, test outputs, and configuration metadata so the table can be regenerated.
- Compare results with the published CodeXGLUE fine-tuned-CodeBERT F1 sanity reference (approximately 0.95, subject to the cited protocol/model differences). Treat a large discrepancy as a harness/protocol check, not as a reason to tune against the test set. Verify the metric implementation, pair ordering, labels, and threshold handling before interpreting results.

## 4. Phase 2 deliverables

Phase 2 is complete when all of the following are available and reproducible:

1. Tested training and evaluation code, including a working smoke test.
2. Nine completed runs (C1/C2/C3 × three fixed seeds) and one C0 evaluation, with checkpoints/configuration/predictions saved outside Git as appropriate.
3. A regenerated main-results table in `report/measurements.md` with per-seed F1, precision, recall, MAP@R, validation threshold, and mean/spread.
4. The completed blind C2/C3 audit, its scored results and uncertainty, and a documented interpretation of the audit.
5. A G2 review: explain any unexpected or implausible metrics and fix the harness before claiming results. Record whether the sanity comparison is within a reasonable range and why.
6. A concise conclusion limited to what the main benchmark and audit support, including dataset/label and truncation limitations.

**Not Phase 2:** training on s′ or computing the unseen-functionality generalisation gap, UMAP plots, the demo, and the paper/deck. Those belong to Phase 3 and later in `IMPLEMENTATION_PLAN.md`.

## 5. Risks and controls

- **Hardness-rule hindsight:** freeze the scale-aware rule before training; preserve the original FAIL and all later gate runs.
- **False negatives / incomplete labels:** use the blinded audit and report its uncertainty; token-Jaccard is only a proxy and must not be treated as ground truth.
- **Seed/condition confounding:** same code, split, budget, objective, and settings across all conditions; retain per-run metadata.
- **GPU OOM or interrupted sessions:** smoke-test on the target runtime, checkpoint each run, and resume only from verified artifacts.
- **Test leakage through threshold selection:** choose thresholds on validation data only; evaluate the test split once per finalized run.
- **Overclaiming:** describe the result as the effect of negative-selection strategies under this dataset/model/protocol, not as general proof of semantic clone detection.

## 6. Phase 2 execution log

No Phase 2 training or evaluation has been completed yet. Append dated entries here as work is actually performed; include command/configuration, artifacts, outcome, and any deviations from this plan.

| Date | Work performed | Evidence / output | Status / deviations |
|---|---|---|---|
| — | Planning document created; execution not started | `PHASE2_PLAN.md` | Pending entry criteria |
