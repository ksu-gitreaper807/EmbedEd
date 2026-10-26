# EmbedEd — current handoff

Updated 26 October 2026. This replaces the older downloaded handoff as the
current-state guide; that version is preserved under
artifacts/phase4_exports/2026-10-23/pre_finalization/HANDOFF.md.
Do not run destructive commands from historical handoffs.

## Goal and boundaries

Finish the existing seven-condition Java clone study, verify its evidence,
reconcile claims and produce a reviewed local release candidate. No new
training, mining, thresholds, audit relabeling or extra seeds. On 26 October the
owner authorized a GitHub `main` push of the organized project; HF publication,
public hosting and paper submission are not included. On 25 October the owner selected the seven-model
webpage as Phase 6, separate from Phase 5's historical closure. Its private
Colab prototype, notebook and live evidence are in `PHASE6_PLAN.md` and
`output/phase6/closure.json`; see `demo/PHASE6_README.md` to launch it.

## Source and immutable evidence

- Existing branch: arena/01a0e714-embeded, base HEAD 13264f112e5f257bbc1bf786dbcda56cd5f56981.
  This is a dirty source tree, not a released commit. Preserve unrelated changes
  and the user's notebooks/Phase4_Colab_output.ipynb.
- HF input: Kusshal/Embed, revision e3144bea7732c20ab2803782282282be4fede7b4.
- Local analysis copy:
  artifacts/phase4_exports/2026-10-23/hf_snapshot_e3144bea.
  snapshot_manifest.json records 524 verified files, SHA-256 hashes and omissions.
  Total 1,474,661,955 bytes. Checkpoints/embeddings/redundant archives are omitted.
- Earlier threshold-repair evidence: a77f8d2d1f26b5168a88822a7d1138f027eee4af.
- New audit protocol P4-AI-AUDIT-2026-10-23; frozen CSVs and rationales are in
  report/phase4_ai_review_c4.csv, c5.csv, c6.csv. Never edit frozen judgments.

## Read first

1. PROJECT_REPORT.md — current plain-language findings.
2. report/final/tables.md and results.json — generated numbers and 111 input hashes.
3. PHASE5_PLAN.md and notebooks/Phase5_Colab.ipynb — guarded closure workflow.
4. report/phase4_audit_completion.md and sanitycheck.md §§9.6–9.9 — voided
   copied-label incident, amendment and score-bound outcomes.

## Scientific facts not to regress

Primary benchmark = C1–C6 seeds 13/14/15, C0 once. Six-seed extension stays
separate. Benchmark F1: C0 .2572; C1 .7533; C2 .2542; C3 .2728; C4 .4208;
C5 .7288; C6 .4688. Primary C3 MAP@R .0770; .1826 is six-seed C6.

All-negative F1=0; all-positive F1=.2406424 at the benchmark prevalence.
Saturated thresholds do not mean all-negative predictions. Replay actual NPZ.
The bouncer compares candidate with anchor, not positive; Jaccard ≥.40 then
skip ten survivors. C5/C6 candidate rejection .27%/13.6% is not contamination.

New AI audit rates: C4 12% [5.6–23.8%], C5 0% [0–7.1%],
C6 22% [12.8–35.2%]; unsure-inclusive 16%/0%/24%.
Historical C2/C3 are also agent review, not human ground truth. C1 is an estimate.
Review passes differ. No independent human validation or causal mechanism proof.

Unseen F1: C5 .7244, C3 .7057, C1 .7031, C0 .6934, C6 .6865,
C4 .6687, C2 .6674. C3 exceeds C0 numerically; do not assert all mined
conditions are below C0 or that nearby means are statistically equal.
Seen F1 uses the threshold-selection validation slice, not an independent test.
Do not compare absolute scores across the two pair regimes.

## Execution evidence

CPU Colab session created for Phase 5: embeded-phase5. The verification
notebook executed five code cells successfully, including the full 176-test
suite, 37 F1 replays with zero mismatches, three audit gates and 300 text checks.
That is an earlier-source pass; final tests/review status belongs in the
Phase 5 closure receipt, not an assumed repeat of that pass.

Executed notebook:
artifacts/phase4_exports/2026-10-23/Phase5_verification_executed.ipynb.
Initial Colab proof:
artifacts/phase4_exports/2026-10-23/phase5_colab_proof.
Generated tables use scripts/phase5_report.py and can rebuild from the local
pinned snapshot without model weights. Paper source is paper/paper.tex;
compiled PDF output is output/pdf/paper.pdf.

On 26 October generated files were moved without changing bytes: the PDF/build
evidence is in `paper/compiled/`, portable source in `paper/latex/`, the figure
ZIP in `report/figures/`, and phase releases/evidence in
`artifacts/releases/phase5/` and `artifacts/releases/phase6/`. Relative symlinks
under `output/` preserve all original receipt paths. Check
`report/output_relocation.json` with `python -m scripts.verify_output_layout`.

Preserve and verify remote outputs before stopping the Phase 5 runtime.
Never stop an unrelated session. No credentials belong in Git, reports,
notebook outputs, command output or release ZIPs.

## Remaining release actions

Local closure, including the later IEEE paper and reference-styled figures, is
recorded in `output/phase5/closure.json`, with current-source test evidence and
the verified ZIP under `output/phase5/`. Check its hashes before reuse.
`report/phase5_closure.json` and the earlier Phase 5 ZIP are historical pre-IEEE
records, not evidence for later changes. All 37 F1 replays pass within 5e-4;
six legacy count differences remain disclosed in the paper/figure JSON. Exact
count/precision/recall equality must not be inferred from the F1 gate.

The historical receipts describe their then-dirty, unpublished source, not
the later GitHub publication. The owner authorized committing and pushing
this organized project to GitHub `main` on 26 October; use normal fast-forward
updates and verify the remote commit. HF publication still requires separate
approval. Six author/affiliation slots remain blank intentionally; venue rules
and contributor details are owner decisions before submission. Never
force-push, reset, clean or overwrite unrelated work.

## Phase 6 demo handoff

Use `notebooks/Phase6_Colab.ipynb` with `output/phase6/EmbedEd_Phase6_source.zip`.
The two editable, unlabeled Java examples are evaluated by C0-C6, all seed 13.
There is one active encoder with sequential checkpoint swapping, not seven
resident models. No Java execution, request-body logging or public tunnel.
235 full-suite tests passed; 14 real T4 scores/decisions matched the original
offline inference path exactly. Cold comparison: 18.0865 seconds, peak GPU
allocation: 673.5 MB. Check the receipt hashes before reusing those claims.
The Phase 6 test runtime was stopped after evidence export. Start a fresh T4
with the notebook when demonstrating; no always-on endpoint is being left up.
No study results, audit labels, thresholds or historical release ZIPs changed.
