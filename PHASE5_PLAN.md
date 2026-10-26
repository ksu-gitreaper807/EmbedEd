# Phase 5 — Evidence freeze, final write-up and release

**Written:** 23 October 2026. **Status (24 October):** local IEEE release
candidate complete subject to the hash-bound gates in `output/phase5/closure.json`;
owner publication approval remains pending. The earlier pre-IEEE receipt is historical.
**Companion:** `notebooks/Phase5_Colab.ipynb` (the standard Jupyter extension is
`.ipynb`, not `.jpynb`). **Runtime:** fresh Colab CPU; no GPU required.

## 1. Goal and boundary

Finish the existing seven-condition study as a reproducible, accurately framed
report and LaTeX paper. Phase 5 completes the table/write-up work inherited
from Phase 4; it is not a new experiment or evidence that Phase 4's entire
closure was already complete. Stop when the verified table, reconciled report,
compiled and visually reviewed paper, evidence manifest and approved release
exist. No training, re-mining, relabeling, threshold tuning, new seeds, or new
dataset/encoder is authorized by this plan.

The initial request was for planning documents. The owner subsequently requested
execution using Colab and a local HF snapshot. Notebook outputs and review
checkboxes start incomplete; only observed checks may complete them.
Publication is a separate, explicit action after review, never a Run-all side
effect. Preserve the existing dirty checkout and user output notebook; do not
reset, clean, force-push, replace an active runtime or overwrite old evidence.

## 2. Verified starting point

| Evidence | Current state | Source |
|---|---|---|
| Threshold persistence | Fixed: save the exact validation-selected threshold, not a rounded display value | `embeded/evaluate.py`; sanitycheck §9.7 |
| Benchmark integrity | 37 F1 replays pass within 5e-4; this is not exact count/precision/recall agreement. Six legacy count differences remain disclosed in the paper/figure record | Evidence HF commit `a77f8d2d1f26b5168a88822a7d1138f027eee4af`; `report/confusion_matrices_export.json` |
| C4/C5/C6 audits | 150 labels frozen before scoring, with rationales and provenance | `report/phase4_ai_audit_freeze.json` |
| Audit results | C4 12% [5.6–23.8%], C5 0% [0–7.1%], C6 22% [12.8–35.2%] | **AI-reviewed; not independently human-validated**; `P4-AI-AUDIT-2026-10-23` |
| Audit artifact sync | 15 uploaded files hash-verified | HF commit `e3144bea7732c20ab2803782282282be4fede7b4` |
| Tests | Earlier full Colab suite: 165 passed; later focused audit/closure suite: 23 passed, three unrelated cases deselected | These are separate historical checks, not a fresh full-suite result |
| Final table/paper | Generated tables, five-page IEEE draft and 54 Matplotlib figures complete; six author blocks intentionally blank | Current receipt: `output/phase5/closure.json` |

Sensitivity rates including unsure are C4 16%, C5 0%, C6 24%. The historical
C2 42% and C3 30% judgments were also **agent-reviewed**, as
`artifacts/audit/AUDIT_JUDGEMENTS.md` states; do not describe them as human
validation. The review passes and some functional-granularity conventions
differ. C1 ≈0–2% is a historical estimate, not a new comparable 50-pair audit.
Wilson intervals quantify sample variation, not reviewer error or bias.

`HANDOFF.md` is useful historical context, not the current state or a source
of permission to run destructive commands. Later append-only records win.

## 3. Stage 5.1 — Freeze inputs and recheck gates

1. Use the existing project branch `arena/01a0e714-embeded`. Record the actual
   source commit and a file-hash manifest, including uncommitted source changes.
   Never claim a clean/released commit when using an uncommitted checkout.
2. Ensure the current closure helper, amended scorer, updated Phase 4 notebook,
   protocol/freeze files and outcome records are present. A fresh clone of the
   remote branch may lack these local changes: synchronize them deliberately
   first. The Phase 5 notebook stops rather than silently using old code.
3. Pull a selective snapshot at immutable HF commit
   `e3144bea7732c20ab2803782282282be4fede7b4`: benchmark metrics + NPZ for
   all 37 runs, 18 s′ metrics + C0 transfer, the audit sheets/keys/results,
   C4/C5/C6 triples, canonical fragments, test pairs and audit provenance.
   No checkpoints, embeddings, raw s′ archive, or GPU allocation are needed.
4. Keep inputs and generated outputs in a separate Phase 5 workspace. Preserve
   the original report as input; never replace the repository's report on pull.
5. Run the full offline suite on the actual source and save its log. Re-run
   `audit_mismatches()` across all 37 runs and `verified_audits()` across the
   three current sheets. Verify reviewed code against key-indexed corpus text.
   Stop on any missing artifact, mismatched frozen hash, stale score or drift.

**Exit:** fresh passing gates and a source/input manifest. A network or source
availability failure is not a reason to regenerate mining or relax checks.

## 4. Stage 5.2 — Assemble the one-table deliverable

Reuse the tested Phase 4 section-5b builder; do not hand-transcribe means.
Retain the section name `## Phase 4 — the one table` so report consumers and
history keep one canonical location. Phase 5 records that it finished it.

- **Panel A:** C0 once; C1–C6 over exactly seeds 13/14/15. Include F1 mean ±
  sample SD, mean MAP@R, thresholds in seed order, and pinned counts with the
  matching denominator. Seeds 16–18 remain a separate six-seed extension
  summary/appendix, not mixed into a three-seed caption.
- **Panel B:** 18 s′ runs over seeds 13–15, F1_seen, F1_unseen, and the mean
  and sample SD of each run's Δ = seen − unseen. Include the single untuned
  C0 transfer anchor. Do not compare absolute F1 across different pair regimes.
- **Panel C:** audit counts, unsure counts, point/sensitivity rates, Wilson
  intervals and reviewer provenance. C1 is an estimate; C2/C3 are historical
  agent review; C4/C5/C6 use the disclosed post-results AI amendment.
- Label 0.27% (C5) and 13.6% (C6) as **candidate-filter rejection statistics**,
  not residual false-negative contamination rates.

**Exit:** generated table, exact run inventory, SHA-256 of its source inputs,
and correct seed/condition/metric provenance. No threshold reselection.

## 5. Stage 5.3 — Reconcile claims and terminology

Create the limitations and claim ledger before revising conclusions. All
headlines must map to a generated table row or named evidence artifact.

| Existing problem | Required correction |
|---|---|
| “All-negative F1 ceiling ≈0.2406” | All-negative F1 is 0. At the measured prevalence π, all-positive F1 is 2π/(1+π), approximately 0.2406. Neither is a performance ceiling for a useful classifier. |
| C3 MAP@R copied as 0.1826 | Recompute from the selected C3 metric files; 0.1826 belonged to the historical C6 summary. Never substitute a six-seed value into a three-seed table. |
| C3 = C4 = 30% audit argument | Void under §9.6; do not restore it. New C4 is 12% under a different review pass. |
| AI audit called human/fully blind/independent | Correct all surfaces, including historical C2/C3 attribution. Disclose prior outcome knowledge and rubric/granularity differences. |
| C5 “confirmed clean control” | Its AI point estimate is consistent with 0–2%; 0/50 has Wilson upper bound 7.1%. Retain C1 as the headline condition. |
| C6 “H1 survives/H2 refuted” from audit | Point 22% is outside 25–35%, with an overlapping interval. Preserve scorer-state → F1 → audit endpoint order; the audit does not isolate dose from similarity. |
| Every contaminated condition ≤ untuned C0 unseen | False as stated: historical C3 mean 0.7057 exceeds C0 0.6934. Report actual means, spread and saturation; do not infer significance or equality without an appropriate analysis. |
| “0.95 discrepancy resolved” | Protocol/model differences are a limitation or hypothesis, not a demonstrated explanation of the entire gap. No tuning to chase the published number. |
| Bouncer compares candidate with positive | Code applies Jaccard against the **anchor**. Describe the actual implementation. |
| Supply rejection = contamination remaining | Distinct quantities with different denominators; keep them separate. |

Reconcile `report/measurements.md`, `PROJECT_REPORT.md`, `paper/paper.tex`,
`TIMELINE.md`, `HANDOFF.md`, relevant README text and current plan status.
Preserve historical append-only findings in `sanitycheck.md`/SCOPE; add dated
corrections rather than erasing the record. The source notebook currently
reuses some older wording: the final table needs the explicit method note
included in the Phase 5 notebook, followed by editorial review.

**Exit:** reviewed claim ledger, consistent denominators/attribution and a
bounded conclusion: negative selection changes performance, but this audit
does not independently certify the causal mechanism.

## 6. Stage 5.4 — Paper and reproducibility review

1. Keep the manuscript in LaTeX. Compile the saved draft using an available
   compiler (the notebook does not install a TeX distribution automatically).
   Record the real compiler result/log; opening a preview is not compilation.
2. Visually review the generated PDF: tables fit, figures/captions agree,
   references resolve, and no draft/pending placeholders imply completed work.
   Repair `\texttt{}` escaping and malformed bibliography entries if present.
3. Verify the report and paper against the generated table and audit metadata.
   Reuse figures unless source metrics changed; UMAP is illustration, not proof.
4. Run `bash -n scripts/run_all.sh`, the full offline suite, and the integrity
   gates again. Do not execute `run_all.sh`: it trains/re-mines and is outside
   this phase. Demo inference checking is optional and separate if weights are
   unavailable; do not download checkpoints merely to claim closure.

**Exit:** compiled PDF, visual review, passing tests and content-hash-bound
review sign-off. The notebook's review file begins with every field false.
It cannot judge a paper's truth or visual quality automatically.

## 7. Stage 5.5 — Package, approve, publish

The notebook creates a local release ZIP only after the review gates are
explicitly completed. Its allowlisted contents are the reconciled report,
draft manuscript/PDF, reviewed project narrative/timeline/handoff, claim
ledger, audit protocol/freeze/results, test log, table inputs and source/input
manifests. No token files, cache directories, `.git`, checkpoints, raw
downloads, or broad workspace copies are bundled.

Then, from the consolidating writer only:

1. Review the exact Git diff, source and input hashes, target repository,
   destination branch and staged upload paths.
2. Obtain the owner's release approval. This request does not publish a release.
3. Synchronize the reviewed docs/source to the existing branch (no force push),
   and publish only approved report/release files to `Kusshal/Embed@main`.
   Use a current-parent guard so a stale writer cannot replace newer work.
4. Download from the resulting immutable commit and compare hashes; record
   Git and HF commit IDs separately. An upload attempt is not verification.
5. Release only the runtime created for Phase 5 after preserving its outputs.
   Do not stop an unrelated/active Phase 4 session or delete unsaved files.

**Exit:** release receipt with verified immutable links and no unresolved
required checks. The companion notebook contains **no automatic Git/HF push**.

## 8. Later work — proposed, not authorized Phase 5 work

After the current report is complete, the owner may select a separate amended
experiment: human replication with explicit adjudication, a denoising δ-ladder,
FN-weighted loss, or a second encoder. Fix the exact algorithm/threshold
definition, sample/rubric, compute budget and endpoints before execution.
These are proposals, not an automatic sequence of Phase 5 jobs. No experiment
starts merely because this notebook reaches its end.

## 9. Acceptance checklist / execution log

- [x] Current local source identified by hashes without discarding local work; Git publication remains pending.
- [x] Pinned inputs downloaded selectively and hash-manifested: 524 files.
- [x] Full current-source CPU suite passed; all 37 F1 replays pass at 5e-4. Six legacy count discrepancies disclosed, not silently overwritten.
- [x] All audit identity/provenance/outcome gates pass; 150 frozen judgments and 300 fragments verified; no relabeling.
- [x] Three-panel table rebuilt with exact seed denominators; all seven result files identical.
- [x] Claims, reviewer attribution and terminology reconciled across current documents.
- [x] Five-page IEEE draft compiled and visually reviewed; 54 figure PNG hashes verified.
- [x] Current local release ZIP allowlisted, payload-hash verified and credential-pattern scanned. See receipt before treating this checklist as evidence.
- [ ] Owner-approved Git/HF release verified by immutable hashes.

| Date | Action | Evidence / status |
|---|---|---|
| 2026-10-23 | Phase 5 plan and companion notebook authored | Planning deliverables only; real-data execution, editorial review and publication remain pending. |
| 2026-10-24 | Local closure and later IEEE/figure revisions consolidated | Current-source CPU tests, repeated pinned-input/audit/count gates and current allowlisted release candidate; actual results/hashes in `output/phase5/closure.json`. No publication or new benchmark training. |

Local closure does not mean publication or conference submission is complete.
Six author/affiliation slots remain blank as requested; venue-specific rules
were not supplied. The seven-model webpage and human replication remain parked.
