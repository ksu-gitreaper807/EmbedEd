# EmbedEd — evidence-reconciled project report

Updated 24 October 2026. This supersedes the 17 October narrative, preserved
locally under artifacts/phase4_exports/2026-10-23/pre_finalization.
The study's current results are in [the generated tables](report/final/tables.md)
and [machine-readable results](report/final/results.json). No new training was
performed during this reconciliation. Public release is a separate approval step.

## What we tested, and why

Can changing only the way training negatives are picked change Java clone
detection and generalisation to unseen functionality?

Seven conditions use GraphCodeBERT weights as a mean-pooled text encoder
(no explicit data-flow graph input): C0 untuned; C1 random negatives; C2 BM25;
C3 semantic mining; C4/C5/C6 apply the bouncer to C2/C1/C3 respectively.
The bouncer cuts candidate–anchor token Jaccard ≥0.40 and skips the first ten
survivors. It does not compare candidates against the positive snippet.

The objective, margin 0.10, 512-token limit, batch 8, learning rate 2e-5 and
one-epoch recipe are held fixed. Benchmark training is capped at 32,000 triples.
Known-clone exclusion cannot remove clones that the dataset never labelled.

## Current findings and their boundaries

- The primary benchmark uses seeds 13–15, not all available seeds. C0 is one
  untuned reference. C1 F1 is 0.7533 ±0.0073, versus C2 0.2542 ±0.0108 and
  C3 0.2728 ±0.0136. Filtered C4/C6 rise to 0.4208/0.4688, below C1.
  C5 is 0.7288. These are descriptive mean comparisons, not significance tests.
- Benchmark C3 MAP@R is 0.0770. The old 0.1826 transcription belongs to the
  separate six-seed C6 extension. The repository's MAP@R ranks all pairs in a
  split; it is not averaged over retrieval queries.
- All-negative F1 is **0**. All-positive F1 is **0.2406424** for the benchmark's
  56,820 positives among 415,416 pairs. This is a weak baseline, not a ceiling.
  C2/C3 do not answer non-clone to everything: their primary predicted-positive
  ranges are 48.1–87.3% and 33.6–42.7%. A near-one threshold alone cannot tell us
  what a classifier predicts.
- The separate s′ experiment uses 18 trained runs (three seeds each) and a single
  C0 transfer reference. Unseen means are C5 0.7244, C3 0.7057, C1 0.7031,
  C0 0.6934, C6 0.6865, C4 0.6687, C2 0.6674. C3 numerically exceeds C0;
  the old blanket claim to the contrary is withdrawn. Saturation warrants
  inspection, not replacing C3's measured number with an asserted tie.
- The gap Δ is seen minus unseen F1; its SD is computed from per-seed gaps.
  Seen F1 is on the threshold-selection validation slice (400 pairs), not an
  independent seen test; unseen evaluation uses 600 pairs. Small gaps may mean
  both endpoints are weak. Do not compare absolute F1 across the benchmark and s′.
- New C4/C5/C6 audits are 6/50, 0/50 and 11/50 reviewed clones:
  **12% [5.6–23.8%], 0% [0–7.1%], 22% [12.8–35.2%]** (Wilson 95%).
  Unsure-inclusive sensitivity rates are 16%, 0%, 24%.
  All judgments are **agent/AI review, not independently human-validated**.
  Historical C2/C3 42%/30% also came from agent review; C1 ≈0–2% is an estimate.
  Methods differ, so these are not controlled cross-reviewer comparisons.
- C5 is not proved contamination-free. C6's interval overlaps its predicted
  25–35% band; neither mechanism explanation is isolated. Candidate-rejection
  shares 0.27%/13.6% are not residual contamination rates.

## Work completed across the phases

| Stage | Work | Why it matters |
|---|---|---|
| Phase 0 | Verified data counts, source identity, lengths and T4 sizing; acquired s′ | Fixes the input and compute budget |
| Phase 1 | Shared-anchor negative mining, exclusions and measured hardness gates | Checks that mining changes the intended training input |
| Phase 2 | Benchmark C0 plus C1–C6, with six trained seeds available | Measures recipe differences; primary table still uses three seeds |
| Phase 3 | Functionality-held-out experiment, fixed holdout IDs 10/13/14; illustration and single-model demo tooling | Tests a different generalisation endpoint |
| Phase 4 | Repaired threshold persistence; voided copied audit labels; amended and completed 150 new reviews with hashes/rationales | Restores artifact consistency and discloses the audit's limits |
| Phase 5 | CPU Colab gates, regenerated tables, local pinned HF analysis copy and evidence-reconciled write-up | Makes the existing results inspectable without new experiments |

The historical G1 rule change and earlier failed gate remain in the records.
An earlier equal-rate C3=C4 mechanism argument depended on invalid copied labels
and is withdrawn. Better traceability does not convert AI labels into human
ground truth. Similarity, contamination dose and hardness were not independently
manipulated, so their causal roles are unresolved.

The published ~0.95 gap is not fully explained by identifying model/protocol
differences. This study uses a constrained mean-pooled adaptation, truncated
inputs, imperfect labels and few seeds. UMAP remains illustration, not proof.

## Verification and reproducibility

The evidence input is pinned to HF Kusshal/Embed revision
e3144bea7732c20ab2803782282282be4fede7b4. The analysis snapshot has 524
hash-verified files (1,474,661,955 bytes); weights, embedding arrays and redundant
archives are omitted and listed in its manifest. It is not a complete model backup.

The Colab notebook verification passed 176 tests before the final reporting
helper was added. All 37 benchmark NPZ files replay stored F1 within 5e-4;
all 150 new audit identities and 300 displayed fragments pass the closure gates.
Later checks and content-hash-bound review are recorded in the Phase 5 closure
receipt rather than silently treating this earlier pass as a final-source pass.

- [Phase 5 workflow](PHASE5_PLAN.md) and [notebook](notebooks/Phase5_Colab.ipynb)
- [Audit completion](report/phase4_audit_completion.md)
- [Paper source](paper/paper.tex); local compiled PDF under output/pdf/paper.pdf
- [Current handoff](HANDOFF.md); [append-only corrections](sanitycheck.md)
- Rebuild tables: python -m scripts.phase5_report --snapshot SNAPSHOT --output report/final

## What is next, and what is parked

The current IEEE local release candidate, tested-source manifest and reviewed
payload hashes are under `output/phase5/`; its `closure.json` is the current
completion receipt. The older Phase 5 receipt/ZIP remains a historical pre-IEEE
record. The current paper has five pages and 54 Matplotlib figures are provided
separately. Six legacy confusion-count discrepancies are disclosed; F1 agreement
within 5e-4 does not imply exact count/precision/recall equality.

Obtain explicit owner approval before publishing a Git/HF release. Six author
blocks remain blank as requested, and venue-specific requirements await an
owner choice before submission. Do not run the training driver as a verification
step. No new seeds, re-mining or relabeling belong to closure.

On 25 October the owner selected the [Phase 6 webpage](PHASE6_PLAN.md): two
editable Java pairs evaluated by all seven conditions on Colab, with scores,
frozen thresholds and decisions. The private prototype and notebook are built;
14 real T4 results matched the existing inference path, and 235 tests passed.
The verification runtime was stopped after export. This demonstrator is not
benchmark evidence, human validation or public permanent hosting, and does not
alter the historical Phase 5 release. Independent human audit replication and
controlled denoising experiments remain separate future decisions.
