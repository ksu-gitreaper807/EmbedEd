# Phase 5 local release candidate

The current completion receipt is `closure.json`; `review.json` binds the
AI technical/editorial review to exact payload hashes. Neither is owner
publication approval, independent human validation or a submission certificate.
The package is generated only after the full offline suite and evidence gates pass.
Current-source validation: **204 tests passed, no skips**, on a fresh CPU-only
Colab session. Its 71 source/input hashes and downloaded test log were checked
locally before the runtime was stopped. Initial transfer-fixture failures are
retained in the evidence; no test was removed or skipped to obtain the final pass.

## Contents and evidence

- Current five-page IEEE manuscript, source, class, and portable LaTeX ZIP.
- All 54 Matplotlib PNGs: 37 model/seed plots, 14 condition means, two mean
  comparison boards and the 37-run overview. Row shares and frozen thresholds
  are preserved; the six legacy JSON count discrepancies remain disclosed.
- Generated numerical tables, frozen AI-audit protocol/reviews/results, current
  narrative/handoff/plans, the tested source and the Phase 4/5 source notebooks.
- `evidence/`: full-suite log and dependency/source hashes, scientific acceptance,
  seven reviewed claim-ledger entries and bibliographic identity checks.
- `release_manifest.json` inside the ZIP hashes every payload file. The external
  completion receipt hashes the ZIP itself, avoiding a circular manifest.

Inputs are pinned to HF `Kusshal/Embed` revision
`e3144bea7732c20ab2803782282282be4fede7b4`. The ZIP includes its snapshot
manifest, not the 1.47 GB analysis snapshot, model weights or embeddings.
Credentials, caches, `.git` and the user's separate output notebook are excluded.
The credential-pattern scan has zero non-fixture matches. Three token-shaped
synthetic literals in the redaction unit tests are explicitly reviewed and
recorded by location/hash, not silently treated as live credentials. Pattern
scanning cannot prove the absence of every possible secret.

## Scope and remaining owner decisions

This is a local, dirty-source release candidate, not a published Git/HF commit.
The earlier pre-IEEE Phase 5 ZIP and `report/phase5_closure.json` are historical
and remain untouched; this package includes the later paper and figure revisions.

Six author/affiliation blocks remain blank intentionally. Before submission,
the owner must supply contributor details and confirm the venue's page limit,
anonymity/copyright and submission requirements. IEEE class formatting is not
IEEE PDF eXpress certification. Before publication, review the exact files and
destinations and explicitly approve a guarded Git/HF upload; then download and
verify immutable release hashes. Nothing in this workflow publishes automatically.

The seven-model Colab webpage, independent human audit replication and new
experiments remain separate future work. No new benchmark training, mining,
threshold selection, relabeling, GPU allocation or remote write is performed.
