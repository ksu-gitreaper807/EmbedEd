# Phase 4 audit completion — 23 October 2026

All **150 judgments are complete and scored**, under amendment
`P4-AI-AUDIT-2026-10-23`. These are **AI-reviewed; not independently
human-validated**. They must not be described as human or fully blind judgments.

| Condition | Clone | Not clone | Unsure | AI-reviewed rate; not independently human-validated | Wilson 95% | Rate including unsure |
|---|---:|---:|---:|---:|---|---:|
| C4: filtered BM25 | 6 | 42 | 2 | 12% | 5.6–23.8% | 16% |
| C5: filtered random | 0 | 50 | 0 | 0% | 0–7.1% | 0% |
| C6: filtered semantic | 11 | 38 | 1 | 22% | 12.8–35.2% | 24% |

Wilson intervals quantify sampling uncertainty only. They do not account for
AI-review error, shared systematic mistakes, or knowledge of benchmark outcomes.
There is one AI reviewer and no measured inter-rater agreement.

## What the results permit

- C4 is below the registered 42% ceiling but not contamination-free. The voided
  C3=C4 same-rate comparison is removed, not restored.
- C5's point estimate is within the predicted 0–2% band. Zero observed clones
  does not demonstrate that its population contamination is at most 2%; its
  sampling upper limit is 7.1%. Keep C1 as the report/demo headline condition.
- C6's point estimate is below the predicted 25–35% band, but its interval
  overlaps that band. Neither mechanism explanation is independently confirmed.
  Retain the observed scorer/F1 rescue; withdraw the stronger claim that it
  isolates similarity from contamination dose.

The original numeric bands and scorer-state-first endpoint ordering are
unchanged. The earlier C2/C3 audits were also agent-reviewed, not human
validation. Their review passes and some functional-granularity conventions
differ, so these are not controlled contamination comparisons.

## Evidence and verification

The completed CSVs (`phase4_ai_review_c4.csv`, `phase4_ai_review_c5.csv`,
`phase4_ai_review_c6.csv`) contain one pair-specific functional rationale and
reviewer/amendment attribution per judgment. The original blank pack is retained.

Labels were frozen at **2026-10-23 11:38:33 UTC**, before joining audit keys or
calculating rates. `phase4_ai_audit_freeze.json` commits their SHA-256 hashes.
`phase4_ai_audit_protocol.json` preserves the original pre-label amendment,
including its historical pending status; this completion report supersedes that
status without changing the committed protocol file.

For each condition, all 50 IDs exactly match the fixed sheet and reproduce its
seed-13 sample; all keys match current mining triples. Result provenance records
the label, sheet, protocol and freeze hashes. Score-bound outcome records are
in `sanitycheck.md` §9.9. Modified scoring refuses altered frozen labels,
missing rationales and missing AI attribution in the amended closure gate.

Additionally, all **300 code fragments** match the key-indexed canonical corpus
text. The focused audit/scoring and closure suite passed **23 tests** (three
unrelated evaluation/checkpoint tests were deselected locally). This is separate
from the earlier 165-test Colab run; no new full-suite pass is claimed.

The audit is complete. Full-project closure additionally requires reconciliation
of the combined benchmark/generalisation table and remaining paper claims;
audit completion alone is not proof those separate tasks are finished.
