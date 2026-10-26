# Future prospects — seven-model interactive comparison

**Recorded:** 23 October 2026, at the project owner's request.
**Status (25 October 2026):** the owner selected this proposal as Phase 6.
The private Colab prototype is implemented and live-verified; see
`PHASE6_PLAN.md`, `demo/PHASE6_README.md`, `notebooks/Phase6_Colab.ipynb` and
`output/phase6/closure.json`. It is not part of the historical Phase 5 release
and is not public, permanent hosting. The original prospect/design notes follow.

## Proposed demonstrator

A webpage starts with two example Java code pairs (four fragments). Each pair
is sent to all seven project conditions, C0–C6, with the models hosted on Colab.
The page displays each condition's cosine similarity, its recorded validation
threshold and its clone/non-clone decision side by side for both pairs. Editing
or adding pairs can be an optional extension, not a requirement for this demo.

| Condition | Model shown |
|---|---|
| C0 | Untuned GraphCodeBERT |
| C1 | Random-negative fine-tuning |
| C2 | BM25-negative fine-tuning |
| C3 | Semantic-negative fine-tuning |
| C4 | BM25 + bouncer fine-tuning |
| C5 | Random + bouncer fine-tuning |
| C6 | Semantic + bouncer fine-tuning |

## Scope to decide before implementation

- Select one recorded seed/checkpoint per condition in advance and identify
  them on the page; do not silently ensemble seeds or choose checkpoints after
  seeing a user's example. C0 has no project fine-tuning checkpoint.
- Reuse the existing `demo/app.py`, encoder and checkpoint-loading interfaces
  before adding a separate inference service. Audit their current capabilities
  first; this proposal does not assert that a seven-model endpoint already exists.
- Use the exact training-time tokenizer, mean pooling, 512-token limit and
  validation-selected thresholds. Display truncation and saturated-score warnings.
- Decide whether one T4 can keep all seven models resident. Measure memory and
  latency; use sequential loading or a bounded cache if necessary. Do not
  promise simultaneous residency before testing it.
- Show similarities as scores, not calibrated probabilities. A clone decision
  is a model prediction, not ground truth or proof of semantic equivalence.
- Colab is ephemeral demo compute, not reliable production hosting. Define
  session start/reconnect/release behaviour, access control, request limits and
  a temporary endpoint before exposing it publicly.
- User-entered code must never be executed. State whether it is transmitted or
  retained; avoid logging proprietary code by default. Reject oversized inputs.
- Keep example selection transparent; do not cherry-pick inputs as evidence of
  benchmark superiority or infer performance from a handful of examples.

## Proposed acceptance conditions

The page evaluates both example pairs with the same preprocessing for all seven conditions,
shows checkpoint/seed identity and threshold provenance, produces decisions
consistent with offline inference, and handles unavailable runtimes/errors
without fabricating results. Verify a shared fixture against the existing
inference path and measure actual latency/memory on Colab.

The owner's Phase 6 selection authorized implementation and checkpoint downloads.
Public deployment, production hosting and new experiments still require separate
decisions; this demonstrator does not change the study's results or Phase 5 release.
