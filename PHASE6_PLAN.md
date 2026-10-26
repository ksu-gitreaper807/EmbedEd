# Phase 6 - Seven-model Java clone demonstrator

**Authorized:** 25 October 2026: owner selected the seven-model webpage.
**Scope:** two illustrative Java pairs, compared by C0-C6 on Colab. No new
training, mining, audit judgments, threshold tuning or Git/HF publication.
Phase 5's immutable release remains untouched.

## Fixed decisions and acceptance

- Seed 13 for every condition, chosen before viewing example predictions.
  C0 is the untuned reference, not an extra trained checkpoint. No ensembles.
- HF dataset revision `e3144bea7732c20ab2803782282282be4fede7b4` is fixed.
  Exact thresholds come from saved NPZ arrays, never rounded JSON displays.
- Base model revision `2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d` is fixed.
  The historical training configuration named the base model but did not pin
  its revision; this explicit demo pin is disclosed, not retroactive provenance.
- Reuse `embeded.encoder` preprocessing and `embeded.evaluate.pair_cosines`.
  Token-only GraphCodeBERT, masked mean pooling, L2 normalization, max 512
  tokens, recorded fp16 inference on CUDA / fp32 on CPU, inclusive threshold.
- One active encoder; restore the untuned weights for C0, then swap six
  checkpoint state dictionaries sequentially. No seven-model GPU-residency claim.
- Each of 14 results must show score, exact threshold, decision, seed, model
  identity, input truncation and saturation warnings. Similarity is not a
  probability, semantic-equivalence proof or benchmark result.
- Validate both examples against the existing offline inference path for all
  seven conditions and record measured latency and memory on Colab.
- A working local/browser preview must handle missing runtime, failed inference,
  oversized input and concurrent requests without fabricated scores.

## Deliverables

1. `demo/compare.py` and `demo/web/`: a small webpage/API, preserving the old demo.
2. `demo/prepare_compare.py`: allowlisted, revision/hash-verified inference assets.
3. `notebooks/Phase6_Colab.ipynb`: prepare, verify, serve privately in Colab, stop.
4. Focused tests, live verification evidence, setup documentation and a local ZIP.

## Privacy and lifecycle

Java input is text only: never compiled, imported or executed. Request bodies
and user snippets are not logged or written to disk. Four bounded snippets,
bounded queue/concurrency, same-origin request header and no public CORS.
Only the checked-in illustrative examples may appear in verification evidence.

The server binds to loopback. Colab's authenticated notebook/port proxy is the
default access path; no public tunnel or durable hosting is enabled. The owner
controls who can access their Colab notebook. Do not paste proprietary code.
Secrets, if needed, belong in Colab Secrets/environment, never notebook source,
logs, browser JavaScript or ZIPs. The chosen HF inputs are publicly readable.

Runtime loss means restart and rehydrate pinned assets, not invent cached
predictions. Save verification results before stopping only this phase's runtime.
Stop the test session after verification; the notebook can start a fresh demo
when the owner wants to use it. A stopped runtime is not an always-on service.

## Completion checklist

- [x] Fixed catalog: 39 hash-verified files, about 3.51 GB, seed 13 throughout.
- [x] Functional webpage/API with 14 real results and metadata.
- [x] 31 focused tests and 235 full-suite tests pass without skips; browser preview/error/reset checks pass.
- [x] T4 parity: all 14 scores/decisions match the original inference path exactly; cold comparison 18.0865 s, peak comparison GPU allocation 673.5 MB. Evidence exported.
- [x] Colab session stopped; allowlisted local release packaged without credentials or weights. Consult the receipt, not this checkbox, for verified ZIP hashes.

Actual completion status belongs in `output/phase6/closure.json`. No acceptance
box is evidence by itself, and no remote publication is authorized here.
