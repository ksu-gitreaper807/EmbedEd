# TIMELINE.md — the whole project, Phase 0 to today

> **24 October 2026 — current correction.** The entries below retain historical
> interpretations, not all current findings. Copied C4–C6 audit scores were
> voided; replacement AI-review rates are 12%/0%/22%, not human validation.
> All-negative F1 is 0; 0.2406424 is all-positive F1, not a ceiling. Saturation
> does not mean all-negative predictions. C3 unseen mean 0.7057 exceeds C0
> 0.6934 numerically, so the old universal untuned-floor claim is withdrawn.
> See PROJECT_REPORT.md and report/final/tables.md for current results.
>
> Phase 5 now has working CPU Colab verification, a verified 524-file pinned HF
> analysis snapshot and generated three-seed primary/six-seed extension tables.
> The current IEEE local release candidate and source-bound checks are recorded
> in output/phase5/closure.json; the pre-IEEE receipt is historical. No publication
> is implied. Six legacy confusion-count discrepancies are retained/disclosed.
> The requested two-pair, seven-model Colab webpage is parked in FUTURE_PROSPECTS.md.

Reconstructed from the repo's dated records (`PROGRESS.md`, `PHASE2_PLAN.md` §9,
`PHASE3_PLAN.md` §9, `sanitycheck.md` §5–§9, `report/measurements.md`) and the
commit history on `arena/01a0e714-embeded` (plus its predecessor PRs #6/#11).
Dates are commit/record dates; where a record is undated the entry says so.
"VM" = the Colab execution fleet; the working pattern throughout is
user-executed notebook cells + repo-side deliverables landed per designed stop.

---

## Phase 0 — corpus verification + feasibility freeze (≈ 2026-09-24/26; PR #6 era)

- **Gate G0 PASS** — BigCloneBench-via-CodeXGLUE verified: 9,126 fragments /
  8,063 unique texts / 901,028 train / 415,416 valid / 415,416 test pairs.
- Train/test fragment **overlap measured** (76/860 test fragments, 8.8%) and
  token lengths → **`MAX_LEN = 512`** (p50 474, p90 1,535) (`7437c90`).
- **Phase 0.5 throughput sweep on the T4** → recipe frozen: `BATCH = 8`,
  `TRAIN_PAIRS_CAP = 32,000`, `EPOCHS = 1` (`f01a150`).
- **Phase 0.7 s′ acquisition gate PASS** — Kitsios et al.'s BCB s′
  (4,600 pairs, 23 functionalities, Zenodo DOI 10.5281/zenodo.17238379).
- **Incidents that shaped the protocol:** a VM recycle lost the un-checkpointed
  embeddings + mined triples → the **checkpoint-to-HF-after-every-heavy-cell**
  resume protocol (`ff212c9`); a CPU-only reconnect redid Phase 0 in 3m42s and
  produced the first successful HF artifact push.

## Phase 1 — mining + the hardness gate (through 2026-09-27; PR #6)

- 32,000 triples per condition mined (v1–v5 era artifacts later lost with the
  VM and regenerated under v7/v8 with bit-identity checks).
- **Gate G1 FAIL recorded verbatim** under the legacy 0.02 absolute margin
  (`502ef54`); `scripts/hardness_diagnostics.py` + label-noise floors added
  (`3a8fdf5`) — the measurement that later explains C2/C3 (monotone poison).
- Blind 50+50 FN-audit tooling (`make`/`score`, Wilson CIs) pulled forward to
  *before* training (`4146485`).
- **2026-09-27 — decision D1 recorded** (gate rule: ordering `C1 < C2 ≤ C3`,
  standardised gap d ≥ 0.5) → **G1 PASS at d = 0.614**. Same day: the Phase 2
  runner (`embeded/encoder/train/evaluate.py`) implemented + 70 offline tests.

## Phase 2 — the main campaign and the collapse finding (2026-09-28 → 10-01)

- **2026-09-28** — `PHASE3_PLAN.md` written from repo state; Phase-3 sequencer
  notebook authored (36 cells — explicitly "written ahead of its code");
  `embeded/generalize.py` landed + 17 tests (suite 115); the **resume incident**
  (a pull restoring finished runs nearly got clobbered by §7's smoke) →
  smoke runs get their own directory, permanently (`06215d8`, `ba225be`).
- **2026-09-29/30 — the 9-run main campaign** (C0 + C1–C3 × seeds 13/14/15)
  across a five-VM fleet, checkpoint-pushed per run; **2026-09-30
  consolidation**: **C0 0.2572 · C1 0.7464 ± 0.0176 (2.90×) · C2 0.2478 ·
  C3 0.2707** — C2/C3 at the 13.7%-positive base rate (ceiling 0.2406) with
  10/12 thresholds pinned at 1.0 → **collapse = fp32-saturation erasure**;
  `sanitycheck.md` opens (G2 cross-examination), hinge-geometry /
  poison-monotonicity closed-form accounts (§5.1–5.3); literature-alignment
  thread (PR discussion 2026-09-30).
- **2026-10-01** — falsifier (i) REFUTED (§9.1, push `0ce5bf4`): C2/C3 rank
  *below C0* — not a benchmark artifact; 5-VM config/fingerprint audit block;
  collapse-signature check (27–67% of test pairs at cosine 1.0); FN-audit
  second pass 100/100 re-adjudicated, 0 flips; **C4 joins** (filtered = BM25 +
  token-Jaccard ≥ 0.40 denoise + head-skip, `56822ba`) with §9.4
  pre-registration written **before** any C4 run (`847a698`); `pull --light`;
  the 2×5 confusion-matrix board; causal-hierarchy note (FN rate = cause,
  margin = amplifier).

## The bouncer factorial — C4/C5/C6 (2026-10-02)

- C4 campaign notebook (sanctioned **v7 re-mine**, C1–C3 verified bit-identical
  to backups); the **committed-chimera lesson** (`91d2f7c`: a Jaccard pad landed
  inside the shared `clean_candidates` — NameError at scale).
- **C4_13 first result 0.4108, unpinned** — first escape from saturation;
  **C4 n = 6: 0.4349 ± 0.0246, MAP@R 0.2760**.
- **§9.4 FINAL — the C4 audit gate PASSES**: 15/50 = 30% [0.191, 0.438] ≤ 42%,
  identical rate to C3 yet opposite outcome → **the registered verdict:
  collapse is governed by the FN *surface* (functional clone ∧ near-duplicate),
  not the FN rate; rate = dose, margin = amplifier; partial denoising escapes
  the degenerate mode without reaching the clean control.**
- **Factorial completion (v8)**: C5 = C1+bouncer, C6 = C3+bouncer (`e8c04ac`);
  §9.5 pre-registration with fixed decision rules (`1c8ccd4`); the **v8 re-mine
  verified supply predictions**: C5 residual 0.27% (predicted ≈0), C6 13.6%
  (predicted 10–25%), C1–C4 bit-identical to v7 (`f18c3ea`); **C5/C6 trained to
  n = 6**: C5 0.7399 ± 0.0140 (control confirmed, ≈ C1 band), C6 0.4722 ±
  0.0282 (unpinned, between clusters) — **both headline-gated on their pending
  50-pair audits**.
- The **naming-key decision** (`fb07f51`, resolving the C4↔C5 swap request as
  NO-SWAP): numbers are build order; C4 = C2+bouncer, C5 = C1+bouncer,
  C6 = C3+bouncer. Parallel-VM workflow + quota budget (50 units) standing.
- **2026-10-02/03 — the filtered set joins Phase 3** (`38b984e`, `5529433`):
  recipe v8, six-condition generalisation design, D3 amendment recorded.

## Phase 3 — design amendments, then execution (2026-10-03; this session's first half)

- **Three scope amendments in one day, all user decisions, all recorded before
  any s′ run:** six seeds (`acc634e`, SPRIME_SEEDS 13–18, 36-run campaign) →
  plan remainder (`585ea55`) → **symmetric entry criterion** (`e279bfd`, C4's
  16–18 = "+ extra", runs stay published) → **re-scope to 6 × 3 = 18**
  ("only 6 models over the same 3 seeds would be enough", `39888f7`, ≈ 6 GPU-h;
  the two pre-re-scope v8 C1 runs retrain under the new fingerprint).
- **The preflight pytest catch**: the VM's first green run of the readable gate
  exposed a real repo bug — condition sets were frozen at import time
  ("iff triples exist"), poisoning fixtures on any machine where C4–C6 triples
  existed → **call-time `conditions()`** (`06171e9`); stale-kernel guards
  (`23f47f7`).
- **HF infra hardening after the 429**: pull **sections** (mining / runs-meta /
  runs / audit / sprime / report — virtual views, ~300 MB instead of ~4.5 GB),
  the one-request identity probe, patient 429 handling honouring Retry-After,
  and §8 refusing to start without secrets (`2f3fcea`, `213622c`, `06ae1aa`);
  the legacy raw-dataset copy (2,358 CloneNNN.java) removed from the sprime
  section (`b919488`).

## Phase 3 — campaign, consolidation, and the two missing modules (2026-10-04 → 10-05)

- **18/18 s′ runs trained and scored on the hub** (§8 across VMs, push per
  run). Δ table: C1 +0.054 / C5 +0.036 / C6 +0.030 / C2 +0.011 / C4 +0.004 /
  C3 −0.005 (thresholds pinned 0.9998–0.9999); the untuned encoder transfers
  at **0.6934 unseen → the registered finding: no poisoned-mining condition
  beats not fine-tuning at all**; the ordering and the saturation signature
  replicate across rulers.
- **`5ba985a` — the results-table latent bug** (directory regexes still
  C1–C3-only) fixed before §10 ever ran on real C4–C6 rows; off-ladder seed
  guard (the pre-re-scope `sprime_C1_16` excluded, visibly).
- **§9's transfer readings exit 1 for C1–C4** — the fingerprint gate demanded
  era equality; **era-tolerant scoring** lands (`f68aaf1`): recorded v6/v7
  checkpoints are scored as-is with provenance notes; training resumes stay
  strict. §9/§12 pulls narrowed to the 19 Phase-2 run dirs (`2ede09f`).
- **§11: `embeded/visualize.py` never existed** (notebook-ahead-of-code, item 7)
  → built to P2-18 (`9cca4b2`): seven panels, shared seeded sample, fixed UMAP
  params recorded in settings, provenance per checkpoint; `91d8642` fixes the
  two-element `--include` (comma-join contract); **the figure renders**.
- **§12: `demo/app.py` never existed** (item 8) → built to D6/P2-21 (`6f1d206`):
  real held-out pairs, four outcome cells asserted headless, threshold bars,
  `--share` (`cf5ee21`), unbuffered streaming (`a913118`); the notebook flag
  lint makes the `--include` bug class impossible (`70cf6f5`).
- **2026-10-05 — consolidation shipped by the user**: report + figure + demo
  check pushed (`ede587a`), closing the Phase-3 pipeline end to end.
- **Fresh-VM provisioning closed out**: §4's decisions cell replay-safe
  (`960bc5a`), `phase3_decisions.json` joins the sprime pull section
  (`a6a0964`), and "missing record" is disambiguated from "genuinely lost"
  (`d43002f`).

## Closure opens (2026-10-05)

- **`PHASE4_PLAN.md` (`43f62f2`)** — SCOPE Part 7's stop condition acknowledged
  as met; Phase 4 = closure: §3 the literal one-table deliverable, §4 the two
  blocking audits, §5 the nine §9.2 repairs, §6 the write-up (limitations
  before claims; `run_all.sh`; D4-1..D4-4; the δ-ladder extension caged behind
  a SCOPE amendment + a completed draft).
- **`paper/paper.tex` (`bc9e65b`)** — full LaTeX draft with verified
  bibliography, both results tables, the mechanism account, future-work
  ordering, and honest `[pending]` markers where gates are still open.

---

## Open items (as of 2026-10-17)

1. **`audit_c5` / `audit_c6`** — the last mandatory gate (predictions: C5
   0–2%, C6 25–35% under H1); until scored, no C5/C6 headline.
2. **§9.2** — nine C2/C3 `predictions.npz`↔stored-metrics repairs (deltas
   ≤ +0.027; era-tolerant scoring makes them mechanical).
3. **The write-up** — limitations first, claims ledger, `run_all.sh`, the
   one-table section (PHASE4_PLAN §3/§6), D4-1..D4-4 recorded.
4. **The extension** — the denoising δ-ladder (recommended), only after the
   SCOPE amendment and a completed draft.
