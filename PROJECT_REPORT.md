# EmbedEd — Consolidated Project Report
**Everything done so far, decisions taken (with reasons), results at each stage, and the outputs/evidence behind them.**

*Compiled 2026-10-17 from the repo's own records: `PROGRESS.md`, `PHASE2_PLAN.md` §8–§9,
`PHASE3_PLAN.md` §8–§9, `PHASE4_PLAN.md`, `sanitycheck.md` §5–§9,
`report/measurements.md` (canonical number store), the shipped consolidation commit
`ede587a`, and the dated commit history on `arena/01a0e714-embeded`.*
**Task:** Java code-clone detection with GraphCodeBERT + triplet contrastive learning;
the independent variable is the **negative-selection strategy**; benchmarks are the
CodeXGLUE/BigCloneBench pair split and the functionality-labelled s′ corpus.

---

## 0. Executive summary — the five headline findings

| # | Finding | Where proven |
|---|---|---|
| **F1** | On the benchmark split, mined **hard negatives collapse training to the all-negative base rate**: C2 (BM25) 0.2478 ± 0.0231, C3 (semantic) 0.2707 ± 0.0095 vs the ceiling **0.2406** (13.7% positives) — below the untuned C0 (0.2572), 2.9× below random negatives C1 (0.7464 ± 0.0176). 10/12 validation thresholds per condition pinned at 1.0 → the failure mode is **fp32-saturation erasure** ("predict non-clone for everything"), visible in 27–67% of test pairs sitting at cosine 1.0. | Table `## Phase 2 — main results` in `report/measurements.md`; board push `0ce5bf4`; falsifier (i) REFUTED, `sanitycheck.md` §9.1 |
| **F2** | The mechanism is **surface, not rate**: C3 and C4 carry the *same* audited false-negative rate (30%, 15/50) yet differ by 0.16 F1 with opposite scorer states. The trigger is a contaminant that is **simultaneously a functional clone and a near-duplicate**; the FN *rate* is the dose, the 0.10 triplet **margin** is the amplifier. | `sanitycheck.md` §9.4 FINAL (audit gate passed 2026-10-02) |
| **F3** | A one-line **positive-aware bouncer** (drop negatives with token-Jaccard ≥ 0.40 to the anchor) **escapes the degenerate mode** (C4 0.4349 ± 0.0246, all thresholds unpinned, MAP@R 2.7× C3) without reaching the clean control — partial denoising, quantitatively matched to the ~40% in-sample contamination cut, with **zero audited clean pairs discarded**. | §9.4 + `embeded/negatives.py` filtered strategy (`56822ba`) |
| **F4** | On **unseen functionality** (s′, holdout {10,13,14}), the ordering replicates across corpora — and the balanced ruler exposes a floor the benchmark hid: the **untuned encoder scores 0.6934 unseen, and no poisoned-mining condition beats doing nothing** (C2 0.6674, C4 0.6687 < 0.6934; C6 0.6865 ≈; C3 0.7057 "ties" it with a saturated scorer). Only clean training converts fine-tuning into gains (C1 +0.054 Δ, C5 +0.036). | `## Phase 3 — generalisation` table, shipped `ede587a`; 18/18 runs |
| **F5** | The benchmark's celebrated ≈0.95 regime and our 0.75 are **different rulers**: corpus-wide natural prevalence (ceiling 0.24, base-rate trap) vs near-clone-heavy published splits; s′'s balanced slices lift the same model family ~0.45. Cross-ruler replication of ordering + saturation signature is evidence *for* the Phase-2 numbers, not against. | Non-comparability note in the s′ table block; C0 0.2572 vs C0-transfer 0.6934 |

---

## 1. The method invariants (what never changed)

- **Model/loss:** `graphcodebert-base`, mean pooling, `MAX_LEN=512`, batch 8, 1 epoch, AdamW
  LR 2e-5, explicit-triplet cosine hinge, `TRIPLET_MARGIN=0.10`, fp16 autocast, up to 32,000
  triples/condition (s′: cap 16,000), `K_NEGATIVES=20` identical across conditions.
- **Seeds 13/14/15** everywhere (Phase 2's C4 and Phase-2-filtered extras also hold 16–18 on
  the hub); thresholds chosen on the **seen/validation slice only**, applied unchanged to
  test, **the threshold itself always reported**.
- **Two pre-registered gates** before any outcome reading: hardness gates G1/G1s
  (ordering `C1 < C2 ≤ C3`, standardised gap d ≥ 0.5) and **blind 50-pair FN audits**
  (sheet committed before unblinding; Wilson CIs).
- **Determinism:** seeded anchors/positives/splits; per-run `run_config.json`
  **content-fingerprinted**; artifacts hash-checked at consolidation.
- **Anti-hindsight discipline:** every design decision recorded **before** the runs it
  affects (PHASE*_PLAN §8/§9 logs); amendments appended, never edited.

---

## 2. Stage-by-stage: done, decisions, results, evidence

### S1 — Phase 0: corpus verification + feasibility freeze *(≈ 2026-09-24/26)*
**Done:** loaded and verified BigCloneBench-via-CodeXGLUE; measured train/test fragment
overlap; measured token lengths; ran the T4 throughput sweep; acquired s′; survived the
project's first infrastructure lessons.
**Decisions:** `MAX_LEN=512` (p50 474 / p90 1,535 + T4 budget); fp16 autocast (fp32 OOMs);
`BATCH=8 / TRAIN_PAIRS_CAP=32,000 / EPOCHS=1` from the sweep; **checkpoint every heavy cell
to HF** (`Kusshal/Embed`) after a VM recycle lost embeddings + triples (~35 min GPU).
**Results (Gates G0, s′-gate):** 9,126 fragments / 8,063 unique texts / 901,028 train /
415,416 valid / 415,416 test; overlap 76/860 test fragments (8.8%); s′ = 4,600 pairs,
23 functionalities — both PASS.
**Evidence:** `report/measurements.md` §Phase 0/0.5/s′; commits `7437c90`, `f01a150`,
`ff212c9`; incident register in `PROGRESS.md` §5 (incl. the revoked-token incident —
tokens live only in Colab secrets from then on).

### S2 — Phase 1: mining + the hardness gate arc *(→ 2026-09-27)*
**Done:** mined 32,000 triples/condition (identical anchors/positives across conditions —
test-enforced invariant); ran G1; built hardness diagnostics + label-noise floors; built
blind 50+50 audit tooling (pulled forward to *before* training).
**Decisions:** **D1 (2026-09-27, recorded before training):** G1's legacy 0.02 absolute
margin ignored the scale of the similarity space (diagnostics I1/I3) → re-operationalise
"visible gap" **scale-free: ordering `C1 < C2 ≤ C3` AND `d(C1→C2) ≥ 0.5`**, percentiles
reported, margin reported but not decisive, **the FAIL kept verbatim** (P6-3 transparency).
**Results:** G1 FAIL→(D1)→**PASS, d = 0.614** (0.63 full-population); poison floors
measured monotone in hardness (~0/30/42% for C1/C3/C2 later confirmed by audit).
**Evidence:** `502ef54`, `3a8fdf5`, `4146485`; `report/measurements.md` G1 history
(FAIL kept); `embeded/hardcheck.py --recorded` re-judges offline.

### S3 — Phase 2 build + the pre-training audit *(2026-09-27)*
**Done:** Phase 2 runner implemented and unit-tested offline (encoder/train/evaluate,
70 tests); fingerprinted resume protocol; **blind audit executed before any training**.
**Decisions:** **D2 (2026-09-27): option (a) — keep the mining, measure and report the FN
rate** rather than pre-filter near-verbatim negatives. Reason: the audit returned C3 = 30%
[19, 44] (trigger ≳30% met at the point estimate), but the separability measurement showed
the *specified* fix does not work — near-verbatim cuts (J/difflib ≥ 0.8) remove only 5–10%
of contamination because functional equivalents differ ~70% of characters; a J ≈ 0.40 cut
looked precision-perfect **in-sample** (every audited not_clone ≤ 0.345) but was a 50-pair
threshold; a cosine ≥ 0.99 cut would alter C3's defining variable. So: rate reported as a
36% pooled label-noise floor; a **filtered variant explicitly reserved as a future v7
experiment with a fresh audit** — which is exactly what C4 later became.
**Results:** audit C3 30%, C2 42%, C1 ≈0–2% (the FN ladder that later explains everything).
**Evidence:** `PROGRESS.md` §4.2 (full D2 text); `artifacts/audit/AUDIT_JUDGEMENTS.md`;
audit tooling with sheet-staleness refusal.

### S4 — Phase 2 campaign: the collapse *(2026-09-28 → 09-30)*
**Done:** 9 runs (C0 + C1/C2/C3 × 3 seeds) across a five-VM Colab fleet, checkpoint-push
per run; consolidation on 09-30; `sanitycheck.md` opened as the standing G2
cross-examination; closed-form accounts of both floors (§5.1–5.3); literature alignment.
**Decisions:** threshold policy frozen (validation-only); smoke runs get their own
directory (after the resume incident where a §7 re-run nearly clobbered pulled runs —
`06215d8`); the collapse-signature check (exact base rate 0.13678; cosine-1.0 mass) added
from external review.
**Results (Phase 2 main table):** **C0 0.2572** (thr 0.9779) · **C1 0.7464 ± 0.0176**
(MAP@R 0.6629) · **C2 0.2478 ± 0.0231** · **C3 0.2707 ± 0.0095** (MAP@R 0.1826) —
**F1 landed at the base-rate ceiling with 10/12 thresholds pinned at 1.0**: the fp32-
saturation erasure mode. Seed-robust (2.90× inversion).
**Evidence:** `report/measurements.md` `## Phase 2 — main results`; board push `0ce5bf4`;
config/fingerprint audit block (`0c87d96`); per-run `runs/C*_13..15/` on the HF snapshot.

### S5 — Cross-examination: the falsifier wave *(2026-09-30 → 10-01)*
**Done:** three registered falsifiers tested and resolved.
**Decisions/Results:** **(i) "benchmark artifact" — REFUTED** (§9.1): C2/C3 rank *below
C0*; **(ii)** NaN/loss spikes → fp16 instability, not eval noise; **(iii)** oracle bounds
rule out evaluation noise; the improvement is **monotone in the measured poison rate**
(§5.2) and the hinge geometry explains both floors (§5.3). Causal hierarchy recorded:
**FN rate = cause, margin = amplifier** (`44db4bd`).
**Evidence:** `sanitycheck.md` §5, §9.1; commits `b63f693`, `fd97aab`, `769921a`,
`e58d1d9`, `e46b8ec`.

### S6 — The bouncer: C4 *(2026-10-01 → 10-02)*
**Done:** implemented the **filtered strategy** (BM25 ranking + token-Jaccard ≥ 0.40
denoise + head-skip); §9.4 **pre-registered before any C4 run** (denoisers, expectations,
mandatory audit gate); sanctioned **v7 re-mine** with C1–C3 verified **bit-identical** to
backups; C4 campaign to n = 6; audit gate executed.
**Decisions:** the C4 audit gate **must pass before any C4 headline** (50-pair blind,
≤42%); C4_13's first result deliberately left uninterpreted until the audit landed.
**Results:** C4_13 0.4108 (unpinned) → **C4 n=6: 0.4349 ± 0.0246, MAP@R 0.2760**;
**audit gate PASSED: 15/50 = 30% [0.191, 0.438] ≤ 42%** — identical rate to C3, opposite
outcome → **§9.4 FINAL: surface-over-rate; rate = dose, margin = amplifier; partial
denoising escapes the degenerate mode without reaching the clean control.** The J-cut
removes ~40% of C2's contamination in-sample, zero audited clean pairs lost (clean max
0.345 < cut 0.40 < clone median 0.418).
**Evidence:** `56822ba`, `847a698`, `c502697`, `d9ffd7c`, `d561b25`, `912b0d0`,
`e9d83fb`; `artifacts/audit_c4/`; `sanitycheck.md` §9.4 (execution log + FINAL).
*Incident en route:* the committed-chimera bug (`91d2f7c` — a Jaccard pad landed inside
the shared `clean_candidates`, NameError at scale) → rule: unique-context anchors,
regression tests for count-based top-ups.

### S7 — Factorial completion: C5/C6 *(2026-10-02 → 10-03)*
**Done:** generalised the bouncer to a **ranker parameter** → the full 3-supply × bouncer
factorial (v8); §9.5 **pre-registration** (C5 = control prediction ≈ C1, contamination
0–2%; C6 fork H1: ≥ C4 band & contamination 25–35%, H2: ≈ C3 pinned ~0.27 — with fixed
decision rules); **v8 re-mine** with C1–C4 verified bit-identical; C5/C6 campaigns to
n = 6; confusion-matrix grid board; the naming-key record.
**Decisions:** **naming key (fb07f51, resolving the C4↔C5 swap request as NO-SWAP):**
numbers are build order — C4 = C2+bouncer, C5 = C1+bouncer, C6 = C3+bouncer; the key lives
in sanitycheck (pre-§9.4), `negatives.py`, and both filtered notebooks' intro cells.
**C5/C6 numbers are headline-GATED on their pending blind audits** (predictions: C5 0–2%,
C6 25–35% under H1).
**Results:** v8 supply predictions **hit**: C5 residual 0.27% (predicted ≈0), C6 13.6%
of candidates (predicted 10–25%), C1–C4 bit-identical; **C5 0.7399 ± 0.0140** (≈ C1 band
→ control confirmed at the scorer endpoint), **C6 0.4722 ± 0.0282** (unpinned 0.63–0.82,
above C4's band, 1.7× C3). Audits: **still open**.
**Evidence:** `e8c04ac`, `1c8ccd4`, `0259ed7`, `f18c3ea`, `5161fcb`, `fb07f51`;
`artifacts/audit_c5/`, `artifacts/audit_c6/` (created when run); `sanitycheck.md` §9.5.

### S8 — Phase 3: amendments, then the 18-run campaign *(2026-10-03 → 10-05)*
**Done:** the six-condition generalisation design (`5529433`, `38b984e`); **three scope
amendments in one day, all user decisions, all recorded before any s′ run**: six seeds
(`acc634e`) → plan remainder (`585ea55`) → symmetric entry (`e279bfd`: C4's 16–18 are
"+ extra", runs stay published) → **re-scope to 6 × 3 = 18 runs ≈ 6 GPU-h** ("only 6
models over the same 3 seeds would be enough", `39888f7`); s′ census (D4:
`max_pair_count_then_lowest_id` → holdout {10,13,14}), six-condition mining, **gate G1s
PASS (d = 0.886)**, smoke, then **18/18 runs trained and scored on the hub**; §9 free
transfer reading (19 Phase-2 checkpoints + C0, era-tolerantly scored); §10 table; §11
figure; §12 demo check; **user consolidated + shipped everything (`ede587a`, 2026-10-05)**.
**Decisions:** D3 = B (fine-tune on s′ with negatives mined from s′ — keeps strategy the
independent variable; (A) kept as the free secondary transfer reading; (C) impossible);
D4 = measured census rule; D5 = code lives in `embeded/` (matches tested runners + both
call sites); D6 = "all four outcome modes" read as the 2×2 of (C0 decision) × (C_best
decision), ≥1 real held-out pair per cell. The re-scope re-fingerprints the two pre-scope
runs (retrain ~40 min) rather than shimming the resume gate — provenance stays truthful.
**Results (s′ table, holdout {10,13,14}, seeds 13–15):** means — **C1 0.7570/0.7031/Δ
+0.054 (sd 0.022)** · C5 0.7606/0.7244/+0.036 · C6 0.7169/0.6865/+0.030 · C2
0.6781/0.6674/+0.011 · C4 0.6731/0.6687/+0.004 · **C3 0.7004/0.7057/−0.005 with
thresholds pinned 0.9998–0.9999**; **C0 untuned transfer 0.6934 unseen** → **F4: no
poisoned condition beats not fine-tuning**; ordering + saturation signature replicate
across rulers (F5). Transfer block: benchmark checkpoints score 0.63–0.75 unseen (C3's
rows again saturated); C1 v6-era rows reproduce (0.7230/0.7564 reading consistent).
**Evidence:** `report/measurements.md` `## Phase 3 — generalisation` + transfer block +
`## Gate G1s` + census/decisions sections; `report/figures/umap_panels.png` +
`umap_params.json` (P2-18 params, shared seeded sample, per-panel checkpoint
fingerprints); `demo/app.py --check` (four D6 cells populated); commits `acc634e` →
`39888f7`, runs `sprime_{C1..C6}_{13,14,15}` on `Kusshal/Embed`.

### S9 — Infrastructure & integrity workstream *(continuous, 10-03 → 10-05)*
**Done (each landed as its own tested commit):** call-time `conditions()` after the VM
preflight pytest exposed import-time FS-gated constants (`06171e9`); stale-kernel guards
(`23f47f7`); **HF pull sections** (mining/runs-meta/runs/audit/sprime/report — virtual
views; ~300 MB instead of ~4.5 GB; the legacy 2,358-file raw-dataset copy excluded,
`b919488`); the identity probe + patient 429 policy (Retry-After honoured, one retry,
wait-it-out message) + §8 refusing to start without secrets (`2f3fcea`, `213622c`,
`06ae1aa`); **era-tolerant checkpoint scoring** (`load_trained_checkpoint`) after §9's
transfer readings exit-1ed on v6/v7-era fingerprints — provenance notes + persisted
`transfer_provenance.json`, training resumes remain strict (`f68aaf1`); §9/§12 pulls
narrowed to the 19 Phase-2 run dirs via one glob (`2ede09f`); the **results-table latent
bug** (C1–C3-only regexes would have silently dropped C4–C6) + off-ladder seed guard
(`5ba985a`); `visualize.py` **built** (it had never existed — notebook-ahead-of-code,
`9cca4b2`); `demo/app.py` **built** (`6f1d206`, `cf5ee21` `--share`, `a913118`
unbuffered streaming); the notebook flag lint making the two-element `--include` bug
class impossible (`70cf6f5`); §4 decisions cell replay-safe + `phase3_decisions.json`
delivered in the sprime section + missing-record disambiguation (`960bc5a`, `a6a0964`,
`d43002f`). Suite: 29 → **150 offline tests**, all green.
**Evidence:** the commits named; the notebook green-runs from a cold VM (validated by
your consolidation + fresh-VM runs).

### S10 — Closure begins *(2026-10-05 → today)*
**Done:** `PHASE4_PLAN.md` (`43f62f2`) — SCOPE Part 7's stop condition acknowledged as
met; Phase 4 = closure (§3 the literal one-table deliverable; §4 the two blocking audits;
§5 the nine §9.2 repairs; §6 the write-up with limitations-before-claims + `run_all.sh`;
D4-1..D4-4; the δ-ladder extension caged behind a SCOPE amendment + completed draft).
`TIMELINE.md` (`ade0220`). `paper/paper.tex` (`bc9e65b`) — full LaTeX draft, verified
bibliography (Krinke IWSC'22; Krinke & Ragkhitwetsagul arXiv 2505.04311; Kitsios et al.
ASE'25; Soft-InfoNCE EMNLP'23; CoCoHaNeRe TOSEM'25; RocketQA; DPR; NV-Retriever; debiased
contrastive; MLRC; FaceNet/triplet; CodeBERT/GraphCodeBERT/CodeXGLUE/BigCloneBench),
both results tables, `[pending]` markers where gates are open.
**Evidence:** the files themselves; all pushed on `arena/01a0e714-embeded`.

---

## 3. Master decision register

| ID | Decision | Reason (as recorded) | When | Where |
|---|---|---|---|---|
| Recipe | MAX_LEN 512; BATCH 8 / cap 32k / 1 epoch; fp16 | Measured token lengths + T4 sweep; fp32 OOM | Phase 0/0.5 | `settings.py`, PROGRESS §4.1 |
| **D1** | G1 rule = ordering `C1<C2≤C3` AND d≥0.5; FAIL kept verbatim | 0.02 absolute margin ignored the space's scale (I1/I3) | 09-27 | PHASE2 §8/§9; `hardcheck.py` |
| **D2** | Keep mining; **measure + report** the FN rate; filtered variant reserved as future v7 | Audit hit ≳30%, but the specified near-verbatim fix removes only 5–10%; J≈0.40 was in-sample; cosine cut would change the independent variable | 09-27 | PROGRESS §4.2 — and **vindicated**: the v7 filtered experiment became C4 |
| Loss/thresholds | Triplet hinge margin 0.10; validation-only thresholds, reported | SCOPE P1-4; PHASE2 §3.2–3.3 | 09-27 | `settings.py` |
| Audit-first | 50-pair blind audit **before training**; sheets committed before unblinding; audits gate headlines | Label-contaminated supplies make "hard" == "possibly false" | 09-27 → standing | `scripts/audit_sample.py`; §9.4/§9.5 |
| **D3** | s′ fine-tuning with negatives mined from s′ (option B); (A) = secondary transfer reading | Only option keeping strategy the independent variable on functionality-labelled data | 10-03 | `artifacts/phase3_decisions.json`; PHASE3 §8 |
| **D4** | Census rule `max_pair_count_then_lowest_id`; holdout {10,13,14}, k=3, identical across all runs | "Best-represented" measured, not assumed; committed before any F1 | 10-03 | census artifact + report |
| **D5** | Phase-3 code in `embeded/` | Matches tested runners + both call sites | 10-03 | PHASE3 §8 |
| **D6** | "Four outcome modes" = 2×2 (C0 × C_best), ≥1 real pair per cell | The phrase was never enumerated; this is the confirmable reading | 10-03 | PHASE3 §8; `demo.app --check` |
| Amendments | Six s′ seeds → symmetric entry (C4 16–18 = extras) → **6×3 = 18 runs** | User decisions, recorded before any 16–18 s′ run | 10-03 | PHASE3 §9 rows; `acc634e`→`39888f7` |
| Naming key | C4=C2+b, C5=C1+b, C6=C3+b; numbers = build order, **no swap** | Resolves the label-confusion request without rewriting history | 10-02 | `fb07f51`; sanitycheck pre-§9.4 |
| §9.4 verdict | Surface-over-rate; partial denoising | C3 = C4 = 30% FN yet 0.16 F1 apart, opposite scorer states | 10-02 | sanitycheck §9.4 FINAL |
| Headline gates | No C5/C6 headline until `audit_c5`/`audit_c6` scored | The standing 50-pair gate, extended to the factorial | 10-02 | §9.5 pre-registration |
| Truthful provenance | Re-scope retrains the 2 pre-scope runs; era-tolerant **scoring** but strict **resume**; paper carries pointers, not copies | Fingerprints are the experiment's identity; scoring a recorded artifact ≠ mixing experiments | 10-03→05 | `39888f7`, `f68aaf1` |
| **D4-1..4** (open) | One-table layout; C_best headline (C1, C5 only if audit passes); 0.95-framing; extension = δ-ladder behind SCOPE amendment + draft | PHASE4 §9 | 10-05 | PHASE4_PLAN |

---

## 4. Master results tables

**Benchmark (CodeXGLUE test, n=415,416, pos 13.7%, ceiling 0.2406):**
| cond | F1 | MAP@R | thresholds | audited FN |
|---|---|---|---|---|
| C0 | 0.2572 | — | 0.978 | — |
| C1 | **0.7464 ± 0.0176** | 0.6629 | unpinned | ≈0–2% |
| C2 | 0.2478 ± 0.0231 | — | 10/12 pinned | 42% |
| C3 | 0.2707 ± 0.0095 | 0.1826 | 10/12 pinned | 30% [0.19, 0.44] |
| C4 | 0.4349 ± 0.0246 | 0.2760 | 6/6 unpinned | 30% [0.19, 0.44] |
| C5 | 0.7399 ± 0.0140 | 0.6629 | unpinned | 0.27% supply (audit pending) |
| C6 | 0.4722 ± 0.0282 | — | unpinned 0.63–0.82 | 13.6% supply (audit pending) |

**s′ generalisation (holdout {10,13,14}; F1_seen/F1_unseen/Δ ± sd):**
C1 0.7570/0.7031/**+0.054±0.022** · C5 0.7606/0.7244/+0.036 · C6 0.7169/0.6865/+0.030 ·
C2 0.6781/0.6674/+0.011 · C4 0.6731/0.6687/+0.004 · C3 0.7004/0.7057/−0.005 (pinned) ·
**C0 untuned anchor 0.7118/0.6934** (transfer).

**Mechanism numbers:** trigger = FN ∧ (functional clone ∧ J ≥ 0.40 near-dup); J-cut: −40%
of C2 contamination, 0 clean pairs lost (clean max 0.345 < 0.40 < clone median 0.418);
supplies: C1–C3 ≈0/42/30% audited; C5 0.27%, C6 13.6% measured at v8; hardness gaps
d = 0.614 (G1) / 0.886 (G1s).

---

## 5. Evidence index — claim → artifact → reproducing command

| Claim/number | Artifact of record | Reproduce |
|---|---|---|
| Corpus counts, overlap, MAX_LEN, sweep | `report/measurements.md` §Phase 0/0.5 | `python -m embeded.data.prepare_data` |
| G1 arc | G1 history in the report (FAIL kept) | `python -m embeded.hardcheck --recorded report/gate_g1/run_2026-09-26_phase01-v5.json` |
| Mining + poison floors | `artifacts/mining_summary.json`, diagnostics sections | `python -m embeded.negatives`; `python -m scripts.hardness_diagnostics` |
| Phase 2 table | `## Phase 2 — main results` | `python -m embeded.evaluate --results-table` |
| Audits | `artifacts/audit*/` + `AUDIT_JUDGEMENTS.md` | `python -m scripts.audit_sample make/score` |
| §9.4 verdict | `sanitycheck.md` §9.4 (log + FINAL) | the audit sheet + board pushes |
| s′ census/decisions/G1s | census/decisions artifacts + report sections | `python -m embeded.generalize --census/--mine/--hardness` |
| s′ table + transfer | `## Phase 3 — generalisation` (+ transfer block) | `python -m embeded.generalize --results-table` |
| Figure | `report/figures/umap_{panels.png,params.json}` | `python -m embeded.visualize --panels C0..C6` |
| Demo contract | demo check output in the report | `python -m demo.app --check` |
| Run identity | `runs/*/*/run_config.json` (fingerprint) | any run dir on `Kusshal/Embed` |
| Consolidated state | commit `ede587a` (user) | `git log`/HF snapshot |
| Paper draft | `paper/paper.tex` | `pdflatex paper/paper.tex` |

**Commit chain (this branch):** Phase-2 era via PRs #6/#11 → `8214c85`…`e58d1d9` (10-01)
→ `847a698`…`e9d83fb` (10-02 §9.4) → `e8c04ac`…`f18c3ea` (10-02 v8/§9.5) → `38b984e`/
`5529433`/`fb07f51` → `acc634e`/`585ea55`/`e279bfd`/`06171e9` (10-03) → `23f47f7`/
`2f3fcea`/`213622c`/`06ae1aa`/`b919488`/`39888f7` (10-03/04 infra + re-scope) →
`5ba985a`/`f68aaf1`/`2ede09f`/`9cca4b2`/`91d8642`/`6f1d206`/`70cf6f5`/`cf5ee21`/
`a913118` (10-04/05 modules) → **`ede587a` (user consolidation)** → `960bc5a`/`a6a0964`/
`d43002f`/`43f62f2`/`bc9e65b`/`ade0220`.

---

## 6. Incidents → lessons (the protocol's origin story)

| Incident | Lesson now baked in |
|---|---|
| Live HF token pasted in a cell | revoked; secrets only in Colab Secrets |
| VM recycle lost embeddings/triples | checkpoint after every heavy cell; artifact-gated stages |
| §7 re-run nearly clobbered pulled runs | smoke runs get their own directory |
| Stale v5 artifacts under a v6 checkout | version gates fail loudly with a fix message |
| Committed chimera (`91d2f7c`) | unique-context anchors + regression tests; heredoc atomicity |
| Import-time `CONDS` froze at pytest collection (VM preflight caught it) | never gate module constants on FS state; call-time discovery (`06171e9`) |
| HF 429 (anonymous ~200-file bursts; retries extended the window) | identity probe; Retry-After policy; token required before §8 |
| "Gate input with no pull section" ×3 (pairs, frag-map, decisions) | every gate input has a named delivery path; lint-tested flag usage |
| Results-table C1–C3-only regexes (latent) | six-condition regexes + off-ladder guard before §10 ran on real rows |
| `visualize.py`/`demo/app.py` never existed | sequencer cells now proven against modules; items #7/#8 closed |
| Notebooks never auto-refresh; kernel caches imports | checkout → reopen → restart sequence; stale-kernel guards |

---

## 7. Open items (as of 2026-10-17)

1. **`audit_c5` / `audit_c6`** — the last mandatory gate (predictions 0–2% / 25–35%);
   unblocks every C5/C6 sentence.
2. **§9.2 repairs** — nine C2/C3 npz↔metrics re-evaluations (deltas ≤ +0.027; mechanical
   with era-tolerant scoring).
3. **The write-up** — limitations first, claims ledger, the one-table section,
   `run_all.sh` (PHASE4 §3/§6).
4. **The extension** — δ-ladder {0.00, 0.05, 0.10, 0.20} on C3's supply (turns F2 into a
   dose-response curve), only after a SCOPE amendment and a completed draft.
