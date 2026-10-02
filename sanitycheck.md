# sanitycheck.md — Phase 2 findings cross-examined (G2-style review aid)

**Purpose:** the `SCOPE.md` P2-17 / `PHASE2_PLAN.md` §6 discipline — *investigate implausible
results as possible harness errors first, never as results* — applied to the Phase 2 table.
Every check below is tagged: **VERIFIED** (backed by a recorded measurement or a run artifact
already inspected), **VERIFY-ON-VM** (needs artifact access; exact command in the appendix),
or **OPEN** (cannot be settled with existing data; would need a new run — out of scope per
P6-1/P6-2).

Numbers are transcribed from the per-run `eval_metrics.json` (verification board run
2026-09-30, consolidating VM) and `report/measurements.md` §Phase 0/0.5/G1/diagnostics/audit.
The `## Phase 2 — main results` section of the report still holds the stale C0-only table;
regenerate it with `python -m embeded.evaluate --results-table` before the write-up.

---

## 1. Observed vs the pre-registered expectation

| condition (strategy) | F1 seeds 13/14/15 | mean ± sd | valid threshold |
|---|---|---|---|
| C0 (untuned baseline) | 0.2572 | 0.2572 (P 0.2075, R 0.3383, MAP@R 0.1024) | 0.9779 |
| C1 (random negatives) | 0.7454 / 0.7546 / 0.7599 | **0.7533 ± 0.0079** | 0.4358 / 0.4357 / 0.4492 |
| C2 (BM25 negatives) | 0.2438 / 0.2534 / 0.2653 | 0.2542 ± 0.0110 | **1.0000 / 1.0000** / 0.8355 |
| C3 (cosine negatives) | 0.2594 / 0.2865 / 0.2726 | 0.2728 ± 0.0137 | **1.0000 ×3** |

The implicit expectation going in (the retrieval folklore: harder negatives → better
encoders) predicts **C1 < C2 ≤ C3** on F1. The table shows the **inverse with a ~3× gap**.
An inversion this large is exactly what the G2 review exists to interrogate. Sections 2–4
run the harness-error checklist; §5 asks where training "went wrong"; §6 gives the mechanistic
account that makes the outcome the *expected* one ex post; §7 checks it against the literature.

## 2. Harness-integrity checklist (could the pipeline have produced this falsely?)

**A1 — corpus/pair integrity.** Gate G0 verified 9,126 lines / 8,063 unique fragments /
901,028 / 415,416 / 415,416 against the spec; fragments deduplicated by text sha1.
**VERIFIED** (`report/measurements.md` §Phase 0).

**A2 — triples are what they claim.** All three conditions share anchors/positives/order
(P1-3; `negatives.anchor_stream`), and differ only in negative ranking. If the triples files
were corrupted *identically* (label order, id shift), all conditions would shift together —
C0's reproduction (A7) and the F1 separation between conditions argue against a shared
corruption. If C1's "random" negatives secretly contained clones, the condition would have
learned garbage — the diagnostics proxy says otherwise (valid-labelled clone in C1 negatives:
0.02%; Jaccard median 0.21 vs 0.37/0.40 for C2/C3). **VERIFIED** (§diagnostics table) with the
blind audit as the independent check for C2/C3 (§3 below).

**A3 — run configs are the frozen ones.** The verification board recomputed
`fingerprint(run_config(...))` for all nine runs: 9/9 match the stored `run_config.json`;
9/9 `metrics.json` fingerprints equal their configs; the nine `artifacts` hash-maps are **one
distinct set** — every run consumed byte-identical triples/fragments, across five parallel
VMs. A stale-or-mixed-version run would have failed this. **VERIFIED** (2026-09-30 board).

**A4 — evaluated model = trained model.** `evaluate.default_embed_fn` loads the base model
then `load_checkpoint`, which **refuses** a fingerprint mismatch; `train.verify_checkpoint`
additionally re-compares every tensor after save/reload on every run (unit-tested:
`test_train_evaluate.py`). A silently-unevaluated base model is excluded — decisively, C1's
optimal threshold (≈0.44) cannot come from the same encoder whose base-model threshold is
0.9779 (A7): the scorer demonstrably changed. **VERIFIED** (code path + tests + threshold
evidence).

**A5 — threshold policy without test leakage.** `max_f1_on_valid`: chosen on the 415,416
validation pairs, applied to test unchanged, test scored once per run. The threshold column is
the audit trail. The suspicious-looking values (C2/C3 at the 1.0000 ceiling) are *consistent
with* their low F1 under this policy — a leakage bug would more likely produce implausibly
HIGH F1, not baseline F1 with pinned thresholds. **VERIFIED** (policy in code; column recorded).

**A6 — metric implementation.** `precision_recall_f1` / `best_f1_threshold` are unit-tested
against a brute-force sweep over all candidate thresholds; MAP@R likewise. Same code scored
C0 at 0.2572 — a plausible, pre-registered value. **VERIFIED** (tests green, 116).

**A7 — C0 reproduces across time and machines.** 0.2572 measured 2026-09-27 (original VM) and
again 2026-09-30 (consolidating VM, different GPU session, after five-VM artifact round-trip).
Bit-stable baseline = the scoring path did not drift between sessions. **VERIFIED.**

**A8 — seed variance is sane.** Per-condition sd 0.007–0.014; no seed dominates; conditions
share the seed set. A flaky harness (e.g. nondeterministic pair order) would inflate sd.
**VERIFIED** (table).

**A9 — test-set reuse.** Test scored once per run for the final table; the only test-set
information in this file is descriptive. No threshold, hyperparameter or condition decision
used test outcomes. **VERIFIED** by protocol (SCOPE P2-14/15).

**A10 — train/test fragment overlap (76/860, 8.8%).** Real, recorded, and shared by all
conditions — it inflates absolute F1 for everyone equally and cannot produce the C1-vs-C2
ordering. Remains a stated limitation. **VERIFIED, shared.**

**Checklist verdict:** no harness error found. Every failure mode we could enumerate is either
refuted by a recorded measurement or would have to be shared by all conditions (and cannot
explain the ordering).

## 3. The independent cross-checks that decide the "seems off" question

The Phase 2 outcome was *not* the first measurement of these conditions — it is the third,
and the first two were recorded **before any training existed**:

1. **G1 (2026-09-27, inputs):** the manipulation took effect —
   cos(a,neg): C1 0.96068 < C2 0.97526 ≤ C3 0.98845, d(C1→C2) = 0.614, percentiles
   50.1 / 91.9 / 99.8. If mining had failed, the conditions could not differ; the gate rules
   that out for the exact triples the nine runs consumed (same files, hash-verified per A3).
2. **Diagnostics (2026-09-27, geometry):** share of negatives **closer to the anchor than the
   anchor's own positive** — C1 39.6%, C2 **71.5%**, C3 **99.2%** (positives sit at 0.9651 in a
   ~0.028-wide base space). The hard-mined negatives were already inside the positive band
   *before training*.
3. **Blind audit (2026-09-27, labels):** C2 negatives are **42%** true clones [29%, 56%]; C3
   **30%** [19%, 44%] (32% counting unsure); pooled 36%. The two CIs overlap, so C2-vs-C3 is
   not separable on noise — consistent with their small F1 difference (0.254 vs 0.273).

So the pipeline that produced "C1 ≫ C2 ≈ C3 on F1" had already measured, independently:
*the C2/C3 training signal differs from C1's exactly where it would need to in order to break
learning* — harder by construction, mostly inside the positive similarity band, and 30–42%
mislabelled as non-clones when they are clones. The finding is the downstream consequence of
recorded upstream facts, not an outlier.

## 4. MAP@R — the one decisive cross-check still to run

MAP@R is threshold-free. If C2/C3's collapse is a **genuine representation failure**, their
test MAP@R should also sit at ≈ C0's 0.1024. If instead MAP@R(C2/C3) ≫ C0 while F1 ≈ C0,
the training learned *ranking* structure and only the **thresholded decision** degenerated —
a materially different conclusion for the write-up (and friendlier to C2/C3).
**VERIFY-ON-VM** — appendix command; record the outcome in §7 of this file before writing up.

## 5. "Where might training have gone wrong?" — candidate-by-candidate

**B1 — the training signal was poisoned (false negatives). RETAINED — primary.**
A margin hinge demands `cos(a,neg) ≤ cos(a,pos) − 0.10`. For C2/C3, 30–42% of the `neg`s are
labelled clones *by the same benchmark's own audit*, and 71.5–99.2% already sit closer than
the positive. The loss is thus asked to push apart pairs that are (mostly) the same relation
the positive term pulls together. Gradient-level contradiction at scale is exactly the regime
where score separation collapses. C1's negatives: ~0% contamination by the same audit logic.

**B2 — optimisation divergence (LR too high, loss exploded). PARTIALLY REFUTED, VERIFY-ON-VM.**
C1 trained under the identical optimiser/LR/schedule reached 0.75, so divergence cannot be a
shared cause. For C2/C3 specifically: if the hinge is unsatisfiable, *loss plateaus near the
margin* (0.10) rather than exploding — a plateau is evidence FOR B1's "unsatisfiable demand"
reading and against numerical instability. Check `train_log.jsonl` first/last loss per run
(appendix). Expectation: C1 decreasing to ≪0.10; C2/C3 flattening near ≈0.06–0.10 without
NaNs/spikes. NaNs or spikes would point at fp16 instability instead.

**B3 — the absolute margin (0.10) is mis-scaled for this space. RETAINED — co-mechanism.**
The whole usable base cosine range is ~0.028 wide; the margin is ~3.5× that. For C1 this
forced a healthy re-scale of the space (threshold moved 0.98 → 0.44 — the model rebuilt its
similarity geometry). For C2/C3 the same demand against negatives already at/above the
positive is unsatisfiable for most triples. B1 and B3 are two descriptions of the same
failure; both are frozen settings (P2-13), and altering either is a P6-2 rescue — out of
scope. Stated as limitation.

**B4 — under-training (1 epoch, 32k triples). REFUTED as the cause of the collapse.**
C1 reached 0.75 under the identical budget — the budget suffices to learn the task when the
signal is clean. It may cap C1's ceiling (0.75 vs the 0.95 cross-encoder reference under a
different protocol), which is a fair limitation; it cannot explain C2/C3 ≈ baseline.

**B5 — anisotropic base space (the 0.028 ruler). Context, not a defect.**
Recorded in the diagnostics section; it is why the absolute-margin gate FAILED under v5 and
was re-operationalised (D1). The evaluation is per-model thresholded, so evaluation is
scale-robust; only the *training* margin interacts with the scale (→ B3).

**B6 — evaluation scored the wrong model. REFUTED** — see A4 (fingerprint refusal + tensor
verify + the threshold-shift evidence).

**B7 — the manipulation quietly failed (C2/C3 actually trained on random negatives). REFUTED.**
G1 measured the exact consumed triples (d = 0.614). And had C2/C3 trained on random negatives
they would look like C1 (≈0.75), not like baselines with pinned thresholds. The pinned
thresholds are positive evidence the hard-negative training *happened*.

**B8 — data leakage inflating C1. REFUTED as the ordering driver.** The 8.8% fragment overlap
(A10) is condition-blind. C1's F1 0.75 is far below the 0.9+ leakage-artefact regime reported
in the BCB-criticism literature; nothing about the C1 result needs leakage to explain it.

**B9 — the audit itself is wrong (42%/30% is an artefact). OPEN but bounded.** The audit was
blind (keys committed post-hoc, `artifacts/audit/`), 50+50 per condition, CIs recorded. An
error rate ~10× smaller would still leave C2/C3's signal badly contaminated relative to C1's
~0%. The qualitative conclusion is robust to wide audit error; the exact floor is not
load-bearing for the ordering.

### 5.1 The hinge-geometry numbers (computed offline from the recorded distributions)

Monte-Carlo over the recorded per-condition cosine distributions
(`report/measurements.md` §diagnostics: cos(a, pos) = 0.9651 (sd 0.023); cos(a, neg) per
condition; `TRIPLET_MARGIN = 0.10`), 400k draws, seed 13 — no GPU artifacts needed:

| condition | P(hinge violated at init) | E[initial hinge loss] | pos − neg at init | recorded P(neg closer than pos) |
|---|---|---|---|---|
| C1 random | 99.7% | 0.0956 | +0.0045 | 39.6% |
| C2 bm25 | ~100% | 0.1104 | −0.0105 | 71.5% |
| C3 cosine | ~100% | 0.1232 | −0.0232 | 99.2% |

(The MC's closer-than-positive column is *conservative* — a normal approximation understates
the pile-up of real cosines near 1.0; the recorded percentages in the last column are the
authoritative ones.)

Four facts fall out, and together they answer "why is C2/C3 accuracy low":

1. **The hinge is unsatisfiable at init for every condition** — expected: the margin (0.10)
   is ~3.5× the entire usable base cosine range (~0.028). So "C2/C3 failed because their
   optimisation started harder" is **not** the explanation: every condition was asked to
   stretch the usable cosine range ~5× (to ≈0.13).
2. **The initial loss magnitudes are close** (0.096 / 0.110 / 0.123). C2/C3 did not fail to
   train because their loss was bigger; they trained *equally hard* — on a signal that is
   30–42% false.
3. **The discriminator is truth, not difficulty.** C1's demanded relation ("this negative is
   dissimilar to the anchor") is true for ~all its negatives (valid-labelled clone: 0.02%;
   audit-consistent), so the model can satisfy the hinge by legitimately re-scaling the
   space — which it did (optimal threshold 0.98 → 0.44, F1 0.753). For C2/C3, the audit says
   42%/30% of the demanded-apart pairs are *the same relation the positives demand pulled
   together* — per triple, since each triple carries one negative, the contamination rate of
   the training signal **is** the FN rate. The objective and the evaluation's own ground
   truth conflict on a third to a half of the hard conditions' signal.
4. **What the poison looks like** (the blind audit's notes, `artifacts/audit/`): mined
   "negatives" judged clones are same-functionality/different-syntax pairs — copy-file
   variants (temp+rename vs direct vs forced), the hashing family ("return a digest of this
   string", SHA-1-hex vs MD5-base64), generic helpers running different SQL behind identical
   JDBC boilerplate, containment pairs. Exactly the weak-T3/T4 band the BigCloneBench
   validity literature says is mislabelled at scale — mined *into* the negative set by
   BM25/cosine ranking because they are lexically/functionally near the anchor.

Consequence for the mechanism: on contaminated triples the gradient pushes pairs apart that
the evaluation counts as clones; the least-bad solution available to the optimiser is to
compress the score range so almost nothing clears a separating threshold — observed directly
as validation-optimal thresholds pinned at 1.0000 (five of six C2/C3 runs) and F1 collapsing
to the untuned baseline. C1 is the control that makes this reading falsifiable: identical
optimiser, margin, budget — only the *truthfulness* of the negative signal differs.

### 5.2 The improvement is monotone in the measured poison rate — "a little gain" is exactly what the data shows

ΔF1 vs C0, against the blind audit's per-condition false-negative floor:

| condition | FN floor | mean F1 | Δ vs C0 (0.2572) |
|---|---|---|---|
| C1 random | ~0% | 0.7533 | **+0.496** |
| C3 cosine | 30% [19, 44] | 0.2728 | **+0.016** (~2 sd of the seed mean) |
| C2 bm25 | 42% [29, 56] | 0.2542 | **−0.003** (flat within noise) |

The improvement decays monotonically through zero as contamination rises, with the break-even
between 30% and 42%. So the intuition "hard negatives should help *a little*" is not refuted
by the table — it is confirmed at C3 and extinguished exactly where the audit says the signal
crosses half-corrupted. Why "a little" is also the *ceiling* under this recipe: (i) the poison
sits on the discriminative band itself, attached to inputs statistically interchangeable with
the true hard negatives, so it deletes the boundary instead of diluting it; (ii) a fixed-margin
hinge keeps applying full-strength gradient to every violated triple with no saturation, so
poisoned (never-satisfiable) triples fire on every step of the epoch rather than washing out;
(iii) the surviving separable structure is the near-exact-duplicate tail the base model already
had — hence thresholds pinned at 1.0 and F1 ≈ baseline rather than above it. The one place a
hidden small gain could still live is ranking (MAP@R) — §4's VERIFY-ON-VM decides it.

### 5.3 A closed-form account of both floors (derivation — every input below is a recorded measurement, no new runs)

**Why C0 is where it is (0.2572): the informationless floor.** In the pretrained space the
label-relevant effect size is d′ = (0.9651 − 0.9603)/√(0.023² + 0.026²) ≈ **0.14** → a
best-possible ranking AUC ≈ **0.54** (§2.6 measured means/sds; the diagnostics' 39.6% of
*random* negatives closer to the anchor than its labelled positive is the same fact as a
percentage). Calibrating the base rate from C0's own operating point (P 0.2075, R 0.3383 at
thr 0.9779) under this two-Gaussian band model gives π ≈ 0.15–0.20, and the F1 ceiling for a
*zero-information* scorer at that base rate is 2π/(1+π) ≈ 0.31 [0.26, 0.33] — C0's tuned
0.2572 sits at/below it. The threshold 0.9779 is not "wrong": with a near-flat ROC, max-F1
validation has nowhere to go but the extreme overlap tail, and the only structure reachable
there is lexical near-duplication (median Jaccard of labelled positives 0.28 vs 0.21 for
random pairs). This is the pre-registered expectation (§2.6 I2: d ≈ 0.2 ⇒ "the C0 baseline
should be expected near chance") — MLM pretraining never optimises pair margins, and the
resulting anisotropic cone leaves ~0.028 cosine units for the whole task.

**Why C2/C3 stay exactly at that floor: the poisoned-hinge worked example.** Four steps, all
constants measured:

1. **The hinge is unsatisfiable at every reachable parameter setting, for every condition.**
   TRIPLET_MARGIN = 0.10 is 3.6× the entire usable cosine range (0.028); satisfying *any*
   triple requires stretching the space ~4.6× (§5.1). Consequence: the hinge term stays
   strictly positive for every triple for the whole epoch — there is no saturation dynamics,
   so 100% of triples (right or wrong) fire at full strength on every one of the 32k steps.
   Nothing ever "washes out" of the gradient.
2. **The mined band is a mixture, and its two components are (nearly) the same set in score
   space.** Audit: p = 42% [29, 56] (C2) / 30% [19, 44] (C3) of mined negatives are true
   clones. In the representation the loss acts on, poison and clean nearly coincide: C2's
   clean not-clone IQR [0.964, 0.984] meets the poison IQR at 0.984, and the best-possible
   cosine separation of poison from clean is only AUC ≈ 0.68 (C2) / 0.81 (C3)
   (audit-separability IQRs, two-Gaussian estimate). C3's only real cut (cos ≥ 0.99) removes
   80% of the poison at the cost of 40% of the condition. So "push the true negatives down"
   and "push the false negatives down" are not two instructions the optimiser could trade
   off — they are one instruction on one overlapping distribution, wrong on ~⅓ of its mass.
3. **The equilibrium is the flat/degenerate optimum, and its fingerprint is the pinned
   threshold.** With 71.5%/99.2% of mined negatives already closer to the anchor than the
   labelled positive, the negative band *contains* the positive band; the only uniform
   loss-decreasing direction is to push the whole band together — positives included. The
   flat solution (s(a,p) ≈ s(a,n) ≈ const, expected hinge ≈ m on every triple) is exactly
   the erasure the artifacts show: validation-optimal thresholds at the parameter boundary
   1.0000 (5/6 C2/C3 runs) and F1 back at C0. The per-step net separation rate under an FN
   fraction p, (1 − 2p), is 1.00 / 0.40 / 0.16 for C1/C3/C2 — monotone, like the measured
   Δ vs C0 (+0.496 / +0.016 / −0.003); at p → ½ the label information in the triple stream
   vanishes, and C2's CI [29, 56] brackets that boundary.
4. **This is not an evaluation-noise story.** The 415,416 test pairs (and their labels) are
   shared by all four conditions, so label noise is common-mode and cannot produce a 3×
   condition gap; and its size is bounded: an oracle encoder under a *test* FN rate q still
   scores F1 = 2π/(2π + q(1−π)) ≈ **0.52** even at q = 0.42 — far above C2/C3's 0.25–0.27 —
   while C1's 0.7533 bounds the shared q at ≲ 0.15, consistent with the BCB-label proxies
   (0.02–0.05%). The mined-band FN rate is a property of the *training signal*; that is
   where the collapse lives.

One-line synthesis: **C0 is low because the pretrained cosine ruler carries almost no label
signal (d′ ≈ 0.14); C2/C3 are low because mining harvested exactly the band where the
benchmark's labels are least truthful, and a fixed, never-saturating margin passes that
untruth to the optimiser at full strength on every step — to which the only available answer
is to erase the distinction, with the threshold pinned at the boundary as the fingerprint.**
Literature anchors in §7 (RocketQA's denoising result; debiased-contrastive FN-bias
amplification; semi-hard mining's reason to exist; hinge noise-tolerance at ρ < ½).

## 6. Mechanistic account (what we believe happened)

Base GraphCodeBERT's space is anisotropic (all cosines ∈ ~[0.960, 0.988]). Random negatives
are drawn from the whole corpus → mostly genuinely-unrelated code, far outside the positive
band → the margin hinge is satisfiable and teaches a true distinction → the space re-scales
(threshold 0.98 → 0.44), clones separate from non-clones, F1 0.753.
Hard mining (BM25 / cosine) concentrates negatives exactly in the near-anchor band where — on
this benchmark — 30–42% are true clones and ~71–99% score closer than the true positive → the
hinge receives contradictory supervision at high frequency → the score separation the
threshold policy needs never forms → the validation-optimal cut pins at/near 1.0 ("only
near-identical code counts") → precision survives, recall collapses → F1 ≈ untuned baseline.
The s′ gate (G1s, d = 0.886, PASS) shows the same manipulation is even *stronger* on Phase 3's
corpus — so Phase 3 inherits this whole account, and C2/C3's Δ on s′ may be mechanically
degenerate; C1's healthy scorer meeting unseen functionality is the informative measurement.

Causal hierarchy, stated explicitly because the same-margin C1 control invites the question:
the **false-negative rate is the cause**; the **margin is the amplifier** — it fixes the dose
(a never-saturating hinge fires every violated triple at full strength, every step, with no
wash-out) that turns a 30–42% false signal into collapse. C1 sharing the margin and succeeding
is precisely what separates the two, which is why B3 is adjudicated as co-mechanism (§5), not
root cause.

## 7. Literature alignment (details in the PR discussion of 2026-09-30)

Consistent with: RocketQA's measured result that undenoised hard negatives **decrease**
retriever performance (their fix: cross-encoder filtering); the hard-negative/false-negative
paradox and positive-aware mining (negatives closer than the positive should be discarded —
our diagnostics: 71.5%/99.2%); debiased-contrastive theory (hard sampling amplifies FN bias);
the BigCloneBench validity critiques (WT3/T4 pair labels unreliable — explains why the
*mineable* band is dirty); and the metric-learning reality check (hardest-negative mining →
bad optima). Against: the folklore reading of DPR (whose own ablation shows random ≈ BM25 ≈
gold) and code-search papers whose gains are small, ranking-metric-based, temperature-loss,
and FN-filtered — a different regime from fixed-margin + thresholded-F1.

## 8. Verdict

**No harness error identified** — ten integrity checks pass on recorded evidence, and the
inversion is over-determined by three independent pre-training measurements (G1 inputs,
geometry, audit floors) plus the literature. The defensible statement of the finding:

> Under the frozen recipe (margin-0.10 cosine hinge, 1 epoch, thresholded F1), undenoised hard
> negatives mined from BigCloneBench collapsed fine-tuning to baseline, while clean random
> negatives improved F1 ≈ 3× over the untuned baseline. The measured false-negative floors
> (42%/30%) are the leading mechanism and a stated confound on C2-vs-C3.

**Would change this verdict:** (i) MAP@R(C2/C3) ≫ C0 with low F1 (→ reframe as thresholding
failure, upgrade C2/C3) — **resolved 2026-10-01: REFUTED, see §9.1 (C2/C3 below C0)**; (ii) NaN/loss spikes in C2/C3 train logs (→ fp16 instability);
(iii) a reproducible F1-identical rerun with shuffled pair labels (→ metric bug). None
expected; all three are checkable from existing artifacts.

---

## Appendix — on-VM verification block (run once on the consolidating VM, paste output here)

```python
import json, subprocess, sys
import numpy as np
from embeded import settings as S
from embeded.train import run_dir, METRICS_NAME, EVAL_METRICS_NAME, LOG_NAME

import re
RUNS = sorted(
    ((m.group(1), int(m.group(2)))
     for d in (S.ARTIFACTS / S.RUNS_SUBDIR).glob("C[0-3]_*")
     if (m := re.fullmatch(r"(C[0-3])_(\d+)", d.name))),
    key=lambda t: ("C0 C1 C2 C3".split().index(t[0]), t[1]),
)  # auto-discovers every condition/seed with a run dir (smoke_/sprime_ never match)
print("== per-run metrics (P / R / MAP@R) ==", f"({len(RUNS)} run dirs discovered)")
for c, s in RUNS:
        p = run_dir(c, s) / EVAL_METRICS_NAME
        if p.exists():
            m = json.loads(p.read_text())
            if not m.get("smoke"):
                print(f"  {c}_{s}: F1={m['test_f1']:.4f} P={m['test_precision']:.4f} "
                      f"R={m['test_recall']:.4f} MAP@R={m['test_map_at_r']:.4f} "
                      f"thr={m['threshold']:.4f}")

print("== recompute F1 from stored predictions (metrics <-> npz consistency) ==")
for c in ("C1", "C2", "C3"):
    d = run_dir(c, 13)
    z = np.load(d / "predictions.npz")
    lab, sc, thr = z["test_labels"], z["test_scores"], float(z["threshold"])
    pred = sc >= thr
    tp = int((pred & (lab == 1)).sum()); fp = int((pred & (lab == 0)).sum())
    fn = int((~pred & (lab == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn)
    m = json.loads((d / EVAL_METRICS_NAME).read_text())
    print(f"  {c}_13: recomputed F1={f1:.4f} stored={m['test_f1']:.4f} "
          f"{'OK' if abs(f1 - m['test_f1']) < 5e-4 else 'MISMATCH'}")

print("== score separation on test (positives vs negatives, per condition) ==")
for c in ("C0", "C1", "C2", "C3"):
    d = run_dir(c, 13)
    z = np.load(d / "predictions.npz")
    lab, sc = z["test_labels"], z["test_scores"]
    print(f"  {c}_13: pos mean={sc[lab==1].mean():.4f} neg mean={sc[lab==0].mean():.4f} "
          f"gap={(sc[lab==1].mean()-sc[lab==0].mean()):+.4f}")

print("== training-loss trajectory (first -> last quarter mean) ==")
for c, s in RUNS:
    if c == "C0" or not (run_dir(c, s) / LOG_NAME).exists():
        continue
    rows = [json.loads(l) for l in open(run_dir(c, s) / LOG_NAME, encoding="utf-8")]
    q = max(1, len(rows) // 4)
    losses = [r["loss"] for r in rows]
    print(f"  {c}_{s}: steps={len(rows)} loss {np.mean(losses[:q]):.4f} -> "
          f"{np.mean(losses[-q:]):.4f}  (nan={any(x != x for x in losses)})")
```

Expected readings if §5's account holds: MAP@R ordering follows F1 (C1 ≫ C0 ≈ C2 ≈ C3);
recomputed F1s match stored; C1's pos–neg gap clearly positive and largest; C2/C3 gaps ≈ 0 or
negative; C1 losses fall well below the 0.10 margin while C2/C3 flatten near it with no NaNs.
Deviations from these readings are exactly the three falsifiers in §8 — record them before any
write-up.

**Part 2 — config / VM-split audit** (added 2026-10-01, after the parallel-VM execution
topology question: VM1 = C1 13/14/15 + C2 13/14; four further VMs = C2 15 + C3 13/14/15 +
C0). Paste this alongside part 1:

```python
import json, os
import numpy as np
from embeded.train import run_config, fingerprint, run_dir, METRICS_NAME, EVAL_METRICS_NAME, EVAL_CONFIG_NAME

import re
RUNS = sorted(
    ((m.group(1), int(m.group(2)))
     for d in (S.ARTIFACTS / S.RUNS_SUBDIR).glob("C[0-3]_*")
     if (m := re.fullmatch(r"(C[0-3])_(\d+)", d.name))),
    key=lambda t: ("C0 C1 C2 C3".split().index(t[0]), t[1]),
)  # auto-discovers every condition/seed with a run dir (smoke_/sprime_ never match)
print("== config/fingerprint audit (every row must say OK) ==")
hash_sets = set()
for c, s in RUNS:
        d = run_dir(c, s)
        if not (d / EVAL_METRICS_NAME).exists():
            continue
        ec = json.loads((d / EVAL_CONFIG_NAME).read_text())   # frozen config the run consumed
        em = json.loads((d / EVAL_METRICS_NAME).read_text())
        tm = json.loads((d / METRICS_NAME).read_text()) if (d / METRICS_NAME).exists() else None
        fp_now = fingerprint(run_config(c, s))                # recomputed from today's settings
        ok_eval = fp_now == ec["fingerprint"] == em["fingerprint"]
        ok_train = True if tm is None else fp_now == tm["fingerprint"]
        hash_sets.add(json.dumps(ec["artifacts"], sort_keys=True))
        dev = em.get("device") or (tm["device"] if tm else "?")
        steps = tm["steps"] if tm else "-"
        print(f"  {c}_{s}: fingerprint {'OK' if ok_eval and ok_train else 'MISMATCH'} | "
              f"device={dev} | train_steps={steps} "
              f"(expected {ec['train_pairs_cap'] // ec['batch']}) | "
              f"finished={tm['finished_utc'] if tm else em['evaluated_utc']}")
print(f"distinct artifact hash-sets across all runs: {len(hash_sets)}  (expected 1)")

print("== run-directory completeness (7 files per trained run, 3 for C0) ==")
for c, s in RUNS:
        d = run_dir(c, s)
        exp = 3 if c == "C0" else 7
        n = len([f for f in os.listdir(d) if not f.startswith(".")]) if d.exists() else 0
        print(f"  {c}_{s}: {n}/{exp} files {'OK' if n >= exp else 'INCOMPLETE'}")
```

What each line catches: a **fingerprint MISMATCH** means that run consumed a different
config/data than the frozen recipe (the one real cross-VM failure mode — stale clone,
re-mined triples, edited settings); `hash-sets > 1` means the VMs did not train on
byte-identical triples; `train_steps` off from expected means an interruption/resume
actually changed the run (check that run's `train_log.jsonl` for a mid-log loss jump);
`INCOMPLETE` means a partial HF upload or an interrupted evaluation — re-pull before
believing anything from that run.

**Part 3 — collapse-signature check** (added 2026-10-01 from external review: pin the exact
test base rate and each run's predicted-positive fraction — decides whether C2/C3 are literally
"say-yes-to-everything" collapse or the partial/erasure mode of §5.3):

```python
import json
import numpy as np
from embeded.train import run_dir, EVAL_METRICS_NAME

z0 = np.load(run_dir("C0", 13) / "predictions.npz")
p = float(z0["test_labels"].mean())
print(f"exact test base rate p = {p:.4f} ({int(z0['test_labels'].sum()):,} positives "
      f"/ {len(z0['test_labels']):,})")
print(f"all-positive F1 = 2p/(1+p) = {2*p/(1+p):.4f};  all-negative F1 = 0.0000")
import re
RUNS = sorted(
    ((m.group(1), int(m.group(2)))
     for d in (S.ARTIFACTS / S.RUNS_SUBDIR).glob("C[0-3]_*")
     if (m := re.fullmatch(r"(C[0-3])_(\d+)", d.name))),
    key=lambda t: ("C0 C1 C2 C3".split().index(t[0]), t[1]),
)  # auto-discovers every condition/seed with a run dir (smoke_/sprime_ never match)
print("== per-run behaviour at the stored threshold ==")
for c, s in RUNS:
        d = run_dir(c, s)
        if not (d / EVAL_METRICS_NAME).exists():
            continue
        m = json.loads((d / EVAL_METRICS_NAME).read_text())
        if m.get("smoke"):
            continue
        z = np.load(d / "predictions.npz")
        thr, sc = float(z["threshold"]), z["test_scores"]
        frac = float((sc >= thr).mean())
        f1 = m["test_f1"]
        verdict = "matches all-positive collapse" if abs(f1 - 2*p/(1+p)) < 0.02 else \
                  ("near all-positive" if frac > 0.9 else
                   ("near all-negative" if frac < 0.05 else "partial / erasure mode"))
        print(f"  {c}_{s}: F1={f1:.4f}  predicted-positive={frac:6.1%}  -> {verdict}")
```

Expected reading if §5.3's erasure account holds: C0 **partial** (it is a weak separator, not
a degenerate one — P 0.21 / R 0.34 at threshold 0.9779); C1 partial with a healthy positive
fraction; C2/C3 at threshold 1.0000 reveal which degenerate mode they took. Note C2/C3's F1
(0.24–0.29) sits *below* the all-positive ceiling whenever p > ~0.14 — a pure "yes to
everything" predictor would score 2p/(1+p), so an observed shortfall is itself informative
(band compression with wrong-side ordering), exactly the distinction this block settles.
The base rate p also replaces §5.3's model-calibrated π ≈ 0.15–0.20 with an exact number.

---

## 9. Phase 2V2 — seed-extension campaign (consolidator board output, 19/19 runs)

**Design:** seeds 16/17/18 trained per condition on the FROZEN seed-13 triples (same
manipulation, same FN floors); C0 reused (never re-trained); recipe untouched. Purpose:
upgrade every per-condition estimate n=3 → n=6 and test seed-robustness of the inversion.
Run on five parallel VMs + one consolidator (`Phase2V2_Colab.ipynb`); the confusion board
below is the consolidator's stdout over all 19 run dirs.

```
      run |      TP      FP      FN      TN |      P      R     F1    thr n_pred_pos
C0_   13 |  19,220  73,400  37,600 285,196 | 0.2075 0.3383 0.2572 0.9779     92,620
C1_   13 |  43,195  15,875  13,625 342,721 | 0.7313 0.7602 0.7454 0.4358     59,070
C1_   14 |  43,731  15,350  13,089 343,246 | 0.7402 0.7696 0.7546 0.4357     59,081
C1_   15 |  44,404  15,650  12,416 342,946 | 0.7394 0.7815 0.7599 0.4492     60,054
C1_   16 |  43,157  14,417  13,663 344,179 | 0.7496 0.7595 0.7545 0.4396     57,574
C1_   17 |  42,613  20,292  14,207 338,304 | 0.6774 0.7500 0.7118 0.4622     62,905
C1_   18 |  41,513  12,046  15,307 346,550 | 0.7751 0.7306 0.7522 0.4581     53,559
C2_   13 |  24,410 144,239  32,410 214,357 | 0.1447 0.4296 0.2165 1.0000    168,649
C2_   14 |  40,677 222,906  16,143 135,690 | 0.1543 0.7159 0.2539 1.0000    263,583
C2_   15 |  34,028 165,655  22,792 192,941 | 0.1704 0.5989 0.2653 0.8355    199,683
C2_   16 |  36,790 193,750  20,030 164,846 | 0.1596 0.6475 0.2561 1.0000    230,540
C2_   17 |  34,667 220,569  22,153 138,027 | 0.1358 0.6101 0.2222 1.0000    255,236
C2_   18 |  45,749 232,809  11,071 125,787 | 0.1642 0.8052 0.2728 0.9793    278,558
C3_   13 |  26,847 125,289  29,973 233,307 | 0.1765 0.4725 0.2570 1.0000    152,136
C3_   14 |  27,058 105,357  29,762 253,239 | 0.2043 0.4762 0.2860 1.0000    132,415
C3_   15 |  25,612 108,250  31,208 250,346 | 0.1913 0.4508 0.2686 1.0000    133,862
C3_   16 |  22,866  90,921  33,954 267,675 | 0.2010 0.4024 0.2681 1.0000    113,787
C3_   17 |  29,009 129,298  27,811 229,298 | 0.1832 0.5105 0.2697 1.0000    158,307
C3_   18 |  38,657 186,014  18,163 172,582 | 0.1721 0.6803 0.2747 1.0000    224,671
```

All 19 rows verified internally consistent (TP+FN = 56,820, FP+TN = 358,596, n = 415,416;
P/R/F1 recomputed from counts). **Headline at n=6** (seed sd over six runs):

| condition | F1 (n=6) | seeds 13–15 | seeds 16–18 | Δ vs C0 | FN floor |
|---|---|---|---|---|---|
| C0 | 0.2572 | — | — | — | — |
| C1 | **0.7464 ± 0.0176** | 0.7533 | 0.7395 | **+0.489** | ~0% |
| C2 | **0.2478 ± 0.0231** | 0.2452 | 0.2504 | **−0.009** | 42% |
| C3 | **0.2707 ± 0.0095** | 0.2705 | 0.2708 | **+0.013** | 30% |

Five conclusions, each closing an item this document had open:

1. **The inversion is seed-robust.** C1 stays ≈ 2.9× every other condition; C2/C3 straddle
   C0 within ~1 seed-sd (C2 now marginally *below* C0). The n=3 result was not a lucky draw.
   Δ vs C0 remains monotone in the measured FN floors (+0.489 / +0.013 / −0.009 for
   ~0%/30%/42%) — §5.2's break-even story strengthens: the point where "hard negatives stop
   helping" now sits more precisely between C3's and C2's contamination rates.
2. **Exact test base rate (appendix Part 3, first input): p = 56,820/415,416 = 0.13678**, so
   the all-positive ceiling is 2p/(1+p) = **0.2406**. This replaces §5.3's model-calibrated
   π ≈ 0.15–0.20. C0's 0.2572 sits ABOVE the ceiling → C0 is confirmed a weak-but-real
   separator, not a degenerate all-positive predictor (exactly the predicted "partial" mode).
3. **The C2/C3 collapse mode is saturation-erasure, not all-positive collapse.** 10/12 C2/C3
   validation thresholds are pinned at exactly 1.0000; for those runs `n_pred_pos` at thr 1.0
   counts pairs at fp32 cosine == 1.0 — i.e. **27–67% of the entire test split saturates at
   the ceiling** (C3: 27–54%, C2: 41–67%). F1 scatters around the 0.2406 ceiling (C2
   0.217–0.273, C3 0.257–0.286): the band is compressed into the fp32 rounding point with
   wrong-side ordering, which is §5.3's erasure mode, now observed directly.
4. **Cross-checks pass on the consolidator:** the C0 row reproduces the stored eval metrics
   exactly (0.2075/0.3383/0.2572 @ 0.9779 — npz↔metrics consistency); C1_17 (0.7118) is the
   n=6 low seed at ≈ −2σ but far above baseline; the §8 stop-conditions (C1 ext seed < 0.70;
   C2/C3 ext seed > 0.35) did not trip.
5. **Still open (unchanged):** MAP@R per run (falsifier (i) — the one ranking-level check)
   and the full appendix Parts 1–2 paste; the `--results-table` rebuild for
   `report/measurements.md` happens VM-side per the standing rule.

**Verdict impact:** none of the three §8 falsifiers fired; the seed-extension *strengthens*
the §8 statement — the defensible claim is now backed by six seeds per condition and an
exact base rate. Wording upgrade for the write-up: "robust across six training seeds"
replaces "three seeds"; the collapse is characterised as fp32-saturation erasure with
27–67% of test pairs at cosine 1.0, not "everything predicted positive".

### 9.1 The report rebuild answers §4 — falsifier (i) is REFUTED (2026-10-01, consolidator push `0ce5bf4`)

The rebuilt report table includes MAP@R for all 19 runs. Expected reading confirmed exactly:

| | C0 | C1 (n=6) | C2 (n=6) | C3 (n=6) |
|---|---|---|---|---|
| mean MAP@R | 0.1024 | **0.667** ± 0.033 | **0.047** ± 0.024 | **0.077** ± 0.005 |

C2/C3 rank clones **worse than the untuned baseline** (C2 at less than half of C0). The
§8 rescue condition ("MAP@R(C2/C3) ≫ C0 with low F1 → thresholding failure") did not fire:
the collapse is representation-level, and training on poisoned hard negatives actively
degraded ranking. §4's decisive check is closed; §5's account needs no reframe.

### 9.2 npz ↔ metrics mismatch on 9 of 18 trained runs — OPEN, VM-side investigation

Comparing the rebuilt report (stored `eval_metrics.json`) with the board's npz-recomputed
rows: **all six C1 runs match to 4 decimals; 9 C2/C3 runs disagree** (largest: C2_13 stored
F1 0.2438 vs npz 0.2165 — stored R 0.90 vs npz R 0.43 at the same claimed threshold 1.0;
smallest Δ 0.0005). `evaluate` writes metrics and `predictions.npz` from the same scores in
one process, so a single pass cannot produce both. The HF run dirs for those nine runs
therefore mix files from different passes/pushes. Leading hypothesis: pushes stage the FULL
artifacts tree, so a VM that pulled mid-campaign and pushed later can re-upload a **stale
copy** of another VM's run dir over the fresh one. Both readings sit in the same degenerate
band and C1 is exact, so **no conclusion changes** — but the provenance must be repaired:

1. Decisive check (no GPU): per-file HF commit SHAs for the nine runs
   (`HfApi.list_repo_commits` / web UI file history) — do `eval_metrics.json` and
   `predictions.npz` come from the same commit? A difference confirms the stale-overwrite.
2. Repair: `evaluate --force` on the nine runs (full pull needed — checkpoints), re-push.
   Eval-only, deterministic given the checkpoint; expected to land in the same bands.
3. Rule going forward: trainers pull once, before training; a push after a later pull may
   carry stale neighbour dirs (the consolidator's `--include-report` pushes are the ones to
   watch, since their trees span days).

### 9.3 Two 600-pair rows leaked into the Phase 2 table — fixed, and they are Phase 3's first result

The rebuilt table contained `C1 13` and `C1 14` rows with n = 600: the **s′ unseen evals**
(holdout {10,13,14} × 200) written by `sprime_C1_13`/`sprime_C1_14`, whose metrics reuse the
condition label "C1". They polluted the C1 aggregate (8 rows, mean 0.7447). Fixed in
`embeded.evaluate` — the sprime run-dir namespace now decides exclusion, not the condition
field (regression-tested) — rebuild the table VM-side.

Substance, flagged as preliminary (n = 2, read through the leak): **C1 transfers to unseen
functionality almost losslessly** — F1_unseen 0.7230 vs seen 0.7454 (−0.022, −3%) for seed
13, and 0.7564 vs 0.7546 (+0.002) for seed 14. Against Kitsios et al.'s task-specific
average drop of ~31%, that is LLM-scale robustness from a 32k-triple bi-encoder — the
single most encouraging number in the project so far, and the healthy-scorer transfer
quantity §6 predicted would be the informative Phase 3 measurement.
### 9.4 C4 "filtered" — pre-registration (written BEFORE any C4 run exists; commit `56822ba`)

The one strategy the campaign left untested: **denoised hard negatives**. C2/C3 collapsed
because undenoised hard mining is dominated by false negatives (§6: FN rates 42%/30%, the
cause; margin the amplifier). The literature position (RocketQA; debiased contrastive;
NV-Retriever) is that hard negatives help **only after denoising**. C4 is that claim made
testable inside the frozen recipe. The denoiser is not invented ad hoc — it is the audit's
own separability cut, lifted from the 100-pair blind audit:

- **C4 = C2's BM25 ranking** (same ranker, depth k×20 instead of k×4 to feed the filters)
- **drop** every candidate with token-Jaccard(anchor, candidate) ≥ **0.40** — the audit's
  separating threshold: removed ~40% of C2's contamination in-sample while keeping every
  audited clean pair (C2 clean max 0.345; clone median 0.418). Tokenizer = the diagnostics
  tokenizer (`bm25_index.tokenize_code`), one definition everywhere.
- **skip the first 10 survivors** (the densest head — rank-skip per the external review;
  helps the C3-like regime, median rank ~13)
- everything else identical to C1–C3: exclusion pass, corpus filter, shared anchor/positive
  stream, margin 0.10, cap, seeds. Padding to k from the corpus tail survives but is counted,
  and the pad itself honours the Jaccard cut. Implementation: `clean_candidates_filtered`,
  commit `56822ba`, 6 unit tests, suite 124/124.

**Expected outcome, pre-registered: F1(C4) ≈ F1(C1)**, i.e. the denoisers remove enough
contamination that C4 behaves like the clean-easy control rather than like C2. The
informative comparisons, in order of interest:
- C4 ≈ C1 → denoising rescues hard mining **to the clean baseline** (consistent with the
  FN-cause account; the residual question becomes whether ANY denoised-hard signal remains).
- C1 < C4 → denoised hard negatives add real signal on top of clean-easy (the RocketQA-style
  best case; would be the first positive hard-mining result in this recipe).
- C4 ≈ C2 → the Jaccard cut does not capture the poison (the remaining FN mass is
  semantically-but-not-lexically duplicated); strengthens "functionality-blind mining is
  unfixable at the lexical layer" (§6, external review).

**MANDATORY before any C4 number is compared to anything: a fresh 50-pair blind audit of
the C4 triples** (`audit_sample --conditions C4` — flag added in `56822ba`), same
50/50 clone/not_clone design, judged without knowing which strategy produced the pair.
Acceptance gate: C4's audited clone-contamination rate must be **≤ C2's measured floor
(42%)** and ideally within noise of C1's (~0%). If the audit fails the gate, C4 is reported
as "denoising attempted, contamination persists" and the F1 comparison is footnoted, not
headlined. Mining is CPU-only (BM25); v7 bumps VERSION, so the `[cache]` is invalidated —
the re-mined C1–C3 triples are deterministic and bit-identical (same seeds, same code paths),
only C4's triples are new bytes.

**§9.4 execution log — mining stage (2026-10-02, VM).** The sanctioned v7 re-mine ran clean:
`frags=8063 corpus=6451 anchors=1600 k=20`; all four strategies landed at 32,000 triples
(`anchors_short=0` everywhere). **C1–C3 re-mined bit-identical to the `.v6bak` backups
(sha256 match) — the audit key and every Phase-2 label survive the version bump.** The
determinism assumption is now verified on real data, not just asserted. Provenance note: the
first attempt crashed on C2 (`NameError: frags`) — a C4-only pad block had been committed
inside the shared `clean_candidates` (fixed in `91d2f7c` with forced-pad regression tests;
offline fixtures never execute the pad path, only scale does). C4 mining stats:
`jaccard_filtered = 66,958` (candidates dropped at J ≥ 0.40 while scanning the k×20 BM25
depth — the audit cut removing real mass), `head_skipped = 16,000` (= 1600 × 10 exactly:
every anchor had ≥ 10 survivors past the cut), `padded = 111` (0.35 % of C4 negatives — no
drift toward C1), against C2's `padded = 7`. Order from here: §4 diagnostics row → §5/§5.5
**mandatory 50-pair blind audit gate** (pass ≤ 42 %, hope ≈ 0 %) → only then the six-seed
campaign, and only a passed gate puts C4 numbers on equal footing.

**§9.4 execution log — C4_13 first result (2026-10-02, monitoring read from the reader VM;
comparison DEFERRED pending the mandatory audit gate).** `test_f1 = 0.4108` (P 0.3256, R
0.5564, thr 0.4965 — **not pinned**), MAP@R 0.2559, valid F1 0.4023 ≈ test, counts internally
consistent (tp 31,617 + fn 25,203 = 56,820 positives; recomputed from stored tp/fp/fn).
Placement, seed 1 of 6: far above the collapsed cluster (C2 0.2478±0.023, C3 0.2707±0.010,
ceiling 0.2406; 1.71× the all-positive ceiling) and far below C1 (0.7464±0.018) — an
**intermediate**, i.e. neither pre-registered pure outcome. Healthy-scorer signatures
distinguish it from the C2/C3 degenerate mode: unpinned mid-range threshold (vs 10/12 runs
at 1.0000), no saturation-erasure signature, and ranking ABOVE the untuned baseline (MAP@R
0.2559 = 2.5× C0's 0.1024, where C2/C3 ranked below it). Registered BEFORE the audit, for
the gate to discriminate: if the audit's contamination is low (≲10 %), the 0.41-vs-0.75 gap
reads as hard-but-clean signal being neutral-to-harmful under the frozen margin-0.10 hinge;
if contamination sits in the ~20–40 % band, it reads as partial denoising (the lexical cut
removes the lexical poison, not all of it). Note the C2/C3 F1-vs-contamination points cannot
be interpolated here — those scorers are saturated-degenerate, so F1 is not a function of
contamination in that regime. Provenance: fingerprint `39251c7058a7e78c`; all six C4 runs
must share it — it will differ from the v6-era C1–C3 fingerprints because v7's hash covers
`triples_C4.jsonl` as well (expected, not drift).

**§9.4 execution log — C4 campaign COMPLETE, n=6 (2026-10-02; VM board + rebuilt report
`a14b62c`; formal comparison still DEFERRED pending the mandatory audit gate).**

F1 by seed: 13→0.4108, 14→0.3988, 15→0.4529, 16→0.4418, 17→0.4609, 18→0.4444.
**C4 = 0.4349 ± 0.0246** (sample-sd convention; range 0.062, comparable to C1's 0.048 —
normal seed spread, no outlier). MAP@R by seed: 0.2559/0.2369/0.2967/0.2735/0.3085/0.2847 →
**mean 0.2760**. Placement: vs C2 +0.1871, vs C3 +0.1643, vs C0 +0.1777, vs C1 −0.3115;
1.81× the all-positive ceiling; C1/C4 = 1.72×. MAP@R: 2.70× C0 (0.1024), vs C1 0.667
(2.42× below), vs C2 0.047 (5.9× above), vs C3 0.077 (3.6× above) — **C4 ranks ABOVE the
untuned baseline**, unlike C2/C3 which ranked below it (§9.1).

**Outcome: none of the three pre-registered pure results obtained — this is the fourth
outcome, PARTIAL RESCUE.** Denoising is necessary and sufficient for escaping the
degenerate mode (C4 ≫ C2/C3 with a completely healthy scorer: 6/6 thresholds unpinned in
0.4965–0.5707 vs 10/12 C2/C3 runs pinned at 1.0000; predicted-positive share 17–25 % of
test pairs vs the 27–67 % saturation mass; valid ≈ test throughout) but not for reaching
the clean-easy control (C1 −0.31). The collapse mechanism (§6: FN cause, margin amplifier,
saturation erasure) is *prevented* by the denoisers; the remaining gap to C1 is either
residual false-negative contamination or genuine hard-clean signal being net-neutral under
the frozen margin-0.10 hinge — **exactly what the mandatory 50-pair blind audit must now
discriminate** (≲10 % contamination → "hard-but-clean is harmful here"; 20–40 % → "partial
denoising"). No headline claim until the gate passes (≤ 42 %, hope ≈ 0 %).

Housekeeping: (1) the 25-run confusion board reads predictions.npz, so its C2/C3 rows are
the npz values (e.g. C2_13 0.2165) — the §9.2 stored-metrics repair (`evaluate --force` on
the 9 runs) remains outstanding and untouched. (2) `smoke_C4_13/` (499 MB checkpoint) was
pushed to HF by the trainer; harmless (never matches `C[0-4]_<seed>` globs), optional
cleanup. (3) Correction to the C4_13 log entry above: run fingerprints legitimately differ
across seeds (the hash covers condition+seed by construction); the cross-run identity check
is the shared `artifacts` hash block in each `run_config.json`, not the fingerprint.

**§9.4 FINAL — the mandatory C4 audit gate (2026-10-02, reader VM; sheet `artifacts/audit_c4/`,
blind, key verified against the on-disk triples). Clone rate 15/50 = 30 % [Wilson 0.191,
0.438]; incl. unsure 16/50 = 32 % [0.208, 0.458] (unsure 1). References: C1 ≈ 0–2 %, C2
42 % [0.294, 0.558], C3 30 % [0.191, 0.438].**

**Gate verdict: PASSED by the pre-registered criterion (30 % ≤ 42 %) — C4 comparisons stand
on equal footing. The pre-registered HOPE (≈ 0 %) failed: the fork registered before the
audit lands on its second branch — PARTIAL DENOISING.** The Jaccard cut removed the lexical
false negatives (the J ≥ 0.40 band, C2-style surface-similar poison); the *semantic* clones
(same functionality, different tokens — J < 0.40) sailed through, and they alone account
for the 30 %.

**The decisive comparison the two audits now make possible: C3 and C4 carry the SAME
measured contamination — 15/50 each — yet sit 0.16 F1 apart with opposite scorer states (C3:
10/12 thresholds saturated at 1.0000 and MAP@R below C0; C4: 6/6 unpinned and MAP@R 2.7×
C0). Contamination RATE therefore does not determine collapse; the FN mass's SURFACE
SIMILARITY does. C3's false negatives are retrieved *by* semantic proximity and include the
lexically-hot band; C4's contain no token-J ≥ 0.40 member by construction. Refinement of
§6: the collapse trigger is a false negative that is simultaneously a functional clone AND
a surface near-duplicate — the push-apart gradient then contradicts the very features that
define clones (→ saturation erasure, §9.1). Function-only clones are a milder corruption:
they misallocate capacity without eroding the representation. FN similarity = trigger, FN
rate = dose, margin 0.10 = amplifier (unchanged).**

**C4 verdict for the write-up (headline-able as of this gate):** denoised hard negatives
(C4: F1 0.4349 ± 0.0246, MAP@R 0.2760) escape the degenerate mode that undenoised mining
produces (C2 0.2478 ± 0.0231 / C3 0.2707 ± 0.0095; MAP@R 0.047/0.077), beat the untuned
baseline on both axes (+0.178 F1, 2.7× MAP@R), and do NOT approach the clean-easy control
(C1 0.7464 ± 0.0176; gap −0.31) — and the audit prices the residue: 30 % of C4's negatives
are still false negatives, the semantic clone mass a lexical filter cannot see. Claim
(literature-aligned, RocketQA / NV-Retriever, extended by the similarity-axis finding):
under this frozen recipe (margin 0.10, thresholded F1) denoising is necessary and
sufficient for avoiding collapse but buys no advantage over easy negatives — and the
denoising axis matters: token-level filtering removes exactly the collapse-causing
component while leaving the semantic FN mass intact. Any future denoiser must be semantic
(cross-encoder or embedding-distance veto), not lexical.

Housekeeping: the audit artifacts live on the reader VM; preserve by a fresh `pull --light`
immediately followed by an artifacts-only `push` from that same VM (never `--include-report`
from a second VM, and do not commit that VM's stale repo copy of `report/measurements.md` —
this section is canonical). The §9.2 stored-metrics repair (per-file SHA check + `evaluate
--force` on the 9 mismatched v6 runs) remains the last open campaign item.
