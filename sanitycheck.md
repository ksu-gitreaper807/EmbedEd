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
failure, upgrade C2/C3); (ii) NaN/loss spikes in C2/C3 train logs (→ fp16 instability);
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
