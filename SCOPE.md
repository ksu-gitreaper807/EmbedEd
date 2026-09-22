# Scope — Java code clone detection (BigCloneBench / GraphCodeBERT)

Read this before writing code, and re-read it in week 3 when you are tempted to add a fifth
condition or switch benchmarks.

Companion documents: [`FINAL_SPEC.md`](FINAL_SPEC.md) (the specification) and
[`GROUND_TRUTH.md`](GROUND_TRUTH.md) (the validity threats).

---

## Part 1 — What is fixed

These six are not negotiable. If one of them breaks, stop and fix it rather than working around
it.

| # | Fixed thing | Why it is fixed |
|---|---|---|
| 1 | **One independent variable:** how the negative was selected | The entire project. Everything else is held constant so this is the only explanation left |
| 2 | **Four conditions:** C0 baseline, C1 random, C2 BM25, C3 semantic | Three points are the minimum that can distinguish "hardness helps" from "hardness helps then hurts" |
| 3 | **Same positive pairs, sampled once, shared across conditions** | Re-sampling per condition would confound the variable with data variation |
| 4 | **One loss, identical across conditions** | Comparing losses is a different project |
| 5 | **The true-clone exclusion check, unit-tested** | A leaked clone in a negative set is not a hard negative, it is a corrupted label |
| 6 | **The hardness check runs before training** | An experiment whose manipulation did not take effect produces no evidence at all |

---

## Part 2 — In scope (MUST)

Everything here is required for the project to be complete.

### Data

1. **Dataset:** `load_dataset("google/code_x_glue_cc_clone_detection_big_clone_bench")`, one line.
2. **Verify** the fragment count against the reported 9,134 and the split sizes against
   901,028 / 415,416 / 415,416.
3. **Corpus for mining:** unique fragments appearing in the **train split only**.
4. **Fragment-overlap measurement:** count how many test-split fragments also appear in a training
   pair; **report the number** in the write-up.
5. **Token-length distribution** of the corpus, measured, used to choose max sequence length.

### Negatives

6. **Three strategies** (random / BM25 / semantic), fixed `k`, same `k` for all conditions.
7. **True-clone exclusion**, implemented as a standalone function with an adversarial unit test.
8. **Hardness check:** mean `cos(anchor, negative)` per condition; require `C1 < C2 ≤ C3` with a
   visible gap. **Run before training.**
9. **False-negative measurement:** 50 mined negatives from C2 and 50 from C3, manually judged for
   functional equivalence, reported per strategy.

### Training

10. **Model:** `microsoft/graphcodebert-base`, loaded via `Transformer` + mean `Pooling`
    (Option A), with the data-flow caveat stated. `microsoft/codebert-base` is the one-line
    fallback.
11. **One loss**, identical across conditions. Either explicit triples or MNRL-with-composed
    batches — chosen once, in week 1, not per condition.
12. **Runs:** C0 evaluated once; C1, C2, C3 each with **3 fixed seeds**.
13. **Subset size and epochs fixed in week 1** from a measured throughput test, then held constant.

### Evaluation

14. **Primary metric: F1**, with precision and recall reported alongside. Threshold chosen on the
    **validation** split per condition, and **the chosen threshold reported**.
15. **Secondary metric: MAP@R**, threshold-free, as the robustness check on the F1 ordering.
16. **Sanity check** against CodeXGLUE's published fine-tuned CodeBERT F1 (≈0.95 `[check]`). If
    your numbers are not in that neighbourhood, fix the harness before writing anything.
17. **Generalisation test:** hold out whole functionalities using Kitsios et al.'s released
    BCB s′; report `F1_seen`, `F1_unseen`, and `Δ = F1_seen − F1_unseen` per condition.
18. **Visualisation:** UMAP, **fixed seed and hyperparameters**, **the same sample of fragments**
    in all four panels.

### Write-up and demo

19. **Three tables:** standard benchmark (four conditions × two metrics × 3 seeds);
    generalisation (`F1_seen` / `F1_unseen` / `Δ`); measured false-negative rates.
20. **Limitations section** (`FINAL_SPEC.md` §12) written **before** the claims are finalised.
21. **Demo:** real held-out test pairs with gold labels as the default; free-text paste clearly
    labelled illustrative; all four outcome modes supported.
22. **`run_all.sh`** reproducing every number in the report.

---

## Part 3 — SHOULD HAVE

Only after all of Part 2 is done, in priority order.

1. **Hard-negative mixing ratio sweep** on C3 — for example `{0.25, 0.5, 1.0}` hard negatives
   mixed with random ones. Traces the hardness curve continuously and is the cheapest way to
   distinguish a genuine inverted-U from a single unlucky operating point.
2. **Leave-one-functionality-out rotation** for the generalisation test, reporting mean and range
   of `Δ` across folds. A single 3-functionality holdout on 23 functionalities will be noisy.
3. **A second base model** (`microsoft/codebert-base`) on the **winning condition only**, 3 runs.
   Tests whether the strategy ranking is an artefact of one encoder.
4. **External comparison against `mchochlov/codebert-base-cd-ft`** — a published
   contrastively-fine-tuned CodeBERT for clone detection. Free sanity check.
5. **Breakdown by clone type** (T1/T2 vs WT3/T4) if the type labels can be recovered — the 93%
   mislabelling figure applies specifically to WT3/T4, so this split is where the noise lives.
6. **ANCE-lite refresh:** re-mine C3's neighbourhoods once after epoch 1 using the
   partially-trained model. One extra matmul. Risks the plumbing, so it stays a SHOULD.
7. **Cross-check F1** with an independent implementation (scikit-learn `f1_score` on binarised
   predictions vs your own accumulation).

---

## Part 4 — Optional extensions

Nothing here may start before the main table exists.

* **Denoising** C3's neighbourhoods — drop candidates whose similarity to the anchor exceeds
  `cos(anchor, positive) − δ`, RocketQA-style. The natural next experiment if you observe the
  inverted-U **and** a high measured false-negative rate. It needs a threshold choice and a
  second mechanism measurement, so it is a week by itself.
* **Full GraphCodeBERT with data flow** (Option B in `FINAL_SPEC.md` §5). Days of work with a real
  chance of not working; do not start it as a rescue attempt.
* **A second language or benchmark** to test whether the strategy ranking is Java-specific.
* **More seeds** (5), or bootstrapping over seeds as well as test pairs.
* **Symmetric training pairs** (add `(p, a)`).
* **A cross-encoder reranker** on the borderline pairs.

---

## Part 5 — Out of scope

Not "later". Not at all, in this project.

**Data:** POJ-104, OJClone, CodeNet, GPTCloneBench, any self-built or LLM-generated clone corpus;
more than ~10,000 fragments; attaching functionality labels to CodeXGLUE by hand (use BCB s′
instead); any dataset without fixed splits.

**Models:** training from scratch; any encoder above ~500M parameters; generative LLMs as clone
detectors (Kitsios et al. already did that comparison); AST/GNN architectures; custom attention or
pooling; multi-GPU or distributed training.

**Training:** hyperparameter search over learning rate, warmup, schedulers, or weight decay;
more than one loss; more than one negative strategy *per run* outside the declared mixing-ratio
sweep; adversarial or curriculum training.

**Evaluation:** FAISS or any approximate-index library (9,134 fragments is one matmul);
significance tests, p-values, multiple-comparison correction; executing code to measure
behavioural similarity; inter-dataset duplication analysis.

**Infrastructure:** Docker, experiment trackers, config frameworks, CI, serving.

**Claims, and this list matters most:**

* ~~"our model detects semantic clones with F1 = X"~~ → *"our model agrees with BigCloneBench's
  labels at F1 = X"*.
* ~~"harder negatives are better for code clone detection"~~ → *"on this corpus, with these
  labels, ..."*
* ~~"the generalisation test is novel"~~ → Kitsios et al. (ASE 2025) established that
  unseen-functionality F1 drops by ~31% on average. **Cite it.** Your contribution is the
  *interaction* with negative strategy.
* ~~"GraphCodeBERT's data-flow signal helps"~~ → not under Option A.
* Any claim that generalises beyond Java and BigCloneBench.
* Any claim to have separated hard negatives from false negatives. You have not — you have
  *measured* the false-negative rate and reported it.

---

## Part 6 — Anti-scope-creep rules

In the order you will be tempted to break them.

1. **If C2 and C3 come out identical, do not add a fourth strategy** to find a difference.
   "Beyond lexical similarity, negative hardness changes nothing here" is a finding.
2. **If C3 is worse than C2, do not lower the learning rate to rescue it.** That is the predicted
   result. Measure the false-negative rate and write it up.
3. **If the hardness check fails**, stop everything and fix the mining. No amount of extra runs
   will salvage an experiment whose manipulation did not take effect.
4. **If every condition beats every other by 20 points**, do not celebrate — re-run the
   fragment-overlap measurement (Part 2, item 4). You have most likely measured memorisation.
5. **Do not switch benchmarks in week 3** because you read `GROUND_TRUTH.md` and got cold feet.
   Every benchmark has defects; these ones are documented, quantified, and partially measurable.
   Decide now or never.
6. **If the generalisation numbers are wildly noisy, report the noise.** Do not add holdout folds
   until the variance goes away.
7. **If you finish early, do not start an optional extension.** Improve the report. A clear
   9-page write-up beats 14 pages with a half-finished denoising experiment.
8. **If you are behind, cut seeds before strategies, and strategies before the two checks.** Never
   cut the true-clone exclusion, the hardness check, the baseline condition, or the
   fragment-overlap measurement.

---

## Part 7 — Stop condition

> **The project is finished when C0, C1, C2 and C3 have been evaluated on the fixed CodeXGLUE
> test split with F1 (threshold chosen on validation and reported) and MAP@R over 3 seeds, and
> on the held-out functionality split with `F1_seen`, `F1_unseen` and `Δ` — and those numbers,
> together with the measured false-negative rates, are in one table.**

At that point, stop running experiments and start writing.

---

## Part 8 — Fallback

**Drop to two negative strategies** (C1 random + C3 semantic, skipping BM25) `[spec]`.

Endorsed, with the caveat that is now stated in two places: **with two conditions you cannot
detect an inverted-U.** If C1 and C3 come out equal you will not be able to distinguish "hardness
does not matter" from "hardness helped and then hurt, and the two cancelled out". If you take this
fallback, say so explicitly in the write-up.

**Second fallback, if `negatives.py` is not working by the end of week 2:** drop to C0 + C1 only
and ship the "does fine-tuning help, and does it generalise?" comparison. It is the weaker
project — it has no independent variable — but it is complete, and a complete weak project beats
an incomplete strong one. Note that in this case you are essentially reproducing a small part of
Kitsios et al., and should say so.
