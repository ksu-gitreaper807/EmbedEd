# Recommended direction

**Start with C1 — Disagreement Negatives.**

This is a recommendation about *finishing*, not about scientific superiority. I am not claiming
C1 is the best idea in [`CANDIDATE_EXPERIMENTS.md`](CANDIDATE_EXPERIMENTS.md); I am claiming it is
the best trade-off between what you can build, what you can keep clean, and what you can defend.

---

## Candidate A — Disagreement Negatives *(recommended)*

### Name
Disagreement Negatives: partitioning the negative pool by lexical/dense agreement.

### Core idea
Every candidate negative gets **two** scores: BM25 (lexical) and cosine (semantic). Existing
methods pick negatives using one of them. Instead, ask whether the two *agree*, and use that to
split the candidate pool into four disjoint sets. Train one model on each. The "embedding-high /
lexical-low" quadrant is the one the literature assumes is full of false negatives and deletes — so
train on it and find out.

### Closest literature
* [CuSINeS](https://arxiv.org/abs/2404.00590) — combines lexical and semantic rankings, but
  **fuses** them with reciprocal rank fusion and then applies a curriculum.
* [ECI](https://arxiv.org/html/2603.20990v1) — evaluates BM25 vs cross-encoder hard-negative sets,
  but for *evaluation*, not selection.
* [SimANS](https://aclanthology.org/2022.emnlp-industry.56/) and
  [NV-Retriever](https://arxiv.org/abs/2407.15831) — select on a single similarity signal.
* Query-by-committee / BALD (active learning, [survey](https://arxiv.org/abs/2306.08954)) —
  disagreement as a selection principle, but selecting points **to label**, not **to train against**.

### Novelty status
**Underexplored.** Targeted search surfaced no direct prior work on partitioning the negative
candidate pool by two-signal agreement and training a model per partition. Medium rediscovery risk;
see [`NOVELTY_AUDIT.md`](NOVELTY_AUDIT.md) Part 5 for the exact searches to run before writing the
word "novel" anywhere.

### Implementation
~30 lines on top of the pipeline that already exists in
[`project/distilled`](../../project/distilled):

```python
# negatives.py — the only function that changes
def candidate_pool(anchor_id, strategy, bm25_rank, cos_rank, dup_groups, k=64):
    """Return up to k candidate neighbour ids for this anchor.
    bm25_rank, cos_rank: dict anchor_id -> ordered list of candidate ids (both already computed)
    """
    if strategy == "random":        return shuffled(all_others)
    if strategy == "lex":           return bm25_rank[anchor_id][:k]
    if strategy == "model":         return cos_rank[anchor_id][:k]

    if strategy.startswith("quad"):                       # the new part
        bm  = set(bm25_rank[anchor_id][:k])               # lexically close
        co  = set(cos_rank[anchor_id][:k])                # semantically close
        if strategy == "quad_agree_hard":  return list(bm & co)      # both signals: hard
        if strategy == "quad_lex_only":    return list(bm - co)      # words match, meaning doesn't
        if strategy == "quad_emb_only":    return list(co - bm)      # meaning matches, words don't
        if strategy == "quad_agree_easy":  return list(all_others - bm - co)
```

Everything downstream — the sampler, `train.py`, `evaluate.py` — is unchanged.

### Experiment
4 quadrants + 3 baselines (BM25, zero-shot, random in-batch) × 3 seeds, Recall@10 / MRR@10 /
nDCG@10 on the fixed test split, with a paired bootstrap on three pre-declared contrasts. Plus a
manual false-negative rate per quadrant (30 sampled negatives each, marked duplicate / related /
unrelated).

### Risk
* **Rediscovery (medium).** Mitigation: run the Part 5 searches before writing.
* **Empty or tiny quadrants.** If `bm ∩ co` is nearly empty at k=64, widen k. Check the quadrant
  sizes in week 1 — this is a data-dependent failure, and it is the first thing to verify.
* **The two signals may be too correlated.** If |bm ∩ co| ≈ |bm| ≈ |co|, the quadrants are not
  distinct populations and the experiment is void. **Check this before training anything**: report
  the 2×2 contingency table of quadrant sizes in week 1. This is the equivalent of the "hardness
  manipulation check" in the existing plan, and it is a hard gate.

---

## Candidate B — Hub-aware negatives

**Name:** hub-aware negative down-weighting.
**Core idea:** exclude documents that are nearest neighbours of disproportionately many anchors.
**Closest literature:** popularity-bias correction in
[recommender negative sampling](https://doi.org/10.1145/3793855); hubness in high-dimensional NN
search. No text-embedding application found.
**Novelty status:** **Potentially novel** — highest novelty uncertainty of the five.
**Implementation:** ~15 lines (a `bincount` over the mined neighbourhood matrix).
**Experiment:** 3 conditions (hub-filtered / hub-only / unfiltered) × 3 seeds; report the hubness
histogram first.
**Risk:** the idea is simple enough that it may exist under another name. Search "hubness" +
"hard negative mining" before committing.

## Candidate C — Reciprocal-NN filter

**Name:** reciprocal-nearest-neighbour false-negative filter.
**Core idea:** drop candidate `c` for anchor `a` when `a` is also in `c`'s top-k.
**Closest literature:** [Zerveas et al., EMNLP 2023](https://aclanthology.org/2023.emnlp-main.665/)
— same mathematics, but used for label smoothing and reranking, **not** as a negative filter.
**Novelty status:** **Variation.**
**Implementation:** ~10 lines. **Cheapest candidate on the list** (6 runs).
**Experiment:** 2 conditions × 3 seeds + manual audit of the removed set.
**Risk:** medium–high rediscovery. Low interpretation risk.

## Candidate D — Band position × width

**Name:** band position × band width.
**Core idea:** sweep which similarity slice negatives come from, and how wide it is.
**Closest literature:** [SimANS](https://aclanthology.org/2022.emnlp-industry.56/),
[Ring NS](https://arxiv.org/abs/2010.02037),
[NV-Retriever](https://arxiv.org/abs/2407.15831).
**Novelty status:** **Variation.** High rediscovery risk.
**Implementation:** ~15 lines.
**Experiment:** 5 band positions + 2 width variants × 3 seeds.
**Risk:** this is a replication-with-measurement. Frame it that way and it is fine; frame it as a
new method and it is not.

## Candidate E — Slot-swap negatives

**Name:** slot-swap structural negatives.
**Core idea:** swap version numbers / package names in a real duplicate to make a near-identical
non-duplicate.
**Closest literature:** [SNCSE](https://arxiv.org/abs/2201.05979) (rule-based negation),
[EASE](https://doi.org/10.1145/3593590) (entity-aware), [MoCHi](https://arxiv.org/abs/2010.01028)
(feature mixing).
**Novelty status:** **Potentially novel**, but the weakest fit.
**Implementation:** ~60 lines plus regexes plus a manual correctness audit.
**Experiment:** 2 conditions × 3 seeds + 30-pair audit.
**Risk:** **high interpretation risk** — the model may learn to ignore exactly the surface features
that matter in Ubuntu support. Only attempt after C1 is finished.

---

## Why C1 is the right place to start

On the five axes that decide whether a one-month undergraduate project finishes:

| Axis | C1 | Why it wins |
|---|---|---|
| **Implementation simplicity** | ★★★★★ | Both rankings already exist in the current pipeline. The new code is a set intersection. |
| **Experimental cleanliness** | ★★★★★ | One variable (which quadrant), one loss, one model, one test set. No confound between *what* and *when*. |
| **Literature foundation** | ★★★★☆ | Sits directly on five well-cited papers; the related-work section writes itself. |
| **Novelty uncertainty** | ★★★☆☆ | Medium — genuine uncertainty, which is the honest maximum you should hope for at this level. |
| **1-month feasibility** | ★★★★★ | ~21 runs × ~5 minutes ≈ 2 hours of GPU. The rest is analysis and writing. |

Two further reasons that are not in the table:

1. **It has a built-in explanation.** Every quadrant carries a different *expected* false-negative
   rate, and you measure that rate by hand. So whichever way the table comes out, you can say why —
   and if it comes out flat, the measured false-negative rates tell you whether that is because
   hardness is irrelevant or because contamination cancelled the gain. A flat table is still a
   report.
2. **It degrades gracefully.** If the quadrants turn out not to be distinct populations, you find
   out in week 1 from a contingency table, and you fall back to C3 or C4, both of which are under
   20 lines. The failure mode is *early and cheap*, which is the single most valuable property a
   student project can have.

**What it is not:** it is not a new algorithm. The selection rule is a 2×2 mask. If you need the
project to sound like a new method, this is the wrong idea — and wanting that is itself a warning
sign.

---

## Novelty audit for C1

| Claim | Evidence | Closest prior work | Difference |
|---|---|---|---|
| Existing negative-selection methods use one similarity signal | SimANS, NV-Retriever, GISTEmbed, DPR, ANCE all reviewed | — | — |
| Two signals are sometimes combined | [CuSINeS](https://arxiv.org/abs/2404.00590) fuses rankings by RRF | CuSINeS | We partition by agreement instead of fusing |
| Disagreement is a known selection principle | QBC / BALD in active learning | [survey](https://arxiv.org/abs/2306.08954) | They select points to label; we select points to train against |
| The "embedding-high / lexical-low" quadrant is assumed to hold the false negatives | [Lei et al.](https://aclanthology.org/N16-1153/) 5% figure; [NV-Retriever](https://arxiv.org/abs/2407.15831) 38.8%/47% FN rates | RocketQA / NV-Retriever | They delete that region; we train on it as an explicit condition |
| No published false-negative rate for AskUbuntu with a modern encoder | Not found | — | Our measurement would be new (but small) |

### The five questions, answered

1. **What exactly is already known?** Hard > random; an intermediate band beats the extreme; the
   hardest negatives are disproportionately false negatives; positive-relative thresholds and
   cross-encoder denoising mitigate this; measured false-negative rates in retrieval are 38–70%.
2. **What exactly would our experiment add?** (a) A measured false-negative rate for AskUbuntu per
   quadrant — new. (b) A four-way controlled comparison of two-signal partitions — uncertain
   novelty. (c) A direct test of whether the suspected false-negative quadrant really is one.
3. **New algorithm, new finding, or new application?** **New empirical finding in a new
   application, plus one small measurement.** Not a new algorithm.
4. **How strong is the novelty claim?** **Weak to moderate.** Write: *"We are not aware of prior
   work that partitions the negative candidate pool by lexical/dense agreement and evaluates each
   partition separately; the closest is CuSINeS, which fuses the two rankings."*
5. **What search before claiming anything?** The nine searches listed in
   [`NOVELTY_AUDIT.md`](NOVELTY_AUDIT.md) Part 5.

---

## How this changes [`project/distilled`](../../project/distilled)

If you adopt C1, the existing Version B plan changes in three places and nowhere else:

| Current Version B | With C1 |
|---|---|
| Three conditions: N1 random / N2 lexical / N3 model-hard | **Four conditions:** the four quadrants. N1/N2/N3 stay as reference rows. |
| One ablation (`hard_fraction` ∈ {0.25, 0.5, 1.0}) | Keep it, or drop it to fund the extra seeds. |
| Manual false-negative rate: 50 queries × top-5 × 2 systems | Extend to **per quadrant** — 30 sampled mined negatives per quadrant. |

`negatives.py` gains one function. `train.py`, `evaluate.py` and `utils.py` are untouched. The four-week
plan holds: week 1 data + baselines + **the quadrant contingency check**, week 2 the sampler and a
first run, week 3 the twelve runs and the quadrant labelling, week 4 statistics and write-up.

**The week-1 gate, restated:** before training a single model, print the 2×2 contingency table of
quadrant sizes and the mean BM25/cosine rank per quadrant. If the quadrants are not distinct
populations, stop and switch to C3 or C4.
