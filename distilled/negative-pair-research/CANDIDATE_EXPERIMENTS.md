# Five candidate experiments

Shortlisted from the twelve in [`NEGATIVE_STRATEGIES.md`](NEGATIVE_STRATEGIES.md) on the basis of
*clear mechanism + low implementation complexity + measurable effect + literature to build on +
novelty uncertainty + one-month fit*.

**Not** shortlisted, with reasons: C6 PosRelFilter (established — use as a control condition, not
a contribution); C7 Curriculum3 and C9 DistanceWeighted (easy but near-certain rediscovery, and
C7 confounds *when* with *what*); C10 ClusterBoundary (variation on cluster-aware sampling);
C8 ScoreVariance (too many extra runs for the month); C11 TemporalNeg (no timestamps in the data).

---

## Shared experimental scaffolding

Everything below is held **identical** across all five candidates. Only the negative-construction
function changes. State this table in the report; it is what makes the comparison interpretable.

| Element | Fixed value |
|---|---|
| **Dataset** | [`sentence-transformers/askubuntu`](https://huggingface.co/datasets/sentence-transformers/askubuntu) |
| **Corpus** | Distinct texts from all `query` and `positive` fields, ~13,000–15,000 documents |
| **Qrels** | The `positive` lists, mapped to corpus ids (~1–2 gold duplicates per query) |
| **Splits** | train 12,745 rows · dev 200 queries (epoch selection only) · test 200 queries (headline) |
| **Positive pairs** | `(anchor, positive)` from **train rows only**, ~17,000–20,000 pairs, no synthetic augmentation, no reverse direction |
| **Pretrained encoder** | `sentence-transformers/all-MiniLM-L6-v2` (23M, 384-d) — one model |
| **Loss** | `MultipleNegativesRankingLoss` (in-batch InfoNCE), scale 20 (τ = 0.05) — one objective |
| **Negative delivery** | `NeighbourhoodBatchSampler`: one seed pair + 31 others drawn from a candidate pool. **The candidate pool is the experimental variable.** |
| **Batch size** | 32 (therefore 31 in-batch negatives) — fixed |
| **Epochs** | Selected on dev from {1, 2, 3} using the **random** condition; reused everywhere |
| **Optimiser** | AdamW, lr 2e-5, warmup 10%, fp16 on T4 |
| **Seeds** | 13, 42, 1337 |
| **Baselines** | B1 BM25 · B2 zero-shot MiniLM · **B3-Random** = in-batch random negatives |
| **Metrics** | Recall@10 (primary) · MRR@10 and nDCG@10 (secondary) |
| **Uncertainty** | mean ± range over 3 seeds + paired bootstrap over the 400 eval queries on the pre-declared contrasts |
| **Mechanism measurement** | Manual false-negative rate: 50 test queries × top-5 non-gold results, labelled duplicate / related / unrelated |

**Why the sampler is the right lever.** With in-batch negatives, "which negatives?" *is* "which
pairs share a mini-batch?". All five candidates below are different answers to that one question,
so the loss function is byte-identical across conditions. That is a cleaner comparison than
swapping between `MultipleNegativesRankingLoss` and `TripletLoss`, which would confound the
objective with the sampling.

---

## C1 — Disagreement Negatives ⭐

### The experiment chain

| | |
|---|---|
| **Dataset** | AskUbuntu (as above) |
| **Positive pairs** | ~19,000 `(anchor, positive)` from train rows |
| **Negative generation** | For every training anchor, compute two rankings over all other training anchors: (a) BM25, (b) zero-shot MiniLM cosine. Exclude the anchor and its duplicate group. Split each ranking at the median of the pool. Assign every candidate to one of four quadrants: **AGREE-HARD** (both high), **LEX-ONLY** (BM25 high, cosine low), **EMB-ONLY** (cosine high, BM25 low), **AGREE-EASY** (both low). Build batches from one quadrant at a time. |
| **Baseline negative generation** | **B3-Random**: batches drawn uniformly at random. Plus **C0b (LexBM25)** and **C0c (ModelHardStatic)** as reference rows, since both are already implemented. |
| **Pretrained encoder** | `all-MiniLM-L6-v2` |
| **Contrastive training** | MultipleNegativesRankingLoss, batch 32, epochs from dev, lr 2e-5, 3 seeds |
| **Fixed test set** | The 200 test queries, ~13–15K corpus, never touched during selection |
| **Evaluation** | Recall@10 / MRR@10 / nDCG@10; bootstrap on the contrasts AGREE-HARD−Random, LEX-ONLY−Random, EMB-ONLY−Random |

**Runs:** 4 quadrants × 3 seeds = 12, plus 3 baselines × 3 seeds = 9. ~21 runs × ~5 minutes ≈
**2 hours of GPU**. If that is too many, drop to 2 seeds for the quadrants and keep 3 for the
winner.

**Mechanism measurement:** for each quadrant, sample 30 mined negatives and label them by hand as
genuine duplicate / related / unrelated. Report the false-negative rate per quadrant. **This is
the table that makes the result meaningful.**

### Research question

> Does selecting in-batch negatives by the **agreement or disagreement between BM25 and embedding
> similarity** improve duplicate-question retrieval (Recall@10) on AskUbuntu, compared with
> selecting them at random, when fine-tuning `all-MiniLM-L6-v2`?

### Hypothesis

**H1:** Quadrant selection matters, and the ranking is
`AGREE-HARD > LEX-ONLY > Random > EMB-ONLY`. The reasoning: AGREE-HARD gives the sharpest
gradient with mutual confirmation from two independent signals; LEX-ONLY is the standard DPR
recipe; EMB-ONLY should be **worst of all** because it is the quadrant where unlabelled
duplicates concentrate.

**H0:** All four are within noise of each other. The two signals are highly correlated in this
corpus, so the quadrants are not meaningfully different populations.

### What a negative result would mean

*If all four are equal:* BM25 and dense cosine rank this corpus too similarly for disagreement to
carry extra information — itself a useful thing to have measured, and it would explain why every
published method gets away with a single signal.

*If EMB-ONLY is not the worst:* the standard assumption ("the dense model's nearest unlabelled
neighbours are false negatives") is wrong here, or the model is robust to it. Either is a finding.

*If the per-quadrant false-negative rates are all similar:* the disagreement partition did not
isolate the false negatives, and the mechanism story in the whole literature needs a caveat for
this domain.

---

## C2 — Band position × band width

### The experiment chain

| | |
|---|---|
| **Dataset / positives / encoder / training / test / evaluation** | As in the shared scaffolding |
| **Negative generation** | Rank candidates by zero-shot cosine. Define five equal-width bands by rank percentile (0–20, 20–40, 40–60, 60–80, 80–100). Also define a **wide** band (everything below the positive's score) and a **narrow** band (a single decile centred on the positive's rank). Build batches from one band at a time. |
| **Baseline negative generation** | B3-Random, plus C0c (plain top-64 = the hardest band) |

**Runs:** 5 bands × 3 seeds + 2 width variants × 3 seeds = 21. Same budget as C1.

### Research question

> Does the **width** of the negative similarity band, as well as its position, affect Recall@10
> when fine-tuning `all-MiniLM-L6-v2` on AskUbuntu?

### Hypothesis

**H1:** performance is single-peaked in band position (an inverted-U, consistent with
[SimANS](https://aclanthology.org/2022.emnlp-industry.56/) and
[NV-Retriever](https://arxiv.org/abs/2407.15831)), and **narrow bands underperform wide bands at
the same centre**, because a narrow band reduces negative diversity.

**H0:** only the centre matters; width is irrelevant.

### What a negative result would mean

If the curve is monotone rather than peaked, the false-negative mechanism is not biting in this
corpus — which would be surprising given [Lei et al.](https://aclanthology.org/N16-1153/)'s 5%
figure, and would be worth reporting as a discrepancy with the retrieval literature.

**Novelty caution:** this is the closest to published work of all five candidates. Its value is
mostly as a *replication with measurement*, and it should be framed that way.

---

## C3 — Hub-aware negatives

### The experiment chain

| | |
|---|---|
| **Dataset / positives / encoder / training / test / evaluation** | As in the shared scaffolding |
| **Negative generation** | Mine top-64 for every training anchor (either signal; use cosine). Count how many anchors list each document as a neighbour. Plot the histogram — it will be heavy-tailed. Define **hubs** as documents in the top 1% by occurrence count. Condition A: sample batch neighbours from **non-hubs only**. Condition B: sample from **hubs only**. Condition C: standard sampling (no filter). |
| **Baseline negative generation** | Condition C is the baseline |

**Runs:** 3 conditions × 3 seeds = 9. **Cheapest candidate on this list.**

### Research question

> Does excluding "hub" documents — those that appear as nearest neighbours of disproportionately
> many anchors — from the negative pool improve Recall@10, compared with unfiltered hard-negative
> sampling, when fine-tuning `all-MiniLM-L6-v2` on AskUbuntu?

### Hypothesis

**H1:** hub-filtered sampling beats unfiltered. Hubs are documents like "how do I install
ubuntu?" that are lexically and semantically near everything; training against them pushes the
model away from a region that contains many true duplicates, i.e. they are systematic
false-negative generators.

**H0 (arguably more likely):** hubs are genuinely informative hard negatives — they are popular
*because* they are central to the domain — and removing them loses signal.

### What a negative result would mean

Either way you learn something concrete about the geometry: if hubs matter, you have found a
failure mode of hard-negative mining that the text-IR literature does not discuss; if they do not,
the popularity-bias lesson from recommender systems does not transfer to text embeddings, which
is a clean cross-field negative result.

**Novelty note:** this is the candidate with the highest novelty uncertainty, because the idea is
simple. Do the search in [`NOVELTY_AUDIT.md`](NOVELTY_AUDIT.md) Part 5 with "hubness" added as a
keyword before committing to it.

---

## C4 — Reciprocal-NN false-negative filter

### The experiment chain

| | |
|---|---|
| **Dataset / positives / encoder / training / test / evaluation** | As in the shared scaffolding |
| **Negative generation** | Mine top-64 by cosine for every training anchor. Keep candidate `c` for anchor `a` **only if** `a ∉ top-64(c)`. (Reciprocal neighbours are mutually close; that mutual relationship is a much stronger similarity signal than one-directional distance — [Zerveas et al.](https://aclanthology.org/2023.emnlp-main.665/).) Build batches from the survivors. Condition B: no filter. |
| **Baseline negative generation** | Condition B |

**Runs:** 2 conditions × 3 seeds = 6, plus the manual audit of the removed set.

**Mechanism measurement:** take 30 removed (reciprocal) pairs and label them by hand. If most are
genuine duplicates, the filter is doing what it claims. If most are not, the filter is removing
useful hard negatives and you have a clean explanation for the result.

### Research question

> Does removing reciprocal nearest neighbours from the negative pool improve Recall@10, compared
> with unfiltered hard-negative sampling, when fine-tuning `all-MiniLM-L6-v2` on AskUbuntu?

### Hypothesis

**H1:** the rNN-filtered model beats the unfiltered one, because the removed pairs are
disproportionately unlabelled duplicates.

**H0:** filtering helps retrieval but not training — removing the hardest negatives costs more
gradient signal than the false negatives were costing.

### What a negative result would mean

[Zerveas et al.](https://aclanthology.org/2023.emnlp-main.665/) show rNN helps *label smoothing*
and *reranking*. If it does not help as a negative filter, the honest conclusion is that the
benefit of rNN is in *softening targets*, not in *cleaning negatives* — a useful distinction, and
a clean negative result.

---

## C5 — Slot-swap structural negatives

### The experiment chain

| | |
|---|---|
| **Dataset / positives / encoder / training / test / evaluation** | As in the shared scaffolding |
| **Negative generation** | Regex-extract technical slots from the corpus: version numbers (`12.04`, `14.04`), package names (`vlc`, `skype`), desktop environments (`unity`, `kde`). For each positive pair `(a, p)`, produce a negative by substituting a different value into one slot of `p`. Condition A: standard hard negatives. Condition B: standard + slot-swapped negatives. |
| **Baseline negative generation** | Condition A |

**Mandatory manual audit:** 30 swapped pairs must be checked by hand to confirm they are genuinely
**not** duplicates. This is the only candidate that requires a correctness audit of the data
generator itself, and skipping it invalidates the experiment.

### Research question

> Does adding slot-swapped near-duplicates (identical except for a version number or package name)
> to the negative pool improve Recall@10, compared with standard hard negatives alone?

### Hypothesis

**H1:** yes — these are maximum-hardness negatives with **guaranteed correct labels**, so they
avoid the false-negative problem entirely, unlike every mined negative.

**H0 / risk:** no, and worse — the model learns that version numbers and package names are
irrelevant, which is exactly wrong for Ubuntu support, where "install X on 12.04" and "install X on
14.04" have different answers.

### What a negative result would mean

A negative result here is *predicted by the domain* and is the more interesting outcome: it would
show that synthetic negatives constructed by surface-level perturbation damage a domain where
surface details carry the answer. That is a cautionary result for the synthetic-negative
literature.

**Why this is last on the list:** it requires a data generator whose correctness must be audited by
hand, and its most likely outcome is a confounded one. Only attempt it if C1 is finished and you
have a week to spare.

---

## Comparison for planning purposes

| | C1 Disagreement | C2 BandWidth | C3 HubAware | C4 ReciprocalNN | C5 SlotSwap |
|---|---|---|---|---|---|
| New code beyond the existing pipeline | ~30 lines | ~15 lines | ~15 lines | ~10 lines | ~60 lines + regexes |
| Training runs (3 seeds) | 12 | 21 | 9 | 6 | 6 |
| Extra manual labelling | 120 items | 0 | 0 | 30 items | 30 items + 30 audits |
| Variables held constant | 1 | 2 (position, width) | 1 | 1 | 1 (plus a generator) |
| Literature support | Strong | Very strong | Moderate (cross-field) | Strong | Weak |
| Novelty uncertainty | Medium | Low | **High** | Medium | High |
| Rediscovery risk | Medium | High | Medium | Medium–high | Low |
| Interpretation risk | Low | Low | Medium | Low | **High** |
| **1-month feasible** | **Yes** | **Yes** | **Yes** | **Yes** | Marginal |

**Interpretation risk** = the chance that a result, once obtained, cannot be cleanly attributed to
the mechanism you claim.
