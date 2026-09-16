# Project specification — Version B (the middle version)

If a component is not described here, it is not part of the project.

---

## Title

**How Hard Should the Negatives Be? In-Batch Negative Hardness in Contrastive Fine-Tuning for
Duplicate Question Retrieval**

---

## Research question

> When contrastively fine-tuning a small pretrained encoder for duplicate-question retrieval,
> does the **hardness of the in-batch negatives** determine whether fine-tuning beats (a) BM25
> and (b) the same encoder with no fine-tuning?

One independent variable with three levels. Everything else — model, loss, training pairs,
batch size, epochs, learning rate, corpus, queries, metrics — is held constant.

---

## Hypotheses (write these down before you train anything)

**H1 (monotone):** Recall@10 improves with negative hardness — `N1 ≤ N2 ≤ N3` — and all three
beat the zero-shot encoder. This is the naive transfer of
[DPR](https://aclanthology.org/2020.emnlp-main.550/)'s result to a new domain.

**H0a (flat):** all three conditions are within noise of each other. Negative hardness is not a
decisive variable here.

**H0b (inverted-U):** `N2 > N1` but `N3 < N2`. Hardness helps up to a point and then hurts.
**This is the prediction the literature actually supports**, for two independent reasons:

* [Robinson et al. (ICLR 2021)](https://openreview.net/forum?id=CR1XOQ0UTh-) show that negative
  hardness is a *controllable* quantity and that pushing it too far degrades the
  representation. Their method "requires only few additional lines of code to implement, and
  introduces no computational overhead" — which is why it is affordable here.
* [Lei et al. (NAACL 2016)](https://aclanthology.org/N16-1153/) measured, on this exact corpus,
  that *"only 5% of similar pairs have been annotated by the users, with a precision of around
  79%"*. So the most similar unlabelled questions in this corpus are disproportionately
  **genuine but unmarked duplicates**. The harder your negatives, the more of them are false
  negatives.

**Why all three outcomes are publishable.** H1 is a clean positive result. H0a says something
useful about a technique everyone assumes matters. **H0b is the best outcome**: it says
hard-negative mining, standard practice in web retrieval, has a *measurable failure point* in
duplicate-question retrieval, and you can put a number on why (your measured false-negative
rate in N3's mined set).

**What would falsify the whole framing:** if BM25 beats every neural system by a wide margin.
Then the story is "keyword overlap is the signal in this domain", which is
[BEIR's](https://arxiv.org/abs/2104.08663) and
[Jiang et al.'s](https://doi.org/10.1016/j.jss.2023.111607) finding. Write that paper instead.

---

## Dataset

**AskUbuntu duplicate questions** —
[`sentence-transformers/askubuntu`](https://huggingface.co/datasets/sentence-transformers/askubuntu)

| Property | Value |
|---|---|
| Origin | AskUbuntu (Stack Exchange) 2014 dump; task and splits from [Lei et al., NAACL 2016](https://aclanthology.org/N16-1153/) |
| Rows | **13,145** — train 12,745 / dev 200 / test 200 |
| Row format | `query` (str), `positive` (list of 1–3 duplicates), `negative` (list of ~100 non-duplicates) |
| Text | already lowercased and tokenised — **no HTML, no cleaning** |
| Domain | Ubuntu/Linux technical support |
| Loading | `load_dataset("sentence-transformers/askubuntu")` |
| Licence | not stated on the card; underlying Stack Exchange content is CC BY-SA — check before publishing |

Kept from Version A unchanged. The dataset was never the reason the small project felt small.

### Why not CQADupStack

457K documents, 12 subforums, raw HTML to strip, ~1 week of data engineering before the first
experiment, and no published reference number for a split you build yourself. AskUbuntu is the
same task at 3% of the size and loads in one line. The small corpus is what makes hard-negative
mining affordable: mining top-64 neighbours for 12,745 anchors is one 12,745 × 12,745 matmul,
i.e. **seconds, with no FAISS**. On a 457K corpus the same operation needs an index.

### What you build (in `prepare_data.py`)

```text
corpus       = distinct texts from every `query` and every `positive`     ~13,000–15,000 docs
qrels        = {qid: [gold corpus ids]}   from the `positive` lists
queries      = the 200 dev + 200 test `query` fields
train_pairs  = [(anchor, positive)]       from the 12,745 train rows only  ~17,000–20,000 pairs
```

The shipped `negative` lists are **not used**. Not for training, not as the corpus. Ignore them.

### Splits — fixed, never re-derived

| Split | Size | Used for |
|---|---|---|
| train | 12,745 rows | training pairs **and** neighbourhood construction |
| dev | 200 queries | **only** to choose the epoch count, using N1 |
| test | 200 queries | the headline number, looked at once |

---

## Model

**One model: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)**
— 23M params, 384-d output, 6 layers, Apache 2.0.

Runs take minutes, which is what buys 3 seeds × 3 conditions inside four weeks. Its zero-shot
form *is* baseline B2, so the before/after comparison is exactly controlled.

**Input:** the question text as-is. **Output:** a 384-d L2-normalised vector.

| Step | What happens | What it causes |
|---|---|---|
| text → tokens | word-piece tokenisation, `[CLS] … [SEP]`, pad to batch max | only in-vocabulary pieces can be read |
| tokens → encoder | 6 self-attention layers | each token's vector absorbs context from the others |
| vectors → embedding | mean pooling over real tokens, L2-normalise | the sentence becomes one point on a 384-d sphere; cosine = dot product |
| embeddings → similarity | `cos(a, b) = a · b` | the single number the whole evaluation rests on |
| similarity → loss | InfoNCE over the batch | the gradient that reshapes the space |
| loss → fine-tuned encoder | backprop through all 23M params | "duplicate" now means *duplicate in Ubuntu* |

---

## Loss

**One objective: `MultipleNegativesRankingLoss`** (in-batch InfoNCE), used identically in all
three conditions. You do not implement it. You read it and understand the formula.

For a batch of `N` pairs `(a₁,p₁) … (a_N,p_N)`, cosine similarity, scale `s`:

```text
            exp( s · cos(aᵢ, pᵢ) )
L = − log  ─────────────────────────────
            Σⱼ exp( s · cos(aᵢ, pⱼ) )
```

averaged over the batch. `s = 20` (τ = 0.05) is the library default.

| Term | What it is | What it causes |
|---|---|---|
| `cos(aᵢ, pᵢ)` | anchor vs. **its own** duplicate | numerator; training pushes this up |
| `cos(aᵢ, pⱼ)`, j ≠ i | anchor vs. **every other pair's** duplicate | the `N−1` in-batch negatives; training pushes these down. **This is where the independent variable lives.** |
| `s = 20` (τ = 0.05) | inverse temperature | sharpens the softmax — the model is punished for near-misses, not just gross errors |
| `log` + minus | cross-entropy over "which of `N` candidates is mine?" | turns retrieval into N-way classification |
| `1/N · Σᵢ` | average over the batch | the usual |

**The sentence you must be able to say:** *"Each question has to pick its own duplicate out of
N candidates; the other N−1 duplicates in the batch are the negatives, for free. So 'which
negatives?' is really 'which other pairs end up in my batch?' — and that is a batch-construction
problem, not a loss-function problem."*

This is the design insight that makes the whole project affordable: by varying **batch
composition** rather than the loss, you get three genuinely different negative-sampling regimes
with a byte-identical objective function.

---

## Positive examples

> **(a, p) is a positive pair iff `p` appears in `a`'s `positive` list** — i.e. the AskUbuntu
> community marked `p` as a duplicate of `a`.

* Built **only** from the 12,745 train rows.
* **Naturally occurring only.** No paraphrasing, no back-translation, no LLM-generated pairs.
* No reverse direction (adding `(p, a)` doubles the pairs but raises the chance that two rows in
  one batch share a duplicate group, which manufactures false negatives — a confound you cannot
  afford when false negatives are your explanatory variable).
* Expected: **~17,000–20,000 pairs.** Print the exact number.

---

## Negative examples — the independent variable

> A candidate `c` is a negative for anchor `a` iff `c` is **another pair's positive in the same
> mini-batch**.

Same definition in all three conditions. What differs is **which pairs share a batch**.

### Neighbourhood construction (`negatives.py`)

For every training anchor `a`, build a neighbour list `Nb(a)` of `k = 64` **other training
anchors**:

| Condition | `Nb(a)` = | Cost |
|---|---|---|
| **N1 random** | *(not used — batches are a plain shuffle)* | 0 |
| **N2 lexical** | top-64 by **BM25** among training anchors (`rank_bm25`, already a dependency) | seconds |
| **N3 model-hard** | top-64 by **cosine similarity of the zero-shot MiniLM's embeddings** (one 12,745 × 384 @ 384 × 12,745 matmul) | seconds |

**Hard rule in both N2 and N3:** remove from `Nb(a)` every anchor in `a`'s **duplicate group**
and `a` itself. A known duplicate must never be used as a negative — that is not hardness, that
is a labelling bug.

### Batch construction

```python
class NeighbourhoodBatchSampler:
    """Yields batches of pair indices.
    Each batch = 1 seed pair + (B−1) others, of which
    round(hard_fraction × (B−1)) are drawn from the seed's neighbourhood
    and the rest are uniformly random (excluding duplicates of the seed).
    Batches are regenerated each epoch with a different RNG seed."""
```

| Condition | `hard_fraction` | Meaning |
|---|---|---|
| **N1** | `0.0` | batch is a random shuffle → 31 random in-batch negatives |
| **N2** | `1.0` | batch is a BM25 neighbourhood → 31 lexically similar negatives |
| **N3** | `1.0` | batch is an embedding neighbourhood → 31 semantically similar negatives |

`__len__` returns `epochs × n_pairs / B` so the trainer's step count is right.

### Why hardness is manipulated this way

* It keeps the loss constant across conditions — a cleaner experiment than comparing
  `MultipleNegativesRankingLoss` against `TripletLoss`, which would confound objective and
  sampling.
* [Robinson et al.](https://openreview.net/forum?id=CR1XOQ0UTh-) frame hardness as a
  *continuous, user-controllable* quantity, which is exactly what `hard_fraction` is.
* It costs no extra forward passes. Mining is two matmuls.

**The honest alternative, and why we avoid it:** you could train explicit `(a, p, n)` triplets
with `TripletLoss`. That is more standard-looking, but it changes the loss between conditions,
so any difference you measure would be attributable to the loss as much as to the negatives. Do
not do it.

### The honest weakness (state this in the report)

"Not in my `positive` list" ≠ "not a duplicate". Lei et al. measured that only ~5% of similar
pairs are annotated, with ~79% precision. So in N3, the questions most similar to your anchor
are very likely to be **genuine duplicates that nobody marked**, and you will train the model to
push them away. That is not a flaw in the experiment — **it is the experiment.** You will
measure it (see "Mechanism measurement") and report it.

---

## Training configuration — fixed, not tuned

| Setting | Value | Source |
|---|---|---|
| Batch size | **32** (so 31 in-batch negatives) | fixed across conditions — do not vary it |
| Neighbourhood size `k` | 64 | must exceed `B−1 = 31` with margin |
| Epochs | chosen on **dev** from {1, 2, 3} **using N1**, then reused for N2/N3 | the only hyperparameter you select |
| Learning rate | **2e-5**, AdamW | standard BERT-family value |
| Warmup | 10% of steps | library default |
| Precision | fp16 on T4 | optional |
| Max seq length | 256 (text is short) | library default |
| Seeds | **13, 42, 1337** | 3 seeds × 3 conditions = 9 runs |
| Negative refresh | **none** (N3 neighbourhoods mined once with the zero-shot model) | keeps it simple; refresh is optional |

Do not tune the learning rate. Do not try more batch sizes. If the model does not learn at
2e-5, the bug is in your data.

**One allowed hardening (SHOULD):** prevent two pairs from the same duplicate group landing in
one batch (sentence-transformers' `NO_DUPLICATES` sampler, or a filter in your own sampler).
In N2/N3 this matters *more*, because neighbours are likelier to be duplicates — which is
precisely the contamination you are trying to measure as an outcome rather than manufacture as
a bug.

---

## Baselines

| ID | System | Why |
|---|---|---|
| **B1** | **BM25** (`rank_bm25.BM25Okapi`, default k1/b, ~15 lines) | [BEIR](https://arxiv.org/abs/2104.08663): "BM25 is a robust baseline"; [Jiang et al.](https://doi.org/10.1016/j.jss.2023.111607): deep models do not reliably beat IR. Without B1 your strongest claim is "better than random". |
| **B2** | **`all-MiniLM-L6-v2`, zero-shot** | **The control.** Identical to B3 in every respect except the fine-tuning. |

**Not included:** TF-IDF (strictly weaker, equally easy — it is only the fallback if `rank_bm25`
will not install), hybrid BM25+dense, any second encoder in the core comparison.

---

## Evaluation

Same corpus, same 400 queries (200 dev + 200 test), same code path, all systems.

| Metric | Definition | Why this metric |
|---|---|---|
| **Recall@10** (primary) | of the duplicate questions that exist for a query, what fraction appear in the top 10? | The user-facing question: if the forum shows 10 "possible duplicate" suggestions, did the real one make the list? 0–1 scale, needs no graded relevance, and has headroom on a 15K corpus so a real effect can appear. |
| **MRR@10** (secondary) | mean over queries of `1 / rank` of the first correct duplicate; a miss at rank > 10 counts 0 | Recall@10 is blind to *where* in the top 10 the answer landed. |
| **nDCG@10** (secondary) | rank-discounted gain over the top 10 with binary relevance | Rewarding finding *all* duplicates high up, not just the first. Also the metric the domain-adaptation literature reports, so your numbers are at least in principle comparable. |

Three metrics, ~30 lines of numpy in one shared function.

**Excluded:** MAP, Recall@1/100, alignment/uniformity, Spearman. Each adds code and a paragraph
of justification and none changes the conclusion.

### Statistics — more than Version A, still not a statistics project

1. **Three seeds per condition.** Report `mean ± (max − min)/2`.
2. **Paired bootstrap over queries** — the addition. For each contrast, compute per-query
   differences `Δ_q = metric_A(q) − metric_B(q)`, resample the 400 queries with replacement
   10,000 times, and report the 2.5th and 97.5th percentiles of the mean `Δ`. ~20 lines.
   Run it on exactly three contrasts:
   * **N1 vs. B2** (does plain fine-tuning help at all?)
   * **N2 vs. N1** (does lexical hardness help?)
   * **N3 vs. N2** (does semantic hardness help, or hurt?)
3. **Report dev and test side by side.** Two independent 200-query sets agreeing is the cheapest
   reality check available.
4. **State the resolution limit:** with 200 test queries, sub-3-point differences are noise, and
   the bootstrap interval is what tells you where that line actually falls in your data.

**Excluded:** significance tests, p-values, stars, multiple-comparison corrections. Three
pre-declared contrasts, reported with intervals. That is the whole inferential apparatus.

### Mechanism measurement — the false-negative rate

**This is what makes a null result interpretable, and it costs one afternoon.**

For 50 test queries, take the top-5 results that are **not** in the gold set, for **two** systems
(B2 zero-shot and the best fine-tuned condition). That is ≤500 items. Read them and mark each
one: **genuine duplicate** / **topically related but different** / **unrelated**.

Output: *"X% of the top non-gold results for the fine-tuned model are, by my judgement, genuine
duplicates that the corpus does not label."*

If that number is high — and given Lei et al.'s 5% figure, expect it to be — it explains pattern
3 and predicts pattern 2. It is the single most valuable number in the project after the main
table.

---

## The ablation (SHOULD HAVE)

**`hard_fraction` ∈ {0.25, 0.5, 1.0} on N3 only**, one seed each, evaluated on dev.

This traces the hardness curve continuously instead of at three points, and it is the direct
empirical test of [Robinson et al.](https://openreview.net/forum?id=CR1XOQ0UTh-)'s claim that
hardness is a dial with an optimum rather than a switch. Two extra runs.

---

## Expected outputs

### Main table (defines the project)

| System | Recall@10 (test) | MRR@10 (test) | nDCG@10 (test) | Recall@10 (dev) |
|---|---|---|---|---|
| B1 BM25 | **?** | **?** | **?** | **?** |
| B2 MiniLM zero-shot | **?** | **?** | **?** | **?** |
| B3-N1 random negatives | **?** | **?** | **?** | **?** |
| B3-N2 lexical negatives | **?** | **?** | **?** | **?** |
| B3-N3 model-hard negatives | **?** | **?** | **?** | **?** |

Each B3 row is the mean over 3 seeds, with the range shown.

### Contrast table (the bootstrap)

| Contrast | Δ Recall@10 | 95% CI | Reading |
|---|---|---|---|
| N1 − B2 | **?** | **[?, ?]** | does fine-tuning help at all? |
| N2 − N1 | **?** | **[?, ?]** | does lexical hardness help? |
| N3 − N2 | **?** | **[?, ?]** | does semantic hardness help, or hurt? |

### Supporting outputs

* **Ablation table:** Recall@10 (dev) vs. `hard_fraction` ∈ {0.25, 0.5, 1.0}.
* **False-negative rate:** % of top-5 non-gold results judged genuine duplicates, per system.
* **Error analysis:** 30 worst queries, five labels each.
* **Three figures:** (1) five systems × Recall@10 with seed error bars; (2) Recall@10 vs.
  `hard_fraction`; (3) the three bootstrap intervals as a horizontal interval plot.

### Write-up

8–10 pages: intro, related work (8 papers), method, setup, results (4 tables, 3 figures),
mechanism measurement, error analysis, threats, conclusion.

---

## Threats to validity — the five you must discuss

1. **Incomplete labels.** ~5% of similar pairs are annotated
   ([Lei et al.](https://aclanthology.org/N16-1153/)). All metrics are lower bounds; the bias
   works *against* the harder-negative conditions more than against N1. **This is a differential
   bias and you must say so explicitly** — it is a plausible alternative explanation for
   pattern 2 that your FN measurement is designed to adjudicate.
2. **Hardness and contamination are confounded.** In N3, "hard" and "likely to be an unlabelled
   duplicate" are nearly the same property in this corpus. You cannot fully separate them with
   this design. Say so. Do not claim you have isolated hardness.
3. **Small test set.** 200 queries. The bootstrap interval is the honest answer to "is this real?".
4. **Corpus contains the training questions.** ~96% of corpus documents are train-split
   questions the model saw during training. B2 saw the same corpus, so the comparison is fair,
   but absolute numbers are optimistic. (Cheap check: report Recall@10 separately for gold
   duplicates in the train split vs. the eval split.)
5. **Single domain, single encoder.** AskUbuntu, MiniLM. No generalisation claim is available and
   none should be made.

---

## You are done with the design when you can answer these out loud

1. Why this dataset? *(small, pre-cleaned, fixed splits, real duplicate labels, and small enough that mining is a matmul)*
2. What exactly is an embedding? *(one point on a 384-d sphere; mean-pooled, L2-normalised)*
3. What makes two examples positive? *(the community marked one as a duplicate of the other)*
4. What makes them negative? *(another pair's positive in the same mini-batch)*
5. **What is your independent variable, and what is held constant?** *(batch composition; the loss, model, pairs, batch size and epochs are all fixed)*
6. What does contrastive learning change? *(which directions in the 384-d space mean "same question")*
7. What does the loss encourage? *(rank my duplicate above the 31 other duplicates in my batch)*
8. What is the baseline? *(BM25, plus the un-fine-tuned version of my own model)*
9. How do we know it improved? *(Recall@10 on a fixed test split, 3 seeds, paired bootstrap)*
10. What could cause the result to be wrong? *(incomplete labels, hardness/contamination confound, 200 queries, train-split corpus, broken harness)*

If you cannot answer one, re-read the relevant section rather than moving on.
