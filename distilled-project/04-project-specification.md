# 04 — The project specification

This is the single project you will build. Everything below is chosen from the evidence in [`03`](03-feasibility-matrix-and-paths.md).

---

## Project title

**Which Negatives? The Effect of Negative-Sampling Strategy on Contrastive Fine-Tuning for Duplicate Question Retrieval**

(Working title. Subtitle if you need one: *a controlled study on 12 technical StackExchange domains*.)

---

## Research question

> In duplicate-question retrieval, does the **choice of negative examples** determine whether contrastive fine-tuning of a small general-purpose sentence encoder improves retrieval over (a) a lexical baseline and (b) the same encoder used without fine-tuning?

It is **one** question with **one** independent variable (negative-sampling strategy). Everything else — model, data, epochs, evaluation — is held fixed. That is what makes it answerable in four weeks.

---

## Hypothesis (falsifiable)

**H1 (the "textbook prediction", carried over from web retrieval):** Contrastive fine-tuning improves nDCG@10 over the zero-shot encoder, and the improvement is ordered by negative strategy:

```
random  <  in-batch  <  lexical-hard  <  model-hard  <  denoised-hard
```

with denoised hard negatives best, because they are hard enough to produce a gradient but filtered enough to avoid false negatives.

**H0 (the alternative I actually think is at least as likely):** All strategies are statistically indistinguishable from one another, because in a duplicate-labelled corpus the hardest negatives are disproportionately **unlabelled duplicates** — so mining harder negatives adds gradient signal and label noise in roughly equal measure, and the two cancel.

**Both are pre-registered and both are interpretable.** You do not need H1 to be true. If H0 holds, the paper becomes: *"hard-negative mining, which is standard practice in web retrieval, does not transfer to duplicate-question retrieval, and here is the measured false-negative rate that explains why."* That is a legitimate, citable finding.

**What would falsify the whole framing:** if *every* fine-tuned configuration, including random negatives, beats BM25 and the zero-shot encoder by a large margin. Then negative sampling is not the interesting variable and you should pivot the write-up to a straight domain-adaptation result (still fine, just less interesting).

---

## Dataset

**CQADupStack** — Hoogeveen, Verspoor & Baldwin, ADCS 2015.
Paper: <https://eltimster.github.io/www/pubs/adcs2015.pdf>
Access: [BEIR](https://github.com/UKPLab/beir) or `ir_datasets` (`beir/cqadupstack/<subforum>`), catalogue at <https://ir-datasets.com/beir.html>

| Property | Value |
|---|---|
| Source | 12 StackExchange subforums (android, english, gaming, gis, mathematica, physics, programmers, stats, tex, unix, webmasters, wordpress) |
| Size | **13,145 queries / 457K documents** in the BEIR version |
| Per-subforum queries | ~652 (stats) to ~2,900 (tex) |
| Labelled duplicates per query | **~1.4 on average** |
| Positives | human community duplicate marks — **naturally occurring, never synthetic** |
| License | Released for research use; included in BEIR, which is Apache-2.0 |

**Why this dataset, in one line each:**
1. Loads in one line — no crawling, no cleaning of scraped HTML dumps.
2. **Published reference numbers exist** (CQADupStack nDCG@10: 29.6 zero-shot, 31.8 TSDAE, 34.5 GPL, 35.1 TSDAE+GPL — [GPL Table 7](https://aclanthology.org/2022.naacl-main.168/)), so you can **verify your evaluation harness** in week 1.
3. 12 subforums = 12 natural technical domains → a built-in domain-generalisation experiment.
4. Sparse labels make the false-negative question sharp and measurable.

**Optional second dataset (cheap, for robustness):** the `triplet` subset of [`sentence-transformers/quora-duplicates`](https://huggingface.co/datasets/sentence-transformers/quora-duplicates), which already ships **hard negatives mined and denoised by a cross-encoder**. Use it only as an external check on your own mining code — do not mix it into your main training set.

---

## Input and output

* **Input to the encoder:** a single string = `"<title>. <body>"`, HTML and code blocks stripped, truncated to **256 tokens**. Identical preprocessing for every system.
* **Output:** a **384-dimensional L2-normalised vector** that represents the question's *information need*. Two questions that a human would call duplicates should have vectors pointing in nearly the same direction.
* **At inference:** embed all documents in the pool once, build a FAISS inner-product index, embed the query, retrieve top-100.

---

## Positive pair — precise definition

> **(a, p) is a positive pair iff `p` is a question that the community explicitly marked as a duplicate of `a`.**

Additional rules you must apply:

1. **Use the full duplicate group, not just the direct link.** Build an undirected graph over duplicate marks and take **connected components**. If A→B and B→C, then (A,C) is also a positive.
2. **No synthetic positives.** No paraphrasing, no back-translation, no LLM-generated duplicates. This is a hard rule: generated positives teach paraphrase-invariance, not domain similarity, and [the literature on synthetic negatives](https://arxiv.org/abs/2606.01304) shows how badly generated data can mislead contrastive training.
3. **Positives are constructed only from training-period data.**

---

## Negative pair — precise definition (this is the independent variable)

> A candidate `c` is a valid negative for anchor `a` iff `c` is **not in `a`'s duplicate group**.

Note the honest weakness: "not marked duplicate" is weaker than "not a duplicate". **That weakness is the subject of the study**, and you will measure it.

Five strategies — all built from the same anchor/positive pool, differing only in which negatives are attached:

| ID | Strategy | Definition | Where the idea comes from |
|---|---|---|---|
| **N1** | Random | Uniform sample of `k` documents from the same subforum, excluding the group | Naive baseline |
| **N2** | In-batch only | No explicit negatives; the other `B−1` positives in the mini-batch act as negatives | [Henderson 2017](https://arxiv.org/abs/1705.00652), [DPR](https://arxiv.org/abs/2004.04906) |
| **N3** | Lexical-hard | Top-`k` BM25 matches for `a`, excluding the group | [DPR](https://arxiv.org/abs/2004.04906) ("BM25 negatives"), [AugSBERT](https://aclanthology.org/2021.naacl-main.28/) |
| **N4** | Model-hard | Top-`k` by the **current model's** embedding similarity, re-mined once per epoch with FAISS | [ANCE](https://openreview.net/forum?id=zeFrfgyZln) (lite version: static refresh instead of asynchronous) |
| **N5** | Denoised-hard | N4 candidates filtered: drop any `c` with `sim(a,c) > sim(a,p) − δ` (or use a cross-encoder score threshold) | [RocketQA](https://aclanthology.org/2021.naacl-main.466/), [Robinson et al.](https://openreview.net/forum?id=S4nZh4WBHxq) |

**Practical note:** `sentence-transformers` ships `mine_hard_negatives(...)` with exactly the knobs for N3–N5 (`range_min`, `range_max`, `max_score`, `absolute_margin`, `relative_margin`, `use_faiss`). Use it. Do not re-implement mining.

---

## Model

**Primary:** `sentence-transformers/all-MiniLM-L6-v2`
* 23M parameters, 384-d output, Apache 2.0, 6 Transformer layers, max 256 tokens.
* Why: a full training run takes **minutes**, so you can afford **3 seeds × 5 strategies = 15 runs**. That statistical power is worth far more than a bigger model. [MTEB](https://aclanthology.org/2023.eacl-main.148/) shows performance scales with size, but scale is not your contribution.
* Also: its zero-shot form is the primary baseline, so the "before vs. after" comparison is exactly controlled.

**Secondary (one extra baseline, and optionally one extra fine-tuned run):** `BAAI/bge-small-en-v1.5`
* 33M parameters, 384-d, MIT.
* **Caveat you must handle:** BGE models require a **query prefix** — `"Represent this sentence for searching relevant passages: "`. Forget it and your baseline is artificially bad. This is a real fairness issue of the kind [Musgrave et al. (2020)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf) warn about.

**Explicitly out of scope:** training any encoder from scratch; anything above ~150M parameters; multi-GPU.

---

## Training objective

**MultipleNegativesRankingLoss** (InfoNCE with in-batch negatives). For anchor `a`, positive `p`, negatives `{nᵢ}`:

```
                          exp( sim(a, p) / τ )
L  =  −log  ─────────────────────────────────────────────────
             exp( sim(a, p) / τ )  +  Σᵢ₌₁..ᴋ exp( sim(a, nᵢ) / τ )
```

**In plain English, term by term:**

| Piece | What it makes the model do |
|---|---|
| `sim(a, p)` on top | Bigger is better → **pull the duplicate closer**. |
| `Σ exp(sim(a, nᵢ))` on the bottom | Bigger is worse → **push every non-duplicate away**. |
| The fraction | "The duplicate's share of the total similarity." The loss is the negative log of that share. |
| `−log` | The loss only reaches ~0 when the duplicate dominates *all* the negatives — it is a softmax classifier where the correct answer is "the positive". |
| `τ` (temperature) | **Small τ** = harsh punishment for near-misses = the model focuses on the hardest negatives. **Large τ** = softer, treats all negatives equally. This is why τ and negative hardness interact. |
| `k` (number of negatives) | More negatives = harder task = stronger signal. With in-batch negatives, `k = batch_size − 1` **for free**. |

**Settings (proposed, not established — you will tune):**

| Item | Value | Note |
|---|---|---|
| Loss | `MultipleNegativesRankingLoss` | default scale 20.0, which behaves like τ ≈ 0.05 |
| Batch size | **64** (try 32 and 128) | Gives 63 in-batch negatives. [DPR](https://arxiv.org/abs/2004.04906) used 128; [arXiv:2508.09534](https://arxiv.org/html/2508.09534) shows 16 works on a single 16 GB GPU |
| Extra mined negatives | 0 (N2) or 3 (N3/N4/N5) | [DPR](https://arxiv.org/abs/2004.04906): one BM25 negative helps, two do not |
| Epochs | 2–4, checkpoint on validation | |
| Optimiser | AdamW | |
| LR | **2e-5**, sweep {1e-5, 2e-5, 5e-5} | [Rosane et al.](https://doi.org/10.5753/sbes.2025.9809) found LR sensitivity is severe in this family of tasks |
| Warmup ratio | 0.1 | [HF training config convention](https://huggingface.co/blog/static-embeddings) |
| Max seq length | 256 tokens | Test 128 vs 512 as a cheap ablation |
| Precision | fp16 (bf16 if supported) | Frees memory for a bigger batch |
| Batch sampler | `BatchSamplers.NO_DUPLICATES` | [Docs](https://sbert.net/): "MultipleNegativesRankingLoss benefits from no duplicate samples in a batch" |
| Hard-negative refresh | once per epoch | Static, lite version of [ANCE](https://openreview.net/forum?id=zeFrfgyZln)'s asynchronous index |
| Seeds | **3** (fixed: 13, 42, 1337) | Non-negotiable |

---

## Baselines

All baselines run on the **same** queries, the **same** document pool, the **same** metric code, the **same** truncation.

| ID | Baseline | Why it is necessary |
|---|---|---|
| **B0** | **BM25** (title+body) | Non-negotiable. [BEIR](https://arxiv.org/abs/2104.08663): BM25 is a robust zero-shot baseline; dense models often lose to it out of domain. Without BM25 you cannot tell whether you beat lexical matching |
| **B1** | TF-IDF + cosine | Second lexical point; anchors "how much does word overlap explain?" |
| **B2** | **`all-MiniLM-L6-v2` zero-shot** | The critical control: isolates the effect of **your fine-tuning** from the effect of the pretrained model. Same architecture, same dimensions, same everything |
| **B3** | `BAAI/bge-small-en-v1.5` zero-shot (with the correct query prefix) | A *modern* retrieval-oriented encoder. Your claim is "better than a general-purpose embedding model" — test a good one, not a 2019 one |
| **B4** | BM25 ∪ dense hybrid (score fusion of B0 and B2) | The realistic deployed system; [Jiang et al.](https://doi.org/10.1016/j.jss.2023.111607)'s best result was a hybrid |

---

## Evaluation

**Primary metric: nDCG@10.** One number, decided before you run anything. It is the [BEIR](https://arxiv.org/abs/2104.08663) / [MTEB](https://aclanthology.org/2023.eacl-main.148/) convention, so your results are comparable to the published CQADupStack numbers.

**Secondary:**

| Metric | What it adds |
|---|---|
| Recall@1, Recall@10, Recall@100 | Matches the user experience ("is a duplicate in the list I look at?") |
| MRR@10 | Rewards getting the duplicate *first* |
| MAP | Whole-ranking quality |

**Diagnostics (the interesting part):**

| Diagnostic | How to compute it |
|---|---|
| **Estimated false-negative rate** of each mined negative set | For each strategy, take the mined negatives, and have (a) a lexical check and (b) a **cross-encoder** score against the anchor; report the fraction scoring above the anchor–positive score. Cross-check on a sample of ≥100 by hand |
| Alignment / uniformity | ~10 lines, from [Wang & Isola](https://mlanthology.org/icml/2020/wang2020icml-understanding/) |
| Labels-per-query distribution | Straight from the qrels — justifies your ceiling argument |

**Statistical protocol:**
* **3 seeds** → report mean ± standard deviation.
* **Paired bootstrap over test queries** (10,000 resamples) for headline comparisons — it accounts for the fact that all systems see the same queries.
* Report **per subforum** as well as the mean. [MTEB](https://aclanthology.org/2023.eacl-main.148/)'s headline finding is that no method dominates; averages hide this.

---

## Ablations

You have room for two. Pick these:

**A1 — Negative strategy (N1–N5), 3 seeds each.** This *is* the project.

**A2 — Number of hard negatives (0 / 1 / 3 / 5) on the best strategy.** Directly tests whether [DPR](https://arxiv.org/abs/2004.04906)'s "one hard negative is enough, two do not help" result holds in this domain. Four short runs.

**If time remains (see [`05`](05-clean-experiment-and-scope.md)):** A3 data efficiency (1/5/25/100% of pairs), A4 held-out-subforum generalisation.

---

## Expected results — what patterns mean what

**No numbers are predicted here.** These are the *shapes* to look for, pre-registered.

### Pattern 1 — Monotone improvement with hardness
```
N1 < N2 < N3 < N4 < N5   (each gap > seed-to-seed std)
```
→ **Supports H1.** The web-retrieval result transfers. Your paper: "hard-negative mining transfers to duplicate-question retrieval, and denoising helps." Check: does the false-negative rate stay low (< ~10%) even for N4? If yes, that explains the monotone pattern.

### Pattern 2 — Improvement, then a cliff
```
N1 < N2 < N3 ≈ N4 < N5     or     N1 < N2 < N3 < N5, with N4 < N3
```
→ **Supports H1 with a caveat, and is the most interesting outcome.** Un-denoised model-hard negatives (N4) hurt relative to lexical-hard (N3), but denoising (N5) recovers and exceeds. Your paper: "hardness helps up to the point where mined negatives become dominated by unlabelled duplicates; the denoising filter is what makes hard negatives safe." **Check:** the measured false-negative rate for N4 should be markedly higher than for N3 and N5. If it is, you have a mechanism, not just a number.

### Pattern 3 — Flat
```
N1 ≈ N2 ≈ N3 ≈ N4 ≈ N5     (all gaps within noise)
```
→ **Supports H0.** Your paper: "negative-sampling strategy, which is decisive in web retrieval, does not matter here." **Check:** is the false-negative rate high across *all* mined strategies? If yes → label incompleteness is the binding constraint, and you should say so and quantify the ceiling. Also check alignment/uniformity: if all five have similar alignment but different uniformity, that is a clean mechanistic story.

### Pattern 4 — Everything loses to BM25
```
BM25 > all fine-tuned variants
```
→ **Entirely possible and entirely publishable.** [Jiang et al.](https://doi.org/10.1016/j.jss.2023.111607) found exactly this for duplicate bug reports, and [Zhang et al.](https://doi.org/10.1145/3576042) found simple retrieval beating deep-learning methods. Your paper becomes a replication of that finding in a second domain, with the false-negative measurement as the explanation.

### Pattern 5 — Everything improves hugely (e.g., +15 nDCG@10)
→ **Suspect a bug before celebrating.** Cross-check against the published CQADupStack range (29.6–35.1 nDCG@10). A jump beyond that almost certainly means leakage — a duplicate group straddling your split, or the gold document appearing in the training positives. Audit before writing.

---

## The ceiling you must report

With only ~1.4 labelled duplicates per query, and with many true duplicates **never marked**, there is a hard ceiling on achievable Recall@k and nDCG@10. Two consequences:

1. **Never compare your absolute number to a web-search number.** Compare systems within your own harness.
2. **Report the ceiling analysis.** Sample 100 queries where the top-1 result is *not* a labelled duplicate, and manually judge whether it is nevertheless a true duplicate. If a large fraction are, your system is better than your metric says — and that is a genuine, reportable finding about evaluation in this domain (cf. [Zhang et al.](https://doi.org/10.1145/3576042)'s failure analysis, which found three distinct causes of apparent failure).
