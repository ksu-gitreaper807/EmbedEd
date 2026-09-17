# Five papers that use Random, BM25 and Semantic negatives

A short, presentation-friendly version of [`STRATEGY_EVIDENCE.md`](STRATEGY_EVIDENCE.md). One page
per paper: **what it compared, the numbers, and the merits and demerits as that paper discusses
them.**

## Coverage — which of the three does each paper actually use?

| Paper | Random | BM25 (static) | Semantic (dense mining) | Task |
|---|---|---|---|---|
| 1. DPR (2020) | ✅ | ✅ | ❌ *predates dense mining* — has "gold/in-batch" instead | Passage retrieval |
| 2. ANCE (2021) | ✅ | ✅ | ✅ | Passage retrieval |
| 3. STAR/ADORE (2021) | ✅ | ✅ | ✅ | Passage retrieval |
| 4. AugSBERT (2021) | ✅ | ✅ | ✅ | **Sentence-pair scoring** |
| 5. SimANS (2022) | ✅ | ✅ | ✅ | Passage retrieval |

Only AugSBERT tests all three on **sentence-pair** tasks — the closest match to AskUbuntu
duplicate detection. Papers 2, 3 and 5 are passage retrieval, where "semantic" means mining with
the model's own ANN index.

**Provenance of numbers:** **[read]** = read from the paper while writing this. **[check]** = from
a snippet of the paper, column headers or exact definition not yet confirmed — open the PDF before
quoting it in the report.

---

## 1. DPR — where the loss comes from

Karpukhin et al., EMNLP 2020 · [aclanthology.org/2020.emnlp-main.550](https://aclanthology.org/2020.emnlp-main.550/)

**What it compared.** Table 3, Natural Questions **dev**, top-k retrieval accuracy **[read]**:

| Type of negative | Top-5 | Top-20 | Top-100 |
|---|---|---|---|
| **Random** | 47.0 | 64.3 | **77.8** |
| **BM25** | **50.0** | 63.3 | 74.8 |
| Gold (positives of other questions) | 42.6 | 63.1 | 78.3 |
| Gold + in-batch (31 neg) | 52.1 | 70.8 | 82.1 |
| **Gold + in-batch + 1 BM25 hard** | **65.0** | 77.3 | 84.4 |
| Gold + in-batch + 2 BM25 hard | 64.5 | 76.4 | 84.0 |

**Merits and demerits, as DPR discusses them [read]:**

| | Merit | Demerit |
|---|---|---|
| **Random** | Free; no index; **best Top-100** of the three (77.8) | Weakest Top-5 (47.0); teaches least |
| **BM25** | **Best Top-5** (50.0); one BM25 hard negative on top of in-batch training is worth **+12.9 Top-5** | **Worst Top-100** (74.8) — narrowly lexical; a *second* BM25 negative adds nothing (65.0 → 64.5) |
| In-batch | The main engine: reuses negatives already in the batch, memory-efficient, "accuracy consistently improves as the batch size grows" | Alone, weakest Top-5 of all (42.6) |

**The quotation that should temper H1 [read, verbatim]:**

> *"We find that the choice of negatives — random, BM25 or gold passages … **does not impact the
> top-k accuracy much in this setting when k ≥ 20**."*

**Bearing on the project:** the loss, the in-batch mechanism, and the honest precedent for a null
result. It does **not** licence an expectation that BM25 beats random on its own.

---

## 2. ANCE — Random vs BM25 vs self-mined, and why they differ

Xiong et al., ICLR 2021 · [OpenReview zeFrfgyZln](https://openreview.net/forum?id=zeFrfgyZln) · [arXiv:2007.00808](https://arxiv.org/abs/2007.00808)

**What it compared.** Table 1, MRR@10 **[read]**:

| Negative construction | MRR@10 |
|---|---|
| **Rand Neg** (random, in-batch) | 0.261 |
| NCE Neg | 0.256 |
| **BM25 Neg** (random sample from BM25 top-100) | 0.299 |
| DPR (BM25 + Rand, 1:1) | 0.311 |
| **ANCE** (dynamic, self-mined via ANN index) | **0.330** |

**Merits and demerits, as ANCE discusses them [read]:**

| | Merit | Demerit |
|---|---|---|
| **Random** | Stable, cheap, no index | Its negatives are *local* — ANCE argues local negatives cause "diminishing gradient norms", bounded loss and slow convergence |
| **BM25** | Good warm-up; lifts MRR@10 from 0.261 to 0.299 | **Different population from what the dense model finds hard: only ~15% overlap with top dense-retrieved negatives.** It cannot supply the negatives the model actually needs. Leaves more "holes" — their Hole@10 statistic is 25.8% for BM25 Neg vs 14.8% for ANCE **[check]** |
| **Semantic (ANCE)** | Overlap with dense top results "starts at 63% and converges to 100% by design"; best MRR@10 (0.330) | Needs an ANN index and periodic refresh; negatives become progressively harder as the model improves — which is where false negatives enter |

**The core argument [read, verbatim]:** *"Instead of random or in-batch local negatives, ANCE
constructs global negatives using the being-optimized DR model to retrieve from the entire
corpus. This fundamentally aligns the distribution of negative samples in training and of
irrelevant documents to separate in testing."*

**Bearing on the project:** the theoretical case that random/in-batch negatives are *local* and
therefore weak — and the 15%-overlap statistic, which is the cleanest available argument that
**BM25 and semantic negatives are genuinely different populations**, not two versions of the same
thing. That is what makes the N2-vs-N3 comparison worth running.

---

## 3. STAR/ADORE — the most explicit merits-and-demerits discussion

Zhan, Mao, Liu, Guo, Zhang, Ma, **SIGIR 2021** · [arXiv:2104.08051](https://arxiv.org/abs/2104.08051)

This is the paper to quote if you want a source that says, in plain language, that **static hard
negatives are risky**.

**From the abstract [read, verbatim]:**

> *"…we theoretically investigate different training strategies for DR models and try to explain
> why hard negative sampling performs better than random sampling. Through the analysis, we also
> find that there are **many potential risks in static hard negative sampling**, which is employed
> by many existing training methods."*

**From §8.3.1 (Baselines) [read, verbatim] — the single best merits/demerits sentence in this file:**

> *"Random negative sampling can effectively train DR models … Rand Neg outperforms BM25, DeepCT,
> and LeToR even by a large margin on some metrics. **Static hard negative sampling does not
> necessarily lead to performance improvements compared with random negative sampling. It improves
> the top-ranking performance but may harm the recall capability.**"*

**Error analysis [check]** — errors made by models trained with each strategy:

| Trained with | Total errors | Top-K errors |
|---|---|---|
| In-Batch Neg | 679 | 43.2 |
| **Rand Neg** | **659** | 39.3 |
| **BM25 Neg** | **2432** | 46.4 |
| ANCE (dynamic semantic) | 1448 | 37.3 |
| STAR | 1128 | 35.8 |
| ADORE + Rand Neg | 736 | 36.8 |
| ADORE + BM25 Neg | 1840 | 40.0 |
| ADORE + ANCE | 1345 | 36.5 |

*(Column definitions not confirmed against the paper body — open the PDF before quoting.)*

**Merits and demerits, as this paper discusses them [read]:**

| | Merit | Demerit |
|---|---|---|
| **Random** | Effective and stable; beats BM25 and even learned sparse baselines (DeepCT) on some metrics; **fewest errors of the static strategies** | Simplest training signal; caps ultimate top-ranking quality |
| **BM25 (static hard)** | Improves **top-ranking** performance | **"May harm the recall capability."** Severe risk profile — by far the most errors (2432 vs 659). The risks are the paper's motivation for proposing STAR |
| **Semantic (dynamic)** | ANCE beats BM25 Neg; their ADORE (dynamic) improves further | Dynamic mining is unstable — hence STAR, which "improves the stability of DR training by introducing random negatives" |

**Bearing on the project:** this is the strongest prior support for the **inverted-U (H0b)**, and
it comes from a paper that explicitly set out to explain *why* hard negatives help. It also shows
the fix used in practice: **mix random negatives back in for stability** — a useful v2 idea if N3
turns out unstable.

---

## 4. AugSBERT — the only all-three comparison on sentence pairs

Thakur et al., NAACL 2021 · [aclanthology.org/2021.naacl-main.28](https://aclanthology.org/2021.naacl-main.28/)

**What it compared.** Table 5, mean over **10 seeds** **[read]**:

| Strategy | Spanish-STS | BWS cross-topic | BWS in-topic | Quora-QP (F₁) | MRPC (F₁) |
|---|---|---|---|---|---|
| SBERT lower-bound (no augmentation) | 72.07 | 60.54 | 63.77 | 74.66 | 84.39 |
| **Random (RS)** | **62.05** | 59.95 | 64.54 | 73.42 | 82.28 |
| KDE | 74.67 | 61.49 | **69.76** | **79.31** | 84.33 |
| **BM25** | **75.08** | 61.48 | 68.63 | 79.01 | **85.46** |
| **Semantic Search (SS)** | 74.99 | 61.05 | 68.06 | 77.20 | 82.42 |
| BM25 + SS | 76.24 | 59.41 | 63.30 | **72.45** | 82.68 |

**Merits and demerits, quoted from §3.1 [read, verbatim]:**

| | Merit | Demerit |
|---|---|---|
| **Random** | Simplest possible | *"Randomly selecting two sentences usually leads to a dissimilar (negative) pair … **This skews the label distribution** of the silver dataset heavily towards negative pairs."* On Spanish-STS it scored **62.05 — worse than no augmentation at all** (72.07) |
| **BM25** | *"**Indexing and retrieving similar sentences is efficient** and all weakly labeled pairs will be used."* Best on both duplicate/paraphrase tasks (79.01 Quora, 85.46 MRPC) | *"**A drawback of BM25 is that only sentences with lexical overlap can be found.** Synonymous sentences with no or little lexical overlap will not be returned."* |
| **Semantic (SS)** | Finds exactly what BM25 misses — paraphrases with no word overlap | Close to BM25 overall but **worse on both duplicate/paraphrase tasks** (77.20 vs 79.01; 82.42 vs 85.46) |
| **BM25 + SS** | Best on Spanish-STS only | *"Aggregating the strategies … **skews the label distribution towards negative pairs**."* **Worse than either alone on 4 of 5 datasets** |

**Bearing on the project:** the closest precedent in the sentence-embedding literature, and two
warnings: random pair selection can be *actively harmful*, and **naively combining BM25 and
semantic signals is not the answer** — relevant to the C1 disagreement-quadrant idea.

> ⚠ **Caveat before citing this as "negatives".** AugSBERT selects *sentence pairs to be
> soft-labelled by a cross-encoder* — pair selection for data augmentation, not negative
> construction inside an InfoNCE loss. The mechanism is adjacent, not identical.

---

## 5. SimANS — the "not too hard, not too easy" framing

Zhou et al., EMNLP 2022 Industry · [aclanthology.org/2022.emnlp-industry.56](https://aclanthology.org/2022.emnlp-industry.56/) · [arXiv:2210.11773](https://arxiv.org/abs/2210.11773)

**What it says the existing options are [read, verbatim]:** *"Previous works either randomly sample
negatives … or select the top-k hard negatives ranked by BM25 or the dense retrieval model
itself."*

**What it compared [check]** — adding each strategy to a fixed baseline:

| Negatives added to baseline | R@5 | R@20 | R@100 |
|---|---|---|---|
| **Random Neg** | 39.5 | 59.0 | 76.2 |
| **top-k Neg** (BM25 / dense-mined hardest) | 57.1 | 73.5 | 85.1 |
| **SimANS** (negatives ranked *around* the positive) | **59.1** | **74.9** | **85.6** |

**Merits and demerits, as SimANS discusses them [read]:**

| | Merit | Demerit |
|---|---|---|
| **Random** | Free, safe, cannot manufacture false negatives | By far the weakest — 39.5 vs 57.1 R@5. It leaves almost all of the available signal on the table |
| **top-k hard (BM25 or dense)** | Large jump over random (39.5 → 57.1) | Sits at the extreme of the difficulty range, where the false-negative risk is highest. Loses to a *band* around the positive (57.1 → 59.1) |
| **Band around the positive** | *"Not too hard (may be false negatives) or too easy (uninformative)"* — negatives in this band have **larger gradient means and smaller gradient variances** | Requires choosing where the band sits |

**Bearing on the project:** supplies the one-sentence framing for the whole trade-off, and the
mechanism (gradient mean vs variance) behind the inverted-U. It also means that if N3 underperforms
N2, the explanation is not "hard negatives are bad" but "the *hardest* negatives are bad" — a
different and more interesting claim.

---

## Cross-paper synthesis

| | **Random (N1)** | **BM25 (N2)** | **Semantic (N3)** |
|---|---|---|---|
| **Consistent merit across papers** | Cheapest, safest, cannot produce false negatives; most stable training (STAR); best deep-recall in DPR | Cheapest *informative* option; efficient to build (AugSBERT); reliable top-ranking gains (DPR +12.9 Top-5; ANCE 0.261→0.299) | Finds what BM25 structurally cannot (AugSBERT); aligns negatives with what the model actually finds hard (ANCE, 63%→100% overlap); best single scores (ANCE 0.330) |
| **Consistent demerit across papers** | Weakest by a wide margin (SimANS 39.5 vs 57.1; AugSBERT 62.05 vs 75.08); can be worse than doing nothing | Improves top-ranking but **harms recall** (STAR; DPR's worst Top-100); most errors by far (STAR 2432 vs 659); only 15% overlap with dense-hard population (ANCE) | False negatives: NV-Retriever measured 38.8% (NQ) / 47% (StackExchange); unstable without random negatives mixed back in (STAR) |
| **Cost** | None | Low (one sparse index) | High (ANN index + periodic refresh) |
| **Failure mode** | Teaches nothing | Teaches surface matching only | Teaches the model to repel its own true positives |

### The pattern the five papers agree on

```text
   Random  <<  { BM25 , Semantic }          ← large, consistent gap (all 5 papers)
                    |
                    ├── BM25 ≈ Semantic, differing by task      (AugSBERT)
                    ├── Semantic > BM25 on headline metrics      (ANCE, SimANS)
                    └── but harder ≠ better past a point         (SimANS, STAR)
```

**No paper in this set establishes which of BM25 or Semantic wins for duplicate-question retrieval
with incomplete labels.** AugSBERT's closest tasks (Quora-QP, MRPC) have explicit negative labels;
AskUbuntu does not. That gap is what the project's experiment fills.

---

## Honourable mentions — the false-negative evidence

Not in the five above because they do not compare all three strategies, but they carry the numbers
that matter most for N3 on AskUbuntu:

| Paper | Number | Status |
|---|---|---|
| **Lei et al., NAACL 2016** ([N16-1153](https://aclanthology.org/N16-1153/)) | *"only 5% of similar pairs have been annotated by the users, with a precision of around 79%"* — on AskUbuntu itself | verified in repo |
| **NV-Retriever** ([arXiv:2407.15831](https://arxiv.org/abs/2407.15831)) | 38.8% (NQ) / **47% (StackExchange)** of naive top-k "negatives" are not negatives | [check] |
| **RocketQA** ([NAACL 2021](https://aclanthology.org/2021.naacl-main.466/)) | Non-denoised hard negatives were **worse than not using them**; denoising gained 3.1 points | [check] |

---

## Which to read, in what order

| Order | Paper | Why |
|---|---|---|
| **1** | **STAR/ADORE** | The clearest merits/demerits language; directly supports H0b |
| **2** | **AugSBERT** | The only sentence-pair three-way comparison |
| **3** | **DPR** | The loss, and the honest "doesn't matter much at k ≥ 20" caveat |
| **4** | **ANCE** | Why BM25 and semantic are different populations (the 15% figure) |
| **5** | **SimANS** | One sentence states the whole trade-off |

Two citation traps: Robinson et al.'s OpenReview ID is **`CR1XOQ0UTh-`** (not the ID several
secondary sources list); and the "~70% of top-retrieved passages are relevant" figure attributed to
RocketQA in surveys could not be found in the paper — **do not quote it.**
