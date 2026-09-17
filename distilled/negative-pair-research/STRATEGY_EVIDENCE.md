# What the papers actually say about Random vs BM25 vs Semantic negatives

A reading note on the three negative-construction strategies the project manipulates:

| Project condition | Strategy | Also called |
|---|---|---|
| **N1** | **Random** | uniformly sampled / 1-of-N random negatives |
| **N2** | **BM25** | lexical, static, sparse-retrieval hard negatives |
| **N3** | **Semantic** | model-mined, dense/ANN hard negatives, dynamic |

### Provenance of every number in this file

| Mark | Meaning |
|---|---|
| **[verified]** | Read directly from the primary PDF (table or quoted sentence) while writing this file |
| **[repo note]** | Already in this repository's literature files, verified in an earlier session; re-check before quoting in the report |
| **[unverified]** | Circulating in surveys or secondary sources. **Do not cite without checking the primary paper** |

Nothing here is from a blog post or a survey abstract alone.

---

## 1. One-page summary

| Source | Setting | What it compared | What it found |
|---|---|---|---|
| **DPR** (EMNLP 2020) [verified] | Passage retrieval, Natural Questions | Random · BM25 · Gold, 1-of-N | *"the choice of negatives — random, BM25 or gold passages … does not impact the top-k accuracy much in this setting when k ≥ 20"* |
| **DPR** [verified] | Same | Adding hard negatives to in-batch training | Adding **one** BM25 negative: Top-5 52.1 → **65.0**. Adding a **second**: 65.0 → 64.5 (no further gain) |
| **AugSBERT** (NAACL 2021) [verified] | Sentence-pair scoring, 5 datasets, 10 seeds | Random · BM25 · Semantic Search · KDE · BM25+SS | Random is clearly worst (62.05 vs 75.08 on Spanish-STS); **BM25 ≈ Semantic Search**; **BM25+SS is worse than either alone** |
| **ANCE** (ICLR 2021) [repo note] | Passage retrieval | BM25 static vs self-mined dynamic | BM25 negatives overlap only ~15% with dense top results — they are a *different* population, so they warm up but cannot replace dynamic mining |
| **RocketQA** (NAACL 2021) [repo note] | Passage retrieval | Hard negatives, denoised vs not | Non-denoised hard negatives were **worse than not using them**; denoising gained 3.1 points |
| **NV-Retriever** (2024) [repo note] | Passage retrieval | False-negative rate of naive top-k mining | **38.8% on NQ, 47% on StackExchange** of top-k "negatives" are not negatives |
| **Lei et al.** (NAACL 2016) [verified in repo] | **AskUbuntu** | Annotation completeness | *"only 5% of similar pairs have been annotated by the users, with a precision of around 79%"* |
| **FaceNet** (CVPR 2015) [repo note] | Face recognition | Semi-hard vs hardest | Hardest negatives drove the embedding to collapse |
| **Sampling Matters** (ICCV 2017) [repo note] | Metric learning | Hardest vs distance-weighted vs random | Hardest-negative mining is **noise-dominated**; *"sampling matters as much or more than loss functions"* |
| **SimANS** (EMNLP 2022) [repo note] | Dense retrieval | Negatives ranked *around* the positive | Larger gradient means, smaller variance, lower false-negative risk |

**The two sentences that matter most for this project are in bold below (§2.1 and §5.3).**

---

## 2. Tier 1 — the papers that compare all three directly

### 2.1 DPR — Dense Passage Retrieval for Open-Domain QA

Karpukhin et al., EMNLP 2020 · [aclanthology.org/2020.emnlp-main.550](https://aclanthology.org/2020.emnlp-main.550/)

This is the canonical statement of in-batch negatives and the paper N2 imitates. Its Table 3
[verified] compares negative types directly on the Natural Questions **development** set:

| Type | #N | IB | Top-5 | Top-20 | Top-100 |
|---|---|---|---|---|---|
| **Random** | 7 | 7 | 47.0 | 64.3 | **77.8** |
| **BM25** | 7 | 7 | **50.0** | 63.3 | 74.8 |
| Gold | 7 | 7 | 42.6 | 63.1 | **78.3** |
| Gold | 7 | 3 | 51.1 | 69.1 | 80.8 |
| Gold | 31 | 3 | 52.1 | 70.8 | 82.1 |
| Gold | 127 | 3 | 55.8 | 73.0 | 83.1 |
| Gold + BM25 (1 extra) | 31+32 | 3 | **65.0** | **77.3** | 84.4 |
| Gold + BM25 (2 extra) | 31+64 | 3 | 64.5 | 76.4 | 84.0 |
| Gold + BM25 (1 extra) | 127+128 | 3 | 65.8 | 78.0 | **84.9** |

*(#N = number of negative examples, IB = in-batch training. Top three rows are the 1-of-N
setting with no in-batch negatives.)*

**What the paper says about it, verbatim [verified]:**

> *"We find that the choice of negatives — random, BM25 or gold passages (positive passages from
> other questions) — **does not impact the top-k accuracy much in this setting when k ≥ 20**."*

> *"We find that adding a single BM25 negative passage improves the result substantially while
> adding two does not help further."*

**Merits and demerits as DPR discusses them:**

| Strategy | Merit per DPR | Demerit per DPR |
|---|---|---|
| **Random** | Free; no index; surprisingly competitive — best Top-100 of the three (77.8) | Weakest at Top-5 (47.0); at k≥20 it is indistinguishable from the others, i.e. it teaches little |
| **BM25** | Best Top-5 of the three (50.0); **one** BM25 hard negative on top of in-batch training is worth +12.9 Top-5 | **Worst Top-100 of the three (74.8)** — it is narrowly focused on lexical overlap; a second BM25 negative adds nothing |
| **Gold / in-batch** | The main engine: reuses negatives already in the batch, memory-efficient, and accuracy "consistently improves as the batch size grows" | Alone (1-of-N, no in-batch) it is the weakest Top-5 of all (42.6) |

> **⚠ The finding that should temper the project's H1.** DPR's cleanest negative-type ablation
> concluded the choice *does not matter much* at k ≥ 20. The large gain came from combining
> in-batch negatives **with** one hard BM25 negative — not from random-vs-BM25 alone. If this
> project's N1/N2/N3 come out close, that is consistent with DPR, not a failure.

---

### 2.2 AugSBERT — the closest three-way comparison in the sentence-embedding literature

Thakur et al., NAACL 2021 · [aclanthology.org/2021.naacl-main.28](https://aclanthology.org/2021.naacl-main.28/)

**Read this one for the project.** It is the only paper found that puts Random, BM25 and Semantic
Search side by side on *sentence-pair* tasks — the same family of task as AskUbuntu duplicate
detection — with 10 seeds.

**How each strategy works, quoted from §3.1 [verified]:**

| Strategy | What the paper says it does |
|---|---|
| **Random Sampling (RS)** | *"We randomly sample a sentence pair … Randomly selecting two sentences usually leads to a dissimilar (negative) pair; positive pairs are extremely rare. **This skews the label distribution** of the silver dataset heavily towards negative pairs."* |
| **BM25 Sampling** | *"…we index every unique sentence, query for each sentence and retrieve the top k similar sentences. … **Indexing and retrieving similar sentences is efficient** and all weakly labeled pairs will be used."* |
| **Semantic Search (SS)** | *"**A drawback of BM25 is that only sentences with lexical overlap can be found.** Synonymous sentences with no or little lexical overlap will not be returned, and hence, not be part of the silver dataset. We train a bi-encoder (SBERT) on the gold training set … and retrieve for every sentence the top k most similar sentences."* |
| **BM25 + SS** | *"Aggregating the strategies helps capture the lexical and semantically similar sentences but **skews the label distribution towards negative pairs**."* |

**Table 5 results [verified]** — mean over 10 seeds (± std). Higher is better.

| Strategy | Spanish-STS (ρ×100) | BWS cross-topic | BWS in-topic | Quora-QP (F₁) | MRPC (F₁) |
|---|---|---|---|---|---|
| SBERT lower-bound | 72.07 ± 2.05 | 60.54 ± 0.99 | 63.77 ± 2.29 | 74.66 ± 0.31 | 84.39 ± 0.51 |
| **AugSBERT-R.S. (random)** | **62.05** ± 2.53 | 59.95 ± 0.70 | 64.54 ± 1.90 | 73.42 ± 0.74 | 82.28 ± 0.38 |
| AugSBERT-KDE | 74.67 ± 1.01 | **61.49** ± 0.71 | **69.76** ± 0.50 | **79.31** ± 0.46 | 84.33 ± 0.27 |
| **AugSBERT-BM25** | **75.08** ± 1.94 | 61.48 ± 0.73 | 68.63 ± 0.79 | 79.01 ± 0.45 | **85.46** ± 0.52 |
| **AugSBERT-S.S. (semantic)** | 74.99 ± 2.30 | 61.05 ± 1.02 | 68.06 ± 0.93 | 77.20 ± 0.41 | 82.42 ± 0.32 |
| AugSBERT-BM25+S.S. | 76.24 ± 1.42 | 59.41 ± 0.98 | 63.30 ± 1.34 | **72.45** ± 0.77 | 82.68 ± 0.33 |

**What to take from it:**

1. **Random is clearly worst** — and on Spanish-STS it is *worse than doing nothing*
   (62.05 vs the SBERT lower-bound 72.07). Random pair selection actively hurt.
2. **BM25 and Semantic Search are close** (75.08 vs 74.99 on Spanish-STS). BM25 is better on the
   two duplicate/paraphrase tasks (79.01 vs 77.20 Quora; 85.46 vs 82.42 MRPC).
3. **Combining them is worse than either alone** on 4 of 5 datasets. This is the single most
   relevant result for candidate C1 (disagreement quadrants) and for CuSINeS-style fusion.
4. The paper's own explanation for both failures is **label-distribution skew**, not hardness:
   random over-produces trivial negatives, and BM25+SS over-produces negatives overall.

> **⚠ Two caveats before citing this as "Random vs BM25 vs Semantic negatives".**
> (1) AugSBERT selects **sentence pairs to be soft-labelled by a cross-encoder** — it is pair
> *selection for data augmentation*, not negative construction inside an InfoNCE loss. The
> mechanism is adjacent, not identical.
> (2) The sampled pairs become *both* positives and negatives; "skews the label distribution
> towards negative pairs" refers to the mix, not to hardness.

---

### 2.3 ANCE — why BM25 and semantic negatives are different populations

Xiong et al., ICLR 2021 · [OpenReview zeFrfgyZln](https://openreview.net/forum?id=zeFrfgyZln)

ANCE mines hard negatives from the whole corpus using the model's **own** ANN index, refreshed
asynchronously during training.

**Merits and demerits as discussed [repo note — verify before quoting numerically]:**

| | ANCE's account |
|---|---|
| **BM25 (static)** merit | Excellent warm-up: BM25 negatives are informative enough to bootstrap a dense model from a cold start |
| **BM25 (static)** demerit | Fundamentally a different population: BM25 negatives **overlap only ~15%** with the top dense-retrieved negatives. It cannot supply the negatives the *dense* model actually finds hard |
| **Semantic / self-mined** merit | Negative quality tracks the model as it learns; the overlap with dense top results starts at 63% and converges to 100% |
| **Semantic / self-mined** demerit | Needs an index and periodic refresh (compute), and the negatives become progressively harder as the model improves — which is where false-negative risk enters |
| **In-batch (both)** | ANCE argues local in-batch negatives lead to **diminishing gradient norms** — the theoretical case for mining beyond the batch |

---

## 3. Tier 2 — the false-negative problem: the main demerit of semantic mining

This is the deciding issue for N3 in *this* project, because AskUbuntu's labels are radically
incomplete.

### 3.1 Lei et al. — the domain statistic that governs everything

Lei et al., NAACL 2016 · [aclanthology.org/N16-1153](https://aclanthology.org/N16-1153/) ·
introduced the AskUbuntu task and splits used here.

> *"Our manual inspection of a sample set of questions from AskUbuntu shows that **only 5% of
> similar pairs have been annotated by the users, with a precision of around 79%**."*

**Consequence for N3:** the questions the encoder rates as most similar are disproportionately
**genuine duplicates that nobody marked**. Training the model to push them away is the mechanism
behind the predicted inverted-U. This is a property of *this corpus*, not of semantic mining in
general.

### 3.2 RocketQA — hard negatives can be worse than none

Qu et al., NAACL 2021 · [aclanthology.org/2021.naacl-main.466](https://aclanthology.org/2021.naacl-main.466/)

Uses a cross-encoder to re-score mined negatives and drop likely false negatives.
[repo note — verify the exact figures against the paper before quoting]

| | Finding |
|---|---|
| Demerit of unfiltered semantic mining | Non-denoised hard negatives performed **worse than not using hard negatives at all** |
| Merit of filtering | Denoised hard negatives beat non-denoised by **3.1 points** |
| Cost | A cross-encoder pass over every mined negative — the reason denoising is out of scope here |

The commonly repeated figure that "~70% of top-retrieved unlabelled passages are relevant" is
attributed to RocketQA in secondary surveys — **[unverified]**, do not quote it without finding it
in the paper.

### 3.3 NV-Retriever — the false-negative rate, measured

Moreira et al., 2024 · [arXiv:2407.15831](https://arxiv.org/abs/2407.15831) · [repo note]

Measured false-negative rates for **naive top-k mining**: **38.8%** on Natural Questions and
**47%** on StackExchange — the same Stack Exchange family as AskUbuntu. Their positive-aware
filters cut those rates by 57% and 50% respectively.

Also instructive for N3: **margins that are too large hurt**, because they strip out genuinely
informative negatives. The useful negatives are near the boundary, not at the extreme.

---

## 4. Tier 3 — why "harder" is not monotonically better

These four are the reason H0b (inverted-U) — not H1 (monotone) — is the hypothesis the literature
actually supports.

| Paper | Method it studies | Merit of hard negatives | Demerit it identifies |
|---|---|---|---|
| **FaceNet** (Schroff et al., CVPR 2015) | Semi-hard mining | Semi-hard negatives (violate the margin but are farther than the positive) train well | The **hardest** negatives drove the embedding to a **collapsed** solution |
| **Sampling Matters** (Wu et al., ICCV 2017) | Distance-weighted sampling | *"sampling matters as much or more than loss functions"* — the licence for this whole project | Hardest-negative mining produces **high-variance, noise-dominated** gradients |
| **Robinson et al.** (ICLR 2021) | Hardness as a controllable parameter | Hardness is a **dial, not a switch**; *"requires only few additional lines of code … and introduces no computational overhead"* | There is a limit past which hardness **degrades** the representation |
| **SimANS** (Zhou et al., EMNLP 2022) | "Ambiguous" negatives ranked *around* the positive | Larger gradient **means**, smaller gradient **variances** | Negatives that are too hard "may be false negatives"; too easy is "uninformative" |

SimANS states the trade-off in one sentence, and it is the best single quotation available for the
project's framing: negatives should be *"not too hard (may be false negatives) or too easy
(uninformative)"*.

**Cross-field confirmation [repo note]:** a 2026 time-series contrastive-learning paper
([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417425036681)) states
the mechanism cleanly — *"in the unsupervised case, some of the most difficult samples can degrade
classification performance, while in the supervised case, more difficult samples are beneficial"*.
With complete labels, harder is monotonically better. With incomplete labels, the hardest negatives
are false negatives and the curve turns over. AskUbuntu has ~95% unlabelled positives, which puts
this project firmly in the second case.

---

## 5. Synthesis — merits and demerits side by side

| | **N1 Random** | **N2 BM25** | **N3 Semantic** |
|---|---|---|---|
| **How it works** | Any other pair in the batch | Top-k by BM25 among training anchors | Top-k by cosine of the zero-shot encoder |
| **Merits (with source)** | Free, no index, no mining pass. Best Top-100 of DPR's three. Unambiguously safe: contains no deliberately-similar items, so near-zero false-negative risk by construction | Cheap and *efficient* (AugSBERT: "indexing and retrieving … is efficient"). Best Top-5 of DPR's three; one BM25 negative gave DPR +12.9 Top-5. Best or near-best on AugSBERT's duplicate/paraphrase tasks | Finds what BM25 *cannot*: AugSBERT — "synonymous sentences with no or little lexical overlap will not be returned". Tracks the model as it learns (ANCE: overlap with dense top results rises to 100%) |
| **Demerits (with source)** | Uninformative. AugSBERT: random pairs "skew the label distribution heavily towards negative pairs" and scored 62.05 vs 75.08 — **worse than no augmentation at all** on Spanish-STS | Blind to paraphrase — "only sentences with lexical overlap can be found" (AugSBERT). DPR: **worst Top-100** of the three (74.8); a *second* BM25 negative added nothing | **False negatives.** NV-Retriever measured 38.8% / 47% FN rates for naive top-k; on AskUbuntu, Lei et al.'s 5%-annotation statistic makes this worse. RocketQA: unfiltered hard negatives were worse than none |
| **Compute cost** | None | Seconds (one sparse index) | Seconds (one encode pass + one matmul), but needs the base model |
| **Failure mode** | Teaches nothing | Teaches only surface matching | Teaches the model to repel its own true positives |
| **Per the literature, wins when…** | …nothing else is affordable, or as the null condition | …the task rewards lexical/terminological precision (AugSBERT's Quora-QP and MRPC) | …labels are complete and paraphrase dominates — **rarely true in community-QA corpora** |

### 5.1 The honest three-way ranking the literature supports

```text
   Not:   Random  <  BM25  <  Semantic          ← the naive H1

   But:   Random  <<  { BM25 , Semantic }       ← the gap that is consistently found
                      (close to each other)      ← AugSBERT: 75.08 vs 74.99

          and the winner among the two depends on
          how complete the positive labels are.
```

### 5.2 What is *not* established by any of these papers

* Which of BM25 or Semantic wins for **duplicate-question retrieval with incomplete labels**.
  AugSBERT's closest tasks (Quora-QP, MRPC) have explicit negative labels; AskUbuntu does not.
* Whether the strategies differ at all when negatives are delivered **in-batch** rather than as
  explicit triples. DPR's negative-type ablation was in the 1-of-N setting; ANCE's argument about
  diminishing in-batch gradients is theoretical.
* Anything at all about the **disagreement** between the two signals — no paper in this set
  partitions by agreement. The nearest result is AugSBERT's BM25+SS row, which suggests that
  naively *combining* them is harmful.

---

## 6. What this means for N1 / N2 / N3

| Implication | Where it comes from |
|---|---|
| **H1 (monotone) is the weakest hypothesis.** The literature supports a gap between random and hard, but not a clean BM25 < Semantic ordering | AugSBERT Table 5; SimANS; Robinson et al. |
| **H0b (inverted-U) is the best-supported hypothesis.** Every tier-3 paper found a limit to hardness, and Lei et al.'s 5% makes the false-negative mechanism concrete here | FaceNet; Sampling Matters; Robinson; SimANS; Lei et al. |
| **A null result is consistent with DPR**, which found the negative type "does not impact the top-k accuracy much" in its cleanest ablation | DPR Table 3, verbatim |
| **N3's expected weakness has a measurement already planned**: the manual false-negative rate over 50 queries × top-5 | Project spec; NV-Retriever's 47% StackExchange figure is the number to compare against |
| **Do not add a "BM25 + Semantic" fused condition.** AugSBERT tried it and it lost on 4 of 5 datasets | AugSBERT Table 5 |
| **Consider the cost asymmetry.** Random costs nothing and cannot manufacture false negatives; if BM25 and Semantic come out within the bootstrap interval, the defensible recommendation is the cheaper, safer one | AugSBERT's label-skew explanation; NV-Retriever |

---

## 7. Reading priority

| Priority | Paper | Read | Why |
|---|---|---|---|
| **1** | **AugSBERT** | §3.1 (sampling strategies), Table 5 | The only direct Random-vs-BM25-vs-Semantic comparison on sentence-pair tasks |
| **2** | **DPR** | §3.2, §5.2 (Table 3) | Where the loss comes from; the honest "it doesn't matter much at k≥20" caveat |
| **3** | **Lei et al.** | §1 | The 5% / 79% statistic the whole false-negative argument rests on |
| **4** | **SimANS** | §1, §3 | One sentence states the entire trade-off |
| 5 | Robinson et al. | §1, §3, §4 | Hardness as a dial; justifies `hard_fraction` |
| 6 | NV-Retriever | §3–4 | The measured FN rates, and the simplest filter if one is ever needed |
| 7 | ANCE | §1, §3 | Only if N3's neighbourhoods get refreshed mid-training |
| 8 | FaceNet, Sampling Matters | §3–4 each | Background for why "hardest" fails |

## 8. Citations

Use the primary source. All links below are official venue or arXiv pages.

```text
DPR          Karpukhin et al., EMNLP 2020      aclanthology.org/2020.emnlp-main.550
AugSBERT     Thakur et al., NAACL 2021         aclanthology.org/2021.naacl-main.28
ANCE         Xiong et al., ICLR 2021           openreview.net/forum?id=zeFrfgyZln
RocketQA     Qu et al., NAACL 2021             aclanthology.org/2021.naacl-main.466
NV-Retriever Moreira et al., 2024              arxiv.org/abs/2407.15831
SimANS       Zhou et al., EMNLP 2022 Industry  aclanthology.org/2022.emnlp-industry.56
Robinson     Robinson et al., ICLR 2021        openreview.net/forum?id=CR1XOQ0UTh-
Lei          Lei et al., NAACL 2016            aclanthology.org/N16-1153
FaceNet      Schroff et al., CVPR 2015         cv-foundation.org (CVPR 2015 open access)
Sampling Matters  Wu et al., ICCV 2017         arxiv.org/abs/1706.07567  doi:10.1109/ICCV.2017.309
```

Two specific traps:

* Robinson et al.'s OpenReview ID is **`CR1XOQ0UTh-`**. Several secondary sources list a different
  ID; do not copy theirs.
* Before quoting **any** number from RocketQA, ANCE or NV-Retriever, open the paper and find the
  table. The figures in this file marked `[repo note]` were verified in an earlier session but
  should be re-checked at the moment they go into the report.
