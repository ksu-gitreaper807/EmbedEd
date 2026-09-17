# Literature on negative-pair construction

Only papers that are specifically about **how negatives are chosen**. Nothing here is included
for background on embeddings, transformers or retrieval in general — see
[`project/distilled/LITERATURE.md`](../../project/distilled/LITERATURE.md) for that.

Organised by the taxonomy category each paper supports. Every entry states what to take from it
and how much of it to read.

---

## Category 1 — Random and in-batch negatives

### Dense Passage Retrieval (DPR)
Karpukhin et al., EMNLP 2020 · [aclanthology.org/2020.emnlp-main.550](https://aclanthology.org/2020.emnlp-main.550/)

The canonical statement of in-batch negatives (`N` pairs → `N−1` free negatives per example), and
the canonical demonstration that adding **one BM25 hard negative** per query helps.

**Take:** §3.2 (the loss), §4 (the ablations showing BM25 negatives > random, and that larger
batches help because they supply more in-batch negatives). **Read:** §2, §3.2, §4.1.

### Sampling Matters in Deep Embedding Learning
Manmatha, Wu, Smola, Krähenbühl, ICCV 2017 · [arXiv:1706.07567](https://arxiv.org/abs/1706.07567) · DOI 10.1109/ICCV.2017.309

The metric-learning paper that argues **sampling matters as much as the loss function** — the
sentence to quote if anyone asks why this is a legitimate research direction at all. Proposes
**distance-weighted sampling**: sample negatives uniformly by *distance* rather than taking the
hardest, because hardest-negative mining produces high-variance, noise-dominated gradients.

**Take:** §4 (distance-weighted sampling), §5.1 (convergence comparison against semi-hard and
random). **Read:** §1, §4, §5.1.

### Do More Negative Samples Necessarily Hurt In Contrastive Learning?
Awasthi et al., ICML 2022 · [PMLR v162](https://proceedings.mlr.press/v162/awasthi22b/awasthi22b.pdf)

Pushes back on the "collision-coverage" claim that more negatives eventually hurt: in their
setting, downstream performance does not degrade with the number of negatives. Useful as the
counter-argument when you claim a null result is meaningful.

---

## Category 2 — Lexical (static) hard negatives

Covered by DPR above (BM25 negatives). Two variants worth knowing:

### PassageBM25: negatives similar to the *positive*, not the query
PACLIC 2023 · [aclanthology.org/2023.paclic-1.59](https://aclanthology.org/2023.paclic-1.59/)

Selects hard negatives by their similarity to the **positive passage** rather than to the query.
Cheap, static, no index refresh. Interesting because it is a third similarity signal.

### Augmented SBERT (AugSBERT)
Thakur et al., NAACL 2021 · [aclanthology.org/2021.naacl-main.28](https://aclanthology.org/2021.naacl-main.28/)

Its pair-sampling ablation — **Random vs. BM25 vs. Semantic Search vs. KDE** — is the closest
precedent in the sentence-embedding literature for "how you pick training pairs is a variable
worth ablating". KDE sampling is the nearest thing to "sample from a density region" rather than
"take the top-k".

**Take:** §3 (the four sampling strategies) and the ablation table.

---

## Category 3 — Model-based (dynamic) hard negatives

### ANCE: Approximate Nearest Neighbor Negative Contrastive Estimation
Xiong et al., ICLR 2021 · [OpenReview zeFrfgyZln](https://openreview.net/forum?id=zeFrfgyZln)

Mines hard negatives from the whole corpus with the model's own ANN index, refreshed
asynchronously during training. Also contains a theoretical argument that **local in-batch
negatives lead to diminishing gradient norms**.

**Take:** the "BM25 warm-up" finding (BM25 negatives first, then self-mined) and the measured
overlap statistic: BM25 negatives overlap 15% with top dense-retrieved negatives, while ANCE
negatives start at 63% and converge to 100%.

---

## Category 4 — Semi-hard / band / ring negatives (controlled similarity)

**This is the most crowded category. Read all four before proposing anything here.**

### FaceNet — semi-hard negative mining
Schroff, Kalenichenko, Philbin, CVPR 2015 · [CVF Open Access](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)

Origin of the term. FaceNet deliberately **excludes the hardest negatives** because they drove the
embedding to a collapsed solution; it selects negatives that violate the margin but lie farther
from the anchor than the positive does.

### Conditional Negative Sampling ("Ring Discrimination")
Wu, Xiong, Yu, Lin, ICLR 2021 · [arXiv:2010.02037](https://arxiv.org/abs/2010.02037)

Samples negatives **in a "ring" around the positive** — within specific similarity percentiles,
"difficult but not excessively close" — and anneals the ring inward during training. Improves
IR/CMC/MoCo by 2–5 points. **Most direct prior work on "controlled similarity band".**

### SimANS: Simple Ambiguous Negatives Sampling
Zhou et al., EMNLP 2022 Industry · [aclanthology.org/2022.emnlp-industry.56](https://aclanthology.org/2022.emnlp-industry.56/) · [arXiv:2210.11773](https://arxiv.org/abs/2210.11773)

Shows empirically that **negatives ranked around the positives** have larger gradient means and
smaller gradient variances, and are less likely to be false negatives: "not too hard (may be false
negatives) or too easy (uninformative)". Designs a sampling distribution peaked on that band.

**This is the paper that most directly pre-empts "is there an optimal similarity range?" — the
answer is yes, and it has been measured.**

### nmCSE: distance moderation + spatial uniformity
Machine Learning, 2023 · [doi:10.1007/s10994-023-06408-8](https://doi.org/10.1007/s10994-023-06408-8)

For sentence embeddings specifically: informative negatives should have **moderate distance**
(not nearest, not farthest) and should be **spatially uniform** around the anchor. Proposes
distance-based weighting and "grid sampling".

---

## Category 5 — Adversarial and synthetic negatives

### Hard Negative Mixing (MoCHi)
Kalantidis et al., NeurIPS 2020 · [arXiv:2010.01028](https://arxiv.org/abs/2010.01028)

Synthesises hard negatives by convex combinations of the hardest existing negatives, and by mixing
the query with them. Notably, they also report that ~8% of synthesised points are *fractionally*
false negatives. Cheap: only extra dot products.

### Also in this space (reference only)
* **MixCSE** — mixes positive and negative features to create hard negatives; argues hard
  negatives are essential to maintain gradient signal. (Listed in the survey below.)
* **AdCSE** — adversarial training to generate hard negatives.
* **SNCSE** — [arXiv:2201.05979](https://arxiv.org/abs/2201.05979) — "soft negatives": rule-based
  negation of the sentence, giving textually near-identical but semantically opposite examples.
  Relevant to "high lexical similarity, low semantic similarity".
* **CLINE** — antonym replacement as the negative counterpart of synonym replacement.
* **SyNeg / LLM-generated negatives** — an active 2025–26 thread; out of scope here (no paid APIs,
  no LLM agents).

### Survey: Contrastive Learning Models for Sentence Representations
ACM Computing Surveys, 2023 · [doi:10.1145/3593590](https://doi.org/10.1145/3593590)

**Read this one first if you want the whole map.** It tabulates ~25 sentence-embedding models with
columns for *how positives are made* and *how negatives are chosen* — the fastest way to see which
negative strategies exist in NLP and which are unexplored.

---

## Category 6 — Curriculum / progressive difficulty

### Curriculum learning (the origin)
Bengio et al., ICML 2009 — the "easy first" principle.

### CuSINeS: Curriculum-driven Structure Induced Negative Sampling
arXiv 2024 · [arXiv:2404.00590](https://arxiv.org/abs/2404.00590)

Legal-domain retrieval. Ranks negatives by **both lexical/structural proximity and semantic
proximity** (combining rankings by reciprocal rank fusion), buckets them into three difficulty
levels, and shifts the sampling mix from easy to hard over training. Finds the curriculum
outperforms a fixed schedule. **Closest paper to "combine two signals + curriculum" — read it
before proposing anything in that space.**

### Mining negative samples via curricular weighting
Zhuang, Jing, Jia, Information Sciences 2024 · [doi:10.1016/j.ins.2024.120534](https://doi.org/10.1016/j.ins.2024.120534)

Adaptive curriculum loss: easy negatives early, hard later, with L2 regularisation on the hard
weights to stop false negatives dominating late in training.

### Robinson et al. — hardness annealing
ICLR 2021 · [OpenReview CR1XOQ0UTh-](https://openreview.net/forum?id=CR1XOQ0UTh-)

In the appendix: an annealing variant that starts with a high hardness parameter and *reduces* it
over training. So the curriculum direction is already explored by the hardness paper itself.

### Also: SPCL, ConCur, ACGCL, CCGL, CurNM
Curriculum contrastive learning is an active area across vision, graph and temporal-network
settings (CurNM for temporal graphs: [arXiv:2407.17070](https://arxiv.org/abs/2407.17070)).

---

## Category 7 — False-negative-aware methods

### Debiased Contrastive Learning
Chuang, Robinson, Yen, Torralba, Jegelka, NeurIPS 2020 · [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html)

Estimates and corrects for the "sampling bias" that makes some in-batch negatives share the
anchor's class. The theoretical backbone of the whole false-negative literature.

### RocketQA — denoised hard negatives
Qu et al., NAACL 2021 · [aclanthology.org/2021.naacl-main.466](https://aclanthology.org/2021.naacl-main.466/)

Uses a cross-encoder to re-score mined negatives and drop likely false negatives. Reported as a
3.1-point improvement over *non-denoised* hard negatives — and non-denoised hard negatives were
*worse* than not using them. A commonly cited estimate (attributed to RocketQA in the survey
below) is that **~70% of top-retrieved unlabelled passages are actually relevant**.

### NV-Retriever — positive-aware mining
Xu/Moreira et al., 2024 · [arXiv:2407.15831](https://arxiv.org/abs/2407.15831)

**The simplest effective false-negative filter, and the one to copy.** Uses the *positive's*
relevance score as the anchor for a threshold:
* `TopK-MarginPos` — keep negatives scoring below `score(positive) − margin` (best margin 0.05)
* `TopK-PercPos` — keep negatives scoring below `95% × score(positive)` (best configuration)

Measured false-negative rates for naive top-k: **38.8%** on Natural Questions, **47%** on
StackExchange. Their positive-aware methods cut that by **57%** (NQ) and **50%** (StackExchange).
Also instructive: margins that are *too* large hurt, because they remove genuinely informative
negatives. ~10 lines to implement, no second model needed.

### GISTEmbed / GISTEmbedLoss
Solatorio, 2024 · [arXiv:2402.16829](https://arxiv.org/abs/2402.16829) ·
[sentence-transformers documentation](https://sbert.net/docs/package_reference/sentence_transformer/losses.html)

The same positive-relative idea applied to **in-batch** negatives, using a "guide model": if the
guide scores a candidate negative as more similar to the anchor than the true positive is, mask it
out of the loss. Shipped in `sentence-transformers` as `GISTEmbedLoss` and `CachedGISTEmbedLoss`
with `margin_strategy ∈ {absolute, relative}`. **Check this before implementing any
false-negative filter yourself.**

### Reciprocal Nearest Neighbors (rNN) for sparse annotation
Zerveas, Rekabsaz, Eickhoff, EMNLP 2023 · [aclanthology.org/2023.emnlp-main.665](https://aclanthology.org/2023.emnlp-main.665/)

Two items are reciprocal neighbours if each is in the other's top-k. rNN is "a much stronger
indicator of semantic similarity than simple distance", and they use it for
**evidence-based label smoothing** (assigning non-zero target probability to unlabelled documents
that resemble the ground truth) and for reranking. Also reports a sweet spot: ~60 candidates for
the neighbourhood analysis. **The reciprocal-NN idea is used here for label smoothing and
reranking — not as a negative-selection filter.** That gap is candidate C4.

### Negative samples filter for time series contrastive learning
Expert Systems with Applications, 2026 · [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417425036681)

**Cross-field, and directly on point for the difficulty question.** Filters negatives above a
hardness threshold. Finding: *"in the unsupervised case, some of the most difficult samples can
degrade classification performance, while in the supervised case, more difficult samples are
beneficial"* — and *"an optimal threshold exists for each dataset"*. Confirms the inverted-U
outside NLP, and explains why: with complete labels, harder is better; with incomplete labels,
the hardest are false negatives.

---

## Category 8 — Cross-signal, structural and domain-aware negatives

### Intent-DQD — "related but not duplicate" as hard negatives
Findings EMNLP 2023 · [aclanthology.org/2023.findings-emnlp.596](https://aclanthology.org/2023.findings-emnlp.596/)

**The most relevant paper to directions B/E/G for this task.** For CQA forums: `LinkTypeId = 3`
marks duplicates; `LinkTypeId = 1` marks questions that reference each other. They take
`LinkTypeId = 1` pairs as "hard neutrals" and note that negative intent pairs *"often exhibit
lexical similarity but possess distinct semantic meanings"*, so they serve as hard negatives.

**Implication:** "related but not duplicate" negatives are established for CQA — but they need the
StackExchange `PostLinks` data, which the AskUbuntu HuggingFace dataset does **not** include.

### EASE — entity-aware contrastive learning
Listed in the [ACM survey](https://doi.org/10.1145/3593590): uses Wikipedia entities to build
"hard negative entities". So entity-based negatives exist in sentence-embedding work.

### PassageBM25
(see Category 2) — uses a third signal: similarity to the positive.

### ECI — evaluating hard-negative quality without training
arXiv 2026 · [arXiv:2603.20990](https://arxiv.org/html/2603.20990v1)

Compares hard-negative sets mined by BM25, cross-encoders and LLMs on signal-vs-safety axes, and
finds **BM25 + cross-encoder hybrids** give the best trade-off. Recent, and it treats
BM25-vs-model as two *sources to combine* rather than two signals whose *disagreement* is the
variable.

---

## Cross-field imports (outside NLP)

| Field | Mechanism | Source | Can it become a 20–50 line text negative selector? |
|---|---|---|---|
| **Metric learning** | Distance-weighted sampling (sample uniformly by distance, not by hardness) | [Sampling Matters, ICCV 2017](https://arxiv.org/abs/1706.07567) | **Yes.** Rank candidates, sample with weights ∝ 1/q(d). ~15 lines. |
| **Metric learning** | Semi-hard selection (harder than easy, inside the margin, but not closer than the positive) | [FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf) | **Yes, trivially.** A threshold on `cos(a, n) < cos(a, p)`. |
| **Recommender systems** | Popularity-biased sampling is harmful; correct for it | [Negative Sampling in Recommendation: A Survey](https://doi.org/10.1145/3793855) | **Yes — this is candidate C3.** The text analogue of popularity is **hubness**: documents that are nearest neighbours of everything. Count how often a document appears in others' top-k; down-weight the top 1%. ~15 lines. |
| **Recommender systems** | SRNS: sample negatives that are high-scoring **and high-variance** to avoid false negatives | Ding et al., SIGIR 2020 (summarised in the survey above) | **Yes, but** it needs the score history across refreshes → ~40 lines and extra runs. This is candidate C8. |
| **Recommender systems** | DNS: pick the highest-scoring candidate from a random pool | Zhang et al. (survey above) | Already standard in IR under the name "hard negative mining". |
| **Graph learning** | MixGCF: synthesise hard negatives by injecting positive information into negatives | Huang et al., KDD 2021 | Equivalent to MoCHi's query-mixing; already in CV. |
| **Graph learning** | Debiased sampling via pseudo-labels | NDSCL and related | Needs a classifier — too heavy here. |
| **Active learning** | Query-by-committee / BALD: select points where models **disagree** most | Seung et al. 1992; Houlsby et al. 2011 | **Yes — this is the conceptual origin of candidate C1**, with the crucial difference that here "disagreement" is used to *select training negatives*, not to request labels. |
| **Self-supervised CV** | Anneal the negative distribution toward the positive over training | [Ring/Conditional NS](https://arxiv.org/abs/2010.02037) | Already covered as the curriculum direction. |
| **Time-series** | Filter negatives above a hardness threshold | [ESWA 2026](https://www.sciencedirect.com/science/article/abs/pii/S0957417425036681) | Yes — and it is the cleanest evidence for the inverted-U outside NLP. |

---

## Surveys — read these to orient, not to cite as evidence

* **Does Negative Sampling Matter? A Review with Insights into its Theory and Applications** —
  [arXiv:2402.17238](https://arxiv.org/abs/2402.17238). Taxonomy (static / hard / in-batch /
  hard-in-batch) and an interesting claim pulled from the literature: *"only 5% hardest negatives
  are necessary for high-accuracy contrastive learning."*
* **Negative Sampling in Recommendation: A Survey** —
  [doi:10.1145/3793855](https://doi.org/10.1145/3793855). Five categories: static, dynamic,
  adversarial generation, importance reweighting, knowledge-based.
* **Negative Sampling Techniques in Information Retrieval: A Survey** —
  [arXiv:2603.18005](https://arxiv.org/html/2603.18005v1). Tabulates DPR random in-batch → BM25
  static → ANCE dynamic → RocketQA denoised, and quantifies the cost (dynamic mining and denoising
  cost 3–5× training time; false-negative contamination costs 10–15% performance). **Note: this is
  a recent, unrefereed survey — verify any number against the primary paper before citing it.**
* **Contrastive Learning Models for Sentence Representations** —
  [doi:10.1145/3593590](https://doi.org/10.1145/3593590). The positive/negative-construction table
  for ~25 sentence-embedding models.

---

## Deliberately excluded

* **New loss functions.** The whole point is to hold the loss fixed. SimCSE, supervised
  contrastive loss, margin losses, RINCE, etc. are out of scope by construction.
* **LLM-generated negatives** (SyNeg, HNLMRec and the 2025–26 thread). Needs an LLM; out of scope
  per the project constraints.
* **Cross-encoder reranking.** A different stage of the pipeline.
* **Papers about *positive* pair construction.** Out of scope, though note that
  [AugSBERT](https://aclanthology.org/2021.naacl-main.28/) treats both.
* **Adversarial robustness and fairness work on negative sampling** — interesting, not useful here.
