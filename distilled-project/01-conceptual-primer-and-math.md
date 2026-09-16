# 01 — Understanding the field, and the only maths you need

**Audience note:** written for a 2nd-year CSE student. No prior NLP knowledge assumed. If a sentence uses a term you have not seen, it is defined the first time it appears.

---

## Part A — The conceptual chain, one box at a time

```text
Pretrained Encoder
       ↓
Text Representation
       ↓
Embedding Vector
       ↓
Similarity Function
       ↓
Contrastive Objective
       ↓
Positive / Negative Pairs
       ↓
Negative Sampling
       ↓
Domain Adaptation
       ↓
Retrieval / Similarity Evaluation
```

### 1. Pretrained Encoder

A neural network — in our case a small Transformer — that has already been trained by someone else on a very large text corpus using a **self-supervised** objective (typically "hide some words, predict them"). "Pretrained" means: do not start from random weights, start from these.

Two concrete facts that set up the whole project:

* Raw BERT-style encoders are **bad at producing comparable sentence vectors**. [Reimers & Gurevych (2019)](https://aclanthology.org/D19-1410/) measured averaged BERT embeddings at **54.81** Spearman correlation on semantic-textual-similarity benchmarks, and the `[CLS]` token output at **29.19** — **both worse than averaged GloVe word vectors**, which is a 2014-era method. That is the hole in the ground your project fills.
* The reason is geometric: masked-language-model pretraining leaves the sentence vectors crowded into a narrow cone (an **anisotropic** space), so unrelated sentences still have high cosine similarity. [Gao et al. (2021)](https://aclanthology.org/2021.emnlp-main.552/) showed that a contrastive objective *flattens* this cone; [Wang & Isola (2020)](https://mlanthology.org/icml/2020/wang2020icml-understanding/) proved formally that the contrastive loss optimises exactly two things — **alignment** (positives close) and **uniformity** (all vectors spread out).

**Which encoder:** `all-MiniLM-L6-v2` — 23M parameters, 384-dimensional output, Apache 2.0. Small enough that a full training run takes minutes on a free Colab GPU. Secondary: `BAAI/bge-small-en-v1.5` (33M, 384-d, MIT).

### 2. Text Representation → 3. Embedding Vector

The encoder reads a token sequence and produces one vector per token. To get **one vector for the whole text** you apply a **pooling** operation. The standard choice — and the SBERT default — is **mean pooling**: average the per-token vectors, weighted by the attention mask so padding is ignored.

```text
x  ──►  f_θ(x)  ──►  token vectors  ──(mean pool)──►  z ∈ R^384  ──(L2 norm)──►  ẑ
```

`ẑ` is the **embedding**: a fixed-length list of 384 floating-point numbers that stands in for the text. Two texts are "similar" if their embeddings point in nearly the same direction.

### 4. Similarity Function

Cosine similarity is the angle between two vectors:

```
sim(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

* If both vectors are L2-normalised (‖a‖ = ‖b‖ = 1) this is just the dot product `a · b`.
* Range: 1 = same direction, 0 = unrelated, −1 = opposite.

**Term-by-term, physically:** the numerator rewards dimensions where *both* vectors are large and positive; the denominator removes the effect of vector *length*, so only direction matters. Without the denominator, a long document would look similar to everything.

One literature detail: [Karpukhin et al. (2020)](https://arxiv.org/abs/2004.04906) tested dot product, cosine and L2 and found dot product ≈ L2, both **better than cosine**, for dual-encoder retrieval. In practice, `sentence-transformers` normalises by default and you get cosine/dot for free. The important thing is: **use the same similarity for every system you compare.**

### 5. Contrastive Objective (the heart of the project)

You want an embedding space where "these two texts are duplicates" means "these two vectors are close". You get it by showing the model a **positive** (should be close) and some **negatives** (should be far) and applying a loss that pushes in both directions at once.

The loss you will use is one object with four names: **InfoNCE** ([van den Oord et al., 2018](https://arxiv.org/abs/1807.03748)), **NT-Xent** ([SimCLR](https://mlanthology.org/icml/2020/chen2020icml-simple/)), **N-pair loss** ([Sohn, 2016](https://papers.nips.cc/paper_files/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html)), and **MultipleNegativesRankingLoss (MNRL)** in `sentence-transformers`. For an anchor `a`, one positive `p`, and negatives `n₁ … n_k`:

```
                          exp( sim(a, p) / τ )
L  =  −log  ─────────────────────────────────────────────
             exp( sim(a, p) / τ )  +  Σᵢ exp( sim(a, nᵢ) / τ )
```

**What each term physically causes the model to do:**

| Term | Physical effect |
|---|---|
| `sim(a, p)` in the numerator | As it grows, `L` shrinks. Gradient descent therefore **pulls the anchor and positive together**. |
| `Σᵢ exp(sim(a, nᵢ)/τ)` in the denominator | As any of these grows, `L` grows. Gradient descent therefore **pushes every negative away from the anchor**. |
| The `-log` | Turns "the positive's share of total similarity" into a loss that goes to 0 only when the positive dominates. Equivalent to a softmax classifier whose correct class is "the positive". |
| `τ` (**temperature**) | Scales the sharpness. **Small τ** → the model is punished heavily for *slightly* wrong rankings → it focuses on the hardest negatives. **Large τ** → softer, all negatives are treated more equally. SimCSE and SimCLR both ablate τ and both find it matters. |
| The number of negatives `k` | More negatives = a harder task = a stronger learning signal, but each extra negative costs memory. This is why batch size matters (see below). |

**Why MNRL is the right default here:** when negatives are just *the other positives in the mini-batch*, a batch of size `B` gives you `B − 1` negatives for free, with zero extra compute. That trick originates in [Henderson et al. (2017)](https://arxiv.org/abs/1705.00652) and is the standard in [DPR](https://arxiv.org/abs/2004.04906). Consequence: **your batch size is your number of negatives** — the single most important hyperparameter you have.

**Triplet loss (the alternative you should also implement, for one ablation):**

```
L = max(0,  d(a, p) − d(a, n) + margin)
```

`d` is Euclidean distance (or `1 − cosine`). The loss is zero as soon as the negative is farther than the positive **by at least `margin`**. Physically: it only cares about the *relative ordering* of one positive and one negative, ignoring all other items. That is why [Sohn (2016)](https://papers.nips.cc/paper_files/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html) argues N-pair/InfoNCE converges faster — it compares against `k` negatives at once.

### 6. Positive / Negative Pairs

* **Positive** = two texts that a human (or a naturally occurring signal) says are equivalent. In our dataset: two StackExchange questions the community marked as duplicates. **Never generate these synthetically** — a paraphrase model teaches paraphrase-invariance, not domain similarity.
* **Negative** = two texts that are *not* duplicates. The trap: "not marked duplicate" ≠ "not a duplicate". This is the **false-negative** problem and it is the centre of this project.

### 7. Negative Sampling

Two facts that make this the most consequential design choice in the project:

* **Most random negatives are useless.** [FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf) observed that most randomly-chosen triplets already satisfy the constraint and produce **zero gradient** — the model learns nothing from them. [ANCE](https://openreview.net/forum?id=zeFrfgyZln) proved it for dense retrieval: uninformative in-batch negatives give **diminishing gradient norms and large gradient variance**.
* **Hard negatives are contaminated.** A "hard negative" is a near-miss — a non-duplicate the model currently thinks *is* a duplicate. But the nearer a miss is, the likelier it is an *unlabelled duplicate* — a false negative. [RocketQA](https://aclanthology.org/2021.naacl-main.466/) solves this with **denoised hard negatives** (a cross-encoder filters suspicious candidates); [Robinson et al. (2021)](https://openreview.net/forum?id=CR1XOQ0UTh-) state the two governing principles: a useful negative must (1) genuinely *be* a negative and (2) currently be *believed* similar.

**Hard vs. semi-hard vs. ambiguous**, in the order you will sweep them:

| Kind | Definition | Origin |
|---|---|---|
| Random | uniform sample from the corpus | baseline |
| In-batch | the other positives in the mini-batch | [Henderson 2017](https://arxiv.org/abs/1705.00652), [DPR](https://arxiv.org/abs/2004.04906) |
| Lexical-hard (BM25) | high word overlap, not a labelled duplicate | [DPR](https://arxiv.org/abs/2004.04906), [AugSBERT](https://aclanthology.org/2021.naacl-main.28/) |
| Model-hard | top-k by the *current model's* embedding | [ANCE](https://openreview.net/forum?id=zeFrfgyZln) |
| Semi-hard | farther than the positive but still inside the margin | [FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf) |
| Ambiguous | in the score band near the positive's score | [SimANS](https://doi.org/10.18653/v1/2022.emnlp-industry.56) |
| Denoised hard | model-hard, filtered to remove likely false negatives | [RocketQA](https://aclanthology.org/2021.naacl-main.466/) |

### 8. Domain Adaptation

Take a general embedding model and make it good at *your* domain. Established results:

* Dense retrievers **degrade under domain shift** — this is the opening claim of [GPL](https://aclanthology.org/2022.naacl-main.168/) and the finding of [BEIR](https://arxiv.org/abs/2104.08663), where BM25 remains a robust zero-shot baseline.
* Domain-adaptive pretraining helps: [TSDAE](https://aclanthology.org/2021.findings-emnlp.59/) reaches up to **93.1% of in-domain supervised performance**; on CQADupStack specifically, published nDCG@10 goes **29.6 → 31.8** (TSDAE) → **34.5** (GPL) → **35.1** (TSDAE+GPL).
* Continued pretraining on in-domain text helps generally: [Gururangan et al. (2020)](https://aclanthology.org/2020.acl-main.740/).

**In this project "domain" = a StackExchange subforum.** Training on 11 subforums and testing on the 12th is a controlled, honest domain-shift experiment.

### 9. Retrieval / Similarity Evaluation

Two fundamentally different framings, and the choice is not cosmetic:

| Framing | Question | Typical metrics |
|---|---|---|
| **Classification / pair scoring** | "Are these two specific texts duplicates? yes/no" | Accuracy, F1, ROC-AUC, PR-AUC |
| **Retrieval (what you will use)** | "Here is a new question. Rank all 40,000 existing ones. Is a duplicate in the top 10?" | **nDCG@10**, Recall@k, MRR@10, MAP |

Use retrieval. Two reasons grounded in the literature, both from your candidate domain family: [Jiang et al. (2023)](https://doi.org/10.1016/j.jss.2023.111607) found that duplicate-detection models "achieve outstanding performance when evaluated in the classification scenario" while **losing to IR methods** in the ranking scenario; and [Zhang et al. (2023)](https://doi.org/10.1145/3576042) found that once the benchmark is bias-corrected, a *simple retrieval* method beats the deep-learning methods. Classification numbers on balanced pairs look great and mean little.

**Metrics, concretely:**

* **nDCG@10** — primary. Rewards getting relevant items *high* in the ranking, discounted by position, normalised to [0,1]. This is the [BEIR](https://arxiv.org/abs/2104.08663) / [MTEB](https://aclanthology.org/2023.eacl-main.148/) convention, so your numbers are comparable to published ones.
* **Recall@k** — "was at least one duplicate in the top k?" Directly matches user experience.
* **MRR@10** — average of 1/rank of the first relevant item. Rewards getting it *first*.
* **MAP** — average precision over the whole ranking.
* **Spearman** (only if you also build a graded-similarity set) — and **not Pearson**: [Reimers et al. (2016)](https://aclanthology.org/C16-1009/) showed Pearson ranking of STS systems can have *negative* predictiveness for the downstream task.

**One structural caveat you must handle:** [Reimers & Gurevych (2021)](https://aclanthology.org/2021.acl-short.77/) proved that dense retrieval degrades **faster than sparse retrieval as the index grows**, because a fixed-dimensional embedding admits more accidental near-matches in a bigger pool. Therefore: **fix the candidate pool size across every system you compare**, and report it.

---

## Part B — The only other maths you need

### Alignment and uniformity (your diagnostic)

From [Wang & Isola (2020)](https://mlanthology.org/icml/2020/wang2020icml-understanding/). For a model `f` and a distribution of positive pairs `p_pos`:

```
L_align   = E_(x,y)~p_pos [ ‖ f(x) − f(y) ‖² ]          ← lower is better
L_uniform = log E_(x,y)~iid [ exp( −2 ‖ f(x) − f(y) ‖² ) ]   ← lower is better
```

* `L_align` small ⇒ duplicates really are close together.
* `L_uniform` small ⇒ embeddings use the whole space instead of collapsing into a cone.
* **Physically:** the contrastive loss is (asymptotically, as negatives → ∞) minimising the sum of exactly these two. Computing them costs you ~10 lines and gives you a *mechanistic* story for why a configuration helped: "N5 improved alignment but degraded uniformity" is a much better sentence than "N5 scored higher".

### Number of negatives vs. batch size

With MNRL and no extra mined negatives: `k = B − 1`. Literature reference points:

| System | Batch size | Source |
|---|---|---|
| all-mpnet-base-v2 (production) | 1024 | [HF model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) |
| DPR | 128 | [Karpukhin et al.](https://arxiv.org/abs/2004.04906) |
| SimCLR | up to 4096 | [Chen et al.](https://mlanthology.org/icml/2020/chen2020icml-simple/) |
| DPR-style on one 16 GB GPU | 16 (with multiple positives) | [arXiv:2508.09534](https://arxiv.org/html/2508.09534) |

**You will not have a batch of 1024.** You will have ~64. That is fine and expected: compensate by adding *explicitly mined* hard negatives, which is exactly the variable you are studying. Say this explicitly in your paper — it shows you understand the trade-off rather than hiding it.

### Why "denoising" is a formula, not a vibe

A candidate `c` for anchor `a` with known positive `p` is dropped from the negative set if it is *too* similar. Concretely, `sentence-transformers`' `mine_hard_negatives` exposes:

* `absolute_margin`: keep `c` only if `sim(a,c) < sim(a,p) − δ`
* `relative_margin`: keep `c` only if `sim(a,c) < sim(a,p) · (1 − ρ)`
* `max_score`: hard cap on `sim(a,c)`
* `range_min`: skip the top-N most similar candidates entirely (they are the most likely false negatives)

Physically: these are all versions of Robinson et al.'s Principle 1 — *do not train the model to push away something that is probably a duplicate*.

---

## Part C — A 60-second version you can recite

> A pretrained encoder turns text into a vector, but on its own those vectors are poorly arranged for similarity. We fix that by fine-tuning the encoder with a contrastive loss: show it a genuine duplicate pair, show it `k` non-duplicates, and minimise a softmax-style loss that rewards the duplicate being the most similar item in the set. The loss pulls positives together and pushes negatives apart, with a temperature controlling how sharply the model is punished for near-misses. **Which negatives you choose turns out to matter more than anything else**, because random negatives produce almost no gradient while the hardest negatives are often duplicates that nobody labelled. We therefore compare five negative-selection strategies and measure how many "negatives" are actually unlabelled duplicates. We evaluate by retrieval — rank all existing questions for a new one and score with nDCG@10 — because that is how the system would actually be used.
