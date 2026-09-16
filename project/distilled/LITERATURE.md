# Literature — the 8 papers you actually need

Not a survey. Every paper below fills a specific hole in *this* project. Two of them
(AugSBERT and Robinson et al.) are here **only** because the middle version manipulates
negative hardness — they were optional in the smaller plan and are load-bearing now.

**Total: 8 essential papers, ~10–12 hours across weeks 1–2.** Four read properly, four read for
one table or one figure each.

---

## Group 1 — Foundational (read properly, in this order)

### 1. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
Reimers & Gurevych, EMNLP-IJCNLP 2019 · [aclanthology.org/D19-1410](https://aclanthology.org/D19-1410/)

**Why.** It created the object you are fine-tuning: pretrained encoder + mean pooling → a fixed
vector comparable by cosine. It also explains why anyone bothers — finding the most similar pair
among 10,000 sentences takes ~65 hours with cross-encoded BERT and ~5 seconds with SBERT.

**Take away:** the bi-encoder architecture; mean pooling; cosine as the whole retrieval
mechanism. **Read** §1, §3, §4. **Time:** 90 min.

---

### 2. Dense Passage Retrieval for Open-Domain Question Answering (DPR)
Karpukhin et al., EMNLP 2020 · [aclanthology.org/2020.emnlp-main.550](https://aclanthology.org/2020.emnlp-main.550/)

**Why.** This is where your loss comes from, and it is the canonical statement of in-batch
negatives. It is also the paper your **N2** condition imitates: DPR's strong configuration uses
BM25 negatives.

**Take away:**
* §3.2: `L = −log( exp(sim(q,d⁺)) / Σⱼ exp(sim(q,dⱼ)) )` over the batch — exactly
  `MultipleNegativesRankingLoss`. Read until the denominator is obvious.
* In-batch negatives: `N` pairs give `N−1` negatives for free, and **larger batches mean harder
  training**. This is why batch size is fixed at 32 in your design — you are not allowed to let
  the number of negatives drift between conditions.
* BM25 negatives: DPR's ablations show that *which* negatives you use changes the result. That
  is the precedent for your independent variable.
* Their headline (beating BM25 by 9–19 points top-20 accuracy) is **in-domain** open QA. Note
  that; it does not license an expectation that you will beat BM25 zero-shot.

**Read** §2, §3.2, §4.1. **Time:** 75 min.

---

### 3. Augmented SBERT (AugSBERT)
Thakur, Reimers, Rücklé, Srivastava, Gurevych, NAACL 2021 · [aclanthology.org/2021.naacl-main.28](https://aclanthology.org/2021.naacl-main.28/)

**Why.** **Promoted from optional to essential.** This is the best available evidence that *how
you choose training examples* is a decisive variable in exactly this kind of fine-tuning. It
also uses AskUbuntu in its domain-adaptation experiments.

**Take away:**
* Its pair-sampling ablation — **Random vs. BM25 vs. Semantic Search vs. KDE** — is the direct
  precedent for your N1/N2/N3 design. Cite it as such.
* The finding that BM25 and semantic-search sampling beat random sampling is the literature
  basis for H1.
* Its domain-adaptation section shows gains up to +37 points on some tasks, which is the
  optimistic end of what domain adaptation can do — useful context for "how big an effect should
  I expect?".

**Read** §1, §3 (the sampling strategies), §5 (the ablation table). **Time:** 60 min.

---

### 4. Contrastive Learning with Hard Negative Samples
Robinson, Chuang, Sra, Jegelka, ICLR 2021 · [openreview.net/forum?id=CR1XOQ0UTh-](https://openreview.net/forum?id=CR1XOQ0UTh-)

**Why.** **New, and the theoretical core of the project.** It is the paper that says hardness is
a dial, not a switch — and that turning it to maximum is wrong.

**Take away:**
* The abstract's framing: *"We argue that … learning contrastive representations benefits from
  hard negative samples … The key challenge toward using hard negatives is that contrastive
  methods must remain unsupervised, making it infeasible to adopt existing negative sampling
  strategies that use label information."* Note the tension with your setting: **you do have
  partial labels** (duplicate groups), so you can exclude *known* duplicates but not unlabelled
  ones. That tension is your project.
* Hardness is **user-controllable**, and there is a limiting case that over-tightens. This is the
  license for your `hard_fraction` parameter and for hypothesis H0b (inverted-U).
* *"Requires only few additional lines of code to implement, and introduces no computational
  overhead"* — quote this when you justify that hardness manipulation is affordable in four
  weeks.
* Useful contrast: their method assumes no labels, so it has to *infer* which near neighbours
  are safe. You can partly circumvent that with duplicate marks — and where you cannot, you get
  the contamination that H0b predicts.

**Read** §1, §3, §4. **Time:** 60 min.

---

## Group 2 — The dataset and the domain-adaptation context

### 5. Semi-supervised Question Retrieval with Gated Convolutions
Lei, Joshi, Barzilay, Jaakkola, Tymoshenko, Moschitti, Màrquez, NAACL 2016 · [aclanthology.org/N16-1153](https://aclanthology.org/N16-1153/)

**Why.** It introduced the AskUbuntu duplicate-question task and the splits you use — and it
contains the single most important fact in your report.

**Take away:**
* Setup: 167,765 AskUbuntu questions with user-marked similar pairs; 8K manually annotated pairs
  for a clean dev/test evaluation. Their best model: **MRR 75.6%, P@1 62.0%**, ~8 points above a
  standard IR baseline.
* **The sentence you must quote (§1):** *"Our manual inspection of a sample set of questions from
  AskUbuntu shows that only 5% of similar pairs have been annotated by the users, with a
  precision of around 79%."* Every claim about false negatives in your report rests on this.
  Cite it precisely.
* Their model (gated convolutions) is irrelevant to you — skip §3 and §4.

**Read** §1, §5, §6. **Time:** 40 min.

---

### 6. TSDAE
Wang, Reimers, Gurevych, Findings EMNLP 2021 · [aclanthology.org/2021.findings-emnlp.59](https://aclanthology.org/2021.findings-emnlp.59/)

**Why.** The cleanest demonstration that domain adaptation of sentence embeddings works — and
**AskUbuntu is one of its four evaluation datasets** (the [USEB benchmark](https://github.com/UKPLab/useb),
with CQADupStack, TwitterPara and SciDocs). So it is motivation and dataset provenance at once.

**Take away:**
* Headline: TSDAE reaches **up to 93.1% of in-domain supervised performance** with no labelled
  data. That is the bar your *supervised* fine-tuning has to clear.
* "93.1% of supervised" also means adaptation does not beat in-domain supervision — which is
  exactly the comparison you are making.
* Practical: `pip install useb` ships the official AskUbuntu evaluation harness
  (`python -m useb.examples.eval_sbert_askubuntu`). **Optional week-1 check:** run it on your
  zero-shot MiniLM as an independent validation of your own harness. The repo's example asserts
  `avg == 47.6` for `bert-base-nli-mean-tokens`, which is a usable smoke test.

**Read** §1, §3, §5. **Time:** 45 min.

---

### 7. GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval
Wang, Yang, Li, Reimers, NAACL 2022 · [aclanthology.org/2022.naacl-main.168](https://aclanthology.org/2022.naacl-main.168/)

**Why.** It gives you the *effect sizes* to expect, so you can distinguish a real effect from
noise — the single most useful thing a paper can give a student about to run their first
experiment.

**Take away:**
* Reported **up to +9.3 nDCG@10** — but that is the best case across several datasets.
* The realistic scale: on CQADupStack the published progression is
  **29.6 (zero-shot) → 31.8 (TSDAE) → 34.5 (GPL) → 35.1 (TSDAE+GPL)** nDCG@10. A 2–5 point
  range for serious adaptation machinery. If your single fine-tuning stage moves Recall@10 by
  2 points, **that is in line with the literature, not a failure.**
* Theirs is *unsupervised* (generated pseudo-labels); yours uses real human duplicate marks. Say
  this difference out loud in related work — it is part of your justification.

**Read** §1, §4, §5. **Time:** 45 min.

---

### 8. BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models
Thakur, Reimers, Rücklé, Srivastava, Gurevych, NeurIPS 2021 Datasets & Benchmarks · [arXiv:2104.08663](https://arxiv.org/abs/2104.08663)

**Why.** The reason BM25 is a baseline rather than an afterthought, and the reason a null result
is interpretable.

**Take away:**
* 18 datasets, 10 retrieval systems, zero-shot: **"BM25 is a robust baseline"**; dense models are
  efficient but "often underperform other approaches" out of domain.
* If BM25 beats your fine-tuned model, that is documented, not broken.
* Take its evaluation *discipline*: fixed corpus, fixed queries, fixed metrics, identical
  treatment for every system. That is what your `evaluate.py` must do.

**Read** §1, §4, §5. **Time:** 45 min.

---

## Optional but recommended (if week 1 has slack)

### Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere
Wang & Isola, ICML 2020 · [proceedings.mlr.press/v119/wang20k.html](https://proceedings.mlr.press/v119/wang20k.html)

The paper that tells you *what the loss does to the geometry* — **alignment** (positives close)
and **uniformity** (embeddings spread over the sphere, which is what stops the trivial
"everything to one point" solution). Read §1–§3 (45 min) if you want a sentence for your viva
about why contrastive learning works at all. Skippable; nothing else depends on it.

---

## Reference only — look something up, do not read

| Paper | When you would open it |
|---|---|
| [CQADupStack](https://eltimster.github.io/www/pubs/adcs2015.pdf), Hoogeveen, Verspoor & Baldwin, ADCS 2015 | To justify, in one sentence, why you did **not** use it (457K docs, 12 subforums, raw HTML). |
| [ANCE](https://openreview.net/forum?id=zeFrfgyZln), Xiong et al., ICLR 2021 | Only if you implement the **optional** refresh of N3's neighbourhoods mid-training. ANCE is the asynchronous-refresh idea; your version is a single static re-mine. |
| [Jiang et al., JSS 2023](https://doi.org/10.1016/j.jss.2023.111607) | Only if BM25 beats everything. It is the paper that turns your disappointment into a citation. |
| [RocketQA](https://aclanthology.org/2021.naacl-main.466/) | Only if you attempt the optional denoising extension. Its cross-encoder filter is the standard fix for false negatives — and it is out of scope. |

---

## What you explicitly do NOT need to read

* SimCSE, SimCLR, MoCo, BYOL, SwAV — you are using one loss, not surveying a field.
* ColBERT, SPLADE, late interaction — different architectures, out of scope by construction.
* MTEB — a leaderboard paper; your model is already chosen.
* Cross-encoder reranking literature — optional extension at best.
* Supervised contrastive learning (Khosla et al.) — different label structure.
* Anything published after your project starts.

---

## Reading schedule

| When | Papers | Time |
|---|---|---|
| Week 1, days 1–2 | 1 (SBERT), 5 (Lei et al.) | ~2.5 h |
| Week 1, days 3–4 | 2 (DPR), 3 (AugSBERT) | ~2.25 h |
| Week 2, day 1 | 4 (Robinson et al.) | ~1 h |
| Week 2, day 2 | 6 (TSDAE), 7 (GPL) | ~1.5 h |
| Week 2, day 3 | 8 (BEIR) | ~45 min |
| Slack | Wang & Isola | ~45 min |

Papers **1, 2, 4, 5** you will cite in most sections. Papers **3, 6, 7, 8** you cite once or
twice each, mostly in related work and in interpreting your results.

---

## Citation hygiene

Cite the **primary source** — never a blog post, a README, or a survey that mentions it. All
eight have a DOI or an official venue page; use the URLs above. When you quote a number (93.1%,
+9.3, 29.6→35.1, 75.6% MRR, "only 5%"), name the paper and the section or table. If you cannot
find where a number came from, do not use the number.

One specific trap: the OpenReview ID for Robinson et al. is `CR1XOQ0UTh-`. Several secondary
sources list a different ID for it. Cite the OpenReview page directly, as above.
