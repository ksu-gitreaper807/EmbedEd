# Literature — the 7 papers you actually need

Not a survey. Every paper below is here because the project has a specific hole that only this
paper fills. If a paper is not on this list, you do not need it to finish.

**Total reading: ~7 essential papers, roughly 8–10 hours spread over weeks 1–2.**
Three are read properly; three are read for one table each; one is read for two paragraphs.

---

## Group 1 — Foundational (read properly, in this order)

### 1. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
Reimers & Gurevych, EMNLP-IJCNLP 2019 · [aclanthology.org/D19-1410](https://aclanthology.org/D19-1410/)

**Why you need it.** It is the paper that created the object you are fine-tuning: a
pretrained encoder plus mean pooling, producing a fixed vector you can compare with cosine.
It is also the paper that explains *why* you bother — finding the most similar pair among
10,000 sentences takes ~65 hours with cross-encoding BERT and ~5 seconds with SBERT.

**What to take away.**
* The bi-encoder architecture: two inputs, two independent forward passes, one shared encoder.
* Mean pooling over token embeddings is what turns per-token vectors into one sentence vector.
* Cosine similarity on those vectors is the entire retrieval mechanism.
* Read §3 (the training objectives: classification, regression, triplet) and §4. Skip §5's
  ablation details.

**Time:** 90 minutes. **Read:** §1, §3, §4.

---

### 2. Dense Passage Retrieval for Open-Domain Question Answering (DPR)
Karpukhin et al., EMNLP 2020 · [aclanthology.org/2020.emnlp-main.550](https://aclanthology.org/2020.emnlp-main.550/)

**Why you need it.** This is the paper your loss comes from. It is the canonical statement of
in-batch negatives, and it is the reason you have a BM25 baseline.

**What to take away.**
* The loss, in §3.2: `L = −log( exp(sim(q, d⁺)) / Σⱼ exp(sim(q, dⱼ)) )` over the batch. This is
  exactly `MultipleNegativesRankingLoss`. Read the two paragraphs around it until the
  denominator makes sense.
* **In-batch negatives are the trick**: with a batch of `N` pairs you get `N−1` negatives per
  question for free, and larger batches give strictly harder training. This is the direct
  justification for your batch-size ablation.
* BM25 as the thing to beat: DPR reports beating "a strong Lucene-BM25 system" by 9–19 points
  on top-20 accuracy — but note this is *in-domain* open QA, not zero-shot transfer.
* Skim §4 (experiments) for the shape of the results table you are going to produce.

**Time:** 75 minutes. **Read:** §2, §3.2, §4.1.

---

### 3. Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere
Wang & Isola, ICML 2020 · [proceedings.mlr.press/v119/wang20k.html](https://proceedings.mlr.press/v119/wang20k.html)

**Why you need it.** SBERT and DPR tell you *what* to compute. This paper tells you *what the
loss does to the geometry*, which is the sentence you will be asked in your viva: "so what
does contrastive learning actually change?"

**What to take away.**
* Two properties: **alignment** (positives end up close together) and **uniformity**
  (all embeddings spread evenly over the sphere rather than collapsing to a point).
* The contrastive loss optimises both at once. Uniformity is what stops the trivial solution
  "map everything to the same vector".
* §2 and §3 only. It is a vision-flavoured paper; the language experiments come later and you
  can skip them.

**Time:** 45 minutes (or skip if week 1 runs long — this is the one paper here you may defer).
**Read:** §1, §2, §3.

---

## Group 2 — Directly motivating (read for the finding, not the method)

### 4. TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning
Wang, Reimers & Gurevych, Findings EMNLP 2021 · [aclanthology.org/2021.findings-emnlp.59](https://aclanthology.org/2021.findings-emnlp.59/)

**Why you need it.** It is the cleanest published demonstration that *domain adaptation of
sentence embeddings works*, and it uses **AskUbuntu** as one of its four evaluation datasets
(alongside CQADupStack, TwitterPara and SciDocs — the [USEB benchmark](https://github.com/UKPLab/useb)).
So it is simultaneously motivation *and* your dataset's provenance.

**What to take away.**
* The headline number: TSDAE reaches **up to 93.1% of in-domain supervised performance**
  without any labelled data. That is the bar your *supervised* fine-tuning has to clear.
* Domain adaptation helps, but "93.1% of supervised" also means it does not beat supervised
  training on the target domain — which is precisely what you are testing.
* The shipped `useb` package contains the official AskUbuntu evaluation harness
  (`pip install useb`; `python -m useb.examples.eval_sbert_askubuntu`). **Optional:** run it
  with your zero-shot MiniLM in week 1 as an independent check that your own harness is sane.
  The repo's example asserts `avg == 47.6` for `bert-base-nli-mean-tokens`, which is a usable
  smoke test.

**Time:** 45 minutes. **Read:** §1, §3 (setup), §5 (results tables).

---

### 5. GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval
Wang, Yang, Li, Reimers, NAACL 2022 · [aclanthology.org/2022.naacl-main.168](https://aclanthology.org/2022.naacl-main.168/)

**Why you need it.** It quantifies the problem your project lives inside: dense retrievers
degrade under domain shift, and adapting them is worth something measurable. It also gives you
the *effect sizes* you should expect, so you can tell a real effect from noise.

**What to take away.**
* Reported gains of **up to +9.3 nDCG@10** on domain-specialised datasets — but note those are
  the *best* cases across several datasets, not the typical case.
* The realistic size of the effect: on CQADupStack the published progression is
  **29.6 (zero-shot) → 31.8 (TSDAE) → 34.5 (GPL) → 35.1 (TSDAE+GPL)** nDCG@10. That is a
  2–5 point range for serious domain-adaptation machinery. If your single fine-tuning stage
  moves Recall@10 by 2 points, **that is in line with the literature, not a failure.**
* Their setup is *unsupervised* (they generate pseudo-labels). Yours uses real human duplicate
  marks. Say this difference out loud in your related-work section — it is your justification
  for the project.

**Time:** 45 minutes. **Read:** §1, §4 (the method sketch), §5 tables.

---

### 6. BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models
Thakur, Reimers, Rücklé, Srivastava, Gurevych, NeurIPS 2021 Datasets & Benchmarks · [arXiv:2104.08663](https://arxiv.org/abs/2104.08663)

**Why you need it.** It is the reason BM25 is a baseline in your project instead of an
afterthought, and it is the reason you should not be surprised if your neural model loses.

**What to take away.**
* 18 datasets, 10 retrieval systems, zero-shot. Result: **"BM25 is a robust baseline"**; dense
  models are efficient but "often underperform other approaches" out of domain.
* Concretely: if BM25 beats your fine-tuned MiniLM, that is a documented phenomenon, not a
  bug. Write it up as pattern 4 in the spec.
* Also take the evaluation *discipline*: fixed corpus, fixed queries, fixed metrics, every
  system treated identically. That is what your `evaluate.py` must do.

**Time:** 45 minutes. **Read:** §1, §4, §5 (and Appendix tables if you want per-dataset numbers).

---

## Group 3 — The dataset and the task

### 7. Semi-supervised Question Retrieval with Gated Convolutions
Lei, Joshi, Barzilay, Jaakkola, Tymoshenko, Moschitti, Màrquez, NAACL 2016 · [aclanthology.org/N16-1153](https://aclanthology.org/N16-1153/)

**Why you need it.** It introduced the AskUbuntu duplicate-question retrieval task and the
splits you are using. It also contains the single most important fact in your whole report.

**What to take away.**
* The setup: 167,765 AskUbuntu questions with user-marked similar pairs; they manually
  annotated 8K question pairs for a clean dev/test evaluation, and report MAP, MRR and P@n.
  Their best model: **MRR 75.6%, P@1 62.0%**, about 8 points above a standard IR baseline.
* **The fact you must quote (§1, "noisy annotations"):** *"Our manual inspection of a sample set
  of questions from AskUbuntu shows that only 5% of similar pairs have been annotated by the
  users, with a precision of around 79%."* Your labels are ~95% incomplete and ~21% imprecise.
  Every negative claim in your report rests on this sentence, so cite it properly.
* Their model (gated convolutions) is **not** relevant to you — skip §3 and §4 entirely.

**Time:** 40 minutes. **Read:** §1, §2 (skim), §5 (dataset description), §6 (results).

---

## Reference only — look something up, do not read

| Paper | When you would open it |
|---|---|
| [Augmented SBERT (AugSBERT)](https://aclanthology.org/2021.naacl-main.28/), Thakur et al., NAACL 2021 | Only if you want to cite a paper about *how to choose pairs*. Its domain-adaptation section also uses AskUbuntu. Its pair-sampling ablation (Random / BM25 / Semantic Search / KDE) is the best available, but you are not running it. |
| [CQADupStack](https://eltimster.github.io/www/pubs/adcs2015.pdf), Hoogeveen, Verspoor & Baldwin, ADCS 2015 | Only to justify, in one sentence, why you did **not** use it (457K documents, 12 subforums, raw HTML posts) and chose AskUbuntu instead. |
| [Jiang et al., JSS 2023](https://doi.org/10.1016/j.jss.2023.111607) | Only if your neural system loses to BM25. It is the paper that says deep models do not reliably outperform traditional IR on software-engineering retrieval tasks, and it turns your disappointment into a citation. |

---

## What you explicitly do NOT need to read

Reading these will make you feel informed and cost you a week:

* SimCSE, SimCLR, MoCo, SwAV, BYOL — you are not studying contrastive learning as a field; you
  are using one loss. Paper 3 covers "what the loss does".
* ColBERT, Splade, late-interaction models — different architectures, out of scope by construction.
* ANCE, RocketQA, hard-negative mining literature — hard negatives are an optional extension you
  will probably not do.
* MTEB — a leaderboard paper. Useful for choosing a model, and `all-MiniLM-L6-v2` is already
  chosen for you.
* Anything published after your project starts. You are not doing a systematic review.

---

## Reading schedule

| Week | Papers | Time |
|---|---|---|
| Week 1, days 1–2 | 1 (SBERT), 7 (Lei et al.) | ~2.5 h |
| Week 1, days 3–4 | 2 (DPR) | ~1.5 h |
| Week 2, day 1 | 4 (TSDAE), 5 (GPL) | ~1.5 h |
| Week 2, day 2 | 6 (BEIR) | ~45 min |
| Any slack time | 3 (Wang & Isola) | ~45 min |

Papers 1, 2 and 7 are the ones you will actually cite in every section. Papers 4, 5 and 6 you
cite once each, in related work. Paper 3 you cite if you want a sentence about *why* the loss
works.

---

## Citation hygiene

Cite the **primary source**, not a blog post, not a GitHub README, not a survey that mentions
it. All seven have a DOI or an official anthology/PMLR page — use those URLs, which are what is
listed above. When you quote a number (93.1%, +9.3, 29.6→35.1, 75.6% MRR, "only 5%"), name the
paper and the section or table it came from. If you cannot find where a number came from, do
not use the number.
