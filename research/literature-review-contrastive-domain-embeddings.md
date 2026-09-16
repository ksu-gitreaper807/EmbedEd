# Contrastive Learning for Domain-Specific Embeddings — A Research-Grade Literature Review

**Prepared:** 2026-09-16
**Purpose:** Decide whether a 1-month, second-year undergraduate project of the form *"fine-tune a small pretrained encoder with a contrastive objective so that its embedding space captures a domain-specific notion of similarity better than a general-purpose embedding model"* is well grounded, and if so, which papers, datasets, baselines, metrics and ablations it should rest on.

**Scope note (Project B).** This review deliberately covers only Project B from your brief: take an *existing* pretrained encoder and adapt it with contrastive learning. Project A (use an embedding model as-is) appears only as a baseline. Project C (train an encoder from scratch) is out of scope and is only referenced to justify why it is out of scope.

**Epistemic conventions used below.**
* Every substantive claim about a paper is linked to its primary source (ACL Anthology, PMLR/ICML, NeurIPS Proceedings, OpenReview/ICLR, ACM DL, IEEE, arXiv, or the authors' code).
* Where I could not verify a fact from the paper itself, it is marked **Not reported** or flagged as unverified. I have not filled gaps with plausible-sounding numbers.
* Hyperparameters are split into **literature-supported** (cited) and **proposed** (my suggestion, to be tuned by you).

---

## 1. Executive summary

**Is the project supported by existing research?** Yes — extremely well supported methodologically, but *not* in the sense you might hope. Every component you named (siamese/bi-encoder fine-tuning, contrastive/InfoNCE/MultipleNegativesRanking loss, triplet loss, hard-negative mining, retrieval-style evaluation) is textbook material with canonical, reproducible papers: [SBERT (Reimers & Gurevych, EMNLP 2019)](https://aclanthology.org/D19-1410/), [SimCSE (Gao et al., EMNLP 2021)](https://aclanthology.org/2021.emnlp-main.552/), [DPR (Karpukhin et al., EMNLP 2020)](https://arxiv.org/abs/2004.04906), [ANCE (Xiong et al., ICLR 2021)](https://openreview.net/forum?id=zeFrfgyZln), [TSDAE (Wang et al., EMNLP 2021)](https://aclanthology.org/2021.findings-emnlp.59/) and [GPL (Wang et al., NAACL 2022)](https://aclanthology.org/2022.naacl-main.168/). Domain adaptation of embeddings is itself an established subfield, not a gap.

**The three established techniques you will actually use.** (1) A *bi-encoder* (single encoder, mean pooling, cosine similarity) — [SBERT](https://aclanthology.org/D19-1410/). (2) An *InfoNCE-family loss with in-batch negatives* — variously called NT-Xent, N-pair loss, or MultipleNegativesRankingLoss ([Sohn, NeurIPS 2016](https://papers.nips.cc/paper_files/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html); [Henderson et al., 2017](https://arxiv.org/abs/1705.00652); [Chen et al., ICML 2020](https://mlanthology.org/icml/2020/chen2020icml-simple/)). (3) *Hard-negative mining*, usually with an ANN index refreshed during training ([ANCE](https://openreview.net/forum?id=zeFrfgyZln)) plus *de-noising* to remove false negatives ([RocketQA](https://aclanthology.org/2022.naacl-main.168/); [Qu et al., NAACL 2021](https://aclanthology.org/2021.naacl-main.466/)).

**Where the interesting opportunity actually lies.** Not in "can contrastive fine-tuning help" — it can, and that is not news. It lies in the fact that in the *specific* domain you proposed (duplicate bug reports), the published evidence is that the gains are **small, fragile, and sometimes negative**. Three independent, well-executed studies converge on this:
* [Zhang et al., TOSEM 2023](https://doi.org/10.1145/3576042) built a bias-corrected duplicate-bug-report benchmark and found that a simple retrieval baseline **beats** recently proposed deep-learning methods on most projects, and that reported progress partly rests on optimistic evaluation (old data, single issue tracker).
* [Jiang et al., J. Systems & Software 2023](https://doi.org/10.1016/j.jss.2023.111607) found that deep-learning methods do **not** beat IR-based methods in the ranking formulation, that a DL+IR hybrid improves MAP by a median of only 7.09–11.34%, and — critically for your framing — that **lexical similarity matters more than semantic similarity** for this task.
* [Rosane et al., SBES 2025](https://doi.org/10.5753/sbes.2025.9809) fine-tuned `all-mpnet-base-v2` on four large open-source bug datasets and found **only marginal improvements**, with several learning rates actively harming performance.

That is a *better* research situation for a one-month project than a guaranteed win would be. A carefully designed study that (a) uses strong, current baselines, (b) splits by duplicate group *and* by time, (c) isolates the effect of negative-sampling strategy, and (d) reports effect sizes over multiple seeds is a genuine, publishable-quality empirical contribution even — especially — if the headline result is "the gain is small and here is why".

**Is one month realistic?** Yes, with discipline, if you (i) pick a dataset that is already cleaned and has explicit duplicate links, (ii) use a ≤110M-parameter encoder, (iii) build BM25 + off-the-shelf-embedding baselines in week 2 *before* any training, and (iv) run a small number of pre-registered ablations. The compute is modest: on a single consumer/colab GPU (≥15 GB), a full training configuration on ~20–50k bug reports takes tens of minutes, so a 5-configuration × 3-seed grid is roughly 10–20 GPU-hours. What will sink the project is not compute; it is weak baselines, leaky splits, and an evaluation that does not match how duplicate detection is actually used (retrieval over historical reports, not balanced pair classification).

---

## 2. Research landscape

The chain you asked me to explain is a single technical story with five links. Here is how they connect, with the paper that establishes each link.

### 2.1 Pretrained LM → embedding model

A masked-LM such as BERT produces *contextual token vectors*, and naively pooling them yields a poor sentence embedding: SBERT reports that averaging BERT embeddings gives an average Spearman correlation of only 54.81 on STS tasks and that the `[CLS]` output gives only 29.19 — **both worse than averaged GloVe embeddings** ([Reimers & Gurevych, 2019](https://aclanthology.org/D19-1410/)). The fix is architectural *and* objective-based: add a pooling layer (mean pooling is the SBERT default), and fine-tune the whole encoder in a **siamese / bi-encoder** configuration so that cosine similarity between two independently-computed vectors becomes meaningful. This is what turns a pretrained LM into an embedding model, and it is the exact operation you will perform.

Two follow-ups matter conceptually. [SimCSE](https://aclanthology.org/2021.emnlp-main.552/) showed that even *unsupervised* contrastive fine-tuning (two dropout masks on the same sentence = a positive pair) lifts BERT-base to 76.3% average Spearman, and explained why: the contrastive objective **flattens the anisotropic (cone-shaped) embedding distribution** produced by MLM pretraining, improving *uniformity* while improving *alignment* of positives. That geometric framing is made explicit in [Wang & Isola, ICML 2020](https://mlanthology.org/icml/2020/wang2020icml-understanding/), who prove that, asymptotically, InfoNCE optimises exactly two quantities: alignment of positive pairs and uniformity of the normalised feature distribution. Read that paper before you write your methods section — it is the cleanest way to explain what your loss is doing.

### 2.2 Embedding model → contrastive learning

The loss you will use is one object wearing many names. Formally, for anchor $a$, positive $p$, and negatives $\{n_i\}$:

$$\mathcal{L} = -\log \frac{e^{\operatorname{sim}(a,p)/\tau}}{e^{\operatorname{sim}(a,p)/\tau} + \sum_i e^{\operatorname{sim}(a,n_i)/\tau}}$$

This is **InfoNCE** ([van den Oord et al., 2018](https://arxiv.org/abs/1807.03748), who frame it as a lower bound on mutual information), **NT-Xent** ([SimCLR](https://mlanthology.org/icml/2020/chen2020icml-simple/)), **N-pair loss** ([Sohn, NeurIPS 2016](https://papers.nips.cc/paper_files/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html)), and — when the negatives are simply the other positives in the mini-batch — **MultipleNegativesRankingLoss** in `sentence-transformers`, whose text-retrieval lineage runs through [Henderson et al. (2017)](https://arxiv.org/abs/1705.00652) and the dual-encoder retrieval line (cited in [DPR](https://arxiv.org/abs/2004.04906)). It directly generalises triplet loss: [Sohn](https://papers.nips.cc/paper_files/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html) notes the equivalence between triplet loss and the 2-pair case, and argues that comparing against $N-1$ negatives speeds convergence and improves the metric.

Two classical ancestors are worth knowing because they explain the *sampling* obsession: the **contrastive loss** of [Hadsell, Chopra & LeCun (CVPR 2006)](https://doi.org/10.1109/CVPR.2006.100) (and the discriminative similarity metric of Chopra et al., CVPR 2005), and the **triplet loss with semi-hard mining** of [FaceNet (Schroff et al., CVPR 2015)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf). FaceNet's argument is the one you should internalise: *most* randomly chosen triplets are already satisfied and produce zero gradient, so learning is driven almost entirely by which negatives you bother to show the model.

### 2.3 Contrastive learning → domain adaptation

There is a well-established result that matters for your framing: dense retrievers **degrade badly under domain shift**. [Thakur et al., NeurIPS 2021 (BEIR)](https://arxiv.org/abs/2104.08663) evaluated nine retrieval models zero-shot across 17 datasets and found BM25 to be a robust baseline, with dense models often underperforming out of domain. [GPL](https://aclanthology.org/2022.naacl-main.168/) opens by quantifying this and proposes Generative Pseudo-Labelling, gaining up to **+9.3 nDCG@10** on six domain-specialised datasets; [TSDAE](https://aclanthology.org/2021.findings-emnlp.59/) shows domain-adaptive pretraining reaching up to **93.1% of in-domain supervised performance**; [AugSBERT](https://aclanthology.org/2021.naacl-main.28/) reports gains of up to **+37 points** in a domain-adaptation setting. The related NLP-wide result is [Gururangan et al., ACL 2020](https://aclanthology.org/2020.acl-main.740/): a second phase of pretraining *in domain* (DAPT) helps, and further adapting on the task's unlabelled data (TAPT) helps again.

**Conclusion: "domain-specific contrastive embeddings" is established, not novel.** Anyone who tells you it is a gap is not reading the retrieval literature.

### 2.4 Contrastive learning → hard negatives

This is the technically richest part of the literature and the part most often done badly. The canonical findings:

* **Random / in-batch negatives are usually too easy.** [DPR](https://arxiv.org/abs/2004.04906) found the *type* of negative (random vs. BM25 vs. gold) barely mattered in a 1-of-$N$ setup at $k \ge 20$, but that switching to **in-batch** training helped substantially, and that adding **one** BM25 hard negative helped a lot while two did not.
* **In-batch negatives can be actively uninformative.** [ANCE](https://openreview.net/forum?id=zeFrfgyZln) proves this theoretically: uninformative local negatives yield **diminishing gradient norms and large gradient variance**, and fixes it by selecting global negatives from an asynchronously refreshed ANN index of the whole corpus.
* **Hard negatives are contaminated by false negatives.** [RocketQA](https://aclanthology.org/2021.naacl-main.466/) introduces *denoised* hard negatives (a cross-encoder filters candidates that are probably unlabelled positives) and cross-batch negatives; [SimANS](https://doi.org/10.18653/v1/2022.emnlp-industry.56) samples deliberately *ambiguous* negatives near the positive's score; [Chuang et al., NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html) give a principled *debiased* objective for exactly this sampling bias; [Robinson et al., ICLR 2021](https://openreview.net/forum?id=CR1XOQ0UTh-) formalise it as two principles — a useful negative must (1) genuinely be a negative and (2) currently be *believed* similar — and show hard sampling **with** debiasing beats hard sampling without it.

That second principle is why hard-negative mining on duplicate-bug-report data is genuinely risky: in an issue tracker, only a small fraction of true duplicates are ever *labelled* as duplicates. A "hard negative" retrieved by a decent model is disproportionately likely to be an unlabelled positive.

### 2.5 Contrastive learning → retrieval/similarity evaluation

Your final evaluation should be retrieval, not classification. There is strong literature support for this framing *in your exact domain*: [Zhang et al., TOSEM 2023](https://doi.org/10.1145/3576042) and [Jiang et al., JSS 2023](https://doi.org/10.1016/j.jss.2023.111607) both argue that the classification framing (balanced duplicate/non-duplicate pairs, accuracy/F1/AUC) is what produced the optimistic literature, and that the realistic formulation is: *given a new report, rank all historical reports and measure Recall@k / MAP / MRR*. Standard retrieval metrics (nDCG@10, Recall@100, MRR@10) are what [BEIR](https://arxiv.org/abs/2104.08663) and [MTEB](https://aclanthology.org/2023.eacl-main.148/) use.

Two evaluation caveats from the literature that belong in your threats-to-validity section:
* [Reimers, Beyer & Gurevych, COLING 2016](https://aclanthology.org/C16-1009/) show that ranking systems by **Pearson** correlation on STS can be actively misleading (in one task, predictiveness was *negative*, $\rho=-0.326$, versus $+0.504$ for nDCG) and recommend Spearman / ranking metrics chosen to match the downstream task. Use Spearman, not Pearson, if you report correlations at all.
* [Reimers & Gurevych, ACL 2021](https://aclanthology.org/2021.acl-short.77/) show that dense retrieval degrades faster than sparse retrieval **as index size grows**, because low-dimensional embeddings admit more false positives. This means your Recall@k numbers are a function of your candidate-pool size, and you must hold it fixed across all systems.

---

## 3. The 24 most relevant papers

Ordered by role, not by quality. "Area" uses your categories A–G. Reproducibility is my assessment for a *second-year undergraduate with ~1 month and one GPU*.

| # | Paper | Year | Area | Method (one line) | Dataset(s) | Key contribution | Relevance to your project | Repro | URL |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | 2019 | B | Siamese/triplet BERT + mean pooling; softmax / regression / triplet objectives | SNLI + MultiNLI; STSb, SentEval | Establishes the bi-encoder fine-tuning recipe you will copy; shows raw BERT pooling is *worse than GloVe* | Directly relevant | Easy | [ACL Anthology](https://aclanthology.org/D19-1410/) · [arXiv](https://arxiv.org/abs/1908.10084) · [code](https://github.com/UKPLab/sentence-transformers) |
| 2 | SimCSE: Simple Contrastive Learning of Sentence Embeddings | 2021 | B | Dropout as minimal augmentation; NLI entailment = positive, contradiction = **hard negative** | Wikipedia (1M), SNLI+MNLI; STS | Canonical contrastive recipe + alignment/uniformity explanation; clean ablation of hard negatives | Directly relevant | Easy | [ACL Anthology](https://aclanthology.org/2021.emnlp-main.552/) · [code](https://github.com/princeton-nlp/SimCSE) |
| 3 | FaceNet: A Unified Embedding for Face Recognition and Clustering | 2015 | A/D | Triplet loss + **semi-hard** online negative mining in large mini-batches | LFW, YouTube Faces | Origin of the semi-hard vs. hard distinction and of "mining strategy decides everything" | Methodologically relevant | N/A (CV) | [CVF PDF](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf) |
| 4 | Dimensionality Reduction by Learning an Invariant Mapping (DrLIM) | 2006 | A | The **contrastive loss** (pull neighbours, push non-neighbours past a margin) | MNIST variants | The loss function your "contrastive loss" option refers to | Background | N/A (CV) | [DOI 10.1109/CVPR.2006.100](https://doi.org/10.1109/CVPR.2006.100) |
| 5 | Improved Deep Metric Learning with Multi-class N-pair Loss Objective | 2016 | A/G | Generalises triplet to $N-1$ negatives; efficient batch construction | Fine-grained recognition, clustering, retrieval | **This is InfoNCE / MultipleNegativesRankingLoss.** Read it to know what your loss is | Methodologically relevant | N/A (CV) | [NeurIPS](https://papers.nips.cc/paper_files/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html) |
| 6 | Representation Learning with Contrastive Predictive Coding (InfoNCE) | 2018 | A | InfoNCE as a mutual-information lower bound | Speech, images, text, RL | Theoretical grounding of the loss + the temperature parameter | Background | N/A | [arXiv](https://arxiv.org/abs/1807.03748) |
| 7 | A Simple Framework for Contrastive Learning of Visual Representations (SimCLR) | 2020 | A | NT-Xent, projection head, big batches, strong augmentation | ImageNet | Canonical ablations: temperature, batch size, projection head | Methodologically relevant | N/A (CV) | [PMLR](https://mlanthology.org/icml/2020/chen2020icml-simple/) · [arXiv](https://arxiv.org/abs/2002.05709) |
| 8 | Understanding Contrastive Representation Learning through Alignment and Uniformity | 2020 | A | Decomposes contrastive loss into alignment + uniformity; optimisable metrics | STL-10, NYU-Depth, BookCorpus | The cleanest theory for *why* your loss works; gives you diagnostic metrics | Methodologically relevant | Moderate | [PMLR](https://mlanthology.org/icml/2020/wang2020icml-understanding/) |
| 9 | A Metric Learning Reality Check | 2020 | A (critical) | Re-audits metric-learning literature; recommends MAP@R | CUB200, Cars196, SOP | **Your methodological conscience**: unfair comparisons, test-set feedback, misleading metrics; with proper tuning most losses tie | Evaluation/methodology | Moderate | [ECCV PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf) · [code](https://github.com/KevinMusgrave/powerful-benchmarker) |
| 10 | Debiased Contrastive Learning | 2020 | D | Corrects the contrastive objective for **false negatives** sampled from the marginal | CIFAR/STL/ImageNet, sentence representation, RL | Formal treatment of the false-negative problem you will have | Methodologically relevant | Moderate | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html) |
| 11 | Contrastive Learning with Hard Negative Samples | 2021 | D | Hard sampling conditioned on being a *true* negative; interpolates marginal ↔ worst-case | Vision, graph, **text** | Principle 1 (true negative) + Principle 2 (currently believed similar). Explains why naive hardest-negative mining fails | Methodologically relevant | Moderate | [OpenReview](https://openreview.net/forum?id=CR1XOQ0UTh-) |
| 12 | Dense Passage Retrieval for Open-Domain QA (DPR) | 2020 | B/G | Dual encoder, dot product, **in-batch negatives + one BM25 hard negative** | NQ, TriviaQA, SQuAD, TREC, WQ | The reference ablation of negative types; also shows 1k training pairs already beat BM25 | Directly relevant | Easy | [arXiv](https://arxiv.org/abs/2004.04906) · [code](https://github.com/facebookresearch/DPR) |
| 13 | ANCE: Approximate Nearest Neighbor Negative Contrastive Learning | 2021 | D/G | Global hard negatives from an **asynchronously refreshed ANN index** | MS MARCO, NQ, TREC, commercial search | Theory for why in-batch negatives are uninformative; the standard mining architecture | Directly relevant | Moderate | [OpenReview](https://openreview.net/forum?id=zeFrfgyZln) · [PMLR](https://mlanthology.org/iclr/2021/xiong2021iclr-approximate/) · [code](https://github.com/microsoft/ANCE) |
| 14 | RocketQA: An Optimized Training Approach to Dense Passage Retrieval | 2021 | D/G | Cross-batch negatives + **denoised** hard negatives (cross-encoder filter) + augmentation | MS MARCO, NQ | The denoising idea: mined "hard negatives" are often unlabelled positives | Directly relevant | Difficult (multi-GPU) | [ACL Anthology](https://aclanthology.org/2021.naacl-main.466/) · [arXiv](https://arxiv.org/abs/2010.08191) · [code](https://github.com/PaddlePaddle/RocketQA) |
| 15 | SimANS: Simple Ambiguous Negatives Sampling | 2022 | D | Samples negatives in the ambiguity band near the positive's score | MS MARCO, NQ | A cheap, principled middle ground between random and hardest | Directly relevant | Moderate | [DOI](https://doi.org/10.18653/v1/2022.emnlp-industry.56) |
| 16 | TSDAE: Transformer-based Sequential Denoising Auto-Encoder | 2021 | C | Denoising autoencoder pretraining then supervised fine-tuning | AskUbuntu, CQADupStack, Twitter, SciDocs | **The cleanest demonstration of domain adaptation for sentence embeddings**, with a table comparing MLM / CT / SimCSE / TSDAE | Directly relevant | Easy | [ACL Anthology](https://aclanthology.org/2021.findings-emnlp.59/) · [arXiv](https://arxiv.org/abs/2104.06979) |
| 17 | GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval | 2022 | C/G | Synthetic query generation + cross-encoder pseudo-labelling + mined hard negatives | FiQA, SciFact, BioASQ, TREC-COVID, CQADupStack, Robust04 | +9.3 nDCG@10 from domain adaptation alone; shows hard-negative mining needs denoising | Directly relevant | Easy–Moderate | [ACL Anthology](https://aclanthology.org/2022.naacl-main.168/) · [code](https://github.com/UKPLab/gpl) |
| 18 | Augmented SBERT | 2021 | B/C | Cross-encoder labels unlabelled pairs (silver data); **pair-sampling strategy is decisive** | BWS, Quora-QP, SICK; AskUbuntu domain adaptation | The best available ablation of *how to choose pairs*: Random vs. BM25 vs. Semantic Search vs. KDE | Directly relevant | Easy | [ACL Anthology](https://aclanthology.org/2021.naacl-main.28/) |
| 19 | Don't Stop Pretraining: Adapt Language Models to Domains and Tasks | 2020 | C | Domain-adaptive then task-adaptive pretraining | Biomedical, CS papers, news, reviews | Establishes DAPT/TAPT; justifies a domain-pretraining stage | Domain/dataset relevant | Moderate | [ACL Anthology](https://aclanthology.org/2020.acl-main.740/) |
| 20 | SPECTER: Document-level Representation Learning using Citation-informed Transformers | 2020 | C/G | SciBERT init, **triplet margin loss**, positives = citations, **hard negatives = citations-of-citations** | SciDocs (7 doc-level tasks) | The single best template for your project: a naturally occurring similarity signal → triplet training → retrieval eval | Directly relevant | Easy | [ACL Anthology](https://aclanthology.org/2020.acl-main.207/) |
| 21 | Specialized Document Embeddings for Aspect-based Similarity of Research Papers | 2022 | C | Siamese-SciBERT + **MultipleNegativesRankingLoss**, one embedding per aspect | Papers with Code, 157,606 papers | Small-scale, GPU-feasible specialisation of a general encoder; finds generic embeddings are implicitly biased | Directly relevant | Easy | [DOI](https://doi.org/10.1145/3529372.3530912) · [arXiv](https://arxiv.org/abs/2203.14541) · [code](https://github.com/malteos/aspect-document-embeddings) |
| 22 | Duplicate Bug Report Detection: How Far Are We? | 2023 | E (critical) | Bias-corrected benchmark (age bias + issue-tracker bias); compares research vs. industry tools | Recent 3-year reports from Bugzilla, Jira, GitHub across 6 projects | **Mandatory reading.** Simple retrieval beats the DL methods; RR@5≈0.4–0.6; duplicates are 2.7–10% of reports | Domain/dataset relevant | Moderate | [ACM TOSEM](https://doi.org/10.1145/3576042) · [arXiv](https://arxiv.org/abs/2212.00548) |
| 23 | Does Deep Learning improve the performance of duplicate bug report detection? | 2023 | E (critical) | DL vs. IR in the *ranking* formulation; hybrid DL+IR | 3 projects, >1,000,000 bug reports | DL alone loses to IR; hybrid gains are modest; **lexical > semantic** similarity for this task | Domain/dataset relevant | Difficult (scale) | [DOI](https://doi.org/10.1016/j.jss.2023.111607) |
| 24 | Evaluating Fine-tuning Approaches for Duplicate Bug Report Detection | 2025 | E (critical) | Fine-tunes `all-mpnet-base-v2` for DBRD; LR sweep | Eclipse, OpenOffice, Firefox, NetBeans | **The closest published result to your proposed project: gains are marginal.** Read before you commit | Domain/dataset relevant | Easy | [SBES](https://doi.org/10.5753/sbes.2025.9809) |

### 3.1 Supporting papers (code + retrieval), referenced where used

| Paper | Year | Why it appears | URL |
|---|---|---|---|
| Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5) | 2022 | A strong modern general-purpose baseline; contrastive with in-batch negatives on CCPairs; first to beat BM25 on BEIR zero-shot without labelled data | [arXiv](https://arxiv.org/abs/2212.03533) |
| C-Pack / BGE | 2023 | Another strong open general embedding baseline | [arXiv](https://arxiv.org/abs/2309.07597) |
| Towards General Text Embeddings with Multi-stage Contrastive Learning (GTE) | 2023 | Same — candidate baseline | [arXiv](https://arxiv.org/abs/2308.03281) |
| BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of IR Models | 2021 | Retrieval benchmark, metric conventions, and the "BM25 is a strong baseline" warning | [arXiv](https://arxiv.org/abs/2104.08663) · [code](https://github.com/UKPLab/beir) |
| MTEB: Massive Text Embedding Benchmark | 2023 | 8 tasks / 58 datasets; finding that **no embedding method dominates** | [ACL Anthology](https://aclanthology.org/2023.eacl-main.148/) |
| The Curse of Dense Low-Dimensional IR for Large Index Sizes | 2021 | Why your Recall@k depends on corpus size; false positives in dense spaces | [ACL Anthology](https://aclanthology.org/2021.acl-short.77/) |
| Task-Oriented Intrinsic Evaluation of Semantic Textual Similarity | 2016 | Why Pearson is the wrong correlation to report | [ACL Anthology](https://aclanthology.org/C16-1009/) |
| Efficient Natural Language Response Suggestion for Smart Reply | 2017 | Origin of the in-batch-negatives dual-encoder loss in text | [arXiv](https://arxiv.org/abs/1705.00652) |
| Distance Metric Learning for Large Margin Nearest Neighbor Classification (LMNN) | 2009 | Classical metric learning; the margin instinct | [JMLR](https://mlanthology.org/jmlr/2009/weinberger2009jmlr-distance/) |
| Mitigating the Impact of False Negatives in Dense Retrieval with Contrastive Confidence Regularization | 2024 | Formalises "hard negatives ⇒ more false negatives" and proposes a filter | [arXiv](https://arxiv.org/abs/2401.00165) |
| Contrastive Code Representation Learning (ContraCode) | 2020 (arXiv) / EMNLP 2021 | Contrastive learning *for code*, positives via semantics-preserving compiler transforms | [ACL Anthology](https://aclanthology.org/2021.emnlp-main.482/) · [code](https://github.com/parasj/contracode) |
| CodeBERT | 2020 | Domain-pretrained encoder for code | [ACL Anthology](https://aclanthology.org/2020.findings-emnlp.139/) |
| UniXcoder | 2022 | Unified code encoder; uses multimodal contrastive learning | [ACL Anthology](https://aclanthology.org/2022.acl-long.499/) |
| Corder: Self-Supervised Contrastive Learning for Code Retrieval and Summarization | 2021 | Small-scale, GPU-feasible contrastive code retrieval | [DOI](https://doi.org/10.1145/3404835.3462840) |
| CoCoSoDa: Effective Contrastive Learning for Code Search | 2023 | Soft data augmentation + momentum negatives; clean component ablation | [arXiv](https://arxiv.org/abs/2204.03293) |
| CodeXGLUE | 2021 | Provides the BigCloneBench and POJ-104 splits and baselines | [arXiv](https://arxiv.org/abs/2102.04664) |
| Towards a Big Data Curated Benchmark of Inter-project Code Clones (BigCloneBench) | 2014 | The code-clone dataset used by CodeXGLUE | [DOI](https://doi.org/10.1109/ICSME.2014.77) · [code](https://github.com/clonebench/BigCloneBench) |
| CodeSearchNet Challenge | 2019 | ~6M functions across 6 languages with docstring "queries" | [arXiv](https://arxiv.org/abs/1909.09436) |
| Generating duplicate bug datasets (BugRepo) | 2014 | The classic Bugzilla duplicate-bug corpus | [DOI](https://doi.org/10.1145/2597073.2597128) · [Zenodo](https://doi.org/10.5281/zenodo.1246025) |
| CQADupStack | 2015 | StackExchange duplicate-question retrieval benchmark (also in BEIR) | [PDF](https://eltimster.github.io/www/pubs/adcs2015.pdf) |
| GitBugs | 2025 | 150k+ recent bug reports with duplicate mappings, ready splits | [arXiv](https://arxiv.org/abs/2504.09651) · [code](https://github.com/av9ash/gitbugs) |
| MMTEB: Massive Multilingual Text Embedding Benchmark | 2025 | Current-scale embedding benchmark; useful for choosing a baseline | [arXiv](https://arxiv.org/abs/2502.13595) |

---

## 4. Detailed extraction for the twelve papers that matter most

Fields use **Not reported** where the paper does not state the item. Abbreviations: MNRL = MultipleNegativesRankingLoss (InfoNCE with in-batch negatives).

### 4.1 Sentence-BERT (Reimers & Gurevych, EMNLP 2019)

| Field | Information |
|---|---|
| Paper | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks |
| Year | 2019 |
| Authors | Nils Reimers, Iryna Gurevych |
| Venue | EMNLP-IJCNLP 2019, pp. 3982–3992 · DOI 10.18653/v1/D19-1410 |
| URL | https://aclanthology.org/D19-1410/ · https://arxiv.org/abs/1908.10084 |
| Problem | BERT cannot be used for similarity search / clustering because cross-encoding every pair costs ~65 hours for 10,000 sentences |
| Method | Siamese and triplet network structures over BERT/RoBERTa; pooling (MEAN default, CLS, MAX); three objectives: classification (softmax over `softmax(W_t · [u; v; abs(u−v)])`), regression (cosine + MSE), triplet |
| Model | BERT-base / RoBERTa-base, mean pooling |
| Loss | Softmax classification / regression (MSE) / **triplet** (Euclidean distance, margin ε = 1) |
| Positive construction | SNLI + MultiNLI *entailment* pairs |
| Negative construction | NLI *contradiction* pairs; for triplet objective, negatives drawn so that `d(a,p) < d(a,n)` |
| Hard negatives | **Not reported** as an explicit mining stage — negatives come from the NLI label. This is a genuine gap you can improve on. |
| Dataset | SNLI (570k) + MultiNLI (430k); evaluated on STS benchmark, SentEval |
| Dataset size | ~1M sentence pairs for training; 7 STS tasks for evaluation |
| Evaluation | Spearman/Pearson correlation on STS; SentEval transfer accuracy. Also reports inference speed (65 h → ~5 s) |
| Baselines | Avg. GloVe, InferSent, Universal Sentence Encoder, raw BERT (avg. 54.81 / CLS 29.19 Spearman) |
| Main finding | Siamese fine-tuning on NLI yields SOTA sentence embeddings; **raw BERT embeddings are worse than averaged GloVe** for cosine similarity |
| Code | Yes — https://github.com/UKPLab/sentence-transformers |
| Reproducibility | **Easy** (one GPU, hours) |
| Relevance | This is the architecture and the training loop you will use. Cite it as your base method. |

### 4.2 SimCSE (Gao, Yao, Chen, EMNLP 2021)

| Field | Information |
|---|---|
| Paper | SimCSE: Simple Contrastive Learning of Sentence Embeddings |
| Year | 2021 |
| Authors | Tianyu Gao, Xingcheng Yao, Danqi Chen |
| Venue | EMNLP 2021, pp. 6894–6910 · DOI 10.18653/v1/2021.emnlp-main.552 |
| URL | https://aclanthology.org/2021.emnlp-main.552/ · https://arxiv.org/abs/2104.08821 |
| Problem | How to build sentence embeddings with (a) no labels, (b) labelled NLI data |
| Method | Unsupervised: feed the same sentence twice with independent dropout masks → positive pair. Supervised: NLI pairs folded into the contrastive framework |
| Model | BERT-base / RoBERTa-base, `[CLS]` representation (MLP head at training, discarded at inference) |
| Loss | InfoNCE / NT-Xent with **in-batch negatives**; temperature reported and ablated |
| Positive construction | (unsup) the same sentence under two dropout masks; (sup) NLI **entailment** pairs |
| Negative construction | Other sentences in the batch (in-batch negatives) |
| Hard negatives | **Yes, in the supervised variant:** NLI **contradiction** pairs are explicitly used as hard negatives |
| Dataset | 1M sentences from English Wikipedia (unsup); SNLI (570k) + MNLI (430k) (sup); 7 STS tasks |
| Dataset size | 1M unlabelled / ~1M labelled pairs |
| Evaluation | Spearman correlation on 7 STS tasks (unsup BERT-base 76.3%, sup 81.6%); alignment/uniformity diagnostics; ablation over dropout rate, temperature, pooling, and hard negatives |
| Baselines | Previous unsupervised methods (BERT-flow, whitening), SBERT, RoBERTa variants |
| Main finding | Dropout alone is a sufficient minimal augmentation (+4.2% over previous best unsup); supervised hard negatives add +2.2%; contrastive learning flattens the anisotropic embedding space |
| Code | Yes — https://github.com/princeton-nlp/SimCSE |
| Reproducibility | **Easy** |
| Relevance | Your conceptual template: unsupervised vs. supervised contrastive, temperature, hard negatives, alignment/uniformity diagnostics. |

### 4.3 Dense Passage Retrieval (Karpukhin et al., EMNLP 2020)

| Field | Information |
|---|---|
| Paper | Dense Passage Retrieval for Open-Domain Question Answering |
| Year | 2020 |
| Authors | Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih |
| Venue | EMNLP 2020 · arXiv:2004.04906 |
| URL | https://arxiv.org/abs/2004.04906 |
| Problem | Passage retrieval for open-domain QA using only dense representations |
| Method | Two independent BERT encoders (question, passage), `[CLS]` output, dot-product similarity |
| Model | BERT-base, 768-d |
| Loss | NLL of the positive passage (InfoNCE) over a set of negatives |
| Positive construction | Gold (question, passage) pairs from QA datasets |
| Negative construction | Three types compared: **Random**, **BM25** (high lexical overlap, no answer string), **Gold** (positives of other questions) |
| Hard negatives | **Yes — the key ablation.** In-batch negatives (batch 128) plus **one** additional BM25 negative: "adding a single BM25 negative passage improves the result substantially while adding two does not help further" |
| Dataset | NQ, TriviaQA, SQuAD, TREC, WebQuestions |
| Dataset size | e.g. NQ 58,880 train pairs; passage corpora of millions |
| Evaluation | Top-k retrieval accuracy (top-20/top-100), plus end-to-end QA EM |
| Baselines | BM25, ORQA, previous dense retrievers |
| Main finding | 9–19% absolute top-20 accuracy over BM25; **"a dense passage retriever trained using only 1,000 examples already outperforms BM25"**; dot product ≈ L2 > cosine |
| Code | Yes — https://github.com/facebookresearch/DPR |
| Reproducibility | **Easy–Moderate** (multi-GPU in the paper; smaller scale fine) |
| Relevance | The canonical negative-type ablation, and the data-efficiency result that justifies a small-scale project. Also: use dot product, not cosine. |

### 4.4 ANCE (Xiong et al., ICLR 2021)

| Field | Information |
|---|---|
| Paper | Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval |
| Year | 2021 |
| Authors | Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, Arnold Overwijk |
| Venue | ICLR 2021 |
| URL | https://openreview.net/forum?id=zeFrfgyZln |
| Problem | Dense retrieval underperforms sparse retrieval because of poor negatives |
| Method | Asynchronously updated ANN index over the *whole corpus*; an "Inferencer" re-encodes documents from a recent checkpoint and refreshes the index while the "Trainer" keeps training |
| Model | RoBERTa-base dual encoder, 768-d |
| Loss | ANCE = NCE-style contrastive loss with globally selected negatives |
| Positive construction | Dataset relevance judgements |
| Negative construction | Top-ranked documents retrieved by the current model from the ANN index |
| Hard negatives | **Yes — the central contribution.** Theory: local/in-batch negatives are uninformative, yield **diminishing gradient norms and large gradient variance**; ANCE negatives have much bigger gradient norms and reduce variance |
| Dataset | MS MARCO Passage/Document, NQ, TREC DL, plus a commercial search engine log |
| Dataset size | MS MARCO: 8.8M passages; ~500k training queries |
| Evaluation | MRR@10, NDCG@10, Recall@100, Coverage@20/100 |
| Baselines | Rand Neg, NCE Neg, BM25 Neg, DPR (BM25+Rand), BM25→Rand, BM25→NCE |
| Main finding | ANCE substantially improves over DPR and BM25-negatives baselines; dot-product retrieval nearly matches a BERT cascade IR pipeline |
| Code | Yes — https://github.com/microsoft/ANCE |
| Reproducibility | **Moderate–Difficult** (the paper reports ~1–2 h per ANCE epoch and 10 epochs to converge; LAMB optimizer, lr 5e-6) |
| Relevance | The theory for *why* your negative sampling matters, and the reference architecture for mining. For a 1-month project, do a **cheap static version**: mine once (or twice) per epoch with a FAISS index instead of asynchronous refresh. |

### 4.5 RocketQA (Qu et al., NAACL 2021)

| Field | Information |
|---|---|
| Paper | RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering |
| Year | 2021 |
| Authors | Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, Haifeng Wang |
| Venue | NAACL 2021, pp. 5835–5847 · DOI 10.18653/v1/2021.naacl-main.466 |
| URL | https://aclanthology.org/2021.naacl-main.466/ · https://arxiv.org/abs/2010.08191 |
| Problem | Three obstacles to training dual encoders: train/inference discrepancy, **unlabelled positives**, limited training data |
| Method | (1) cross-batch negatives, (2) **denoised hard negatives**, (3) data augmentation via cross-encoder pseudo-labelling |
| Model | ERNIE / BERT dual encoder + cross-encoder teacher |
| Loss | Ranking loss over positives, cross-batch negatives and denoised hard negatives |
| Positive construction | Labelled (query, passage) pairs |
| Negative construction | Top-K retrieved passages, then a **cross-encoder filters out likely false negatives** |
| Hard negatives | **Yes, and explicitly denoised.** This is the paper's most transferable idea for you |
| Dataset | MS MARCO Passage Ranking, Natural Questions |
| Dataset size | MS MARCO: 8.8M passages, ~500k training queries |
| Evaluation | MRR@10, Recall@50 (MS MARCO); top-20/top-100 (NQ) |
| Baselines | BM25, DPR, ANCE |
| Main finding | Cross-batch negatives, denoised hard negatives and augmentation each help; the combination beats prior SOTA on both datasets |
| Code | Yes — https://github.com/PaddlePaddle/RocketQA (scripts construct each training stage) |
| Reproducibility | **Difficult as published** (multi-GPU, cross-encoder over top-K for every query). **Adaptable**: run the denoising step on a subsample. |
| Relevance | Gives you the *false-negative filter* idea, which is the single most valuable methodological import for bug-report data where duplicate labels are sparse. |

### 4.6 TSDAE (Wang, Reimers, Gurevych, Findings EMNLP 2021)

| Field | Information |
|---|---|
| Paper | TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning |
| Year | 2021 |
| Authors | Kexin Wang, Nils Reimers, Iryna Gurevych |
| Venue | Findings of EMNLP 2021, pp. 671–688 · DOI 10.18653/v1/2021.findings-emnlp.59 |
| URL | https://aclanthology.org/2021.findings-emnlp.59/ · https://arxiv.org/abs/2104.06979 |
| Problem | Sentence embeddings need labels; most domains have none |
| Method | Encode a **corrupted** sentence (deletion-based noise, ratio 0.6) to a fixed-size vector, reconstruct the original with a decoder whose cross-attention keys/values are the sentence embedding only; discard the decoder at inference |
| Model | BERT/RoBERTa encoder + tied decoder; `[CLS]` as sentence representation |
| Loss | Token-level cross-entropy reconstruction loss (**not** contrastive) |
| Positive construction | N/A — unsupervised reconstruction. In the *domain-adaptation* setting, TSDAE pretraining is followed by supervised fine-tuning on an existing labelled set |
| Negative construction | N/A |
| Hard negatives | **Not reported** |
| Dataset | Domain adaptation: AskUbuntu, CQADupStack, Twitter, SciDocs. Retrieval: FiQA, SciFact, BioASQ, TREC-COVID, CQADupStack, Robust04 |
| Dataset size | Datasets are small by retrieval standards (thousands of documents) |
| Evaluation | Retrieval (nDCG, MAP, etc.) per the sentence-transformers domain-adaptation tables |
| Baselines | Zero-shot model, MLM, CT, SimCSE, ICT, CD |
| Main finding | TSDAE reaches **up to 93.1% of in-domain supervised performance**, outperforms MLM as a domain-adaptation method (e.g. retrieval avg. 45.2 → 49.2 nDCG@10 where MLM gives 46.7 and SimCSE 45.0); also beats prior unsupervised methods by up to 6.4 points |
| Code | Yes — https://github.com/UKPLab/sentence-transformers |
| Reproducibility | **Easy** (single GPU). The sentence-transformers docs reproduce the domain-adaptation table. |
| Relevance | Directly answers "does domain-specific training beat a general embedding?" for sentence embeddings — with **honest, modest effect sizes**. This is your calibration for what to expect. |

### 4.7 GPL (Wang, Thakur, Reimers, Gurevych, NAACL 2022)

| Field | Information |
|---|---|
| Paper | GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval |
| Year | 2022 |
| Authors | Kexin Wang, Nandan Thakur, Nils Reimers, Iryna Gurevych |
| Venue | NAACL 2022, pp. 2345–2360 · DOI 10.18653/v1/2022.naacl-main.168 |
| URL | https://aclanthology.org/2022.naacl-main.168/ |
| Problem | Dense retrievers degrade under domain shift; labelled data is unavailable in the target domain |
| Method | T5 generates synthetic queries for target-corpus passages → retrieve 50 candidate negatives per query → cross-encoder pseudo-labels the triples → train the bi-encoder on MarginMSE |
| Model | bi-encoder student + cross-encoder teacher |
| Loss | MarginMSE distillation from the cross-encoder |
| Positive construction | **Synthetic** (generated query, source passage) pairs — note: *artificial*, unlike your naturally occurring duplicates |
| Negative construction | 50 retrieved candidates per query, pseudo-labelled |
| Hard negatives | **Yes**, mined and then denoised by cross-encoder pseudo-labels: "training with mined hard negatives is possible as the pseudo labels performs efficient denoising" |
| Dataset | FiQA, SciFact, BioASQ, TREC-COVID, CQADupStack, Robust04 |
| Dataset size | Six domain-specialised BEIR datasets (corpora of thousands to hundreds of thousands of passages) |
| Evaluation | nDCG@10 |
| Baselines | Zero-shot MS MARCO retrievers, QGen, TSDAE |
| Main finding | Up to **+9.3 nDCG@10** over an out-of-the-box SOTA dense retriever; TSDAE+GPL adds a further +1.4 average |
| Code | Yes — https://github.com/UKPLab/gpl |
| Reproducibility | **Easy–Moderate** on BEIR-sized data |
| Relevance | The strongest published demonstration of domain-adaptation gains for retrieval. Contrast with your setting: you *do* have natural positives, so you should not need synthetic queries. |

### 4.8 Augmented SBERT (Thakur et al., NAACL 2021)

| Field | Information |
|---|---|
| Paper | Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks |
| Year | 2021 |
| Authors | Nandan Thakur, Nils Reimers, Johannes Daxenberger, Iryna Gurevych |
| Venue | NAACL 2021, pp. 296–310 · DOI 10.18653/v1/2021.naacl-main.28 |
| URL | https://aclanthology.org/2021.naacl-main.28/ |
| Problem | Bi-encoders need lots of target-task data; cross-encoders are accurate but slow |
| Method | Fine-tune a cross-encoder on gold data → soft-label a large set of *unlabelled pairs* (silver data) → train the bi-encoder on gold + silver |
| Model | BERT cross-encoder (teacher) → SBERT bi-encoder (student) |
| Loss | **Not reported** as a contrastive loss — regression/classification objectives over pairs |
| Positive construction | Gold pairs; silver pairs selected by a sampling strategy |
| Negative construction | **This is the paper's central contribution**: sampling strategy for unlabelled pairs — **Random**, **BM25**, **Semantic Search (SS)**, **Kernel Density Estimation (KDE)**. The authors note random pairing creates overwhelming class imbalance with easy negatives |
| Hard negatives | **Yes, via sampling strategy** (BM25 sampling = high lexical overlap = hard negatives) |
| Dataset | BWS, Quora-QP, SICK; domain adaptation to AskUbuntu |
| Dataset size | **Not reported** as a single number; task-dependent (tens of thousands of pairs) |
| Evaluation | Task-specific (Spearman / accuracy); in-domain and domain-adaptation |
| Baselines | Original bi-encoder, cross-encoder (upper bound), off-the-shelf USE |
| Main finding | +1 to +6 points in-domain; **up to +37 points for domain adaptation**; pair selection is "non-trivial and crucial" |
| Code | Yes — in sentence-transformers; see the docs' data-augmentation section |
| Reproducibility | **Easy** |
| Relevance | The best available ablation of *pair/negative selection strategy* for pairwise sentence scoring — directly adaptable to bug reports (BM25-sampled hard negatives being the obvious analogue of "same component, different bug"). |

### 4.9 SPECTER (Cohan et al., ACL 2020)

| Field | Information |
|---|---|
| Paper | SPECTER: Document-level Representation Learning using Citation-informed Transformers |
| Year | 2020 |
| Authors | Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, Daniel Weld |
| Venue | ACL 2020, pp. 2270–2282 · DOI 10.18653/v1/2020.acl-main.207 |
| URL | https://aclanthology.org/2020.acl-main.207/ |
| Problem | Document-level embeddings for scientific papers; no task-specific fine-tuning wanted |
| Method | Pretrain a Transformer on the **citation graph**: triplet loss with citations as positives |
| Model | Initialised from **SciBERT**; `[CLS]` representation of title+abstract |
| Loss | **Triplet margin loss** |
| Positive construction | Papers cited by the query paper (natural!) |
| Negative construction | Random papers; **plus "citations of citations": papers cited by the positive but not by the query** |
| Hard negatives | **Yes — citations-of-citations, and the ablation shows they are essential.** This is the cleanest hard-negative construction in the literature |
| Dataset | Full-text scientific corpus with citation graph; SciDocs (7 document-level tasks) |
| Dataset size | **Not reported** as a single figure in the abstract; the training corpus is on the order of hundreds of thousands of papers |
| Evaluation | SciDocs: citation prediction, document classification, recommendation (MAP, Precision@1, nDCG) |
| Baselines | TF-IDF, BM25, Doc2Vec, SPECTER-init, SciBERT, sentence-BERT variants |
| Main finding | Outperforms competitive baselines on SciDocs without task-specific fine-tuning; hard negatives are necessary for the effect |
| Code | Yes — https://github.com/allenai/specter |
| Reproducibility | **Easy–Moderate** |
| Relevance | **Your structural template.** Natural similarity signal → triplet/contrastive training with principled hard negatives → retrieval evaluation. Bug reports give you "duplicate-of" links exactly as citations give SPECTER its positives. |

### 4.10 Specialized Document Embeddings for Aspect-based Similarity (Ostendorff et al., JCDL 2022)

| Field | Information |
|---|---|
| Paper | Specialized Document Embeddings for Aspect-based Similarity of Research Papers |
| Year | 2022 |
| Authors | Malte Ostendorff, Till Blume, Terry Ruas, Bela Gipp, Georg Rehm |
| Venue | JCDL 2022 · DOI 10.1145/3529372.3530912 |
| URL | https://doi.org/10.1145/3529372.3530912 · https://arxiv.org/abs/2203.14541 |
| Problem | A single generic embedding gives one view of similarity; different *aspects* of a document are similar for different reasons |
| Method | Compare retrofitting, fine-tuning and **Siamese networks**; represent each document by multiple specialised embeddings |
| Model | **Siamese SciBERT** (best performing) |
| Loss | **MultipleNegativesRankingLoss** |
| Positive construction | Papers sharing an aspect label (task / method / dataset) from Papers with Code |
| Negative construction | In-batch negatives; aspect-specific training defines the negatives implicitly |
| Hard negatives | **Not reported** |
| Dataset | Papers with Code corpus |
| Dataset size | **157,606 research papers** |
| Evaluation | MAP + Precision/Recall in a k-NN recommendation setting |
| Baselines | 3 generic embeddings (FastText, SciBERT, SPECTER), 6 specialised embeddings, 1 pairwise classification baseline |
| Main finding | Siamese SciBERT scores highest; **generic embeddings are implicitly biased towards the *dataset* aspect and against the *method* aspect** |
| Code | Yes — https://github.com/malteos/aspect-document-embeddings (MIT), models on HF |
| Reproducibility | **Easy** (single-GPU scale) |
| Relevance | A one-month-scale, GPU-feasible example of "specialise a general encoder with MNRL and measure what the generic embedding gets wrong". The "which aspect does the generic embedding bias towards?" question has a direct bug-report analogue. |

### 4.11 Duplicate Bug Report Detection: How Far Are We? (Zhang et al., TOSEM 2023)

| Field | Information |
|---|---|
| Paper | Duplicate Bug Report Detection: How Far Are We? |
| Year | 2023 |
| Authors | Ting Zhang, DongGyun Han, Venkatesh Vinayakarao, Ivana Clairine Irsan, Bowen Xu, Ferdian Thung, David Lo, Lingxiao Jiang |
| Venue | ACM TOSEM 32(4) · DOI 10.1145/3576042 |
| URL | https://doi.org/10.1145/3576042 · https://arxiv.org/abs/2212.00548 |
| Problem | DBRD techniques are compared unfairly; unclear what actually works |
| Method | Empirical study. Identifies two biases (**data age**, **issue-tracking system**), builds a bias-corrected benchmark, re-evaluates research tools and compares to industrial tools (Mozilla's FTS, VSCodeBot) |
| Model | Evaluates REP, Siamese Pair, SABD, DC-CNN, HINDBR |
| Loss | N/A (benchmark paper) |
| Positive construction | Duplicate links extracted from the trackers |
| Negative construction | Non-duplicate reports in the corpus |
| Hard negatives | **Not reported** |
| Dataset | Recent 3-year bug reports from Bugzilla, Jira and GitHub, six projects |
| Dataset size | e.g. Eclipse 3,342 sampled pairs / Mozilla 68,396; VSCode corpus 62,092 reports (11,808 got duplicate recommendations) |
| Evaluation | Recall Rate@k (RR@k), MAP; statistical tests (p-values, effect sizes) |
| Baselines | Research tools vs. each other vs. **industrial tools actually deployed** |
| Main finding | **A simple retrieval-based approach (REP) beats recently proposed deep-learning methods on most projects**, and a simple technique already used in industry matches a recent research tool. Best RR@5 ≈ **0.4–0.6**. Only **2.7–10%** of bug reports are duplicates |
| Code | Yes — replication package released (data + code) |
| Reproducibility | **Moderate** (large corpora, but the replication package exists) |
| Relevance | **Sets the bar you must clear and warns you about evaluation bias.** Also gives you realistic numbers to sanity-check your own: if you report RR@10 = 0.95, you have a leak. |

### 4.12 Evaluating Fine-tuning Approaches for Duplicate Bug Report Detection (Rosane et al., SBES 2025)

| Field | Information |
|---|---|
| Paper | Evaluating Fine-tuning Approaches for Duplicate Bug Report Detection |
| Year | 2025 |
| Authors | Luiz Eduardo Philippi Rosane, Robert Einer, Mert Yurdakul, Francisco Gomes de Oliveira Neto |
| Venue | SBES 2025, pp. 115–125 · DOI 10.5753/sbes.2025.9809 |
| URL | https://doi.org/10.5753/sbes.2025.9809 |
| Problem | Do LLMs / pretrained encoders help DBRD, and does fine-tuning help? |
| Method | Fine-tune a pretrained sentence encoder for the DBRD task; hyperparameter (learning-rate) study |
| Model | `all-mpnet-base-v2` (MPNet-based, BERT-family) |
| Loss | **Not reported** in the abstract — need the PDF for the exact objective |
| Positive construction | Duplicate pairs from the bug datasets |
| Negative construction | Non-duplicate pairs; balance across similarity levels |
| Hard negatives | **Not reported** |
| Dataset | Eclipse, OpenOffice, Firefox, NetBeans (open-source bug-tracking datasets) |
| Dataset size | **Not reported** as a single number (reuses Rocha & Carvalho's release of the Lazar et al. corpus) |
| Evaluation | Recall Rate@k (RR@5 reported), plus discussion of classification metrics |
| Baselines | The base (non-fine-tuned) `all-mpnet-base-v2` model |
| Main finding | **Fine-tuning yields only marginal improvements across all datasets.** LR 2e-7 *decreased* performance significantly; LR 1e-8 gave only marginal RR@5 gains |
| Code | **Not reported** |
| Reproducibility | **Easy** (this is very close to what you plan to do) |
| Relevance | The single most important reality check for your project. It is also **the gap**: the paper does not appear to ablate negative sampling, hard negatives, the loss function, or the data split. That is where you come in. |

---

## 5. Essential reading list (7 papers, organised by role)

Read these in the order given. Together they are ~120 pages; that is a realistic week of reading alongside data work.

**1. SBERT — the method you will implement.**
> **Read this because** it defines the bi-encoder + pooling + similarity setup that your entire project uses, and because its negative result (raw BERT pooling is *worse than GloVe*) is the reason contrastive fine-tuning is necessary at all. Read §3 (model), §3.1 (training details) and §7 (efficiency).
> https://aclanthology.org/D19-1410/

**2. SimCSE — the contrastive objective and its diagnostics.**
> **Read this because** it is the shortest path to understanding dropout/positives, temperature, in-batch negatives, and — critically — because it shows supervised **contradiction pairs used as hard negatives**. Its alignment/uniformity analysis is the theory you will cite when explaining your loss. Read §2–§3 and the ablation tables.
> https://aclanthology.org/2021.emnlp-main.552/

**3. Wang & Isola, Alignment and Uniformity — the theory section of your write-up.**
> **Read this because** it proves that InfoNCE asymptotically optimises alignment + uniformity, and gives you two cheap, *optimisable* diagnostic metrics you can report for your own embeddings. It is 11 pages and will save you pages of hand-waving.
> https://mlanthology.org/icml/2020/wang2020icml-understanding/

**4. DPR + ANCE — the retrieval framing and the negative-sampling argument.**
> **Read DPR §3.2 and §5.2** (https://arxiv.org/abs/2004.04906) for in-batch negatives and the random/BM25/gold ablation, and **ANCE §1–§3** (https://openreview.net/forum?id=zeFrfgyZln) for the proof that uninformative in-batch negatives kill the gradient signal. Together they justify your most important design choice, and give you the exact ablations to replicate at small scale.

**5. Robinson et al., Contrastive Learning with Hard Negative Samples — why "hardest" is wrong.**
> **Read this because** it states the two principles you must respect (a negative must be a *true* negative, and must be one the model currently believes similar) and shows empirically that hard sampling *without* debiasing is worse than hard sampling *with* it. On bug-report data, where duplicate labels are sparse, this is the paper that will stop you from fooling yourself.
> https://openreview.net/forum?id=CR1XOQ0UTh-

**6. SPECTER — your structural template.**
> **Read this because** it is the closest successful analogue of your project in the literature: a *naturally occurring* relatedness signal (citations), converted into triplets, trained with a margin loss, with **hard negatives constructed by a principled domain rule** (citations-of-citations), and evaluated by retrieval. Substitute "duplicate-of links" for "citations" and you have your design.
> https://aclanthology.org/2020.acl-main.207/

**7. Zhang et al., "How Far Are We?" — the domain reality check.**
> **Read this because** it is the paper that makes your project honest. It shows the evaluation biases (data age, tracker choice) that inflate DBRD results, gives you realistic RR@k magnitudes to compare against, and identifies the simple-retrieval baseline you must beat. If you read only one domain paper, read this one.
> https://doi.org/10.1145/3576042

**Recommended eighth (optional):** [Musgrave et al., A Metric Learning Reality Check](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf). Read §3–§4 only. It is the cheapest way to learn the five ways metric-learning papers mislead, all of which are available to you.

---

## 6. Dataset shortlist

I prioritised **naturally occurring** duplicate relationships (i.e. a human marked A as a duplicate of B) over generated ones, exactly as you asked. Sizes are as published; where a "pair count" is not published, I say so rather than estimating.

### 6.1 Bug reports

| Dataset | Source / URL | Domain | Size | Positive relationship | Negative construction | Accessibility | Difficulty | 1-month suitability |
|---|---|---|---|---|---|---|---|---|
| **GitBugs** | [arXiv:2504.09651](https://arxiv.org/abs/2504.09651) · [GitHub](https://github.com/av9ash/gitbugs/) · [Kaggle](https://www.kaggle.com/datasets/av9ash/gitbugs) | Bug reports from 9 OSS projects (Firefox, Mozilla Core, VS Code, Thunderbird, Spark, Cassandra, HBase, Hadoop, SeaMonkey) across GitHub/Jira/Bugzilla | **150,000+ reports**; duplicate rates 2.0%–28.2% by project (e.g. VS Code 9,272/32,829; Mozilla Core 17,899/85,673) | **Explicit duplicate mappings**, with **predefined train/test splits for duplicate detection** and EDA/model scripts included | Non-duplicate reports in the same project; hard negatives minable by BM25 within the same component/period | **Very easy** — ~100 MB zip, CC BY 4.0 | **Low** — standardised fields (Summary, Description, Status, Priority, Resolution, timestamps) | ★★★★★ Best default. Recent data, ready splits, timestamps present (needed for temporal splitting) |
| **Zhang et al. TOSEM 2023 benchmark** | [DOI 10.1145/3576042](https://doi.org/10.1145/3576042) · [arXiv](https://arxiv.org/abs/2212.00548) | Recent 3-year bug reports from Bugzilla, Jira, GitHub; 6 projects | Six projects; e.g. Eclipse 3,342 sampled pairs, Mozilla 68,396, VSCode 62,092 reports | Explicit duplicate relations extracted from trackers | Non-duplicates in corpus | Replication package released (data + code) | **Medium** — you will need to reconstruct the crawling/cleaning or adapt the released artefact | ★★★★ Best if you want *comparable* numbers to a published bias-corrected benchmark |
| **BugRepo (Lazar et al. 2014)** | [DOI 10.1145/2597073.2597128](https://doi.org/10.1145/2597073.2597128) · [Zenodo](https://doi.org/10.5281/zenodo.1246025) | Eclipse, OpenOffice, NetBeans, Mozilla (Bugzilla) | 4 projects; **3.3 GB** total on Zenodo; ~23% of reports are duplicates (as reported in the DBRD literature) | Explicit duplicate links | Non-duplicate pairs | **Easy** — CC BY 4.0; but MongoDB dumps / large archives | **Medium–High** — 3.3 GB, awkward formats, requires cleaning | ★★ Use only for comparability with older papers. **Warning: data ends ~2014**, which is precisely the *age bias* Zhang et al. identify |
| **Eclipse & Mozilla defect-tracking dataset (2013)** | Referenced in GitBugs and DBRD literature | Eclipse (JDT, CDT, Platform, GEF), Mozilla (Firefox, Thunderbird, Core) | ~215,000 reports, XML, full lifecycle | Duplicate links available | Non-duplicates | Moderate | High (XML parsing, huge) | ★★ Only if you have a specific reason |

### 6.2 Duplicate questions (alternative / fallback domain)

| Dataset | Source / URL | Domain | Size | Positive relationship | Negative construction | Accessibility | Difficulty | 1-month suitability |
|---|---|---|---|---|---|---|---|---|
| **CQADupStack** | [Hoogeveen et al., ADCS 2015](https://eltimster.github.io/www/pubs/adcs2015.pdf) · in BEIR via [ir_datasets](https://ir-datasets.com/beir.html) | 12 StackExchange subforums | **13,145 queries / 457K documents** in the BEIR version; ~1.4 labelled duplicates per query | Explicit community duplicate marks | Non-duplicate posts in the subforum | **Very easy** (BEIR loader, one line) | **Low** | ★★★★ Excellent *fallback*: published BEIR numbers let you sanity-check your pipeline. **Caveat:** with only ~1.4 labelled duplicates/query, the false-negative rate is high — good for studying exactly that, bad if you want clean supervision |
| **BEIR Quora** | [BEIR](https://arxiv.org/abs/2104.08663) | Quora questions | 10,000 queries / 522K docs | Duplicate question pairs | Non-duplicates | Very easy | Low | ★★★ Similar caveat |
| **Stack Overflow duplicate pairs (2023 release)** | [arXiv:2312.15068](https://arxiv.org/html/2312.15068v2) | Stack Overflow | **723,008 duplicate post pairs** from a recent SO dump | Community duplicate marks | Non-duplicates | **Not reported** whether the full dump is publicly released — verify before planning around it | Medium | ★★★ Attractive scale; but **do not assume availability** |
| **AskUbuntu** | Used as the domain-adaptation target in [TSDAE](https://aclanthology.org/2021.findings-emnlp.59/) | Ubuntu Q&A | Small (thousands) | Duplicate-related question links | Non-duplicates | Easy | Low | ★★ Small; useful as a second domain for a cross-domain check |

### 6.3 Code (alternative domain)

| Dataset | Source / URL | Domain | Size | Positive relationship | Negative construction | Accessibility | Difficulty | 1-month suitability |
|---|---|---|---|---|---|---|---|---|
| **POJ-104** (via CodeXGLUE) | [CodeXGLUE](https://arxiv.org/abs/2102.04664) · [GitHub](https://github.com/microsoft/CodeXGLUE) | C/C++ solutions to 104 programming problems | **32K / 8K / 12K** (train/dev/test) | **Programs solving the same problem are semantically equivalent** — a natural functional-similarity grouping | Programs solving different problems | **Easy** | **Low** | ★★★★ The cleanest *retrieval*-framed code-similarity dataset (MAP@R-style evaluation, published CodeBERT baseline MAP 84.29). Excellent if you prefer code |
| **BigCloneBench** (via CodeXGLUE) | [DOI 10.1109/ICSME.2014.77](https://doi.org/10.1109/ICSME.2014.77) · [GitHub](https://github.com/clonebench/BigCloneBench) | Java methods | **6M true clone pairs + 260K false clone pairs**; CodeXGLUE split 900K/416K/416K | Human-validated clones across clone types T1–T4 | False clone pairs | Moderate (large download, IJaDataset 2.0 dependency) | **High** — 6M pairs, dominant "weak Type-3/Type-4" bucket, known label noise | ★★ Only use the CodeXGLUE-processed subset, and be aware that most "semantic" pairs are weakly similar |
| **CodeSearchNet** | [arXiv:1909.09436](https://arxiv.org/abs/1909.09436) · [HF](https://huggingface.co/datasets/code_search_net) | 6 languages | ~6M functions; 2M with scraped docstrings; 99 expert-annotated queries (~4k judgements) | (docstring, function) pairs — **weak / noisy** positives | Other functions | Easy | Medium (quality filtering needed) | ★★★ Good for *code-vs-text* contrastive work; the "query" is automatically scraped, so it is not a natural duplicate relation |

### 6.4 Recommendation on datasets

Take **GitBugs** as primary. It is recent, openly licensed, has explicit duplicate mappings and timestamps, and already ships train/test splits. That removes roughly a week of data-engineering risk. Use **CQADupStack (BEIR)** as a secondary domain if you want a second domain to test generalisation, and **POJ-104** if you decide to switch to code.

**Do not build your project on BugRepo/Lazar data as the primary dataset** — it is the corpus whose use is identified as an *age bias* in [Zhang et al.](https://doi.org/10.1145/3576042).

---

## 7. Candidate research questions

Five questions, each with its literature position. I have marked how "crowded" each is and what an undergraduate-scale version could still contribute.

### Q1 — Domain adaptation: does contrastive fine-tuning on duplicate links beat a general-purpose embedding, when the baseline is strong?

* **Existing literature.** Established as a *technique*: [TSDAE](https://aclanthology.org/2021.findings-emnlp.59/), [GPL](https://aclanthology.org/2022.naacl-main.168/), [AugSBERT](https://aclanthology.org/2021.naacl-main.28/), [SPECTER](https://aclanthology.org/2020.acl-main.207/). Directly contradicted *in this domain* by [Rosane et al. (SBES 2025)](https://doi.org/10.5753/sbes.2025.9809), who found only marginal gains from fine-tuning `all-mpnet-base-v2`.
* **What you reproduce.** Fine-tune a bi-encoder with MNRL on (duplicate, duplicate) pairs.
* **What you change.** (a) Use *current* off-the-shelf baselines (`all-mpnet-base-v2`, `bge-base-en-v1.5`, `e5-base-v2`) rather than only one; (b) include BM25 and a BM25+dense hybrid; (c) split by duplicate group *and* by time.
* **What you measure.** Recall@1/5/10/20, MRR@10, MAP on retrieval over historical reports; report per-project, with 3 seeds.
* **Potential contribution.** A credible *positive or negative* result with a properly strong baseline. Under-supplied in the DBRD literature, which mostly uses weak baselines and optimistic splits.
* **Crowding:** the method is crowded; the *honest evaluation* of it in this domain is not. **Recommended.**

### Q2 — Negative sampling: how does negative-construction strategy affect domain-specific embedding quality?

* **Existing literature.** Extremely rich in general retrieval ([DPR](https://arxiv.org/abs/2004.04906), [ANCE](https://openreview.net/forum?id=zeFrfgyZln), [RocketQA](https://aclanthology.org/2021.naacl-main.466/), [SimANS](https://doi.org/10.18653/v1/2022.emnlp-industry.56)) and in pair-sampling for sentence scoring ([AugSBERT](https://aclanthology.org/2021.naacl-main.28/)). Almost absent for duplicate bug reports.
* **What you reproduce.** The DPR-style three-way negative comparison (random / lexical-hard / in-batch) inside `sentence-transformers`.
* **What you change.** Add a domain-specific hard-negative rule with no analogue in the web-search literature: negatives that are **lexically similar and in the same product/component but not marked duplicate**. Also add a **de-noised** variant (drop candidates whose similarity exceeds the anchor–positive similarity, or filter with a cross-encoder).
* **What you measure.** Retrieval metrics per negative strategy, plus the *false-negative rate* of each strategy estimated post hoc.
* **Potential contribution.** A small but genuine empirical result: "in duplicate-bug-report retrieval, component-matched hard negatives help / hurt, and here is the measured false-negative rate that explains it." **Recommended as your primary ablation axis.**

### Q3 — Hard negatives vs. random negatives: do they actually help here?

* **Existing literature.** [FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf) (semi-hard > hard), [Robinson et al.](https://openreview.net/forum?id=CR1XOQ0UTh-) (hard without debiasing can fail), [ANCE](https://openreview.net/forum?id=zeFrfgyZln) (in-batch too easy), [RocketQA](https://aclanthology.org/2021.naacl-main.466/) (denoise). The answer elsewhere is "it depends, and the details matter".
* **What you change.** Sweep hardness explicitly: random → lexical (BM25) → model-mined → model-mined+denoised. Plot the hardness distribution against the metric.
* **What you measure.** Retrieval metrics + alignment/uniformity ([Wang & Isola](https://mlanthology.org/icml/2020/wang2020icml-understanding/)) + an estimate of the false-negative rate in each mined set.
* **Potential contribution.** Good; this is the version of Q2 with the strongest prior literature to anchor you, so it is the easiest to write up defensibly. **This is the safest choice of primary question.**

### Q4 — Data efficiency: how much domain data before fine-tuning pays off?

* **Existing literature.** [DPR](https://arxiv.org/abs/2004.04906) reports a dense retriever trained on **1,000 examples already beating BM25**, with continued gains to 59k. [GPL](https://aclanthology.org/2022.naacl-main.168/) studies how much unlabelled target data is needed. Not studied for duplicate bug reports.
* **What you change.** Train on 1% / 5% / 25% / 100% of the available duplicate pairs, holding everything else fixed.
* **What you measure.** Metric vs. training-set size, with the zero-shot baseline as a horizontal line.
* **Potential contribution.** Clean, easy to run, and genuinely useful. **Best fallback / addition if time runs short** — it is basically free once the pipeline works.

### Q5 — What does the generic embedding get wrong? (error/bias analysis)

* **Existing literature.** [Ostendorff et al.](https://doi.org/10.1145/3529372.3530912) found generic scientific embeddings are implicitly biased towards the *dataset* aspect. [Jiang et al.](https://doi.org/10.1016/j.jss.2023.111607) found **lexical similarity matters more than semantic similarity** for DBRD. [Zhang et al.](https://doi.org/10.1145/3576042) identified three failure causes (short/incomplete descriptions, URL-heavy text, information split across comments).
* **What you change.** Instead of proposing a model, *analyse* the residual errors of the best system: bucket test queries by lexical overlap with the true duplicate, by report length, by component, by product, and report Recall@k per bucket for each system.
* **What you measure.** Per-bucket Recall@k; qualitative inspection of ~30 failures.
* **Potential contribution.** Highly achievable in one month, requires no new model, and is the kind of analysis that reviewers value. **Strongly recommended as a section of the write-up regardless of which Q you pick as primary.**

---

## 8. Recommended experimental design

### 8.1 Pipeline

```
raw dataset (GitBugs: 9 projects, 150k+ reports with duplicate mappings + timestamps)
   │
   ├─► [1] Preprocessing
   │      • keep: id, project, summary(title), description, product/component,
   │        resolution, status, created timestamp, duplicated_to / duplicate list
   │      • drop/flag: reports with empty description; HTML/code-block stripping;
   │        truncate to a fixed token budget (see 8.4)
   │      • build the duplicate graph: connected components over "A is duplicate of B"
   │        ⇒ each component = one "duplicate group"
   │
   ├─► [2] Splitting  (do this BEFORE any pair generation)
   │      • TEMPORAL: sort by creation time per project; train < val < test by time
   │        (e.g. 70% / 10% / 20% by timestamp quantile)
   │      • GROUP-CONSTRAINED: a duplicate group may not straddle the boundary —
   │        assign the whole group to the split of its EARLIEST member, and DROP
   │        from the test set any query whose group has no member in the past
   │      • OPTIONAL cross-project: train on Firefox+Core, test on VS Code
   │
   ├─► [3] Query/candidate construction (retrieval framing)
   │      • test query  = a report R whose duplicate group contains ≥1 EARLIER report
   │      • gold        = all EARLIER reports in R's duplicate group
   │      • candidate pool = all reports from R's project created before R
   │        (pool size FIXED across all systems — see 8.5)
   │      • train triples/pairs built the same way from the TRAIN period only
   │
   ├─► [4] Pair / triple generation  (train split only)
   │      • positive (a, p): a and p in the same duplicate group, p earlier than a
   │      • negatives, one strategy per configuration:
   │          N1 random        – uniform sample from the train pool
   │          N2 in-batch      – other positives in the mini-batch (free)
   │          N3 lexical-hard  – top-k BM25 matches that are NOT in a's group
   │          N4 model-hard    – top-k by current-model embedding similarity
   │          N5 denoised-hard – N4 filtered by cross-encoder score, or by a
   │                             margin rule (sim(a,n) < sim(a,p) − δ)
   │
   ├─► [5] Model
   │      • base encoder (pick ONE primary, see 8.4): all-MiniLM-L6-v2 (22M, 384-d)
   │        or all-mpnet-base-v2 (110M, 768-d)
   │      • mean pooling, L2-normalised embeddings
   │
   ├─► [6] Loss
   │      • Primary: MultipleNegativesRankingLoss (InfoNCE with in-batch negatives)
   │      • Secondary comparison: TripletLoss with margin, following the
   │        sentence-transformers convention (Euclidean distance, small margin)
   │
   ├─► [7] Training
   │      • see 8.4 for hyperparameters; every configuration shares the same budget
   │      • hard-negative refresh: re-mine once per epoch with FAISS (static ANCE-lite)
   │      • checkpoint selection on the VALIDATION period only
   │
   ├─► [8] Embedding generation
   │      • encode the candidate pool once, L2-normalise, build a FAISS flat-IP index
   │      • encode test queries, retrieve top-100 by dot product
   │
   └─► [9] Evaluation
          • primary: Recall@10, MRR@10, MAP over the fixed pool
          • secondary: Recall@1/5/20, nDCG@10
          • diagnostics: alignment/uniformity, similarity-score histograms,
            per-bucket error analysis (Q5)
```

### 8.2 Why the split must be both temporal and group-constrained

* **Random splitting leaks the answer.** If report A and its duplicate B are randomly assigned to train and test, the model can memorise B's surface form and "retrieve" it. In duplicate detection this is not a subtle effect: the whole supervision signal is a symmetric, sparse graph, and a random edge split puts ~both endpoints of many edges on different sides.
* **Temporal splitting matches deployment.** A triager asks: "has this *already* been reported?" The candidate set is, by construction, everything reported earlier. [Zhang et al.](https://doi.org/10.1145/3576042) show that the *age* of the data changes measured accuracy substantially and with large effect sizes, so a non-temporal split is not comparable to anything deployed.
* **Group splitting prevents the residual leak.** Even with a temporal split, a duplicate group can straddle the boundary; a later member in train would reveal the test query's answer. Assign groups, not reports.
* **Drop unanswerable test queries.** A test query whose duplicate group has no earlier member cannot be retrieved by any system. Keeping it silently deflates everyone's Recall@k by a constant; excluding it changes the denominator. **Report both**, and state which is primary. This is exactly the sort of detail that separates a credible study from an optimistic one.

### 8.3 Why retrieval, not classification

The classification framing (balanced duplicate/non-duplicate pairs, then accuracy/F1/AUC) is what produced the optimistic DBRD literature: [Jiang et al.](https://doi.org/10.1016/j.jss.2023.111607) note that DL-based methods "achieve outstanding performance when evaluated in the classification scenario" while failing to beat IR in the ranking scenario, and [Zhang et al.](https://doi.org/10.1145/3576042) show simple retrieval beating research tools once the benchmark is bias-corrected. Your embeddings are intended to be used for search over a historical corpus. **Evaluate them the way they will be used.** Report classification metrics only as a secondary sanity check, and if you do, use a test set with the *natural* duplicate prevalence (2.7–10% per [Zhang et al.](https://doi.org/10.1145/3576042)), not 50/50.

### 8.4 Training configuration

**Literature-supported values** (cite these):

| Item | Value | Source |
|---|---|---|
| Batch size for in-batch negatives | DPR used **128**; SimCLR showed benefits up to **4096**; `all-mpnet-base-v2` was trained at **1024** for 100k steps | [DPR](https://arxiv.org/abs/2004.04906); [SimCLR](https://mlanthology.org/icml/2020/chen2020icml-simple/); [HF model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) |
| Optimiser / LR | Adam(W) with **2e-5**; warmup. DPR used **1e-5**, Adam, linear schedule with warmup, dropout 0.1 | [HF model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2); [DPR](https://arxiv.org/abs/2004.04906) |
| Epochs | DPR: up to 40 epochs (large datasets) / 100 (small). ANCE converged in ~10 epochs | [DPR](https://arxiv.org/abs/2004.04906); [ANCE](https://openreview.net/pdf?id=zeFrfgyZln) |
| Sequence length | `all-mpnet-base-v2` trained at **128** tokens; E5 truncates at 512; SBERT-family models often 384 | [HF model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) |
| Similarity | Dot product ≈ L2 > cosine for DPR; cosine is standard for SBERT-family | [DPR](https://arxiv.org/abs/2004.04906) |
| Warmup ratio | 0.1 used in current sentence-transformers training configs; `all-mpnet-base-v2` used 500 warmup steps | [HF blog](https://huggingface.co/blog/static-embeddings); [HF model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) |
| Batch sampler | `BatchSamplers.NO_DUPLICATES` — "MultipleNegativesRankingLoss benefits from no duplicate samples in a batch" | [sentence-transformers docs](https://sbert.net/) |

**Proposed configuration** (tune this; do not present it as established):

| Item | Suggested starting point | Rationale |
|---|---|---|
| Base model | `sentence-transformers/all-MiniLM-L6-v2` (primary, 22M/384-d) and `all-mpnet-base-v2` (secondary, 110M/768-d) | MiniLM trains ~5× faster and lets you run 3 seeds; MPNet is the model [Rosane et al.](https://doi.org/10.5753/sbes.2025.9809) used, so it gives you a direct comparison |
| Max sequence length | 256 tokens (title + first N tokens of description) | Bug descriptions can be very long; truncation is a real decision — report it, and test 128 vs 512 as a cheap ablation |
| Batch size | **64** with fp16 on a ≥15 GB GPU; 32 if memory-bound | In-batch negatives = batch size − 1, so this is your most important compute/quality trade-off. Try 32 / 64 / 128 |
| Learning rate | 2e-5 **for the primary run**; sweep 1e-5, 2e-5, 5e-5 | [Rosane et al.](https://doi.org/10.5753/sbes.2025.9809) found 2e-7 harmful and 1e-8 marginal — much lower than the usual range, so LR is worth an explicit sweep |
| Epochs | 2–4, with the refresh of mined negatives between epochs | Small data; watch validation Recall@10 |
| Loss | MNRL (primary); TripletLoss with margin ~0.2–0.5 (comparison) | Margin is a hyperparameter for triplets, not a constant |
| Temperature | MNRL default in sentence-transformers (≈0.05 scaled similarity) — sweep if time | SimCSE and SimCLR both ablate temperature; it interacts with batch size |
| Negatives | In-batch (free) + **1–5** mined hard negatives per anchor | DPR found one BM25 negative helped and two did not; that is in a different domain, so test 1/3/5 |
| Mine refresh | Once per epoch (static ANCE-lite) | Cheap approximation of [ANCE](https://openreview.net/forum?id=zeFrfgyZln)'s asynchronous index |
| Seeds | **3** | [Musgrave et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf) show that with proper tuning many losses tie; without seeds you cannot tell a real difference from noise |

### 8.5 Evaluation protocol

**Primary metrics (retrieval):**

| Metric | Why |
|---|---|
| **Recall@k** (k = 1, 5, 10, 20) | Matches the actual use: "is the duplicate somewhere in the top-k the triager will look at?" Directly comparable to [Zhang et al.](https://doi.org/10.1145/3576042)'s RR@k |
| **MRR@10** | Rewards getting the duplicate *first*, standard in retrieval |
| **MAP** | The metric [Jiang et al.](https://doi.org/10.1016/j.jss.2023.111607) report for the hybrid comparison, so it is comparable |
| **nDCG@10** | BEIR/MTEB convention, so it is comparable to the general embedding literature |

**Secondary / diagnostic:**

| Metric | Why |
|---|---|
| Precision@k, ROC-AUC / PR-AUC on pairs | Only as a sanity check, **with natural class prevalence**. PR-AUC is more informative than ROC-AUC at 2.7–10% prevalence |
| Spearman correlation (if you build a graded similarity set) | [Reimers et al. (COLING 2016)](https://aclanthology.org/C16-1009/) show Pearson can be actively misleading; do not report Pearson as your headline |
| Alignment / uniformity | Cheap, and lets you say *why* a configuration helped ([Wang & Isola](https://mlanthology.org/icml/2020/wang2020icml-understanding/)) |
| Estimated false-negative rate of the mined negative set | Your most original measurement. Procedure: take the top-k mined hard negatives, and check what fraction were later marked duplicate (or are in the same duplicate group) in held-out tracker data. This directly tests [Robinson et al.](https://openreview.net/forum?id=CR1XOQ0UTh-)'s Principle 1 |

**Protocol requirements:**
1. **Fix the candidate pool** across all systems and report its size. [Reimers & Gurevych (ACL 2021)](https://aclanthology.org/2021.acl-short.77/) show dense retrieval degrades faster with index size than sparse retrieval; comparing systems at different pool sizes is not a comparison.
2. **Report per project, not just the mean.** [BEIR](https://arxiv.org/abs/2104.08663) and [MTEB](https://aclanthology.org/2023.eacl-main.148/) both emphasise that averages hide large per-dataset variance, and [MTEB](https://aclanthology.org/2023.eacl-main.148/)'s headline finding is that no method dominates.
3. **Statistical reporting.** 3 seeds → mean ± std; a paired test (e.g. Wilcoxon) or bootstrap CI over test queries for the headline comparisons. [Zhang et al.](https://doi.org/10.1145/3576042) report p-values and effect sizes — copy that standard.
4. **Same tokenisation/truncation and same pooling for every system**, including baselines, or you have confounded your comparison ([Musgrave et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf): unfair comparisons from changes in architecture/embedding size).

---

## 9. Baselines

Every baseline below is *scientifically necessary*, not decorative. Each row says what it controls for.

| # | Baseline | What it controls for | Why it is necessary |
|---|---|---|---|
| **B0** | **BM25** (title + description), tuned or default parameters | Pure lexical matching | Non-negotiable. [BEIR](https://arxiv.org/abs/2104.08663) shows BM25 is a robust zero-shot baseline; [Jiang et al.](https://doi.org/10.1016/j.jss.2023.111607) show DL methods lose to IR methods on DBRD ranking; [Zhang et al.](https://doi.org/10.1145/3576042) show a simple retrieval method beating research tools. **If your fine-tuned model does not beat BM25, that is the most interesting possible result and you must report it.** |
| **B1** | TF-IDF + cosine | A second, weaker lexical point of reference | Cheap; anchors the "how much does lexical overlap explain?" question that [Jiang et al.](https://doi.org/10.1016/j.jss.2023.111607) raise |
| **B2** | **`all-MiniLM-L6-v2` (zero-shot, no fine-tuning)** | The general-purpose embedding baseline, matched to your base model | This is the single most important control: it isolates the effect of *your fine-tuning* from the effect of *the pretrained model*. Using a different base for the baseline and the tuned model is exactly the unfair comparison [Musgrave et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf) warn about |
| **B3** | **`all-mpnet-base-v2` (zero-shot)** | A stronger general embedding; also the exact model [Rosane et al.](https://doi.org/10.5753/sbes.2025.9809) fine-tuned | Makes your result directly comparable to the closest published result |
| **B4** | **A current strong general embedding** — e.g. `BAAI/bge-base-en-v1.5` ([C-Pack](https://arxiv.org/abs/2309.07597)) or `intfloat/e5-base-v2` ([E5](https://arxiv.org/abs/2212.03533)) | The "modern strong baseline" | Because your claim is "better than a general-purpose embedding model", you must test a *good* one, not a 2019 one. Check the current MTEB leaderboard when you start and pick a model of comparable parameter count |
| **B5** | **Same base encoder, fine-tuned with a *different* objective** (e.g. TripletLoss, or regression/softmax on pairs) | Isolates the *loss*, holding architecture and data fixed | Without this, any gain is attributable to "contrastive fine-tuning" only in the loosest sense |
| **B6** | **BM25 ∪ dense hybrid** (e.g. score fusion, or dense re-ranking of BM25 top-200) | The realistic deployed system | [Jiang et al.](https://doi.org/10.1016/j.jss.2023.111607)'s best result is a hybrid; if your fine-tuned model cannot beat the hybrid, that tempers the claim |
| **B7** | *(optional)* **Domain-lexicalised BM25** or a cross-encoder re-ranker on top-50 | Upper bound / ceiling | A cross-encoder gives you the "teacher" upper bound in the style of [AugSBERT](https://aclanthology.org/2021.naacl-main.28/); useful for discussing headroom |
| **B8** | *(optional)* **TSDAE-style domain-adaptive pretraining** before contrastive fine-tuning | Separates "domain adaptation by continued pretraining" from "domain adaptation by contrastive pairs" | Directly comparable to [TSDAE](https://aclanthology.org/2021.findings-emnlp.59/); cheap to add if your pipeline is stable by week 3 |

**Baselines that are NOT defensible as your only comparison:** a bag-of-words model alone; a 2019 SBERT without a modern embedding alongside it; a "random pairs" negative baseline with no BM25; or a classification-only evaluation.

---

## 10. Ablation experiments

Ranked by insight-per-GPU-hour. Run A1–A3 for sure; add A4–A6 only if the pipeline is stable by the end of week 2.

| ID | Ablation | Settings | Question it answers | Cost |
|---|---|---|---|---|
| **A1** | **Negative strategy** | N1 random · N2 in-batch only · N3 lexical (BM25) · N4 model-mined · N5 mined + denoised | Does negative selection matter *in this domain*, and does denoising help? (Q2/Q3) | ★★★★★ highest value · 5 runs |
| **A2** | **Fine-tuned vs. zero-shot, same encoder** | `all-MiniLM-L6-v2` zero-shot vs. fine-tuned (best negative strategy) | The core claim, properly controlled | ★★★★★ · free (reuses B2, A1) |
| **A3** | **Number of hard negatives** | 0 / 1 / 3 / 5 mined hard negatives per anchor, on top of in-batch | Is the DPR "one is enough, two is too many" finding domain-specific? | ★★★★ · 4 runs |
| **A4** | **Data efficiency** | 1% / 5% / 25% / 100% of duplicate pairs | How much domain data before it pays off? (Q4) | ★★★★ · 4 runs (cheap at low budget) |
| **A5** | **Loss function** | MNRL vs. TripletLoss (margin sweep) vs. softmax-on-pairs | Is the gain from the loss or from the data? | ★★★ · 2–4 runs |
| **A6** | **Cross-project generalisation** | train on Firefox + Mozilla Core, test on VS Code | Does the domain-specific embedding transfer across projects, or only within one? | ★★★ · 1 extra run, high interpretive value |
| **A7** | *(cheap)* **Pooling / truncation** | mean vs. CLS pooling; 128 vs. 512 tokens | Mostly a hygiene check, but cheap and reviewers ask | ★★ |
| **A8** | *(cheap)* **Hard-negative refresh rate** | mine once vs. per-epoch | Static vs. ANCE-like refresh | ★★ |

**Do not** add ablations that only add model variants (different encoders, different sizes) unless you have GPU time left over — they increase the number of numbers without increasing understanding, which is exactly the failure mode [Musgrave et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf) diagnose in the metric-learning literature.

**Report:** mean ± std over 3 seeds for each configuration, per project, with the primary metric (Recall@10) in the main table and the rest in an appendix.

---

## 11. Compute requirements

### 11.1 What the comparable papers actually used

| Paper | Reported hardware / cost |
|---|---|
| [SBERT](https://aclanthology.org/D19-1410/) | Reports CPU/GPU inference throughput; training on SNLI+MNLI with a single GPU. No multi-GPU requirement stated |
| [SimCSE](https://aclanthology.org/2021.emnlp-main.552/) | Single-GPU-scale BERT-base experiments (the paper reports a standard fine-tuning setup); explicitly designed to be cheap |
| [DPR](https://arxiv.org/abs/2004.04906) | Batch size 128 with in-batch negatives is reported to require 8 × 32 GB GPUs (stated in follow-up work: [arXiv:2508.09534](https://arxiv.org/html/2508.09534)). **But** DPR also shows that 1,000 training examples suffice to beat BM25, and that same follow-up paper trains a DPR-style model with **batch size 16 on a single 16 GB GPU** by using multiple positives per question |
| [ANCE](https://openreview.net/pdf?id=zeFrfgyZln) | **1–2 hours per ANCE epoch**, converging in ~10 epochs; corpus encoding ~10 h for the full MS MARCO/NQ corpora |
| [RocketQA](https://aclanthology.org/2021.naacl-main.466/) | Multi-GPU (cross-batch negatives span GPUs); the released scripts split candidate passages across 8 GPUs |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | **7 × TPU v3-8**, 100k steps at batch 1024, ~1B pairs — *this is the regime you are NOT in, and should not try to be in* |
| [TSDAE](https://aclanthology.org/2021.findings-emnlp.59/) / [GPL](https://aclanthology.org/2022.naacl-main.168/) | Single-GPU feasible on BEIR-scale corpora (thousands–hundreds of thousands of documents) |
| [Ostendorff et al.](https://doi.org/10.1145/3529372.3530912) | Single-GPU: Siamese SciBERT over 157,606 documents |
| [Corder](https://doi.org/10.1145/3404835.3462840) | Single-GPU-scale contrastive pretraining for code retrieval |

### 11.2 Estimate for your project

Assume 50,000 bug reports in one project, mean length 256 tokens, a 22M-parameter MiniLM encoder.

| Stage | Estimate on one ≥15 GB GPU (T4 / RTX 3060+ / Colab / Kaggle) |
|---|---|
| Encode 50k reports once | ~2–6 minutes |
| Build FAISS flat index | seconds |
| One training epoch, ~20k pairs, batch 64, seq 256 | ~5–15 minutes |
| One hard-negative mining pass (encode pool + top-k FAISS search) | ~3–8 minutes |
| **One full configuration** (3 epochs + 2 mining passes) | **~30–60 minutes** |
| Embedding + retrieval evaluation of one configuration | ~3–5 minutes |
| **Total for A1 (5 negative strategies) × 3 seeds** | **~8–15 GPU-hours** |
| **Total for A1–A5 × 3 seeds** | **~20–35 GPU-hours** |

Memory: a 110M-parameter model in fp32 with AdamW needs roughly 4× parameter bytes (params + grads + 2 optimizer states) ≈ 1.8 GB, plus activations. At batch 64 × 256 tokens this fits comfortably in 15 GB; use fp16/bf16 for headroom. MiniLM (22M) is trivially small.

### 11.3 Recommended setup

* **Primary:** Google Colab (free T4, or Colab Pro for longer sessions and better GPUs) or Kaggle Notebooks.
* **Storage:** keep the dataset and all artefacts in Google Drive or as a small HF dataset; do not rely on Colab's ephemeral disk. Budget < 2 GB for the working set if you use GitBugs' per-project slices.
* **Determinism:** fix seeds and record the exact package versions (`sentence-transformers`, `transformers`, `torch`, `faiss`). Log to Weights & Biases or plain JSON — you will need these numbers for the write-up.
* **Guard against session loss:** save checkpoints every N steps and write metrics incrementally to disk.
* **Encouraging precedent for single-GPU work:** [arXiv:2508.09534](https://arxiv.org/html/2508.09534) trains a DPR-style bi-encoder with batch size 16 and multiple positives per question on a **single 16 GB GPU**. In-batch negatives scale with batch size, so a small batch means fewer of them — compensate with explicitly mined hard negatives, which is exactly the variable you are studying.
* **Do not** attempt: training an encoder from scratch (Project C), reproducing ANCE/RocketQA at MS MARCO scale, or multi-GPU cross-batch negatives. All are out of scope for one month and none are needed to answer your question.

---

## 12. Reproducibility check on the top 10 papers

| Paper | Paper accessible? | Code? | Dataset? | Preprocessing scripts? | Trained models? | Reproducible by an undergraduate? |
|---|---|---|---|---|---|---|
| [SBERT](https://aclanthology.org/D19-1410/) | Yes (ACL Anthology + arXiv) | Yes — [sentence-transformers](https://github.com/UKPLab/sentence-transformers) | Yes (SNLI/MNLI public) | Yes | Yes (>25k models on HF) | **Yes** |
| [SimCSE](https://aclanthology.org/2021.emnlp-main.552/) | Yes | Yes — [princeton-nlp/SimCSE](https://github.com/princeton-nlp/SimCSE) | Yes (Wikipedia / NLI) | Yes | Yes (HF) | **Yes** |
| [DPR](https://arxiv.org/abs/2004.04906) | Yes | Yes — [facebookresearch/DPR](https://github.com/facebookresearch/DPR) | Yes | Yes | Yes | **Yes** (at reduced scale) |
| [ANCE](https://openreview.net/forum?id=zeFrfgyZln) | Yes (OpenReview + PMLR) | Yes — [microsoft/ANCE](https://github.com/microsoft/ANCE) | Yes | Yes | Yes | **Partly** — the full pipeline needs substantial GPU time; the *idea* is reproducible in lite form |
| [RocketQA](https://aclanthology.org/2021.naacl-main.466/) | Yes | Yes — [PaddlePaddle/RocketQA](https://github.com/PaddlePaddle/RocketQA) | Yes | Yes (per-stage construction scripts) | Yes | **No, not as published** (multi-GPU). Adopt the denoising *idea*, subsample the pipeline |
| [TSDAE](https://aclanthology.org/2021.findings-emnlp.59/) | Yes | Yes (in sentence-transformers) | Yes | Yes | Yes | **Yes** |
| [GPL](https://aclanthology.org/2022.naacl-main.168/) | Yes | Yes — [UKPLab/gpl](https://github.com/UKPLab/gpl) | Yes (BEIR) | Yes | Yes | **Yes** |
| [AugSBERT](https://aclanthology.org/2021.naacl-main.28/) | Yes | Yes (sentence-transformers docs) | Partly (task datasets public; exact splits in docs) | Yes | Partly | **Yes** |
| [SPECTER](https://aclanthology.org/2020.acl-main.207/) | Yes | Yes — [allenai/specter](https://github.com/allenai/specter) | Yes (SciDocs + S2ORC) | Yes | Yes | **Yes** |
| [Zhang et al., TOSEM 2023](https://doi.org/10.1145/3576042) | Yes (arXiv open + ACM) | Yes — **replication package released** (data + code) | Yes | Yes | N/A | **Yes**, though the corpora are large |

Note on [Rosane et al. (SBES 2025)](https://doi.org/10.5753/sbes.2025.9809): paper accessible; code **not reported** in the record I could access. You should email the authors or simply re-run their setup yourself — it is close to what you plan anyway, and re-running it *is* a valuable contribution.

---

## 13. Weak or misleading papers — critical appraisal

Apply these filters to everything you read, including the papers I have recommended.

**Systemic problems in the duplicate-bug-report-detection (DBRD) literature:**

1. **Age bias.** Much of the DBRD literature evaluates on the [Lazar et al. 2014](https://doi.org/10.1145/2597073.2597128) corpora, whose latest reports date to ~2014. [Zhang et al.](https://doi.org/10.1145/3576042) demonstrate statistically that this inflates measured accuracy (p-values 0.002–0.012, large effect sizes). ⚠️ Any DBRD paper evaluated only on this corpus should be read as reporting an upper bound.
2. **Issue-tracker bias.** Results on Bugzilla do not transfer to Jira or GitHub (same paper, large effect sizes). ⚠️ Single-tracker evaluations are not generalisable.
3. **Classification framing.** Balanced duplicate/non-duplicate pairs with accuracy/F1 reported as the headline. [Jiang et al.](https://doi.org/10.1016/j.jss.2023.111607) show that DL methods look excellent here and lose in the ranking framing that practitioners need. ⚠️ Treat classification-only DBRD results as non-comparable.
4. **Weak baselines.** Many DBRD papers compare against other learned methods but not against BM25 or a modern general embedding. [Zhang et al.](https://doi.org/10.1145/3576042) show that when you add a simple retrieval baseline and an industrial tool, the ranking of methods changes. ⚠️ "We outperform [other DL method]" is not evidence of practical value.

**Specific papers to treat with caution:**

| Paper | Flag |
|---|---|
| [Isotani et al., Frontiers in Computer Science 2022](https://doi.org/10.3389/fcomp.2022.1032452) (SBERT + triplet network for duplicate bug reports) | Closest published work to your idea and reports positive results for fine-tuning — but the evaluation is on a company-specific maintenance dataset, with TF-IDF/LDA baselines only, no modern embedding baseline, no significance testing, and no comparison against BM25. **Suggestive, not conclusive.** Cite it as motivation, not as evidence of effect size |
| [GitBugs (Patil, arXiv 2504.09651)](https://arxiv.org/abs/2504.09651) | Useful resource, but it is an unrefereed dataset paper: no quantitative validation of the duplicate-label extraction is reported, and a public review notes that the claim of "ready for downstream ML" rests on deterministic field mappings with duplicate labels imported verbatim from source fields. **Verify the duplicate mappings yourself on a random sample of ~100 before trusting them** |
| [AugSBERT](https://aclanthology.org/2021.naacl-main.28/)'s "+37 points domain adaptation" | Real, but from a very specific setup (bi-encoder trained on a *different* labelled source domain, adapted to a target domain with a cross-encoder teacher). It is not a general expectation for contrastive fine-tuning. Do not quote it as a typical gain |
| [GPL](https://aclanthology.org/2022.naacl-main.168/)'s "+9.3 nDCG@10" | Measured against an out-of-the-box MS MARCO retriever, on BEIR domains — i.e. against a *weak-in-domain* baseline. Against a *modern* general embedding the gap is smaller. Check the per-dataset table, not the headline |
| [ANCE](https://openreview.net/forum?id=zeFrfgyZln) and [RocketQA](https://aclanthology.org/2021.naacl-main.466/) | Excellent papers, but the reported gains are at MS MARCO scale with multi-GPU training. Do not assume the same relative gains at 50k documents with one GPU |
| "When Hard Negatives Hurt" (arXiv 2606.01304) | A very recent preprint arguing that LLM-generated hard negatives often *degrade* contrastive retrieval training. The diagnosis is plausible and matches [Robinson et al.](https://openreview.net/forum?id=CR1XOQ0UTh-)'s principles, but a public review notes the proposed fix is presented without supporting results. **Treat as preliminary; do not build on it** |
| Any paper reporting only average MTEB/BEIR scores | [MTEB](https://aclanthology.org/2023.eacl-main.148/)'s own headline finding is that **no method dominates across tasks**; averages hide per-dataset swings. Always read the per-dataset table |
| Any paper using LLM-generated paraphrases as "positives" | This is the failure mode you explicitly wanted to avoid. Paraphrase positives teach paraphrase-invariance, not domain similarity, and they contain no information about which distinctions matter in the domain |

**And the general filter from [Musgrave et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf):** before believing any claimed improvement in a metric-learning paper, ask (a) were the architectures, embedding sizes and optimisers matched across compared methods? (b) were hyperparameters tuned on the test set? (c) is the metric informative (they recommend MAP@R over Recall@1 / NMI)? (d) were enough seeds run? In their re-run, most methods tie once these are fixed.

---

## 14. Tier classification

Categorised by *role in your project*, not by quality.

### Tier 1 — Foundational (the concepts your project rests on)
* [Chopra, Hadsell & LeCun, CVPR 2005](https://doi.org/10.1109/CVPR.2005.202) — discriminative similarity metric for face verification; the predecessor of the contrastive loss.
* [Hadsell, Chopra & LeCun, CVPR 2006](https://doi.org/10.1109/CVPR.2006.100) — the contrastive loss.
* [Weinberger & Saul, JMLR 2009](https://mlanthology.org/jmlr/2009/weinberger2009jmlr-distance/) — LMNN, margin-based metric learning.
* [Schroff et al., CVPR 2015 (FaceNet)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf) — triplet loss, semi-hard mining.
* [Sohn, NeurIPS 2016](https://papers.nips.cc/paper_files/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html) — N-pair loss = InfoNCE = MNRL.
* [Henderson et al., 2017](https://arxiv.org/abs/1705.00652) — in-batch negatives for text dual encoders.
* [van den Oord et al., 2018](https://arxiv.org/abs/1807.03748) — InfoNCE as an MI bound.
* [Chen et al., ICML 2020 (SimCLR)](https://mlanthology.org/icml/2020/chen2020icml-simple/) — the canonical contrastive recipe and its ablations.
* [Wang & Isola, ICML 2020](https://mlanthology.org/icml/2020/wang2020icml-understanding/) — alignment and uniformity.

### Tier 2 — Directly applicable (methods you can adapt almost as-is)
* [SBERT](https://aclanthology.org/D19-1410/) — the architecture and library.
* [SimCSE](https://aclanthology.org/2021.emnlp-main.552/) — the training objective and diagnostics.
* [DPR](https://arxiv.org/abs/2004.04906) — dual encoder + in-batch negatives + the negative-type ablation.
* [ANCE](https://openreview.net/forum?id=zeFrfgyZln) — hard-negative mining architecture.
* [RocketQA](https://aclanthology.org/2021.naacl-main.466/) — denoised hard negatives.
* [SimANS](https://doi.org/10.18653/v1/2022.emnlp-industry.56) — ambiguous negative sampling.
* [AugSBERT](https://aclanthology.org/2021.naacl-main.28/) — pair/negative sampling strategies for sentence scoring.
* [TSDAE](https://aclanthology.org/2021.findings-emnlp.59/) and [GPL](https://aclanthology.org/2022.naacl-main.168/) — domain adaptation recipes with honest effect sizes.
* [SPECTER](https://aclanthology.org/2020.acl-main.207/) — the natural-signal → triplets → retrieval template.
* [Ostendorff et al., JCDL 2022](https://doi.org/10.1145/3529372.3530912) — small-scale Siamese + MNRL specialisation.

### Tier 3 — Domain-specific (bug reports, code, retrieval)
* [Zhang et al., TOSEM 2023](https://doi.org/10.1145/3576042) — DBRD benchmark and reality check.
* [Jiang et al., JSS 2023](https://doi.org/10.1016/j.jss.2023.111607) — DL vs. IR for DBRD; lexical > semantic.
* [Rosane et al., SBES 2025](https://doi.org/10.5753/sbes.2025.9809) — fine-tuning an encoder for DBRD yields marginal gains.
* [Lazar et al., MSR 2014](https://doi.org/10.1145/2597073.2597128) — the classic Bugzilla corpus (use with the age-bias caveat).
* [Hoogeveen et al., ADCS 2015](https://eltimster.github.io/www/pubs/adcs2015.pdf) — CQADupStack.
* [ContraCode](https://aclanthology.org/2021.emnlp-main.482/) · [CodeBERT](https://aclanthology.org/2020.findings-emnlp.139/) · [UniXcoder](https://aclanthology.org/2022.acl-long.499/) · [Corder](https://doi.org/10.1145/3404835.3462840) · [CoCoSoDa](https://arxiv.org/abs/2204.03293) — the code branch, if you choose it.
* [Gururangan et al., ACL 2020](https://aclanthology.org/2020.acl-main.740/) — DAPT/TAPT.

### Tier 4 — Evaluation and methodology
* [Musgrave et al., ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf) — **the methodological conscience.**
* [Reimers, Beyer & Gurevych, COLING 2016](https://aclanthology.org/C16-1009/) — task-oriented evaluation; Pearson is misleading.
* [BEIR](https://arxiv.org/abs/2104.08663) · [MTEB](https://aclanthology.org/2023.eacl-main.148/) · [MMTEB](https://arxiv.org/abs/2502.13595) — benchmarks and metric conventions.
* [Reimers & Gurevych, ACL 2021](https://aclanthology.org/2021.acl-short.77/) — index-size effects on dense retrieval.
* [Chuang et al., NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html) · [Robinson et al., ICLR 2021](https://openreview.net/forum?id=CR1XOQ0UTh-) · [arXiv 2401.00165](https://arxiv.org/abs/2401.00165) — false negatives: the theory and the remedies.
* [sentence-transformers documentation](https://sbert.net/) — loss reference, `mine_hard_negatives()` (with `range_min`, `range_max`, `absolute_margin`, `relative_margin`, `max_score`, `min_score`, `use_faiss`) and the domain-adaptation tables.

---

## 15. The research gap

### 15.1 What is already extensively studied
* **Contrastive sentence-embedding pretraining at scale** — [SimCSE](https://aclanthology.org/2021.emnlp-main.552/), [E5](https://arxiv.org/abs/2212.03533), [BGE/C-Pack](https://arxiv.org/abs/2309.07597), [GTE](https://arxiv.org/abs/2308.03281). Saturated and industrialised.
* **Hard-negative mining for web/QA retrieval** — [DPR](https://arxiv.org/abs/2004.04906), [ANCE](https://openreview.net/forum?id=zeFrfgyZln), [RocketQA](https://aclanthology.org/2021.naacl-main.466/), [SimANS](https://doi.org/10.18653/v1/2022.emnlp-industry.56). Rich and mature, with clear theoretical accounts.
* **Domain adaptation of embeddings** — [TSDAE](https://aclanthology.org/2021.findings-emnlp.59/), [GPL](https://aclanthology.org/2022.naacl-main.168/), [AugSBERT](https://aclanthology.org/2021.naacl-main.28/), [DAPT/TAPT](https://aclanthology.org/2020.acl-main.740/). Established, with measured (modest) effect sizes.
* **Code representation learning with contrastive objectives** — [ContraCode](https://aclanthology.org/2021.emnlp-main.482/), [Corder](https://doi.org/10.1145/3404835.3462840), [CoCoSoDa](https://arxiv.org/abs/2204.03293), [UniXcoder](https://aclanthology.org/2022.acl-long.499/). Well populated.
* **Duplicate bug report detection as a task** — decades of work; [Zhang et al.](https://doi.org/10.1145/3576042) show that much of the reported progress does not survive a bias-corrected benchmark.

### 15.2 What is relatively underexplored
1. **Negative-sampling strategy *within* the duplicate-bug-report domain.** The retrieval literature studies it on MS MARCO/NQ; the DBRD literature studies models, not sampling. The intersection is nearly empty.
2. **The false-negative structure of issue trackers.** Because only a small fraction of true duplicates are labelled (2.7–10% of reports are duplicates, and many duplicates are never marked), the "unlabelled positive" problem is far more severe here than in web search, and nobody has measured it. [RocketQA](https://aclanthology.org/2021.naacl-main.466/) and [Robinson et al.](https://openreview.net/forum?id=CR1XOQ0UTh-) give you the tools; the measurement is open.
3. **Whether contrastive fine-tuning beats a *strong modern* general embedding plus BM25 for this task.** [Rosane et al.](https://doi.org/10.5753/sbes.2025.9809) tested fine-tuning against itself; [Zhang et al.](https://doi.org/10.1145/3576042) tested research tools against IR. Nobody has put "modern general embedding + BM25 hybrid" head-to-head with a properly contrastively fine-tuned model under temporal + group-constrained splits.
4. **Cross-project transfer.** Whether a duplicate-similarity embedding trained on Firefox transfers to VS Code is, as far as I found, **not reported**.
5. **What the generic embedding systematically gets wrong in this domain** — the [Ostendorff et al.](https://doi.org/10.1145/3529372.3530912)-style aspect-bias analysis has no bug-report analogue.

### 15.3 Is "domain-specific contrastive embeddings" novel?
**No.** As a technique it is established ([TSDAE](https://aclanthology.org/2021.findings-emnlp.59/), [GPL](https://aclanthology.org/2022.naacl-main.168/), [SPECTER](https://aclanthology.org/2020.acl-main.207/)). Framing your project as "I will show that domain-specific contrastive embeddings work" is framing it as a reproduction of a known result, and — given [Rosane et al.](https://doi.org/10.5753/sbes.2025.9809) and [Jiang et al.](https://doi.org/10.1016/j.jss.2023.111607) — you might not even reproduce it.

### 15.4 The methodological variable to investigate in one month
**Negative-example construction, with an explicit false-negative measurement.** It is (a) underexplored in this domain, (b) cheap to vary, (c) theoretically grounded in a mature literature you can cite, (d) guaranteed to produce an interpretable result whether it is positive or negative, and (e) has a natural "here is why" story (the false-negative rate of each strategy).

---

## 16. Novelty assessment — an honest verdict

Be careful here, because this is where undergraduate projects most often over-claim.

| Classification | Assessment |
|---|---|
| **Primarily a reproduction?** | Partly. You *will* reproduce [SBERT](https://aclanthology.org/D19-1410/)/[SimCSE](https://aclanthology.org/2021.emnlp-main.552/)-style contrastive fine-tuning and [DPR](https://arxiv.org/abs/2004.04906)/[ANCE](https://openreview.net/forum?id=zeFrfgyZln)-style negative sampling. That is unavoidable and appropriate — it is the "related work you implement" part. |
| **An engineering implementation?** | Partly, and necessarily. Building a correct duplicate-group + temporal split and a retrieval evaluation harness for issue-tracker data is real engineering, and it is where most of your risk sits. |
| **An empirical study?** | **Yes — this is the honest primary characterisation.** You are running a controlled experiment whose outcome is not known in advance. |
| **A potential small research contribution?** | **Yes, conditionally.** It becomes one if and only if: (1) your baselines include BM25 and a strong modern general embedding; (2) your splits are temporal *and* group-constrained; (3) you run ≥3 seeds and report effect sizes with uncertainty; (4) you measure something nobody has measured in this domain — most plausibly the false-negative rate of each negative-sampling strategy; (5) you report the result honestly even if contrastive fine-tuning does not help. |

**What you should not claim:** a new loss, a new architecture, a new state of the art, or that "domain-specific contrastive embeddings" is a novel idea. Reviewers will correctly reject all of those.

**What you can defensibly claim:** *"In duplicate bug report retrieval, under temporal and duplicate-group-constrained splits, contrastive fine-tuning of encoder X with negative strategy Y improves Recall@10 from A to B relative to BM25 and to zero-shot encoder X; the improvement is (or is not) attributable to hard negatives, and the mined hard negatives contain an estimated Z% unlabelled positives, which we argue is the binding constraint."* That is a real sentence with a real contribution in it, whichever way the numbers fall.

**One extra warning about scope.** [Jiang et al.](https://doi.org/10.1016/j.jss.2023.111607)'s finding that **lexical similarity matters more than semantic similarity** for this task is the biggest threat to your thesis. If your fine-tuned model's advantage turns out to be mostly lexical, say so; that is a finding, not a failure. You can test it directly by adding a lexical-overlap feature to the baseline (their hybrid) and by bucketing test queries by lexical overlap with the gold duplicate (Q5).

---

## 17. Final recommendation

### 17.1 The most defensible project formulation

> **Title (working):** *Does the Choice of Negative Examples Determine Whether Contrastive Fine-tuning Helps in Duplicate Bug Report Retrieval?*
>
> **Design:** Take `all-MiniLM-L6-v2` and `all-mpnet-base-v2` as base bi-encoders. On GitBugs (150k+ reports across 9 projects with explicit duplicate mappings and timestamps), construct a duplicate-group graph; split **temporally and group-constrained**; frame the task as **retrieval** over a fixed historical candidate pool; evaluate with **Recall@k, MRR@10, MAP, nDCG@10** over 3 seeds. Baselines: BM25, TF-IDF, both encoders zero-shot, one current strong general embedding, and a BM25+dense hybrid. Then ablate five negative-construction strategies (random, in-batch, BM25-lexical, model-mined, mined+denoised) and, for each, **estimate the false-negative rate** of the mined set.
>
> **Primary hypothesis (pre-register it in writing before you train anything):** hard negatives help on average in web retrieval ([ANCE](https://openreview.net/forum?id=zeFrfgyZln), [RocketQA](https://aclanthology.org/2021.naacl-main.466/)); we predict the effect will be **smaller or absent** here, because the mined "hard negatives" in an issue tracker are disproportionately unlabelled duplicates.
>
> **Secondary questions:** data efficiency (Q4) and per-bucket error analysis (Q5).

**Why this formulation rather than the alternatives:**

* Versus "Option A — does domain adaptation help?": that question is already answered in the general case ([TSDAE](https://aclanthology.org/2021.findings-emnlp.59/), [GPL](https://aclanthology.org/2022.naacl-main.168/)) and, in this domain, appears to be answered *negatively* ([Rosane et al.](https://doi.org/10.5753/sbes.2025.9809)). As a standalone question it is a reproduction with a high chance of a null result you cannot explain. Asking *which variable drives the outcome* turns a null result into an explanation.
* Versus "Option C — do hard negatives help?": this is essentially the same question, but I am recommending you add the false-negative measurement, which is what makes it more than a re-run of [DPR](https://arxiv.org/abs/2004.04906)'s table at small scale.
* Versus "Option D — how much does it improve?": a magnitude question needs an unusually careful setup to be interesting, and it is the easiest to make look vacuous.
* Versus "Option E — data efficiency": excellent, but it is a *fallback* rather than a primary question, because "performance vs. dataset size" curves are less informative about mechanism than a negative-sampling ablation.

**Why bug reports rather than code:** the code branch ([POJ-104](https://arxiv.org/abs/2102.04664), [BigCloneBench](https://doi.org/10.1109/ICSME.2014.77)) has *stronger* natural similarity labels (same problem solved = semantically equivalent), but the code-representation literature is far more crowded ([ContraCode](https://aclanthology.org/2021.emnlp-main.482/), [Corder](https://doi.org/10.1145/3404835.3462840), [CoCoSoDa](https://arxiv.org/abs/2204.03293), [UniXcoder](https://aclanthology.org/2022.acl-long.499/)), so an undergraduate-scale experiment is more likely to be a straight reproduction. Bug reports give you a domain where the *evaluation* is still contested — which is where contribution is cheap. **Caveat:** BigCloneBench's dominant bucket is "weak Type-3 / Type-4" similarity, so "semantic" code clones are the noisiest category; if you go the code route, use POJ-104, not BigCloneBench.

**Why retrieval rather than classification:** because the classification framing is precisely what produced the optimistic DBRD literature ([Jiang et al.](https://doi.org/10.1016/j.jss.2023.111607), [Zhang et al.](https://doi.org/10.1145/3576042)), and because your embeddings will be used for search.

### 17.2 "If I had one month" — week by week

**Week 1 — literature, dataset, preprocessing (no training yet)**
* Read §5's seven papers. Skim: SBERT §3, SimCSE §2–3, DPR §3.2+§5.2, ANCE §1–3, Robinson et al. in full, SPECTER §3–4, Zhang et al. §3+§5.
* Download **GitBugs**; load one project end-to-end. Write and unit-test: HTML/code-block stripping, field extraction, token counting.
* **Build the duplicate graph** (connected components over duplicate links) and print its statistics: number of groups, group-size distribution, duplicate rate per project. Sanity-check against the published rates (VS Code 28.2%, Mozilla Core 20.9%, Spark 2.5%).
* Manually verify ~100 random duplicate links (guards against the [GitBugs](https://arxiv.org/abs/2504.09651) label-quality caveat).
* **Deliverable:** a cleaned per-project dataframe + a split function that is temporal *and* group-constrained, with a written description of the split rule.

**Week 2 — baselines and pair generation (still no training)**
* Implement the retrieval harness: fixed candidate pool, FAISS index, Recall@k / MRR@10 / MAP / nDCG@10.
* Run **B0 BM25**, **B1 TF-IDF**, **B2 MiniLM zero-shot**, **B3 MPNet zero-shot**, **B4 one strong modern embedding**.
* **Deliverable: a complete baseline table before you train anything.** This step is what makes the project defensible. If you are short on time later, you will still have a complete, honest empirical result.
* Build pair generation for the train period: positives from the duplicate graph (p earlier than a), negatives under all five strategies. Verify by eye that N3 (BM25-lexical) negatives are "same area, different bug" and not obviously duplicates.
* **Pre-register** your hypothesis, primary metric and ablation list in a short markdown file. Commit it.

**Week 3 — contrastive training and ablations**
* Train with MNRL: `all-MiniLM-L6-v2`, batch 64, lr 2e-5, 3 epochs, max_seq 256, fp16.
* Run **A1** (five negative strategies) × 3 seeds. This is your main result.
* If time permits: **A3** (number of hard negatives) and **A4** (data efficiency).
* Implement the **false-negative estimate** for each mined negative set.
* Log everything; checkpoint to Drive.

**Week 4 — evaluation, analysis, writing**
* Full evaluation of every configuration; assemble the main table (per project, mean ± std).
* **Q5 error analysis:** bucket test queries by lexical overlap with the gold duplicate, by report length, by component; report Recall@10 per bucket for BM25, zero-shot and best-tuned.
* Compute alignment/uniformity for the best and worst configurations ([Wang & Isola](https://mlanthology.org/icml/2020/wang2020icml-understanding/)).
* Manually inspect ~30 failures ([Zhang et al.](https://doi.org/10.1145/3576042) style: short descriptions, URL-heavy text, information split across comments).
* Write: intro → related work → method → experimental setup → results → **threats to validity** → discussion. Be explicit about the possibility that gains are small; that section is where the credibility lives.

**Buffers and triage.** If you fall behind: drop A3/A4, drop the second encoder, keep A1 and the baselines. If you are ahead: add A6 (cross-project) — it is one extra run and a genuinely open question.

### 17.3 The five things most likely to sink this project

1. **Skipping BM25.** Without it you cannot tell whether you beat lexical matching, and [Jiang et al.](https://doi.org/10.1016/j.jss.2023.111607) say you might not.
2. **A leaky split.** Random or non-group-constrained splitting will give you beautiful, meaningless numbers.
3. **No seeds.** [Musgrave et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf)'s core lesson: with proper tuning, most methods tie. One run cannot distinguish a real gain from noise.
4. **Evaluating by classification accuracy.** It does not match the use case and it is the documented source of optimism in this literature.
5. **Not pre-registering.** If you decide what your hypothesis was *after* seeing the results, you will unconsciously shape the story. Write it down in week 2.

---

## 18. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Insufficient data** — too few duplicate pairs in a project | Medium | High | Pick high-duplicate-rate projects (VS Code 28.2%, Thunderbird 27.6%, Mozilla Core 20.9%, Firefox 21.7%) over low ones (HBase 2.0%, Spark 2.5%). [DPR](https://arxiv.org/abs/2004.04906) shows 1,000 pairs can already beat BM25, so you need far less than you think |
| **Poor positive pairs** — duplicate links are noisy or incomplete | **High** | High | Manually verify ~100 links in week 1; report a noise estimate; do not assume the tracker's labels are complete |
| **False negatives** — true duplicates not labelled, treated as negatives | **High** | **High** (this is the central technical risk *and* the central research opportunity) | Use the [RocketQA](https://aclanthology.org/2021.naacl-main.466/) denoising idea; respect [Robinson et al.](https://openreview.net/forum?id=CR1XOQ0UTh-)'s two principles; **measure** the false-negative rate rather than assuming it away; use `mine_hard_negatives` margins (`absolute_margin`, `relative_margin`, `max_score`) |
| **Data leakage** — duplicate groups straddling the split; near-duplicate text appearing on both sides | Medium | High | Group-constrained + temporal splitting (§8.2); additionally de-duplicate exact/near-identical text corpora before splitting |
| **Compute limitations** — Colab timeouts, session loss | Medium | Medium | MiniLM primary; checkpoint every N steps; write metrics incrementally; budget ≤35 GPU-hours total |
| **Weak baselines** — you beat only other learned models | Medium | **High** (invalidates the central claim) | Build BM25 and both zero-shot encoders in **week 2, before training**. Non-negotiable |
| **Domain shift / no cross-project transfer** | Medium | Medium | Treat A6 as an explicit ablation rather than an assumption; report per-project results, never only the mean |
| **Evaluation mismatch** — classification metrics, unbalanced prevalence, varying pool size | Medium | High | Retrieval framing; natural prevalence in any pair-level metrics; **fixed candidate pool**; report pool size ([Reimers & Gurevych, ACL 2021](https://aclanthology.org/2021.acl-short.77/)) |
| **Gains are small or zero** | **High** | Medium (manageable — reframe) | This is *expected* given [Rosane et al.](https://doi.org/10.5753/sbes.2025.9809) and [Jiang et al.](https://doi.org/10.1016/j.jss.2023.111607). Pre-register the mechanism question so a null result is an answer, not a failure |
| **Truncation destroys signal** — long bug descriptions cut to 256 tokens | Medium | Medium | Test 128 vs 512; consider title + first-N-token-of-description vs. full; report the choice |
| **Hyperparameter sensitivity** — LR in particular | Medium | Medium | [Rosane et al.](https://doi.org/10.5753/sbes.2025.9809) found LR 2e-7 harmful and 1e-8 marginal; sweep 1e-5 / 2e-5 / 5e-5 and report the sweep |
| **You over-claim novelty** | Medium | High | Use the framing in §16 verbatim as a checklist before writing the intro |

---

## 19. Minimal bibliography (primary links only)

**Foundations**
1. Hadsell, Chopra, LeCun (2006). *Dimensionality Reduction by Learning an Invariant Mapping.* CVPR 2006, 2:1735–1742. https://doi.org/10.1109/CVPR.2006.100
2. Chopra, Hadsell, LeCun (2005). *Learning a Similarity Metric Discriminatively, with Application to Face Verification.* CVPR 2005, pp. 539–546. https://doi.org/10.1109/CVPR.2005.202
3. Weinberger & Saul (2009). *Distance Metric Learning for Large Margin Nearest Neighbor Classification.* JMLR 10:207–244. https://mlanthology.org/jmlr/2009/weinberger2009jmlr-distance/
4. Schroff, Kalenichenko, Philbin (2015). *FaceNet.* CVPR 2015. https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf
5. Sohn (2016). *Improved Deep Metric Learning with Multi-class N-pair Loss Objective.* NeurIPS 2016. https://papers.nips.cc/paper_files/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html
6. van den Oord, Li, Vinyals (2018). *Representation Learning with Contrastive Predictive Coding.* arXiv:1807.03748. https://arxiv.org/abs/1807.03748
7. Henderson et al. (2017). *Efficient Natural Language Response Suggestion for Smart Reply.* arXiv:1705.00652. https://arxiv.org/abs/1705.00652
8. Chen, Kornblith, Norouzi, Hinton (2020). *A Simple Framework for Contrastive Learning of Visual Representations.* ICML 2020, PMLR 119:1597–1607. https://mlanthology.org/icml/2020/chen2020icml-simple/
9. Wang & Isola (2020). *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.* ICML 2020, PMLR 119:9929–9939. https://mlanthology.org/icml/2020/wang2020icml-understanding/

**Embeddings and retrieval**
10. Reimers & Gurevych (2019). *Sentence-BERT.* EMNLP-IJCNLP 2019, pp. 3982–3992. https://aclanthology.org/D19-1410/ · code https://github.com/UKPLab/sentence-transformers
11. Gao, Yao, Chen (2021). *SimCSE.* EMNLP 2021, pp. 6894–6910. https://aclanthology.org/2021.emnlp-main.552/ · code https://github.com/princeton-nlp/SimCSE
12. Karpukhin et al. (2020). *Dense Passage Retrieval for Open-Domain QA.* EMNLP 2020. https://arxiv.org/abs/2004.04906 · code https://github.com/facebookresearch/DPR
13. Xiong et al. (2021). *ANCE.* ICLR 2021. https://openreview.net/forum?id=zeFrfgyZln · code https://github.com/microsoft/ANCE
14. Qu et al. (2021). *RocketQA.* NAACL 2021, pp. 5835–5847. https://aclanthology.org/2021.naacl-main.466/ · code https://github.com/PaddlePaddle/RocketQA
15. Zhou et al. (2022). *SimANS.* EMNLP 2022 Industry Track, pp. 548–559. https://doi.org/10.18653/v1/2022.emnlp-industry.56
16. Chuang et al. (2020). *Debiased Contrastive Learning.* NeurIPS 2020. https://proceedings.neurips.cc/paper_files/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html
17. Robinson et al. (2021). *Contrastive Learning with Hard Negative Samples.* ICLR 2021. https://openreview.net/forum?id=CR1XOQ0UTh-
18. Wang et al. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5).* arXiv:2212.03533. https://arxiv.org/abs/2212.03533
19. Xiao et al. (2023). *C-Pack / BGE.* arXiv:2309.07597. https://arxiv.org/abs/2309.07597
20. Li et al. (2023). *Towards General Text Embeddings with Multi-stage Contrastive Learning (GTE).* arXiv:2308.03281. https://arxiv.org/abs/2308.03281

**Domain adaptation**
21. Wang, Reimers, Gurevych (2021). *TSDAE.* Findings EMNLP 2021, pp. 671–688. https://aclanthology.org/2021.findings-emnlp.59/
22. Wang, Thakur, Reimers, Gurevych (2022). *GPL.* NAACL 2022, pp. 2345–2360. https://aclanthology.org/2022.naacl-main.168/ · code https://github.com/UKPLab/gpl
23. Thakur, Reimers, Daxenberger, Gurevych (2021). *Augmented SBERT.* NAACL 2021, pp. 296–310. https://aclanthology.org/2021.naacl-main.28/
24. Gururangan et al. (2020). *Don't Stop Pretraining.* ACL 2020, pp. 8342–8360. https://aclanthology.org/2020.acl-main.740/
25. Cohan et al. (2020). *SPECTER.* ACL 2020, pp. 2270–2282. https://aclanthology.org/2020.acl-main.207/
26. Ostendorff et al. (2022). *Specialized Document Embeddings for Aspect-based Similarity of Research Papers.* JCDL 2022. https://doi.org/10.1145/3529372.3530912 · code https://github.com/malteos/aspect-document-embeddings

**Duplicate bug reports**
27. Lazar, Ritchey, Sharif (2014). *Generating duplicate bug datasets.* MSR 2014, pp. 392–395. https://doi.org/10.1145/2597073.2597128 · data https://doi.org/10.5281/zenodo.1246025
28. Zhang et al. (2023). *Duplicate Bug Report Detection: How Far Are We?* ACM TOSEM 32(4). https://doi.org/10.1145/3576042 · https://arxiv.org/abs/2212.00548
29. Jiang, Su, Treude, Shang, Wang (2023). *Does Deep Learning improve the performance of duplicate bug report detection? An empirical study.* J. Syst. Softw. 198:111607. https://doi.org/10.1016/j.jss.2023.111607
30. Rosane, Einer, Yurdakul, Oliveira Neto (2025). *Evaluating Fine-tuning Approaches for Duplicate Bug Report Detection.* SBES 2025, pp. 115–125. https://doi.org/10.5753/sbes.2025.9809
31. Patil (2025). *GitBugs.* arXiv:2504.09651. https://arxiv.org/abs/2504.09651 · https://github.com/av9ash/gitbugs/
32. Hoogeveen, Verspoor, Baldwin (2015). *CQADupStack.* ADCS 2015. https://eltimster.github.io/www/pubs/adcs2015.pdf

**Code (if you take that branch)**
33. Jain et al. (2021). *Contrastive Code Representation Learning (ContraCode).* EMNLP 2021, pp. 5954–5971. https://aclanthology.org/2021.emnlp-main.482/ · code https://github.com/parasj/contracode
34. Feng et al. (2020). *CodeBERT.* Findings EMNLP 2020, pp. 1536–1547. https://aclanthology.org/2020.findings-emnlp.139/
35. Guo et al. (2022). *UniXcoder.* ACL 2022, pp. 7212–7225. https://aclanthology.org/2022.acl-long.499/
36. Bui, Yu, Jiang (2021). *Corder.* SIGIR 2021, pp. 511–521. https://doi.org/10.1145/3404835.3462840
37. Shi et al. (2023). *CoCoSoDa.* ICSE 2023, pp. 2198–2210. https://arxiv.org/abs/2204.03293
38. Lu et al. (2021). *CodeXGLUE.* NeurIPS 2021 Datasets & Benchmarks. https://arxiv.org/abs/2102.04664
39. Svajlenko et al. (2014). *Towards a Big Data Curated Benchmark of Inter-project Code Clones (BigCloneBench).* ICSME 2014, pp. 476–480. https://doi.org/10.1109/ICSME.2014.77
40. Husain et al. (2019). *CodeSearchNet Challenge.* arXiv:1909.09436. https://arxiv.org/abs/1909.09436

**Evaluation and methodology**
41. Musgrave, Belongie, Lim (2020). *A Metric Learning Reality Check.* ECCV 2020. https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf · code https://github.com/KevinMusgrave/powerful-benchmarker
42. Reimers, Beyer, Gurevych (2016). *Task-Oriented Intrinsic Evaluation of Semantic Textual Similarity.* COLING 2016, pp. 87–96. https://aclanthology.org/C16-1009/
43. Thakur et al. (2021). *BEIR.* NeurIPS 2021 Datasets & Benchmarks. https://arxiv.org/abs/2104.08663 · code https://github.com/UKPLab/beir
44. Muennighoff et al. (2023). *MTEB.* EACL 2023, pp. 2014–2037. https://aclanthology.org/2023.eacl-main.148/
45. Reimers & Gurevych (2021). *The Curse of Dense Low-Dimensional Information Retrieval for Large Index Sizes.* ACL 2021 (Short), pp. 605–611. https://aclanthology.org/2021.acl-short.77/
46. Sentence Transformers documentation (losses, `mine_hard_negatives`, domain adaptation). https://sbert.net/

---

### A note on what this review does not settle

I verified facts against abstracts, official proceedings pages and (where available) paper PDFs. Four items remain open and you should confirm them before relying on them:

* **(a)** The exact objective and hyperparameters used by [Rosane et al. (SBES 2025)](https://doi.org/10.5753/sbes.2025.9809). I could only access the conference record, which does not state the loss function. Read the PDF — this is the closest published result to your project, so it is worth the effort.
* **(b)** The precise size of the SPECTER training corpus (not stated in the abstract).
* **(c)** Whether the 723,008-pair Stack Overflow duplicate dataset of [arXiv:2312.15068](https://arxiv.org/html/2312.15068v2) is publicly downloadable. **Do not plan around it until confirmed.**
* **(d)** The sentiment expressed in [Isotani et al. (Frontiers 2022)](https://doi.org/10.3389/fcomp.2022.1032452) — I read the published article, which reports positive fine-tuning results, but the evaluation is small and the baselines weak. Treat as motivation only.

Everything else above is traceable to the linked primary source. Where I could not verify a fact from the paper itself, it is marked **Not reported** in the extraction tables.
