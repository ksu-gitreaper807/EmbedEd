# 09 — Literature report and reading order

Tailored to **the final project** ([`04`](04-project-specification.md)), not to the field in general. Sixteen papers. You will read about seven of them properly.

---

## 1. Foundational papers (4)

### 1.1 Hadsell, Chopra & LeCun (2006) — *Dimensionality Reduction by Learning an Invariant Mapping*
CVPR 2006, vol. 2, pp. 1735–1742 · <https://doi.org/10.1109/CVPR.2006.100>
* **What you need:** the shape of the **contrastive loss** — pull similar pairs together, push dissimilar pairs past a margin — and the idea that the *arrangement* of the space is what you are optimising, not a label.
* **Informs:** your Introduction's one-paragraph history, and the intuition for why a margin/temperature exists at all.
* **Read full paper?** **No.** Read the abstract and Figure 1. Two pages, five minutes.

### 1.2 Schroff, Kalenichenko & Philbin (2015) — *FaceNet*
CVPR 2015 · <https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf>
* **What you need:** (a) most randomly chosen triplets are already satisfied and give **zero gradient** — so the choice of negatives decides everything; (b) **semi-hard** negatives (harder than the positive but not absurdly hard) beat the hardest ones, which can collapse training.
* **Informs:** the entire motivation for your independent variable, and your expectation that "hardest" may not be best.
* **Read full paper?** **Skim §3.2 and §4** (triplet selection). ~15 minutes.

### 1.3 Sohn (2016) — *Improved Deep Metric Learning with Multi-class N-pair Loss Objective*
NeurIPS 2016 · <https://papers.nips.cc/paper_files/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html>
* **What you need:** N-pair loss **is** InfoNCE **is** MultipleNegativesRankingLoss. Triplet loss is the 2-pair special case. Comparing against `N−1` negatives at once converges faster than comparing one.
* **Informs:** your Methods section — this is the citation for "the loss we use is …".
* **Read full paper?** **Skim §3.** ~15 minutes.

### 1.4 Wang & Isola (2020) — *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*
ICML 2020, PMLR 119:9929–9939 · <https://mlanthology.org/icml/2020/wang2020icml-understanding/>
* **What you need:** the proof that, asymptotically, the contrastive loss optimises exactly **alignment** (positives close) + **uniformity** (vectors spread over the sphere), and the two cheap metrics that measure them.
* **Informs:** your diagnostic analysis. "Strategy N5 improved alignment but degraded uniformity" is a mechanistic claim you can otherwise never make.
* **Read full paper?** **Yes — all 11 pages.** It is short, it is the best-written paper on this list, and it will save you pages of hand-waving.

---

## 2. Embedding papers (4)

### 2.1 Reimers & Gurevych (2019) — *Sentence-BERT*
EMNLP-IJCNLP 2019, pp. 3982–3992 · <https://aclanthology.org/D19-1410/> · code: <https://github.com/UKPLab/sentence-transformers>
* **What you need:** siamese/triplet BERT; mean pooling; three objectives; and the crucial negative result — **averaged BERT embeddings score 54.81 Spearman and CLS 29.19, both worse than averaged GloVe.**
* **Informs:** your architecture, your library choice, and the strongest single sentence in your motivation ("pretrained encoders do not give usable sentence embeddings out of the box").
* **Read full paper?** **Yes, §3 and §3.1.** ~20 minutes.

### 2.2 Gao, Yao & Chen (2021) — *SimCSE*
EMNLP 2021, pp. 6894–6910 · <https://aclanthology.org/2021.emnlp-main.552/> · code: <https://github.com/princeton-nlp/SimCSE>
* **What you need:** dropout-as-augmentation for the unsupervised variant; **NLI contradiction pairs used as hard negatives** for the supervised variant; temperature ablation; the uniformity explanation of why it works.
* **Informs:** your training recipe and the framing of hard negatives as an explicit design choice with a measurable effect.
* **Read full paper?** **Yes, §2–§3 and the ablation tables.** ~30 minutes.

### 2.3 Karpukhin et al. (2020) — *Dense Passage Retrieval for Open-Domain QA*
EMNLP 2020 · <https://arxiv.org/abs/2004.04906> · code: <https://github.com/facebookresearch/DPR>
* **What you need:** dual encoders; **in-batch negatives**; the three-way negative comparison (random / BM25 / gold); the finding that **one BM25 hard negative helps substantially and two do not**; dot product ≈ L2 > cosine; and that 1,000 training pairs already beat BM25.
* **Informs:** your baselines, your A2 ablation, your data-efficiency expectations, and your similarity function.
* **Read full paper?** **Yes, §3.2 and §5.2.** ~25 minutes. This is the single most directly useful paper for your experimental design.

### 2.4 Muennighoff et al. (2023) — *MTEB: Massive Text Embedding Benchmark*
EACL 2023, pp. 2014–2037 · <https://aclanthology.org/2023.eacl-main.148/>
* **What you need:** the metric conventions (nDCG@10 for retrieval), and the headline finding that **no embedding method dominates across tasks** — which is your licence (and obligation) to report per-domain results.
* **Informs:** metric choice and your "do not trust averages" caveat.
* **Read full paper?** **No.** Read §1, §3.3 and Table 1. ~15 minutes.

---

## 3. Domain-specific application papers (4)

### 3.1 Cohan et al. (2020) — *SPECTER*
ACL 2020, pp. 2270–2282 · <https://aclanthology.org/2020.acl-main.207/>
* **What you need:** the **template** for your whole project — a naturally occurring relatedness signal (citations) → triplet training → **hard negatives constructed by a domain rule** ("citations of citations") → retrieval evaluation — plus the ablation showing hard negatives are essential.
* **Informs:** your Method section's structure and your justification for rule-based hard negatives (your N3/N5).
* **Read full paper?** **Yes, §3–§4.** ~25 minutes.

### 3.2 Ostendorff et al. (2022) — *Specialized Document Embeddings for Aspect-based Similarity of Research Papers*
JCDL 2022 · <https://doi.org/10.1145/3529372.3530912> · code: <https://github.com/malteos/aspect-document-embeddings>
* **What you need:** a **single-GPU-scale** example of specialising a general encoder with **MultipleNegativesRankingLoss**, using 157,606 documents; and the finding that generic embeddings carry an **implicit bias** towards one aspect.
* **Informs:** proof that your project's scale is publishable, and the model for your error analysis.
* **Read full paper?** **Skim §3–§5.** ~20 minutes.

### 3.3 Zhang et al. (2023) — *Duplicate Bug Report Detection: How Far Are We?*
ACM TOSEM 32(4) · <https://doi.org/10.1145/3576042> · <https://arxiv.org/abs/2212.00548>
* **What you need:** the **two evaluation biases** (data age, issue tracker) that inflate duplicate-detection results; the finding that a **simple retrieval method beats deep-learning methods**; realistic magnitudes (RR@5 ≈ 0.4–0.6); the three documented failure causes.
* **Informs:** your threats-to-validity section and your error-analysis method. Even though your dataset is questions, not bugs, the *methodological* lessons transfer exactly.
* **Read full paper?** **Yes, §1, §3, §5, §6.3.** ~40 minutes.

### 3.4 Jiang et al. (2023) — *Does Deep Learning improve the performance of duplicate bug report detection? An empirical study*
J. Systems and Software 198:111607 · <https://doi.org/10.1016/j.jss.2023.111607>
* **What you need:** DL methods look excellent under **classification** and lose under **ranking**; hybrid DL+IR gains only 7.09–11.34% median MAP; **lexical similarity matters more than semantic similarity**.
* **Informs:** your choice of retrieval over classification (justify it with this citation), and the honest possibility that your gain is partly lexical — which your bucketed error analysis will test.
* **Read full paper?** **Skim §1, §5, §6.** ~20 minutes.

---

## 4. Negative-sampling / hard-negative papers (4)

### 4.1 Xiong et al. (2021) — *ANCE*
ICLR 2021 · <https://openreview.net/forum?id=zeFrfgyZln> · code: <https://github.com/microsoft/ANCE>
* **What you need:** the **theory** — uninformative in-batch negatives yield **diminishing gradient norms and large gradient variance** — and the fix: global negatives from an ANN index refreshed during training.
* **Informs:** why N4 exists, why you re-mine per epoch, and the theoretical justification for your whole independent variable.
* **Read full paper?** **Yes, §1–§3.** ~30 minutes.

### 4.2 Robinson, Chuang, Sra & Jegelka (2021) — *Contrastive Learning with Hard Negative Samples*
ICLR 2021 · <https://openreview.net/forum?id=S4nZh4WBHxq>
* **What you need:** the **two principles** — a useful negative must (1) genuinely be a negative and (2) currently be believed similar — and the empirical result that **hard sampling with debiasing beats hard sampling without it**.
* **Informs:** the design of N5 and the theoretical grounding for your false-negative measurement. This is the paper that explains your result either way.
* **Read full paper?** **Yes, all of it.** ~35 minutes.

### 4.3 Qu et al. (2021) — *RocketQA*
NAACL 2021, pp. 5835–5847 · <https://aclanthology.org/2021.naacl-main.466/> · code: <https://github.com/PaddlePaddle/RocketQA>
* **What you need:** **denoised hard negatives** — a cross-encoder filters mined candidates that are probably unlabelled positives — plus cross-batch negatives.
* **Informs:** the exact recipe for N5.
* **Read full paper?** **Skim §3–§4.** ~20 minutes. (Do not try to reproduce it: it needs multiple GPUs.)

### 4.4 Thakur, Reimers, Daxenberger & Gurevych (2021) — *Augmented SBERT*
NAACL 2021, pp. 296–310 · <https://aclanthology.org/2021.naacl-main.28/>
* **What you need:** the **pair-sampling ablation** — Random vs. BM25 vs. Semantic Search vs. KDE — and the finding that random pairing produces overwhelming easy-negative class imbalance. Also useful as the source of your N3 (BM25 sampling).
* **Informs:** the design and justification of your five negative strategies, and your framing of "pair selection is non-trivial and crucial".
* **Read full paper?** **Skim §3–§4.** ~20 minutes. (Note: its "+37 points domain adaptation" figure comes from a very specific setup — do not quote it as a typical gain.)

---

## 5. Evaluation methodology papers (2 — only where necessary)

### 5.1 Musgrave, Belongie & Lim (2020) — *A Metric Learning Reality Check*
ECCV 2020 · <https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf> · code: <https://github.com/KevinMusgrave/powerful-benchmarker>
* **What you need:** the five failure modes — unfair comparisons (different architectures/embedding sizes/optimisers), test-set feedback in hyperparameter choice, uninformative metrics (Recall@1, NMI), single runs, and the conclusion that **with proper tuning most methods tie**.
* **Informs:** your entire experimental protocol. This is your conscience.
* **Read full paper?** **Yes, §3 and §4.** ~30 minutes. Skim the rest.

### 5.2 Reimers, Beyer & Gurevych (2016) — *Task-Oriented Intrinsic Evaluation of Semantic Textual Similarity*
COLING 2016, pp. 87–96 · <https://aclanthology.org/C16-1009/>
* **What you need:** ranking STS systems by **Pearson** correlation can be actively misleading — in one task its predictiveness was **negative (ρ = −0.326)** versus +0.504 for nDCG. Use Spearman or a ranking metric matched to your task.
* **Informs:** the one line in your paper where you explain why you do not report Pearson.
* **Read full paper?** **No.** Read the abstract and §1. Five minutes.

---

## 6. Reading order

**Total essential reading: ~4.5 hours of paper reading spread over 3–4 days**, plus skimming. Do this *alongside* week-1 data work, not before it.

### Read first (minimum to understand the project) — ~2 hours

| # | Paper | Time |
|---|---|---|
| 1 | [SBERT](https://aclanthology.org/D19-1410/) §3, §3.1 | 20 min |
| 2 | [SimCSE](https://aclanthology.org/2021.emnlp-main.552/) §2–§3 | 30 min |
| 3 | [Wang & Isola](https://mlanthology.org/icml/2020/wang2020icml-understanding/) (all) | 45 min |
| 4 | [DPR](https://arxiv.org/abs/2004.04906) §3.2, §5.2 | 25 min |

After these four you can explain, to another student, what your model is and what your loss does.

### Read second (needed to design the experiment) — ~2.5 hours

| # | Paper | Time |
|---|---|---|
| 5 | [ANCE](https://openreview.net/forum?id=zeFrfgyZln) §1–§3 | 30 min |
| 6 | [Robinson et al.](https://openreview.net/forum?id=S4nZh4WBHxq) (all) | 35 min |
| 7 | [Musgrave et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf) §3–§4 | 30 min |
| 8 | [SPECTER](https://aclanthology.org/2020.acl-main.207/) §3–§4 | 25 min |
| 9 | [Zhang et al., TOSEM](https://doi.org/10.1145/3576042) §1, §3, §5, §6.3 | 40 min |

After these five you can defend every design decision in your Methods section.

### Skim (useful context, 15–25 min each, only if relevant as you write)

| Paper | Skim when |
|---|---|
| [RocketQA](https://aclanthology.org/2021.naacl-main.466/) §3–§4 | You implement N5 |
| [AugSBERT](https://aclanthology.org/2021.naacl-main.28/) §3–§4 | You design the five strategies |
| [Jiang et al.](https://doi.org/10.1016/j.jss.2023.111607) §1, §5, §6 | You write the "classification vs. retrieval" paragraph |
| [Ostendorff et al.](https://doi.org/10.1145/3529372.3530912) §3–§5 | You write the error analysis |
| [FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf) §3.2 | You explain semi-hard negatives |
| [Sohn](https://papers.nips.cc/paper_files/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html) §3 | You write the loss definition |
| [MTEB](https://aclanthology.org/2023.eacl-main.148/) §1, §3.3 | You justify nDCG@10 and per-domain reporting |

### Reference only (cite, do not read)

| Paper | Cite it for |
|---|---|
| [Hadsell et al. 2006](https://doi.org/10.1109/CVPR.2006.100) | The origin of the contrastive loss |
| [van den Oord et al. 2018](https://arxiv.org/abs/1807.03748) | InfoNCE as a mutual-information bound |
| [Chen et al. 2020 (SimCLR)](https://mlanthology.org/icml/2020/chen2020icml-simple/) | Temperature and batch-size ablations |
| [Henderson et al. 2017](https://arxiv.org/abs/1705.00652) | The origin of in-batch negatives for text |
| [Chuang et al. 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html) | The debiased objective for false negatives |
| [SimANS](https://doi.org/10.18653/v1/2022.emnlp-industry.56) | Ambiguous (middle-band) negative sampling |
| [BEIR](https://arxiv.org/abs/2104.08663) | The benchmark your dataset comes from; BM25's strength |
| [GPL](https://aclanthology.org/2022.naacl-main.168/) / [TSDAE](https://aclanthology.org/2021.findings-emnlp.59/) | The published CQADupStack reference numbers (29.6 → 31.8 → 34.5 → 35.1) |
| [Reimers & Gurevych 2021](https://aclanthology.org/2021.acl-short.77/) | Why you fix the candidate-pool size |
| [Reimers et al. 2016](https://aclanthology.org/C16-1009/) | Why you do not report Pearson |
| [Hoogeveen et al. 2015](https://eltimster.github.io/www/pubs/adcs2015.pdf) | CQADupStack itself |
| [Rosane et al. 2025](https://doi.org/10.5753/sbes.2025.9809) | The closest prior result: fine-tuning yields marginal gains |
| [ContraCode](https://aclanthology.org/2021.emnlp-main.482/) / [CodeXGLUE](https://arxiv.org/abs/2102.04664) / [CoCoSoDa](https://arxiv.org/abs/2204.03293) | Only if you switch to the code path |

---

## 7. Time budget sanity check

| Tier | Papers | Time |
|---|---|---|
| Read first | 4 | ~2.0 h |
| Read second | 5 | ~2.7 h |
| Skim | up to 7 | ~2.0 h (optional) |
| Reference only | 12 | 0 h |
| **Total essential** | **9** | **~4.7 h ≈ 3–4 days at 90 min/day** |

Nine papers read properly. That is the real reading load for this project — not twenty-five.
