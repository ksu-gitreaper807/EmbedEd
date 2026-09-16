# 02 — What is already solved, and what is genuinely open

Rule used throughout: **something is a gap only if the question has not been answered, not because a particular combination of dataset + model has not been tried.** Testing `X` on dataset `Y` is not research by itself.

---

## Part A — Already established (do not claim these as contributions)

| Claim | Status | Evidence |
|---|---|---|
| Fine-tuning a BERT/SBERT encoder with a contrastive objective produces usable sentence embeddings | **Standard practice.** Textbook. | [SBERT (2019)](https://aclanthology.org/D19-1410/); [SimCSE (2021)](https://aclanthology.org/2021.emnlp-main.552/) — BERT-base reaches 76.3% (unsup) / 81.6% (sup) average Spearman |
| Raw pretrained encoder outputs are bad sentence embeddings | **Established, with numbers.** Avg. BERT 54.81 / CLS 29.19 Spearman — worse than averaged GloVe | [Reimers & Gurevych (2019)](https://aclanthology.org/D19-1410/) |
| InfoNCE / N-pair / MultipleNegativesRankingLoss are the same object | **Established.** | [Sohn (2016)](https://papers.nips.cc/paper_files/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html); [van den Oord et al. (2018)](https://arxiv.org/abs/1807.03748); [Chen et al. (2020)](https://mlanthology.org/icml/2020/chen2020icml-simple/) |
| In-batch negatives work and are essentially free | **Established.** | [Henderson et al. (2017)](https://arxiv.org/abs/1705.00652); [DPR (2020)](https://arxiv.org/abs/2004.04906) |
| Hard negatives help in web/QA retrieval | **Established.** DPR: one BM25 hard negative helps substantially, two do not. ANCE: global ANN-mined negatives beat in-batch. | [DPR](https://arxiv.org/abs/2004.04906) §5.2; [ANCE (2021)](https://openreview.net/forum?id=zeFrfgyZln) |
| Mining harder negatives eventually hurts, because they become false negatives | **Established, with a fix (denoising).** | [RocketQA (2021)](https://aclanthology.org/2021.naacl-main.466/); [SimANS (2022)](https://doi.org/10.18653/v1/2022.emnlp-industry.56); [Robinson et al. (2021)](https://openreview.net/forum?id=CR1XOQ0UTh-) |
| Domain adaptation improves embeddings / retrieval on a target domain | **Established, with measured (modest) gains.** CQADupStack nDCG@10: 29.6 zero-shot → 31.8 TSDAE → 34.5 GPL → 35.1 TSDAE+GPL. GPL reports up to +9.3 nDCG@10 overall. | [TSDAE](https://aclanthology.org/2021.findings-emnlp.59/); [GPL](https://aclanthology.org/2022.naacl-main.168/) |
| Continued pretraining on in-domain text helps downstream tasks | **Established.** | [Gururangan et al. (2020)](https://aclanthology.org/2020.acl-main.740/) |
| BM25 is a strong baseline that dense models often fail to beat out of domain | **Established.** | [BEIR (2021)](https://arxiv.org/abs/2104.08663) |
| Duplicate bug report detection as a *task* | **Extensively studied for ~20 years.** | [Zhang et al. (2023, TOSEM)](https://doi.org/10.1145/3576042) surveys and re-benchmarks it |
| Semantic code similarity / code clone detection | **Extensively studied.** | [CodeXGLUE](https://arxiv.org/abs/2102.04664); [ContraCode](https://aclanthology.org/2021.emnlp-main.482/); [CoCoSoDa](https://arxiv.org/abs/2204.03293) |
| Dense retrieval in general | **Mature and industrialised.** | [DPR](https://arxiv.org/abs/2004.04906), [ANCE](https://openreview.net/forum?id=zeFrfgyZln), [BEIR](https://arxiv.org/abs/2104.08663), [MTEB](https://aclanthology.org/2023.eacl-main.148/) |

**Conclusion: every single component of your proposed pipeline is established.** Claiming that "domain-specific contrastive embeddings" is a novel idea would be wrong, and a reviewer would say so.

---

## Part B — Also established, and it cuts against the naive version of your project

This is the part most students miss. In the *specific* applied domains you listed, the published evidence says the gains from domain-specific fine-tuning are **small, fragile, and sometimes negative**:

| Finding | Source |
|---|---|
| In duplicate bug report detection, a **simple retrieval method beats recently proposed deep-learning methods** on most projects once the benchmark is corrected for data-age and issue-tracker bias. Best RR@5 ≈ 0.4–0.6. | [Zhang et al., TOSEM 2023](https://doi.org/10.1145/3576042) |
| Deep-learning methods **do not beat IR-based methods** in the ranking formulation; a DL+IR hybrid improves MAP by a median of only 7.09–11.34%; **lexical similarity matters more than semantic similarity** for this task. | [Jiang et al., JSS 2023](https://doi.org/10.1016/j.jss.2023.111607) |
| Fine-tuning `all-mpnet-base-v2` on four large open-source bug datasets yields **only marginal improvements**; LR 2e-7 significantly *decreased* performance. | [Rosane et al., SBES 2025](https://doi.org/10.5753/sbes.2025.9809) |
| No text-embedding method dominates across tasks — averages hide large per-dataset variance. | [MTEB](https://aclanthology.org/2023.eacl-main.148/) |
| Generic embeddings carry implicit biases (SPECTER-style scientific embeddings are biased towards the *dataset* aspect and against the *method* aspect). | [Ostendorff et al., JCDL 2022](https://doi.org/10.1145/3529372.3530912) |

**So the honest framing of your project is not "show that it works." It is: "measure how much it works, identify the variable that decides it, and say why."**

---

## Part C — What is genuinely still open (and undergraduate-scale)

These are ordered by how well they fit a one-month, single-GPU project.

### Gap 1 — Negative-sampling strategy *within* a specialised duplicate-detection domain ⭐

* **Known:** negative sampling is decisive in web/QA retrieval ([DPR](https://arxiv.org/abs/2004.04906), [ANCE](https://openreview.net/forum?id=zeFrfgyZln), [RocketQA](https://aclanthology.org/2021.naacl-main.466/), [SimANS](https://doi.org/10.18653/v1/2022.emnlp-industry.56)) and in pair-sampling for sentence scoring ([AugSBERT](https://aclanthology.org/2021.naacl-main.28/) compares Random / BM25 / Semantic Search / KDE).
* **Open:** nobody has run that comparison *systematically inside a duplicate-question or duplicate-bug-report domain*, where the label structure is completely different (sparse, community-generated, massively incomplete).
* **Why it is a real gap and not a gap-by-combination:** the answer is *not predictable* from the web-search literature, because here the hardest negatives are the most likely to be unlabelled positives. The interaction between "hardness" and "label incompleteness" is domain-specific and unmeasured.

### Gap 2 — The false-negative rate of mined hard negatives ⭐

* **Known:** false negatives are a problem in principle ([Chuang et al., 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html); [Robinson et al., 2021](https://openreview.net/forum?id=CR1XOQ0UTh-)); RocketQA *filters* them; [arXiv:2401.00165](https://arxiv.org/abs/2401.00165) models them.
* **Open:** nobody **reports the actual measured rate** in a duplicate-detection domain. Everyone assumes it, filters it, or ignores it. In CQADupStack there are only ~1.4 human-marked duplicates per query, so the unlabelled-positive rate is almost certainly high — but *how high* is an empirical question that costs you one afternoon to answer.
* **Why it is valuable:** it converts a null result into an explanation. "Hard negatives did not help; we measured that 38% of our mined hard negatives are semantically duplicate questions" is a finding.

### Gap 3 — Cross-domain generalisation of a domain-adapted embedding

* **Known:** domain adaptation helps *on the target domain* ([GPL](https://aclanthology.org/2022.naacl-main.168/), [TSDAE](https://aclanthology.org/2021.findings-emnlp.59/)).
* **Open:** whether a duplicate-similarity embedding trained on some technical domains **transfers to an unseen technical domain**, or merely overfits to the training domains' vocabulary. "No one can win every battle" is literally a conclusion in [Zhang et al.](https://doi.org/10.1145/3576042) — the best method differed by project.
* **Why CQADupStack is ideal:** its 12 subforums are 12 natural domains with a shared task, so the experiment is *designed into the dataset*.

### Gap 4 — Data efficiency at small scale

* **Known:** DPR reported that a dense retriever trained on **1,000 examples already outperforms BM25**, with continued gains to 59k. GPL studies how much unlabelled target data is needed.
* **Open:** the shape of the learning curve *for a specialised duplicate-similarity task*, and specifically whether the "knee" depends on the negative strategy. Cheap to run: once the pipeline works, this is four more short training runs.

### Gap 5 — What the generic embedding systematically gets wrong

* **Known:** the analysis style exists ([Ostendorff et al.](https://doi.org/10.1145/3529372.3530912) found aspect bias in scientific-paper embeddings; [Jiang et al.](https://doi.org/10.1016/j.jss.2023.111607) found lexical > semantic for bug reports; [Zhang et al.](https://doi.org/10.1145/3576042) catalogued three failure causes: short/incomplete descriptions, URL-heavy text, information split across comments).
* **Open:** the bucketed error analysis for duplicate *question* retrieval. Requires no new model — just discipline.

---

## Part D — Things that look like gaps but are not

| Tempting claim | Why it is not a gap |
|---|---|
| "Nobody has applied contrastive learning to dataset X" | Applying a standard method to a new dataset is an **experiment**, not a contribution. It can be a contribution *if* it answers an open question (see Gaps 1–3) — but the dataset switch alone is not the contribution. |
| "Nobody has combined SBERT with hard negatives for bug reports" | Combination claims are only contributions if the interaction is surprising or the result changes practice. |
| "I will build a duplicate-detection system" | That is Project A (application). It is engineering unless it contains a controlled comparison. |
| "I will propose a new loss" | Almost certainly a re-derivation of something in the InfoNCE family. [Musgrave et al. (2020)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf) show that most proposed metric-learning losses tie with each other once hyperparameters are tuned fairly. Do not go here in one month. |
| "I will fine-tune a bigger model" | [MTEB](https://aclanthology.org/2023.eacl-main.148/) shows performance scales with size, but scale is not a research contribution and it costs you the compute you need for seeds and ablations. |

---

## Part E — The one-sentence research position

> The *method* (contrastive fine-tuning of a small encoder for a specialised similarity task) is **established**; whether it beats a strong general-purpose baseline in duplicate-detection domains is **contested** ([Rosane et al.](https://doi.org/10.5753/sbes.2025.9809), [Jiang et al.](https://doi.org/10.1016/j.jss.2023.111607)); and **which negative examples you use** — interacting with the fact that most true duplicates are never labelled — is the **decisive, unmeasured variable**. That is where a one-month undergraduate project can do real, defensible work.
