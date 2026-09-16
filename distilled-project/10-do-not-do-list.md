# 10 — Do not do this

Twenty ways this project becomes weak. Each item says what goes wrong, why, and what to do instead. Several of these are the difference between a project that gets a good grade and one that is quietly wrong.

---

## Data and splits

### ❌ 1. Split randomly over queries
If a question and its duplicate land on opposite sides of the split, the model can memorise surface form and "retrieve" it. Worse, duplicate groups are connected components, so a two-hop chain leaks too.
**✅ Do instead:** assign whole **duplicate groups** to one split; print and assert that train IDs and test-gold IDs do not intersect.

### ❌ 2. Construct the split *after* generating pairs
Pair generation bakes in the split. If you generate pairs over the whole corpus and then split, positives straddle the boundary by construction.
**✅ Do instead:** split first, then generate pairs from the train split only.

### ❌ 3. Use the test set to choose a checkpoint, a learning rate, or an epoch count
This is "test-set feedback", one of the three failures [Musgrave et al. (2020)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf) identify in the metric-learning literature.
**✅ Do instead:** validation for everything; touch test exactly once, at the end.

### ❌ 4. Compare systems at different candidate-pool sizes
[Reimers & Gurevych (2021)](https://aclanthology.org/2021.acl-short.77/) proved dense retrieval degrades faster than sparse retrieval as the index grows, because low-dimensional embeddings admit more accidental near-matches.
**✅ Do instead:** one fixed pool for every system; report its size in every table caption.

---

## Pairs and negatives

### ❌ 5. Use only random negatives
Random negatives give almost no gradient — [FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf) observed that most random triplets are already satisfied, and [ANCE](https://openreview.net/forum?id=zeFrfgyZln) proved the resulting gradients vanish.
**✅ Do instead:** random is your N1 *baseline*, never your only configuration.

### ❌ 6. Assume "not marked duplicate" means "not a duplicate"
With ~1.4 labelled duplicates per query, many true duplicates are unlabelled. Treat them as negatives and you train the model to push away exactly the things it should pull together.
**✅ Do instead:** *measure* the false-negative rate for each strategy and report it. This is your most original contribution — do not skip it.

### ❌ 7. Mine the "hardest possible" negatives without a filter
[Robinson et al. (2021)](https://openreview.net/forum?id=CR1XOQ0UTh-) show that hard sampling **without** debiasing is worse than with it; [FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf) found the hardest negatives can collapse training.
**✅ Do instead:** N5 — filter with a margin rule or cross-encoder score; use `range_min` to skip the top few nearest candidates.

### ❌ 8. Generate all positives synthetically
LLM paraphrases teach paraphrase-invariance, not domain similarity, and they carry no information about which distinctions matter. Synthetic negatives have their own documented failure mode — a generative-discriminative gap that can actively *degrade* contrastive training ([arXiv:2606.01304](https://arxiv.org/abs/2606.01304)).
**✅ Do instead:** positives come only from human duplicate marks. Hard rule.

### ❌ 9. Let a positive appear in its own negative set
Sounds trivial; happens constantly when you exclude only the *direct* link rather than the whole duplicate group.
**✅ Do instead:** exclude by `group_id`, and write `test_pairs.py` to assert it.

---

## Baselines and evaluation

### ❌ 10. Have no generic / zero-shot baseline
Then any improvement is attributable to "I trained a model", not to contrastive fine-tuning.
**✅ Do instead:** **B2 = the same encoder, zero-shot.** Same architecture, same dimensions, same everything. This is the single most important control in the project.

### ❌ 11. Have no lexical baseline
[BEIR](https://arxiv.org/abs/2104.08663) shows BM25 is a robust zero-shot baseline; [Jiang et al. (2023)](https://doi.org/10.1016/j.jss.2023.111607) found DL methods losing to IR methods in duplicate-detection ranking. You may well lose too.
**✅ Do instead:** BM25 + TF-IDF, built in **week 2 before any training**.

### ❌ 12. Forget a model-specific quirk and beat a crippled baseline
BGE models need a query prefix (`"Represent this sentence for searching relevant passages: "`). Omit it and your "modern strong baseline" is artificially bad. This is exactly the unfair-comparison pattern [Musgrave et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf) document.
**✅ Do instead:** read each baseline model card; verify the prefix changes the score.

### ❌ 13. Report only accuracy (or only F1, or only AUC)
[Reimers et al. (2016)](https://aclanthology.org/C16-1009/) showed that even correlation choice can invert system rankings; [Jiang et al. (2023)](https://doi.org/10.1016/j.jss.2023.111607) showed duplicate-detection models look excellent under classification and lose under ranking.
**✅ Do instead:** retrieval metrics — **nDCG@10 primary**, Recall@k, MRR@10, MAP secondary. Classification metrics, if any, at *natural* prevalence.

### ❌ 14. Report only the average across subforums
[MTEB](https://aclanthology.org/2023.eacl-main.148/)'s headline finding is that **no embedding method dominates across tasks**; averages hide per-domain swings.
**✅ Do instead:** a per-subforum table plus the mean.

### ❌ 15. Report Pearson correlation as a headline number
[Reimers et al. (2016)](https://aclanthology.org/C16-1009/) measured its predictiveness for a downstream task at **ρ = −0.326** — worse than useless.
**✅ Do instead:** Spearman if you need a correlation at all; retrieval metrics otherwise.

### ❌ 16. Implement nDCG yourself and get it subtly wrong
A wrong metric invalidates every number in the paper, and the error is invisible.
**✅ Do instead:** `ranx` or `pytrec_eval`, plus a unit test on a toy example you can compute by hand.

---

## Statistics and claims

### ❌ 17. Report a single run
[Musgrave et al. (2020)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf) re-ran the metric-learning literature under a fair protocol and found that **most methods tie** once hyperparameters are tuned properly. One run cannot separate your effect from noise.
**✅ Do instead:** **3 seeds**, mean ± std, and a paired bootstrap over queries for headline comparisons.

### ❌ 18. Claim novelty because a dataset/model combination has not been tried
"Nobody has applied X to Y" is an experiment, not a contribution.
**✅ Do instead:** frame the contribution as the *controlled comparison and the measured false-negative rate*. See [`02`](02-solved-vs-open-gaps.md) Part D.

### ❌ 19. Decide what your hypothesis was after seeing the results
You will unconsciously shape the story to fit, and the write-up will read as if you knew all along.
**✅ Do instead:** `PRE_REGISTRATION.md`, written and committed in week 2, containing the RQ, H1, H0, primary metric, strategies, seeds and ablation plan.

### ❌ 20. Hide a negative or null result
The literature already suggests the gain may be small ([Rosane et al. 2025](https://doi.org/10.5753/sbes.2025.9809); [Jiang et al. 2023](https://doi.org/10.1016/j.jss.2023.111607)). A clean null with a measured mechanism is more impressive than a sloppy positive.
**✅ Do instead:** report it, then use the false-negative rate and the lexical-overlap buckets to explain it. That is the paper.

---

## Also do not (scope and engineering)

### ❌ Train a large model "for better results"
Scale is not a contribution, and it eats the compute you need for **3 seeds × 5 strategies**. [MTEB](https://aclanthology.org/2023.eacl-main.148/) confirms size helps — which is precisely why it is not interesting here. Stay at ≤150M parameters.

### ❌ Train an encoder from scratch
Explicitly out of scope. It needs data and compute you do not have, and it teaches you less about your research question.

### ❌ Propose a new loss function
Almost certainly a re-derivation of something in the InfoNCE family, and [Musgrave et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf) show most proposed losses tie with the classics once fairly tuned. One month is not enough to do this well.

### ❌ Make the project "call an embedding API and search"
That is Project A. It teaches you nothing about contrastive learning and contains no experiment.

### ❌ Tune heavily and then report the best number
Best-of-N reporting is selection bias.
**✅ Do instead:** pre-register hyperparameters where you can; sweep LR once on validation; report the sweep.

### ❌ Pick a dataset because it is easy to load, then discover it has no meaningful similarity labels
This is why requirements-similarity and CVE→CWE were eliminated in [`03`](03-feasibility-matrix-and-paths.md). Check for *naturally occurring* positives before committing.

### ❌ Skip the week-1 validation step
Reproducing a published CQADupStack number is the cheapest insurance you will ever buy. If your harness is wrong, everything downstream is worthless and you will not find out until week 4.

### ❌ Try to reproduce ANCE or RocketQA at full scale
They need multi-GPU training at MS MARCO scale. Adopt the *idea* (global ANN mining; denoising), not the infrastructure.
