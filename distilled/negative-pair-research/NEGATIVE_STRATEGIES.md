# Taxonomy of negative-pair strategies, and 12 candidates

---

# Part 1 — The taxonomy

Nine categories, in rough order of increasing sophistication. For each: how the negative is
generated, why it should help, representative papers, whether it is standard, its computational
cost, its weaknesses, and whether a 2nd-year project can use it.

---

## T1 — Random negatives

| | |
|---|---|
| **How generated** | Sample uniformly from examples not in the anchor's positive set. |
| **Why it should help** | It is the null case. It establishes that any improvement comes from *selection* and not from something else. |
| **Papers** | Universal baseline. Formalised as "static NS" in [Negative Sampling in Recommendation: A Survey](https://doi.org/10.1145/3793855). |
| **Standard?** | Yes — it is the floor every paper reports. |
| **Cost** | O(1). |
| **Weaknesses** | Most random negatives are trivially easy and produce near-zero gradient. Training is slow and the resulting boundary is loose. |
| **2nd-year suitable?** | **Yes, as a baseline only.** Never as a proposed method. |

## T2 — In-batch negatives

| | |
|---|---|
| **How generated** | For a batch of `N` (anchor, positive) pairs, the other `N−1` positives serve as negatives for each anchor. Free. |
| **Why it should help** | Gives `N−1` negatives at zero extra forward cost, and the InfoNCE softmax implicitly weights them by hardness. |
| **Papers** | [DPR](https://aclanthology.org/2020.emnlp-main.550/); [Henderson et al. 2017](https://arxiv.org/abs/1705.00652); standard in `sentence-transformers` as `MultipleNegativesRankingLoss`. |
| **Standard?** | **Yes — completely.** It is the default. |
| **Cost** | Zero extra. |
| **Weaknesses** | Negatives are only as hard as the batch is homogeneous; on a random shuffle most are easy. Assumes no two pairs in the batch are semantically related, which is false. Larger batches help, so quality is coupled to GPU memory. |
| **2nd-year suitable?** | **Yes — as the mechanism through which other strategies are applied.** In this project every strategy is expressed as *which pairs share a batch*, so the loss stays fixed. |

## T3 — Lexical (static) hard negatives

| | |
|---|---|
| **How generated** | For each anchor, take the top-k documents by BM25/TF-IDF, excluding the positive and the anchor's duplicate group. Computed once, before training. |
| **Why it should help** | High word overlap with low semantic relevance forces the model to stop equating keyword match with meaning. |
| **Papers** | [DPR](https://aclanthology.org/2020.emnlp-main.550/) (BM25 negatives), [ANCE](https://openreview.net/forum?id=zeFrfgyZln) (BM25 warm-up), [PassageBM25](https://aclanthology.org/2023.paclic-1.59/) (similarity to the positive instead of the query). |
| **Standard?** | Yes. |
| **Cost** | Low — one pass, no refresh. On a 15K corpus: seconds. |
| **Weaknesses** | Static, so they stop being hard as the model improves. Lexical similarity is a crude proxy for embedding-space hardness. ANCE measured only **15% overlap** between BM25 negatives and the model's own top retrieved negatives. |
| **2nd-year suitable?** | **Yes.** This is the N2 condition already in [`project/distilled`](../../project/distilled). |

## T4 — Model-based (dynamic) hard negatives

| | |
|---|---|
| **How generated** | Encode the corpus with the current model, take each anchor's top-k by cosine, use those as negatives. Refresh the index periodically (ANCE) or once (static variant). |
| **Why it should help** | These are hard *in the space the model actually cares about*, and they stay hard as the model changes. |
| **Papers** | [ANCE](https://openreview.net/forum?id=zeFrfgyZln), [RocketQA](https://aclanthology.org/2021.naacl-main.466/), [ADORE/STAR](https://arxiv.org/abs/2505.18366). |
| **Standard?** | Yes. |
| **Cost** | Medium — one corpus encoding per refresh. On a 15K corpus, seconds; on MS MARCO-scale corpora, minutes per refresh. |
| **Weaknesses** | **The false-negative problem lives here.** The model's nearest neighbours are exactly the unlabelled positives. [NV-Retriever](https://arxiv.org/abs/2407.15831) measured 38.8% (NQ) and 47% (StackExchange) false-negative rates for naive top-k. The asynchronous refresh is also fiddly to implement. |
| **2nd-year suitable?** | **Yes in the static, single-refresh form** (this is the N3 condition already in the project plan). **No** in the asynchronously-refreshed ANCE form. |

## T5 — Semi-hard / band / ring negatives (controlled similarity)

| | |
|---|---|
| **How generated** | Do not take the top-k. Take candidates whose similarity to the anchor falls in a **band** — typically defined relative to the positive's similarity, or by similarity percentile. |
| **Why it should help** | The band avoids both failure modes: too-easy negatives give no gradient, too-hard negatives are disproportionately false negatives. |
| **Papers** | [FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf) (semi-hard), [Ring/Conditional Negative Sampling, ICLR 2021](https://arxiv.org/abs/2010.02037), [SimANS, EMNLP 2022](https://aclanthology.org/2022.emnlp-industry.56/), [nmCSE, 2023](https://doi.org/10.1007/s10994-023-06408-8), [NV-Retriever TopK-PercPos](https://arxiv.org/abs/2407.15831) (95% of the positive score). |
| **Standard?** | **Yes — this is now the consensus best practice.** Four independent communities (face recognition, self-supervised CV, dense retrieval, sentence embeddings) converged on it. |
| **Cost** | Low. A band is a mask over an existing ranking. |
| **Weaknesses** | Introduces one or two hyperparameters (band position, band width) that must be set, and the optimum is dataset-dependent ([time-series evidence](https://www.sciencedirect.com/science/article/abs/pii/S0957417425036681): "an optimal threshold exists for each dataset"). |
| **2nd-year suitable?** | **Yes, but only as a variation.** This is the most crowded category in the whole taxonomy. |

## T6 — Adversarial and synthetic negatives

| | |
|---|---|
| **How generated** | Create negatives rather than find them: mix embeddings of hard negatives ([MoCHi](https://arxiv.org/abs/2010.01028)), mix in the query itself, inject positive information ([MixGCF](https://doi.org/10.1145/3793855)), apply adversarial perturbations ([AdCSE](https://doi.org/10.1145/3593590)), or rewrite the text ([SNCSE negation](https://arxiv.org/abs/2201.05979), antonym replacement in CLINE). |
| **Why it should help** | You control hardness precisely, and synthetic negatives are **guaranteed not to be false negatives** if constructed correctly. |
| **Papers** | [MoCHi](https://arxiv.org/abs/2010.01028), [SSCL](https://arxiv.org/abs/2304.02971), MixCSE, AdCSE, [SNCSE](https://arxiv.org/abs/2201.05979), SyNeg and the LLM thread. |
| **Standard?** | Partially — established in CV, active but contested in NLP. |
| **Cost** | Low for feature mixing (extra dot products); high for LLM generation. |
| **Weaknesses** | Mixed negatives can still be *fractionally* false negatives (MoCHi measured ~8%). Text-rewriting approaches risk teaching surface artefacts. The previous project already banned synthetic *positives*; synthetic negatives are a different case but carry the same "what did I actually teach it?" problem. |
| **2nd-year suitable?** | **Only in the simplest slot-swap form** (candidate C5), and with care. |

## T7 — Curriculum / progressive difficulty

| | |
|---|---|
| **How generated** | Rank candidates by difficulty and shift the sampling distribution from easy to hard over training (or anneal a hardness parameter). |
| **Why it should help** | Early in training the model cannot use very hard negatives; late in training easy ones are worthless. A schedule gives the model the right thing at the right time. |
| **Papers** | Bengio et al. 2009; [CuSINeS](https://arxiv.org/abs/2404.00590); [Zhuang et al. 2024](https://doi.org/10.1016/j.ins.2024.120534); [CurNM](https://arxiv.org/abs/2407.17070); SPCL, ConCur, ACGCL; and the β-annealing variant in [Robinson et al.](https://openreview.net/forum?id=CR1XOQ0UTh-). |
| **Standard?** | Established as a principle; still actively published in new domains. |
| **Cost** | Low — a schedule over a pre-computed ranking. |
| **Weaknesses** | Adds a pacing function as a hyperparameter. Confounds *when* with *what*: a curriculum run and a fixed run differ in both the negatives seen and the training trajectory, so a win does not isolate hardness. |
| **2nd-year suitable?** | **Yes, but it is a replication risk.** [CuSINeS](https://arxiv.org/abs/2404.00590) already reports curriculum > fixed on retrieval. |

## T8 — False-negative-aware methods

| | |
|---|---|
| **How generated** | Score candidates for the probability that they are secretly positives, then filter or reweight. Signals used: the positive's own score ([NV-Retriever](https://arxiv.org/abs/2407.15831), [GISTEmbed](https://arxiv.org/abs/2402.16829)), a cross-encoder ([RocketQA](https://aclanthology.org/2021.naacl-main.466/)), reciprocal-neighbour structure ([rNN](https://aclanthology.org/2023.emnlp-main.665/)), score variance ([SRNS](https://doi.org/10.1145/3793855)), or an explicit bias correction ([Debiased CL](https://proceedings.neurips.cc/paper_files/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html)). |
| **Why it should help** | You keep the gradient signal of hard negatives while removing the supervision that is actively wrong. |
| **Papers** | See above; also [DCLR](https://doi.org/10.1145/3593590) ("punish the false negatives"). |
| **Standard?** | **Yes, and it is the current frontier.** Positive-relative thresholding is now shipped in `sentence-transformers` as `GISTEmbedLoss`. |
| **Cost** | Low for threshold-based methods (~10 lines). High for cross-encoder denoising (a second model, a second scoring pass). |
| **Weaknesses** | Every filter is a heuristic with a tunable threshold, and removing hard negatives can remove genuinely useful ones — [NV-Retriever](https://arxiv.org/abs/2407.15831) found that large margins *hurt* because they removed informative negatives. |
| **2nd-year suitable?** | **Yes — and this is where the defensibly-open questions are.** Filtering is cheap; the open part is *measurement*: nobody has published a false-negative rate for AskUbuntu specifically. |

## T9 — Cross-signal, structural and domain-aware negatives

| | |
|---|---|
| **How generated** | Use a signal other than (or in addition to) one global similarity: agreement/disagreement between a lexical and a dense ranker; entity or slot structure; cluster or topic membership; document structure; time. |
| **Why it should help** | Two representations disagree precisely where the single-signal view is ambiguous — so disagreement is a cheap, unsupervised proxy for "informative" and for "possibly mislabelled". |
| **Papers** | [Intent-DQD](https://aclanthology.org/2023.findings-emnlp.596/) ("related but not duplicate" pairs as hard neutrals — *"lexical similarity but distinct semantic meanings"*); [EASE](https://doi.org/10.1145/3593590) (entity-aware negatives); [CuSINeS](https://arxiv.org/abs/2404.00590) (statute hierarchy + sequence structure); [ECI](https://arxiv.org/html/2603.20990v1) (BM25 + cross-encoder hybrid evaluation); [rNN](https://aclanthology.org/2023.emnlp-main.665/) (neighbourhood structure vs raw distance). |
| **Standard?** | **No — this is the least standardised category.** Each paper picks a different structural signal for its own domain. |
| **Cost** | Low, if the second signal is already computed (BM25 is, in this project). |
| **Weaknesses** | Domain-specific: the useful structure in legal statutes is not the useful structure in Ubuntu questions. Risk of building something that looks novel only because nobody else works on this dataset. |
| **2nd-year suitable?** | **Yes — this is the best category to look in.** Especially the two-signal disagreement variant, because both signals are already computed in the existing pipeline. |

---

# Part 2 — Twelve candidate strategies

**Unifying implementation mechanism.** Every candidate below is expressed the same way, so the
loss never changes: build a candidate pool per anchor, then draw the *other 31 pairs in each
mini-batch* from that pool. This is exactly the `NeighbourhoodBatchSampler` design already in
[`project/distilled`](../../project/distilled) — so switching strategy means changing one function
that returns a candidate list.

| Field | Meaning |
|---|---|
| **Novelty evidence** | What the literature says about this specific rule |
| **Risk of rediscovery** | Probability that a targeted search would surface this exact method |

---

### C0 — RandInBatch *(baseline)*
| Field | |
|---|---|
| **Name** | Random in-batch |
| **Core idea** | Shuffle pairs; the other 31 pairs in each batch are the negatives. |
| **How generated** | `hard_fraction = 0.0` in the existing sampler. |
| **Required info** | Nothing. |
| **Cost** | Zero. |
| **Novelty evidence** | Textbook. |
| **Closest existing method** | [DPR](https://aclanthology.org/2020.emnlp-main.550/) in-batch negatives. |
| **Difference** | None. |
| **Rediscovery risk** | N/A — it is the baseline. |
| **Impl. complexity** | 1/10 |
| **1-month feasible** | Yes |
| **Smallest experiment** | Train once, evaluate. This is the reference row of the table. |

### C0b — LexBM25 *(baseline)*
| Field | |
|---|---|
| **Name** | Lexical hard negatives |
| **Core idea** | Batch neighbours = top-64 by BM25 among training anchors. |
| **How generated** | `rank_bm25` over the training anchors; exclude the duplicate group. |
| **Required info** | Raw text. |
| **Cost** | Low (seconds). |
| **Novelty evidence** | Standard since DPR. |
| **Closest existing method** | [DPR](https://aclanthology.org/2020.emnlp-main.550/) BM25 negatives. |
| **Difference** | None. |
| **Rediscovery risk** | N/A — baseline. |
| **Impl. complexity** | 2/10 |
| **1-month feasible** | Yes |
| **Smallest experiment** | Train once, compare to C0. |

### C0c — ModelHardStatic *(baseline)*
| Field | |
|---|---|
| **Name** | Model-mined hard negatives, mined once |
| **Core idea** | Batch neighbours = top-64 by zero-shot embedding cosine among training anchors. |
| **How generated** | One matmul; exclude the duplicate group. |
| **Required info** | Embeddings from the un-fine-tuned encoder. |
| **Cost** | Low (seconds). |
| **Novelty evidence** | Standard (static variant of ANCE). |
| **Closest existing method** | [ANCE](https://openreview.net/forum?id=zeFrfgyZln), static version. |
| **Difference** | No refresh — deliberately, to keep the variable clean and the cost near zero. |
| **Rediscovery risk** | N/A — baseline. |
| **Impl. complexity** | 2/10 |
| **1-month feasible** | Yes |
| **Smallest experiment** | Train once, compare to C0 and C0b. |

---

### C1 — DisagreementQuad ⭐
| Field | |
|---|---|
| **Name** | Disagreement negatives (BM25 × embedding quadrants) |
| **Core idea** | Every candidate has two scores — BM25 rank and embedding cosine. Partition the candidate pool into four quadrants by whether each score is high or low, and train one model per quadrant. Lexical-vs-semantic disagreement is treated as the *selection variable*, not as noise. |
| **How generated** | 1. Rank all candidates by BM25 and by cosine. 2. Split each ranking at the median (or at the positive's score). 3. Four buckets: **A&G** (both high = agreed-hard), **D1** (BM25 high, cosine low = lexical-hard), **D2** (cosine high, BM25 low = *suspected false negative*), **N/N** (both low = easy). 4. Build batches from one bucket at a time. |
| **Required info** | BM25 ranking + embedding ranking + the duplicate groups. **All three already exist in the current pipeline.** |
| **Cost** | Low — two rankings and a 2×2 mask. |
| **Novelty evidence** | **Underexplored.** Every method I found selects on a *single* similarity signal ([SimANS](https://aclanthology.org/2022.emnlp-industry.56/), [NV-Retriever](https://arxiv.org/abs/2407.15831), [GISTEmbed](https://arxiv.org/abs/2402.16829), [DPR](https://aclanthology.org/2020.emnlp-main.550/)) or *combines* two signals into one fused ranking ([CuSINeS](https://arxiv.org/abs/2404.00590) via reciprocal rank fusion; [ECI](https://arxiv.org/html/2603.20990v1) evaluating BM25+CE hybrids). Using the *disagreement between the two* as the selection variable, with a per-quadrant false-negative measurement, did not surface. |
| **Closest existing method** | [CuSINeS](https://arxiv.org/abs/2404.00590) (combines lexical + semantic rankings — but fuses them, then applies a curriculum; does not partition by agreement); [ECI](https://arxiv.org/html/2603.20990v1) (evaluates BM25 vs CE hard-negative sets, but for *evaluation*, not selection). |
| **Difference** | We do not fuse or choose between the two signals — we use their disagreement to define four disjoint training sets, and we measure the false-negative rate of each. |
| **Rediscovery risk** | **Medium.** The components are all standard; the specific formulation may or may not exist. |
| **Impl. complexity** | 3/10 |
| **1-month feasible** | **Yes** |
| **Smallest experiment** | 4 training runs (one per quadrant) + 1 baseline, 3 seeds, Recall@10 on the fixed test split. Plus 50-query manual false-negative rate per quadrant. |

### C2 — BandWidth
| Field | |
|---|---|
| **Name** | Band position × band width |
| **Core idea** | Instead of "how hard", vary *which slice* of the similarity range negatives come from and *how wide* that slice is. Gives a 2-D hardness map rather than a 1-D dial. |
| **How generated** | Rank candidates by cosine; split into five equal-width bands (0–20%, 20–40%, …); additionally run one "narrow" (single decile) and one "wide" (all below the positive) variant. |
| **Required info** | Embedding ranking (exists). |
| **Cost** | Low. |
| **Novelty evidence** | **Variation.** Band selection is established ([SimANS](https://aclanthology.org/2022.emnlp-industry.56/), [Ring NS](https://arxiv.org/abs/2010.02037)); sweeping *width* as well as position, on a duplicate-detection corpus with a measured false-negative rate, is a small increment. |
| **Closest existing method** | [SimANS](https://aclanthology.org/2022.emnlp-industry.56/) (peaks the sampling distribution near the positive); [Ring NS](https://arxiv.org/abs/2010.02037) (similarity percentiles + annealing). |
| **Difference** | SimANS optimises a distribution; we hold the distribution fixed and vary the support, so the comparison is a plain ablation. |
| **Rediscovery risk** | **High.** This is close to SimANS's own ablations. |
| **Impl. complexity** | 2/10 |
| **1-month feasible** | **Yes** |
| **Smallest experiment** | 5–7 runs over band positions, plus 2 width variants. One line plot: Recall@10 vs. band. |

### C3 — HubAware
| Field | |
|---|---|
| **Name** | Hub-aware negative down-weighting |
| **Core idea** | In high-dimensional embedding spaces some points are nearest neighbours of almost everything ("hubs"). Hub documents will be over-sampled as hard negatives and are rarely genuinely informative. Count how often each document appears in someone else's top-k and down-weight or exclude the worst offenders. **Transferred from popularity-bias correction in recommender systems.** |
| **How generated** | 1. Mine top-k for every anchor. 2. Count per-document occurrence. 3. Compute the occurrence distribution; flag the top ~1% as hubs. 4. Sample batch neighbours from non-hubs only. |
| **Required info** | The mined neighbourhood matrix (already built). |
| **Cost** | Very low — a `np.bincount`. |
| **Novelty evidence** | **Potentially novel as a negative-selection rule.** Popularity-bias correction is standard in recommendation ([survey](https://doi.org/10.1145/3793855)); hubness is a well-documented phenomenon in high-dimensional nearest-neighbour search. I found **no** paper applying hub/popularity correction to hard-negative selection for text embeddings. |
| **Closest existing method** | Popularity-based negative sampling (PNS) and popularity correction in [recommender negative sampling](https://doi.org/10.1145/3793855); hubness literature in metric spaces. |
| **Difference** | Different field, different failure mode, and we use it as a *pre-filter on candidates*, not as a sampling distribution. |
| **Rediscovery risk** | **Medium.** The idea is simple enough that it may exist in the hubness/IR literature under another name. |
| **Impl. complexity** | 2/10 |
| **1-month feasible** | **Yes** |
| **Smallest experiment** | Report the hubness histogram; then 2 runs (hub-filtered vs not) × 3 seeds. |

### C4 — ReciprocalNN
| Field | |
|---|---|
| **Name** | Reciprocal-nearest-neighbour false-negative filter |
| **Core idea** | A candidate is much more likely to be a genuine (unlabelled) duplicate if the relationship is *mutual*: the candidate is in the anchor's top-k **and** the anchor is in the candidate's top-k. Drop those from the negative pool. |
| **How generated** | 1. Mine top-k for every anchor. 2. Keep candidate `c` for anchor `a` only if `a ∉ topk(c)`. 3. Build batches from the survivors. |
| **Required info** | The mined neighbourhood matrix (already built). |
| **Cost** | Very low — a set-membership test. |
| **Novelty evidence** | **Variation.** rNN is established ([Zerveas et al., EMNLP 2023](https://aclanthology.org/2023.emnlp-main.665/)) but used there for *evidence-based label smoothing* and for *reranking* — explicitly not as a negative-selection filter. The transfer is short but real. |
| **Closest existing method** | [rNN for dense retrieval](https://aclanthology.org/2023.emnlp-main.665/); also related to mutual-nearest-neighbour filtering in image retrieval. |
| **Difference** | We use rNN as a **training-time negative filter** with no extra model and no label smoothing — a change of role, not of mathematics. |
| **Rediscovery risk** | **Medium–high.** rNN-based filtering may well have been tried in retrieval. |
| **Impl. complexity** | 2/10 |
| **1-month feasible** | **Yes** |
| **Smallest experiment** | 2 runs (rNN-filtered vs not) × 3 seeds + a manual false-negative rate on the removed set. |

### C5 — SlotSwap
| Field | |
|---|---|
| **Name** | Slot-swap structural negatives |
| **Core idea** | In a technical support domain, swapping a version number or package name in a real duplicate produces a string that is nearly identical but is genuinely a **different** question ("install X on 12.04" vs "on 14.04" have different answers). These are maximum-hardness negatives with **known-correct labels**. |
| **How generated** | Regex-extract version numbers / package names from a duplicate; replace with another value drawn from the corpus; use the result as a negative. |
| **Required info** | Regexes and a value vocabulary. |
| **Cost** | Low. |
| **Novelty evidence** | **Potentially novel, domain-specific.** Synthetic negatives exist ([MoCHi](https://arxiv.org/abs/2010.01028) mixes embeddings; [SNCSE](https://arxiv.org/abs/2201.05979) negates sentences; SyNeg uses LLMs), but slot-swapping *technical identifiers* for a support forum is not something I found. |
| **Closest existing method** | [SNCSE](https://arxiv.org/abs/2201.05979) (rule-based negation as "soft negatives"); entity-aware negatives in [EASE](https://doi.org/10.1145/3593590). |
| **Difference** | We swap domain slots (versions, packages) rather than negating the sentence. |
| **Rediscovery risk** | **Low**, but the *idea risk* is high: the model may learn to ignore version numbers, which is exactly wrong for Ubuntu. |
| **Impl. complexity** | 4/10 (regex work plus a manual check that the swapped pair is genuinely not a duplicate) |
| **1-month feasible** | **Marginal** |
| **Smallest experiment** | 2 runs (with/without swapped negatives added) × 3 seeds + a manual audit of 30 swapped pairs. |

### C6 — PosRelFilter
| Field | |
|---|---|
| **Name** | Positive-relative margin filter |
| **Core idea** | Drop any candidate whose similarity to the anchor exceeds 95% of the positive's similarity — it is probably an unlabelled duplicate. |
| **How generated** | Threshold on the existing cosine ranking. |
| **Required info** | Embeddings + the positive's score. |
| **Cost** | Very low. |
| **Novelty evidence** | **Established.** [NV-Retriever TopK-PercPos](https://arxiv.org/abs/2407.15831) (optimum: 95% of the positive score), [GISTEmbedLoss](https://arxiv.org/abs/2402.16829), and shipped in `sentence-transformers`. |
| **Closest existing method** | [NV-Retriever](https://arxiv.org/abs/2407.15831). |
| **Difference** | None in the core rule. |
| **Rediscovery risk** | **Certain.** |
| **Impl. complexity** | 1/10 |
| **1-month feasible** | Yes — **as a control condition, not as the contribution.** |

### C7 — Curriculum3
| Field | |
|---|---|
| **Name** | Three-bucket easy→hard curriculum |
| **Core idea** | Bucket candidates into easy / medium / hard by cosine; shift the mix from easy to hard across epochs. |
| **How generated** | Extend the existing sampler's `hard_fraction` with a per-epoch schedule (0.0 → 0.5 → 1.0). |
| **Required info** | The existing sampler (one extra function). |
| **Cost** | Low. |
| **Novelty evidence** | **Established as a principle; still published per-domain.** [CuSINeS](https://arxiv.org/abs/2404.00590) reports curriculum > fixed on legal retrieval; [Zhuang et al. 2024](https://doi.org/10.1016/j.ins.2024.120534) on self-supervised CL; [Robinson et al.](https://openreview.net/forum?id=CR1XOQ0UTh-) annealed hardness in the appendix. |
| **Closest existing method** | [CuSINeS](https://arxiv.org/abs/2404.00590). |
| **Difference** | None in mechanism; ours would be a minimal 3-bucket version on a small duplicate-detection corpus. |
| **Rediscovery risk** | **High.** |
| **Impl. complexity** | 3/10 |
| **1-month feasible** | Yes |
| **Smallest experiment** | 2 runs (curriculum vs fixed) × 3 seeds. |

### C8 — ScoreVariance
| Field | |
|---|---|
| **Name** | Score-variance negative selection |
| **Core idea** | Transferred from SRNS in recommender systems: a false negative tends to be *consistently* high-scoring, whereas a genuinely hard-but-true negative has a high score that **fluctuates** as training progresses. Select candidates with high mean score *and* high score variance. |
| **How generated** | 1. Mine negatives at 2–3 checkpoints (e.g. after epoch 0, 1, 2). 2. Track each candidate's rank/score. 3. Sample from the high-mean, high-variance set. |
| **Required info** | Multiple mining passes → 2–3 extra training runs' worth of encodings. |
| **Cost** | Medium. |
| **Novelty evidence** | **Underexplored for text.** SRNS ([Ding et al. 2020](https://doi.org/10.1145/3793855)) applies it to implicit collaborative filtering. I found no text-embedding equivalent. |
| **Closest existing method** | SRNS in recommendation; uncertainty/active-learning selection ([QBC](https://arxiv.org/abs/2306.08954), BALD). |
| **Difference** | Different domain and different uncertainty proxy. |
| **Rediscovery risk** | **Medium.** |
| **Impl. complexity** | 6/10 |
| **1-month feasible** | **Marginal** — the extra mining passes roughly double the experiment count. |

### C9 — DistanceWeighted
| Field | |
|---|---|
| **Name** | Distance-weighted sampling |
| **Core idea** | Transferred from metric learning: sample negatives with probability proportional to the inverse of the distance density, so that *all* distances are represented instead of just the nearest. Corrects the variance problem of hardest-negative mining. |
| **How generated** | Compute all candidate distances; sample with weights ∝ 1/q(d), clipped. |
| **Required info** | Embedding matrix. |
| **Cost** | Low. |
| **Novelty evidence** | **Established in metric learning** ([Sampling Matters, ICCV 2017](https://arxiv.org/abs/1706.07567)); a transfer to text embeddings is a **variation**. |
| **Closest existing method** | [Sampling Matters](https://arxiv.org/abs/1706.07567). |
| **Difference** | Applied to in-batch composition for sentence embeddings. |
| **Rediscovery risk** | **High** (as a method); the transfer is cheap but the increment is small. |
| **Impl. complexity** | 3/10 |
| **1-month feasible** | Yes |

### C10 — ClusterBoundary
| Field | |
|---|---|
| **Name** | Cluster-boundary negatives |
| **Core idea** | Cluster the training anchors by embedding (k-means). Negatives drawn from the *same* cluster are topically hard; from the *nearest neighbouring* cluster they are boundary-hard; from a *distant* cluster they are easy. Compare the three. |
| **How generated** | `sklearn.KMeans(n_clusters=50)` on the training embeddings; define the three pools. |
| **Required info** | Embeddings + sklearn. |
| **Cost** | Low (k-means on 12,745 × 384 is seconds). |
| **Novelty evidence** | **Variation.** Class/cluster-aware sampling is standard in metric learning; "same topic, different intent" is close to [Intent-DQD](https://aclanthology.org/2023.findings-emnlp.596/)'s framing. |
| **Closest existing method** | Class-aware sampling in metric learning; [Intent-DQD](https://aclanthology.org/2023.findings-emnlp.596/). |
| **Difference** | Uses unsupervised clusters rather than topic labels or forum link types. |
| **Rediscovery risk** | **High.** |
| **Impl. complexity** | 4/10 |
| **1-month feasible** | Yes |

### C11 — TemporalNeg
| Field | |
|---|---|
| **Name** | Temporal negatives |
| **Core idea** | Prefer negatives from a different time period, or from the same period, and see which produces a better encoder. |
| **How generated** | Needs a creation timestamp per question. |
| **Required info** | **Timestamps.** |
| **Cost** | N/A |
| **Novelty evidence** | Temporal negative mining is an active area for temporal graphs ([CurNM](https://arxiv.org/abs/2407.17070), ENS) and video (shuffled-frame negatives). |
| **Closest existing method** | [CurNM](https://arxiv.org/abs/2407.17070). |
| **Difference** | N/A |
| **Rediscovery risk** | N/A |
| **Impl. complexity** | 2/10 (given timestamps) |
| **1-month feasible** | **No — the dataset does not support it.** The `sentence-transformers/askubuntu` rows contain question text only; there are no timestamps. Rejected at the data-availability stage, as the brief instructs. |

---

## Feasibility matrix

**Not** ranked by scientific quality. These are project-planning dimensions only.

| Candidate | Coding effort | Data requirement | Compute | Experiment simplicity | Literature support | Novelty uncertainty | 1-month feasible |
|---|---|---|---|---|---|---|---|
| C0 / C0b / C0c (baselines) | 1 | None | Very low | 10 (cleanest) | Very strong | N/A | **Yes** |
| **C1 DisagreementQuad** | 3 | None extra | Low | 9 | Strong | Medium | **Yes** |
| C2 BandWidth | 2 | None extra | Low | 9 | Very strong | Low (≈ high rediscovery) | **Yes** |
| C3 HubAware | 2 | None extra | Very low | 9 | Moderate (cross-field) | **High** | **Yes** |
| C4 ReciprocalNN | 2 | None extra | Very low | 9 | Strong | Medium | **Yes** |
| C5 SlotSwap | 4 | Regexes + manual audit | Low | 6 | Weak | High | Marginal |
| C6 PosRelFilter | 1 | None extra | Very low | 10 | Very strong | None (established) | Yes (as control) |
| C7 Curriculum3 | 3 | None extra | Low | 7 | Strong | Low (≈ high rediscovery) | Yes |
| C8 ScoreVariance | 6 | Multiple mining passes | Medium | 5 | Moderate | Medium | Marginal |
| C9 DistanceWeighted | 3 | None extra | Low | 8 | Strong (different field) | Low (≈ high rediscovery) | Yes |
| C10 ClusterBoundary | 4 | sklearn only | Low | 7 | Moderate | Low | Yes |
| C11 TemporalNeg | — | **Timestamps absent** | — | — | — | — | **No** |

*Coding effort, experiment simplicity: 1 = trivial, 10 = hard (experiment simplicity is inverted —
10 means "easiest to keep clean"). Novelty uncertainty: High = I am genuinely unsure whether this
exists; Low = I am confident it exists.*

**Reading the matrix:** C1, C3 and C4 are the only candidates that combine low coding effort,
clean experiments, *and* genuine novelty uncertainty. C2, C7, C9 and C10 are easy but are
variations on published work. C5 and C8 are interesting but do not fit the month.
