# Novelty audit

Nothing in this folder claims anything is new. This document records what a targeted search
found, what it did not find, and how confident each label is.

## Labels used

| Label | Meaning |
|---|---|
| **Established** | Clearly described in prior literature. Reproducing it is an exercise, not a contribution. |
| **Variation** | A known idea applied differently, or with a minor modification. Defensible as a project; not a novelty claim. |
| **Underexplored** | Related work exists, but the exact formulation or application did not surface in a targeted search. |
| **Potentially novel** | No direct prior work found after a targeted search. **Novelty has NOT been established.** |

**No claim in this folder says "this has never been done."** The search was ~18 queries across
Google Scholar, Semantic Scholar, arXiv, ACL Anthology, ACM DL, IEEE Xplore, OpenReview and library
documentation. That is a targeted search, not a systematic review. A targeted search that fails is
weak evidence.

---

## Part 1 — Direction-by-direction assessment

### A. Controlled similarity (select negatives at a chosen semantic distance)

**Status: Established.**

Five independent strands converge on "pick a band, not the extreme":

| Work | Field | What it establishes |
|---|---|---|
| [FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf) (2015) | Face recognition | Semi-hard negatives; hardest negatives cause collapse |
| [Sampling Matters](https://arxiv.org/abs/1706.07567) (2017) | Metric learning | Hardest-negative mining is noise-dominated; sample by distance instead |
| [Ring / Conditional Negative Sampling](https://arxiv.org/abs/2010.02037) (ICLR 2021) | Self-supervised CV | "Ring" around the positive; similarity percentiles; annealing |
| [SimANS](https://aclanthology.org/2022.emnlp-industry.56/) (EMNLP 2022) | Dense retrieval | Negatives ranked around the positive have larger gradient means, smaller variances, lower false-negative risk |
| [nmCSE](https://doi.org/10.1007/s10994-023-06408-8) (2023) | Sentence embeddings | "Distance moderation": weight negatives at moderate distance, down-weight nearest and farthest |

Plus [NV-Retriever](https://arxiv.org/abs/2407.15831) (2024) puts the optimum at *95% of the
positive's score*.

**Conclusion:** "Is there an optimal similarity range?" has been asked and answered, several
times, in several fields. A project that re-asks it on AskUbuntu is a replication. The only
defensible increment is the *width* dimension (candidate C2) or a *measurement* of where the
optimum falls in a corpus with 95% unlabelled positives — and that is an empirical finding, not a
method.

---

### B. Lexical vs. semantic disagreement

**Status: Partially established; one corner underexplored.**

| Sub-case | Status | Evidence |
|---|---|---|
| **High lexical, low semantic** → use as hard negative | **Established** | This is literally what DPR's BM25 negatives are. Also [Intent-DQD](https://aclanthology.org/2023.findings-emnlp.596/): "related but not duplicate" pairs "exhibit lexical similarity but possess distinct semantic meanings… can serve as hard negative labels". And [SNCSE](https://arxiv.org/abs/2201.05979)'s rule-based negation. |
| **Low lexical, high semantic** → deliberately select these | **Underexplored** | I found no paper that *selects* this quadrant as a training-negative class. The obvious reason is that this quadrant is exactly where false negatives are expected to live, so the literature treats it as contamination to remove ([RocketQA](https://aclanthology.org/2021.naacl-main.466/), [NV-Retriever](https://arxiv.org/abs/2407.15831), [GISTEmbed](https://arxiv.org/abs/2402.16829)) rather than as a condition to test. |
| **Use the disagreement itself as the selection variable** | **Underexplored** | See Part 2. |

**Why this matters:** the asymmetry is the interesting part. One quadrant is the standard recipe;
the other is standard practice to *delete*. Testing them side by side, and measuring the
false-negative rate of each, is a clean experiment whose outcome is not predictable from the
literature.

---

### C. Controlled false negatives

**Status: Established as a technique; underexplored as a measurement.**

* **Detection:** positive-relative threshold ([NV-Retriever](https://arxiv.org/abs/2407.15831),
  [GISTEmbed](https://arxiv.org/abs/2402.16829)), cross-encoder denoising
  ([RocketQA](https://aclanthology.org/2021.naacl-main.466/)), reciprocal-neighbour structure
  ([rNN](https://aclanthology.org/2023.emnlp-main.665/)), score variance
  ([SRNS](https://doi.org/10.1145/3793855)), explicit bias correction
  ([Debiased CL](https://proceedings.neurips.cc/paper_files/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html)).
  **All established.**
* **Estimating the rate:** [RocketQA](https://aclanthology.org/2021.naacl-main.466/) (≈70% of
  top-retrieved unlabelled passages are relevant, as reported in
  [this survey](https://arxiv.org/html/2603.18005v1) — verify against the primary paper);
  [NV-Retriever](https://arxiv.org/abs/2407.15831) (38.8% NQ, 47% StackExchange);
  [Lei et al.](https://aclanthology.org/N16-1153/) (only 5% of similar pairs annotated).
  **No published false-negative rate exists for AskUbuntu specifically with a modern encoder.**
  That is a small, real, citable gap — and it is a measurement, not a method.
* **Removing them:** established, with the caveat that aggressive removal hurts
  ([NV-Retriever](https://arxiv.org/abs/2407.15831): large margins penalise accuracy).

**Keep the implementation extremely simple:** a threshold on the existing cosine ranking
(~10 lines). Do not build a cross-encoder denoiser — that is [RocketQA](https://aclanthology.org/2021.naacl-main.466/)
and it costs a second model and a second scoring pass.

---

### D. Difficulty curriculum

**Status: Variation (established principle, still published per-domain).**

The principle is Bengio et al. 2009. Applied to negatives:
[CuSINeS](https://arxiv.org/abs/2404.00590) (legal retrieval, curriculum > fixed),
[Zhuang et al. 2024](https://doi.org/10.1016/j.ins.2024.120534) (adaptive curriculum + weight
regularisation), [CurNM](https://arxiv.org/abs/2407.17070) (temporal graphs), and — importantly —
the β-annealing variant in [Robinson et al.](https://openreview.net/forum?id=CR1XOQ0UTh-)'s own
appendix.

**Problem specific to a one-month project:** a curriculum confounds *which negatives* with *when
they appear*. A win does not isolate hardness. And the most likely result (curriculum ≥ fixed) is
already reported. Downgraded from the shortlist on that basis.

---

### E. Domain-aware negatives (same topic / different intent)

**Status: Variation.**

[Intent-DQD](https://aclanthology.org/2023.findings-emnlp.596/) is the direct hit: it extracts
"intent" pairs from CQA forums and explicitly uses *"related but not duplicate"* link types as
hard neutrals, noting they have lexical similarity but different meaning. It evaluates on two
datasets across fifteen domains.

**Data constraint:** it relies on StackExchange `PostLinks` (`LinkTypeId = 3` for duplicates,
`LinkTypeId = 1` for linked/related). The AskUbuntu HuggingFace dataset
([`sentence-transformers/askubuntu`](https://huggingface.co/datasets/sentence-transformers/askubuntu))
ships **only** duplicates — no linked/related pairs. Obtaining them means processing the raw
Stack Exchange dump, which is exactly the kind of data engineering that ate the previous project's
first week. **Rejected on data-availability grounds, not on novelty grounds.**

An unsupervised approximation (clusters instead of forum link types) is candidate C10, but that
moves it back towards "variation on cluster-aware sampling", which is standard in metric learning.

---

### F. Feature-disagreement negatives

**Status: Underexplored. This is the most promising corner.**

Closest work found:

| Work | What it does | What it does **not** do |
|---|---|---|
| [CuSINeS](https://arxiv.org/abs/2404.00590) | Ranks negatives by lexical *and* semantic proximity, **fuses** the rankings with reciprocal rank fusion, then applies a curriculum | Does not partition candidates by whether the two signals agree |
| [ECI](https://arxiv.org/html/2603.20990v1) | **Evaluates** hard-negative sets mined by BM25 vs cross-encoder vs LLM, on signal/safety axes | Evaluates sets for quality; does not use disagreement as a training-selection variable |
| [rNN](https://aclanthology.org/2023.emnlp-main.665/) | Uses neighbourhood **structure** as a second signal alongside geometric distance | Uses it for label smoothing and reranking, not for selection |
| [PassageBM25](https://aclanthology.org/2023.paclic-1.59/) | Uses similarity-to-the-positive as an alternative signal | Compares signals; does not cross them |

I found **no** paper that partitions the candidate pool into quadrants by lexical-vs-dense
agreement and trains a separate model on each quadrant, with a per-quadrant false-negative
measurement.

**Caveat, stated plainly:** this is an obvious enough idea that its absence from my search may be
a search failure rather than a literature gap. The conceptual ancestor — disagreement sampling —
is textbook active learning ([QBC](https://arxiv.org/abs/2306.08954), BALD). See Part 4 for the
searches to run before claiming anything.

---

### G. Structural negatives

**Status: Established in specific forms; domain-dependent.**

* Entity-aware negatives: [EASE](https://doi.org/10.1145/3593590) (Wikipedia entities).
* Grammatical structure: [CLINE](https://doi.org/10.1145/3593590) (antonym replacement),
  [SNCSE](https://arxiv.org/abs/2201.05979) (rule-based negation).
* Document structure: [CuSINeS](https://arxiv.org/abs/2404.00590) (statute hierarchy and sequence).
* Forum link structure: [Intent-DQD](https://aclanthology.org/2023.findings-emnlp.596/).

Each of these uses a *different* structural signal for a *different* domain. "Structure-aware
negatives" as a general concept is established. The only version that is cheap here is slot
swapping (C5), and it carries a real risk of teaching the model to ignore the exact surface
features that matter in Ubuntu support.

---

### H. Temporal negatives

**Status: Not applicable to this dataset.**

Temporal negative mining is active for temporal graphs ([CurNM](https://arxiv.org/abs/2407.17070),
ENS) and video (frame-order corruption as a hard negative). But
[`sentence-transformers/askubuntu`](https://huggingface.co/datasets/sentence-transformers/askubuntu)
contains question text only — no timestamps. Per the brief ("only consider this if the dataset
supports it naturally"), **rejected**.

---

## Part 2 — Is "negative difficulty" actually the research question?

**Short answer: no. It has been answered, and the answer is the inverted-U.**

This is the most important negative finding in this folder, because "how hard should a negative
be?" is the first question anyone asks and the literature has settled it:

| Source | Result |
|---|---|
| [FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf) | Hardest negatives → collapsed embedding; semi-hard chosen instead |
| [Sampling Matters](https://arxiv.org/abs/1706.07567) | Hardest-negative mining is dominated by gradient noise |
| [Robinson et al.](https://openreview.net/forum?id=CR1XOQ0UTh-) | Hardness is a dial; too hard degrades the representation |
| [SimANS](https://aclanthology.org/2022.emnlp-industry.56/) | Around-the-positive is the sweet spot (larger gradient mean, lower variance, lower false-negative risk) |
| [NV-Retriever](https://arxiv.org/abs/2407.15831) | Optimum cut-off = 95% of the positive's score; naive top-k carries 38.8%/47% false negatives |
| [Time-series filter (2026)](https://www.sciencedirect.com/science/article/abs/pii/S0957417425036681) | "An optimal threshold exists for each dataset"; unsupervised hardest hurts, **supervised** hardest helps |
| [Survey](https://arxiv.org/abs/2402.17238) | "Only 5% hardest negatives are necessary" |

The last row of that table is the key insight: **the position of the optimum is controlled by how
incomplete your labels are.** With complete labels, harder is monotonically better. With
incomplete labels, the hardest negatives are false negatives and the curve turns over.

That is why the difficulty question is *interesting again* specifically in this project's setting —
AskUbuntu has ~95% unlabelled positives
([Lei et al.](https://aclanthology.org/N16-1153/)). But "what happens to the difficulty curve when
95% of positives are unlabelled?" is a **new application** of a known result, not a new result.

**Recommendation:** do not make "how hard?" the research question. Make it a *measured covariate*
of whatever question you do ask.

---

## Part 3 — The false-negative risk (this gets its own section)

A candidate negative is a false negative if it is semantically equivalent to the anchor but the
data does not label it as a duplicate. In AskUbuntu this is not a tail case — it is the norm:

> *"Our manual inspection of a sample set of questions from AskUbuntu shows that only 5% of
> similar pairs have been annotated by the users, with a precision of around 79%."*
> — [Lei et al., NAACL 2016](https://aclanthology.org/N16-1153/), §1

Three consequences, all of which must appear in the write-up:

1. **All metrics are lower bounds.** A correct retrieval that is not in the gold set is scored as
   an error.
2. **The bias is differential.** A better model finds *more* unlabelled duplicates, so it is
   penalised more. If strategy X beats strategy Y by 2 points, one candidate explanation is that X
   found more unlabelled duplicates, not that X is better.
3. **Therefore the false-negative rate must be measured, not assumed.** Without it, a null result
   is uninterpretable and a positive result is ambiguous.

**Methods for detecting probable false negatives, simplest first:**

| Method | Lines | Needs | Status |
|---|---|---|---|
| Positive-relative threshold (`sim(a,n) > 0.95 × sim(a,p)`) | ~5 | Embeddings | [NV-Retriever](https://arxiv.org/abs/2407.15831) / [GISTEmbed](https://arxiv.org/abs/2402.16829); shipped in `sentence-transformers` |
| Reciprocal-NN test (`n ∈ topk(a)` **and** `a ∈ topk(n)`) | ~10 | Neighbourhood matrix | [rNN](https://aclanthology.org/2023.emnlp-main.665/) (different role) |
| Hub/popularity filter | ~10 | Neighbourhood matrix | Cross-field (recommenders) |
| Score variance across checkpoints | ~30 | 2–3 mining passes | [SRNS](https://doi.org/10.1145/3793855) |
| Cross-encoder rescoring | ~50 | Second model | [RocketQA](https://aclanthology.org/2021.naacl-main.466/) — **out of scope** |

**Manual estimation — the one that matters most.** Take 50 test queries, take the top-5 results
that are *not* in the gold set, read them, and mark each as genuine duplicate / related but
different / unrelated. ~250 judgements, about an hour. This produces the number that explains
your results, and **no published value exists for AskUbuntu with a modern encoder.** That is a
small but real contribution in itself.

**The specific research question worth asking:**

> Does selecting negatives from a **controlled similarity interval** produce better training
> examples than selecting the hardest possible negatives?

**Is it already answered?** Partially. [SimANS](https://aclanthology.org/2022.emnlp-industry.56/)
answers "yes, an interval beats the extreme" for passage retrieval, and
[NV-Retriever](https://arxiv.org/abs/2407.15831) pins down the cut-off. It is **not** answered for
duplicate-question retrieval on a corpus where the unlabelled-positive rate is ~95% — and in that
regime the interval may need to be much gentler, or may collapse entirely. That is a legitimate
empirical question, but it is an *application* finding.

---

## Part 4 — Claim / evidence table for the recommended candidate (C1)

Full specification in [`RECOMMENDED_DIRECTION.md`](RECOMMENDED_DIRECTION.md).

| Claim | Evidence | Closest prior work | Difference from our idea |
|---|---|---|---|
| "Hard negatives help" is established | [DPR](https://aclanthology.org/2020.emnlp-main.550/), [ANCE](https://openreview.net/forum?id=zeFrfgyZln) | — | Not our claim |
| "There is an optimal similarity band" is established | [FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf), [SimANS](https://aclanthology.org/2022.emnlp-industry.56/), [Ring NS](https://arxiv.org/abs/2010.02037), [NV-Retriever](https://arxiv.org/abs/2407.15831) | — | Not our claim; we hold the band fixed |
| Existing methods select on **one** similarity signal | Every method reviewed (SimANS, NV-Retriever, GISTEmbed, DPR, ANCE, RocketQA) uses one score | — | — |
| Two signals are sometimes **combined** | [CuSINeS](https://arxiv.org/abs/2404.00590) fuses lexical + semantic rankings by RRF; [ECI](https://arxiv.org/html/2603.20990v1) evaluates BM25 + cross-encoder hybrids | CuSINeS | **We do not fuse.** We partition the candidate pool by agreement, train one model per quadrant, and compare |
| Disagreement is a known general selection principle | Query-by-committee / BALD in active learning ([survey](https://arxiv.org/abs/2306.08954)) | QBC/BALD | They select points **to label**; we select points **to train against** |
| The "embedding-high / lexical-low" quadrant is where false negatives live | [Lei et al.](https://aclanthology.org/N16-1153/) (5% annotated); [NV-Retriever](https://arxiv.org/abs/2407.15831) FN rates; [rNN](https://aclanthology.org/2023.emnlp-main.665/) | RocketQA/NV-Retriever | They **remove** this region; we **train on it** as an explicit condition and measure what happens |
| No published false-negative rate for AskUbuntu with a modern encoder | Not found in search | — | Our measurement would be new (but small) |

---

## Part 5 — The five honest questions, answered

**1. What exactly is already known?**
That hard negatives beat random; that an intermediate hardness band beats the extreme; that the
hardest negatives are disproportionately false negatives; that positive-relative thresholds and
cross-encoder denoising both mitigate this; and that in retrieval corpora the false-negative rate
is large (38–47% measured, ~70% estimated).

**2. What exactly would our experiment add?**
Three things, in decreasing order of confidence:
* **A measurement:** the false-negative rate of AskUbuntu duplicate retrieval under a modern
  encoder, per quadrant. This is new and citable.
* **An empirical comparison:** four disjoint negative pools defined by lexical-vs-dense agreement,
  trained identically and compared. Whether this is new is uncertain.
* **A mechanism check:** whether the "semantic-hard / lexical-easy" quadrant really is where the
  false negatives are, tested by training on it rather than by assumption.

**3. Is the contribution a new algorithm, a new empirical finding, or a new application?**
**A new empirical finding in a new application, plus a small measurement.** Not a new algorithm.
The selection rule is a 2×2 mask.

**4. How strong is the novelty claim?**
**Weak to moderate.** I would write: *"We are not aware of prior work that partitions the negative
candidate pool by lexical/dense agreement and evaluates each partition separately; the closest
work is CuSINeS, which fuses the two rankings rather than partitioning by agreement."* That is
defensible. "We propose a novel negative sampling method" is not.

**5. What search should be performed before making any final novelty claim?**

Before writing the word "novel", run these and read the abstracts:

* Google Scholar / Semantic Scholar: `"lexical" AND "semantic" AND ("disagreement" OR "discrepancy") AND "negative sampling"`
* `("BM25" OR "lexical") AND ("dense" OR "embedding") AND "hard negative" AND ("combine" OR "hybrid" OR "ensemble")`
* `"reciprocal rank fusion" AND "hard negative mining"` (this is where a CuSINeS-like prior would hide)
* `"false negative" AND ("duplicate question" OR "community question answering") AND "contrastive"`
* `"similar but not duplicate" OR "related but not duplicate" AND negative` (Intent-DQD and neighbours)
* `"agreement" OR "consensus" AND "negative sampling" AND ("retrieval" OR "embedding")`
* arXiv full-text search for `"quadrant"` / `"2x2"` / `"two-signal"` with `negative sampling`
* Check the citation graph of [CuSINeS](https://arxiv.org/abs/2404.00590) and
  [SimANS](https://aclanthology.org/2022.emnlp-industry.56/) forwards and backwards — if this
  exists, that is where it will be
* Check `sentence-transformers` release notes and
  [losses documentation](https://sbert.net/docs/package_reference/sentence_transformer/losses.html)
  for anything recently added that subsumes it (the library already added `hardness_mode` to
  `MultipleNegativesRankingLoss`)

**If any of these turns up direct prior work, downgrade the label to Variation and keep going.** A
clean replication-with-measurement is still a fine undergraduate project.
