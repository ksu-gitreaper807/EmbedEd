# Negative-pair construction: a search for a small, testable idea

**Goal of this folder:** find one simple question about *how negatives are chosen* that a
2nd-year CSE student can answer in about a month — not invent a new retrieval algorithm.

Companion to [`project/distilled/`](../../project/distilled) (the current project plan) and
[`distilled-project/`](../../distilled-project) (the larger version).

---

## The question, in plain language

When you train a sentence-embedding model by contrastive learning, you show it triples like:

```text
this question  →  is like  →  THIS one      (positive)
this question  →  is not like  →  that one     (negative)
```

The positives are given to you by the data (humans marked these two questions as duplicates).
**The negatives are not given to you — you have to choose them.** And the choice turns out to
matter a lot.

Choosing badly goes wrong in two opposite directions:

* **Too easy.** A negative about an unrelated topic teaches the model nothing; it is already far
  away in the embedding space and produces almost no gradient.
* **Too hard.** The hardest negatives are the questions that look *most* like your anchor. In a
  forum corpus, those are very often genuine duplicates that nobody happened to label. Training
  the model to push them away is training it to be wrong.

So the interesting region is in the middle, and where exactly that middle is — and how you find
it — is the research space.

---

## What I found, in one paragraph each

**1. "Hard negatives help" is thoroughly established.** It is the standard recipe in dense
retrieval ([DPR](https://aclanthology.org/2020.emnlp-main.550/) uses BM25 negatives;
[ANCE](https://openreview.net/forum?id=zeFrfgyZln) mines them with the model itself) and a
reproduction study is not a contribution.

**2. "How hard, exactly?" is largely answered too** — and the answer is *a band, not the
extreme*. [FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)
introduced semi-hard negatives; [Robinson et al. (ICLR 2021)](https://openreview.net/forum?id=CR1XOQ0UTh-)
made hardness a user-controllable dial; [Ring/Conditional Negative Sampling (ICLR 2021)](https://arxiv.org/abs/2010.02037)
samples "in a ring around the positive"; [SimANS (EMNLP 2022)](https://aclanthology.org/2022.emnlp-industry.56/)
samples negatives whose score sits near the positive's score;
[NV-Retriever (2024)](https://arxiv.org/abs/2407.15831) found 95% of the positive's score to be
the optimal cut-off. **So the inverted-U (easy < medium > hard) is the expected result, and
re-measuring it on a new dataset is a replication, not a finding.**

**3. The false-negative problem is the live part of the field** — and it is especially bad in
*this* domain. [Lei et al. (NAACL 2016)](https://aclanthology.org/N16-1153/) measured on the very
corpus this project uses that *"only 5% of similar pairs have been annotated by the users, with a
precision of around 79%"*.
[NV-Retriever](https://arxiv.org/abs/2407.15831) measured false-negative rates of **38.8%** (NQ)
and **47%** (StackExchange) for naive top-k mining. Everything interesting happens here.

**4. Almost all existing methods use ONE similarity signal to choose negatives** — either a
lexical score (BM25) or the model's own embedding score. The obvious second signal is the *other*
one. I found only partial work on using the two together
([CuSINeS](https://arxiv.org/abs/2404.00590) combines lexical and semantic rankings for legal
text; [ECI (2026)](https://arxiv.org/html/2603.20990v1) evaluates BM25+cross-encoder hybrids).
**The 2×2 partition of the candidate pool by "does BM25 agree with the embedding model?" is the
corner I could not find directly studied.**

---

## The shortlist

| # | Idea | One line | Novelty status |
|---|---|---|---|
| **C1** | **Disagreement Negatives** | Partition candidates by BM25-vs-embedding agreement into four quadrants; train one model per quadrant | **Underexplored** |
| C2 | Band position × width | Not just "how hard" but "which slice of the similarity range, and how wide" | Variation |
| C3 | Hub-aware negatives | Down-weight documents that are nearest neighbours of *everything* (hubness) — transferred from popularity-bias correction in recommender systems | **Potentially novel** |
| C4 | Reciprocal-NN filter | A candidate is a suspected false negative if the anchor is in *its* top-k and it is in *the anchor's* top-k | Variation |
| C5 | Slot-swap negatives | Swap version numbers / package names in a real duplicate to make a near-identical *non*-duplicate | **Potentially novel**, high risk |

Full specifications in [`CANDIDATE_EXPERIMENTS.md`](CANDIDATE_EXPERIMENTS.md).

---

## The recommendation

**Start with C1 — Disagreement Negatives.** Not because it is the most scientifically promising
(it is not obviously so), but because on the four axes that actually decide whether a one-month
undergraduate project finishes, it wins:

* **Implementation simplicity:** it reuses the BM25 neighbourhoods and the embedding
  neighbourhoods that [`project/distilled/`](../../project/distilled) already builds. The new code
  is a 2×2 mask over two ranked lists — roughly 30 lines.
* **Experimental cleanliness:** one variable (which quadrant the negatives come from), one loss,
  one model, one test set, four runs.
* **Built-in explanation:** each quadrant has a different expected false-negative rate, and you
  measure it by hand on 50 queries. So whatever the table says, you can say *why*.
* **Literature foundation:** it sits directly on top of five well-cited papers rather than in a
  vacuum.

Full write-up, including an honest novelty audit, in
[`RECOMMENDED_DIRECTION.md`](RECOMMENDED_DIRECTION.md).

---

## Read this before you get excited

Two warnings, and they matter more than the recommendation.

**Warning 1: "I could not find it" is not "it has not been done."** Every "potentially novel"
label in this folder means *a targeted search did not surface direct prior work*, nothing more.
The search was 18 queries across Google Scholar/Semantic Scholar/arXiv/ACL/ACM/IEEE/OpenReview
summaries, plus library documentation. It is not a systematic review.
[`NOVELTY_AUDIT.md`](NOVELTY_AUDIT.md) lists exactly what to search before you write the word
"novel" anywhere.

**Warning 2: the library is moving under your feet.** Recent `sentence-transformers` versions
added `hardness_mode` and `hardness_strength` to `MultipleNegativesRankingLoss`, and
`GISTEmbedLoss` / `CachedGISTEmbedLoss` ship positive-relative false-negative filtering out of the
box ([losses documentation](https://sbert.net/docs/package_reference/sentence_transformer/losses.html)).
Some of what was a research question in 2022 is a keyword argument in 2025. **Check what the
library already does before you implement anything** — and if the library already does it, that is
good news (less code) and bad news (less novelty), and you should re-read
[`NOVELTY_AUDIT.md`](NOVELTY_AUDIT.md).

---

## What this folder contains

| File | What is in it |
|---|---|
| [`STRATEGY_EVIDENCE.md`](STRATEGY_EVIDENCE.md) | **Start here for the three strategies you are actually implementing.** What the papers say about Random vs BM25 vs Semantic negatives, with merits and demerits attributed to each source and numbers read from the primary PDFs |
| [`LITERATURE.md`](LITERATURE.md) | Negative-pair-specific literature: the taxonomy's supporting papers, organised by category, plus the cross-field imports |
| [`NEGATIVE_STRATEGIES.md`](NEGATIVE_STRATEGIES.md) | The 9-category taxonomy (each with how/why/papers/cost/weakness/suitability) and 12 candidate strategies with the full field table |
| [`NOVELTY_AUDIT.md`](NOVELTY_AUDIT.md) | Per-direction (A–H) novelty assessment; "is difficulty already answered?"; the false-negative section; claim/evidence table |
| [`CANDIDATE_EXPERIMENTS.md`](CANDIDATE_EXPERIMENTS.md) | Five shortlisted ideas with the complete experiment chain, research question, hypothesis and negative-result reading |
| [`RECOMMENDED_DIRECTION.md`](RECOMMENDED_DIRECTION.md) | The recommended starting point in full, with the novelty audit and the five honest questions answered |
