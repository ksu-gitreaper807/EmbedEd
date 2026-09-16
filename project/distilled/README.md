# The middle version — a serious but finishable 4-week project

**Contrastive learning for domain-specific embeddings, scaled to one undergraduate, one GPU,
four weeks.**

This folder is the **plan of record**. It sits deliberately between two other documents:

| | What it is | Where | Effort |
|---|---|---|---|
| **Version A** | In-batch negatives only, 2 metrics, 4 files, ~500 lines | described compactly in [`SCOPE.md`](SCOPE.md); full text in git history at commit `633db02` | 10–12 days |
| **Version B** ← *this document* | **Three negative-hardness conditions, 3 metrics, bootstrap CIs, 6 files, ~650 lines** | here | 4 weeks |
| **Version C** | CQADupStack, five negative strategies, two encoders, cross-domain, FAISS, denoising | [`distilled-project/`](../../distilled-project) | 8–12 weeks |

Version A was too small: its research question — "does fine-tuning help?" — is a reproduction
question whose answer the literature already roughly predicts, so a null result was both the
most likely outcome *and* an uninteresting one. Version C is a good research design that does
not fit in a month. This is the version in between: it puts back the one thing that turns a
reproduction into an experiment — **a manipulated variable** — while keeping everything that
made Version A finishable.

---

## What the project does

You take a small pretrained sentence encoder that knows general English, fine-tune it with a
contrastive objective on ~19,000 in-domain duplicate-question pairs, and measure whether that
helped — **and whether the answer depends on how hard the negative examples are.**

That second clause is the whole difference from Version A.

---

## Why it matters

Two facts collide in this project, and the collision is the research question.

1. **Domain adaptation of sentence embeddings is an established technique.**
   [TSDAE](https://aclanthology.org/2021.findings-emnlp.59/) reaches up to 93.1% of in-domain
   supervised performance without labels; [GPL](https://aclanthology.org/2022.naacl-main.168/)
   reports up to +9.3 nDCG@10 via pseudo-labelling.
2. **On duplicate-detection tasks specifically, it often does not help much, and keyword
   baselines are hard to beat.** [BEIR](https://arxiv.org/abs/2104.08663): "BM25 is a robust
   baseline", with dense models often underperforming out of domain.
   [Jiang et al.](https://doi.org/10.1016/j.jss.2023.111607): deep models do not reliably
   outperform traditional IR.

So "does it work?" has a likely answer (a little, sometimes) and an uninteresting null. But
*"what decides whether it works?"* has a real answer, and there is a specific candidate:
[Robinson et al. (ICLR 2021)](https://openreview.net/forum?id=CR1XOQ0UTh-) show that harder
negatives help **only up to a point** — the hardness of a negative is controllable, and pushing
it too far degrades the representation. Combine that with
[Lei et al. (NAACL 2016)](https://aclanthology.org/N16-1153/)'s measurement on this very corpus
that *"only 5% of similar pairs have been annotated by the users, with a precision of around
79%"* — and you have a concrete, testable prediction: **in a corpus where 95% of duplicates are
unlabelled, the hardest negatives are disproportionately unlabelled duplicates, so hardness
should help and then hurt.** That is an inverted-U, it is falsifiable, and it is measurable in
four weeks.

---

## Research question (one sentence)

> **When contrastively fine-tuning a small pretrained encoder for duplicate-question retrieval,
> does the hardness of the in-batch negatives determine whether fine-tuning beats (a) BM25 and
> (b) the same encoder with no fine-tuning?**

One independent variable (negative hardness, three levels). Everything else held fixed: same
model, same loss, same pairs, same batch size, same epochs, same corpus, same queries, same
metrics. That is what makes it answerable in four weeks.

---

## The pipeline

```text
text                     "how do i fix a 404 error when updating packages ?"
  ↓  tokenizer           word pieces + [CLS]/[SEP]
pretrained encoder       6-layer MiniLM, 23M params — one vector per token
  ↓  mean pooling        average the token vectors (masking padding), L2-normalise
embedding                a point on a 384-d sphere
  ↓  cosine similarity   a·b — the entire notion of "semantic similarity"
contrastive loss         "among the 32 candidates in this batch, your duplicate ranks first"
  ↓  (InfoNCE)           the other 31 are negatives, and THEIR IDENTITY IS THE VARIABLE
fine-tuned embedding     backprop; the 384-d space re-warps so Ubuntu duplicates align
```

Every arrow is one function call. Nothing else is in the model.

---

## The experiment

| System | What varies | What it tells you |
|---|---|---|
| **B1 — BM25** | — | the "do we need a neural model at all?" floor |
| **B2 — MiniLM, zero-shot** | — | **the control.** Identical to B3 except the fine-tuning |
| **B3-N1** | negatives = **random** in-batch | the plain fine-tuning result (this is all of Version A) |
| **B3-N2** | negatives = **lexically similar** (BM25 neighbourhoods of the anchor) | does topical hardness help? |
| **B3-N3** | negatives = **embedding-similar** (mined with the zero-shot model) | does semantic hardness help, or is it contaminated by unlabelled duplicates? |

All five rank the same ~15,000-question corpus for the same 400 held-out queries.

**How hardness is manipulated without changing the loss.** All three conditions use
`MultipleNegativesRankingLoss` on the same `(anchor, positive)` pairs. The only difference is
*which pairs share a mini-batch*: in N1 the batch is a random shuffle, so the 31 in-batch
negatives are random questions; in N2/N3 the batch is built from the anchor's neighbourhood, so
the 31 in-batch negatives are questions that look like the anchor. One parameter —
`hard_fraction ∈ [0, 1]` — controls the mix. **The loss function is bit-identical across
conditions**, which is a cleaner experiment than comparing different losses.

**Metrics:** Recall@10 (primary), MRR@10 and nDCG@10 (secondary).
**Uncertainty:** mean ± range over 3 seeds, **plus** a paired bootstrap over the 400 eval
queries for the three contrasts that matter.
**Mechanism check:** manually annotate the top-5 non-gold results for 50 queries → a measured
false-negative rate. This is what turns a null result into an explanation.

---

## The five result patterns (pre-registered)

| # | Shape | What you write |
|---|---|---|
| 1 | N1 < N2 < N3, all > B2 | "Harder in-batch negatives monotonically improve duplicate retrieval" — clean positive result |
| 2 | **N2 > N1 but N3 < N2** | **The interesting one.** "Hardness helps then hurts; the measured false-negative rate in N3's mined set is X%", which explains it |
| 3 | All within noise of each other | "Negative hardness does not matter here" — and the FN measurement says whether that is because hardness is irrelevant or because contamination cancels the gain |
| 4 | BM25 ≥ everything | "Keyword overlap dominates this corpus" — consistent with BEIR and Jiang et al. |
| 5 | Anything implausible (Recall@10 > 95%, BM25 < 2%) | **Your harness is broken.** Go to the debugging checkpoints. |

Pattern 2 is the one the literature predicts, and it is why this version is worth the extra
week over Version A.

---

## What will actually be built

```text
duqa/                            (your code repo — ~650 lines of Python)
├── README.md
├── requirements.txt
├── data/                        generated by prepare_data.py; small; committed
├── src/
│   ├── prepare_data.py          dataset → corpus, qrels, queries, train pairs
│   ├── negatives.py             neighbourhood construction + the batch sampler  ← the variable
│   ├── train.py                 fine-tune with MultipleNegativesRankingLoss
│   ├── evaluate.py              rank corpus; Recall@10 / MRR@10 / nDCG@10; bootstrap
│   ├── analyze.py               error analysis, FN annotation sheet, figures
│   └── utils.py                 seeding, encoding, metrics, jsonl I/O
├── experiments/run_all.sh       one command → every number in the report
├── results/                     metrics.json, 4 tables, 3 figures
└── report/report.md             8–10 pages
```

Six source files. `negatives.py` is the only file that contains anything novel — everything
else is glue, and that is the correct ratio.

---

## Stop condition

> **The project is finished when BM25, zero-shot MiniLM, and all three fine-tuned conditions
> (N1/N2/N3) have been evaluated on the fixed test split with Recall@10, MRR@10 and nDCG@10,
> over 3 seeds, and those numbers are in one table with bootstrap intervals on the key
> contrasts.**

If it is day 26 and you have that table, stop running experiments and start writing.

**Fallback:** if `negatives.py` is fighting you on day 20, drop to N1 only. You lose the
independent variable but you keep a complete, correct project — that is Version A, described in
[`SCOPE.md`](SCOPE.md), and it is a legitimate outcome rather than a failure.

---

## Final sanity check (answered honestly)

| Check | Answer |
|---|---|
| Can one student implement the core system in ~2 weeks? | **Yes, just.** Six files, ~650 lines; `negatives.py` is the only non-trivial one (~120 lines). |
| Can the other 2 weeks go to experiments and writing? | **Yes.** ~17 runs × ~5–10 minutes ≈ 2–3 hours of GPU total. The rest is reading output. |
| Does it run on one consumer GPU / Colab? | **Yes.** 23M params, ~15K corpus, no FAISS. A free T4 is plenty. |
| Can the student understand every major component? | **Yes.** Tokenizer → encoder → pool → cosine → InfoNCE → batch composition. Six ideas. |
| Is there exactly one main research question? | **Yes.** "Does negative hardness decide whether fine-tuning helps?" |
| Is there exactly one primary model? | **Yes.** `all-MiniLM-L6-v2`. (A second encoder is a SHOULD-HAVE, on the winning condition only.) |
| Is there exactly one primary training objective? | **Yes.** `MultipleNegativesRankingLoss`, identical across all conditions. |
| Is there a simple baseline? | **Yes, two.** BM25 and the un-trained version of your own model. |
| Is there a clear evaluation? | **Yes.** Recall@10 + MRR@10 + nDCG@10 on a fixed test split, 3 seeds, bootstrap CIs. |
| Is it complete with every optional extension removed? | **Yes.** Drop N3 and the ablation and you have Version A, which is complete on its own. |

---

## Documents in this folder

| File | What it is |
|---|---|
| [`PROJECT_SPEC.md`](PROJECT_SPEC.md) | Title, RQ, hypotheses, dataset, model, loss, pair construction, the three negative conditions, baselines, metrics, statistics, expected outputs |
| [`LITERATURE.md`](LITERATURE.md) | 8 essential papers (+1 optional, +3 reference), grouped and time-boxed |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | Repo structure, libraries, 8 implementation steps, 12 debugging checkpoints, 4-week plan |
| [`SCOPE.md`](SCOPE.md) | Why Version A was too small; in/out of scope; Versions A/B/C; stop condition |
| [`requirements.txt`](requirements.txt) | Five packages |

Start with [`PROJECT_SPEC.md`](PROJECT_SPEC.md). Then [`SCOPE.md`](SCOPE.md) before you write
a line of code, so you know where the walls are.
