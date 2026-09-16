# Project specification

This is the whole project on four pages. If a component is not described here, it is not part
of the project.

---

## Title

**Does It Help? Measuring the Effect of In-Domain Contrastive Fine-Tuning on Duplicate
Question Retrieval**

(Working title. Subtitle if you want one: *a controlled before/after study on AskUbuntu*.)

---

## Research question

> Does contrastive fine-tuning on in-domain duplicate-question pairs improve duplicate
> retrieval over the same encoder used off-the-shelf — and does either beat BM25?

One question, one independent variable (**fine-tuned vs. not**), everything else held constant:
same model, same weights at initialisation, same corpus, same queries, same metrics, same
preprocessing. That is what makes it answerable in four weeks.

---

## Hypothesis (write this down before you train anything)

**H1 (the textbook prediction):** fine-tuning improves Recall@10 over the zero-shot encoder by
a **visible** margin (≥ 3 points), because the model learns that the domain's duplicates share
*intent*, not vocabulary.

**H0 (equally likely, and equally publishable):** fine-tuning changes Recall@10 by **less than
3 points**, because (a) `all-MiniLM-L6-v2` was already trained on over a billion sentence pairs
including question-similarity data, so the general notion of "duplicate" is not new to it, and
(b) the training signal is only ~19,000 pairs, which is small for reshaping a 384-d space.

**Why both are fine.** If H1 holds you have a clean positive result with a measured effect size.
If H0 holds you have reproduced — on a new dataset, with a modern encoder — the finding that
domain-specific contrastive tuning gives *marginal* returns on duplicate detection
([Rosane et al., SBES 2025](https://doi.org/10.5753/sbes.2025.9809)), and you can say *why*
(section "Threats" below gives you three concrete mechanisms). Neither outcome is a failure.

**What would falsify the whole framing:** if BM25 beats both neural systems by a wide margin.
Then the story is "in this domain, keyword overlap is the signal", which is
[BEIR's](https://arxiv.org/abs/2104.08663) and
[Jiang et al.'s](https://doi.org/10.1016/j.jss.2023.111607) finding, and you write *that* paper.

---

## Dataset

**AskUbuntu duplicate questions**, as distributed by sentence-transformers:
[`sentence-transformers/askubuntu`](https://huggingface.co/datasets/sentence-transformers/askubuntu)

| Property | Value |
|---|---|
| Origin | AskUbuntu (Stack Exchange) 2014 dump; task and split introduced by [Lei et al., NAACL 2016](https://aclanthology.org/N16-1153/) |
| Rows | **13,145** — train 12,745 / dev 200 / test 200 |
| Row format | `query` (string), `positive` (list of 1–3 duplicate questions), `negative` (list of ~100 non-duplicate questions) |
| Text | **already lowercased and tokenised** — no HTML, no code blocks, no cleaning needed |
| Domain | Ubuntu/Linux technical support |
| Loading | one line: `load_dataset("sentence-transformers/askubuntu")` |
| Licence | not stated on the card; the underlying Stack Exchange content is CC BY-SA — check before publishing |

### Why this dataset and not CQADupStack

| | CQADupStack (old plan) | AskUbuntu (this plan) |
|---|---|---|
| Documents | 457K | ~15K |
| Loading | `ir_datasets` / BEIR, 12 subforums, per-subforum handling | one HF `load_dataset` call, splits already defined |
| Preprocessing | strip HTML and code blocks from raw StackExchange posts | **none** — text is pre-tokenised |
| Corpus encoding | minutes per run | **seconds per run** |
| Eval set | 13,145 queries across 12 domains | 400 queries in one domain |

The task is identical: *given a question, find its duplicates in an archive.* The smaller
dataset turns every experiment from "wait 20 minutes, hope it worked" into "wait 30 seconds,
look at it again". For a four-week project where you need to run ~10 training runs and inspect
failures by hand, that difference is worth more than the extra scale. The cost is that your
final numbers have a wider confidence band (200 test queries) — which you will report honestly
rather than hide.

### What you build from it (in `prepare_data.py`)

```text
corpus   = {id: text}  ← distinct texts from every `query` and every `positive`
                          (≈ 13,000–15,000 documents; print the exact number)
qrels    = {qid: [gold corpus ids]}  ← the `positive` lists, mapped to corpus ids
queries  = {qid: text} for the 200 dev + 200 test rows only
train_pairs = [(anchor, positive)]   ← from the 12,745 train rows only (≈ 17,000–20,000 pairs)
```

The `negative` lists are **not used** for training and **not used** as the retrieval corpus.
They are simply ignored. (They exist because the dataset ships them; you do not need them.)

### Splits — fixed, never re-touched

| Split | Size | Used for |
|---|---|---|
| train | 12,745 rows | building `(anchor, positive)` pairs |
| dev | 200 queries | **only** to choose the number of epochs from {1, 2, 3} |
| test | 200 queries | the headline number, looked at once |

---

## Model

**One model: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)**

| Property | Value |
|---|---|
| Parameters | 23M |
| Output | 384-d, L2-normalised |
| Layers / max tokens | 6 / 256 |
| Licence | Apache 2.0 |

It is small enough that a 3-epoch run takes minutes, which is what buys you 3 seeds and an
ablation inside four weeks. Model size is not the contribution here; the controlled comparison
is. Its zero-shot form *is* baseline B2, so the before/after comparison is exactly controlled.

**Input:** the question text, as-is. **Output:** a 384-d unit vector. Same preprocessing for
BM25's tokeniser and the encoder — no system gets a cleaner input than another.

### The six steps, explained

| Step | What happens | What it physically causes |
|---|---|---|
| **text → tokens** | word-piece tokenisation, `[CLS] … [SEP]`, pad to batch max | the model can only read text it has a vocabulary entry for; unknown words get split into pieces |
| **tokens → encoder** | 6 Transformer layers of self-attention | each token's vector is repeatedly mixed with the others', so "404" ends up carrying context from "updating packages" |
| **vectors → embedding** | mean pooling over non-padding tokens, then L2 normalise | the whole sentence becomes one point on a 384-d sphere; normalising means cosine = dot product |
| **embeddings → similarity** | `cos(a, b) = a · b` (both unit length) | the single number your entire evaluation is built on: 1 = same direction, 0 = unrelated |
| **similarity → loss** | InfoNCE (next section) | a gradient that pulls duplicates together and pushes everything else away |
| **loss → fine-tuned encoder** | backprop through all 23M params | the 384-d space is re-warped; "duplicate" now means *duplicate in Ubuntu*, not *duplicate in general English* |

---

## Loss

**One objective: `MultipleNegativesRankingLoss`** — the library's name for in-batch InfoNCE.
You do not implement it. You read twelve lines of code and understand the formula.

For a batch of `N` pairs `(a₁,p₁) … (a_N,p_N)` with cosine similarity and scale `s`:

```text
            exp( s · cos(aᵢ, pᵢ) )
L = − log  ─────────────────────────────
            Σⱼ exp( s · cos(aᵢ, pⱼ) )
```

averaged over the batch. `s = 20` (i.e. temperature τ = 0.05) is the library default.

| Term | What it is | What it physically causes |
|---|---|---|
| `cos(aᵢ, pᵢ)` | similarity of the anchor to **its own** duplicate | the numerator. Training pushes this up. |
| `cos(aᵢ, pⱼ)`, j ≠ i | similarity to **every other pair's** duplicate | the `N−1` negatives. Training pushes these down. They cost nothing — they are already in the batch. |
| `s = 20` (τ = 0.05) | inverse temperature | sharpens the softmax. Large `s` ⇒ the model is punished for *near misses*, not just gross errors. This is why the loss is "hard" even with random negatives. |
| `log` + minus | cross-entropy over "which of the `N` candidates is mine?" | makes it a classification problem: pick the right duplicate out of `N`. |
| `1/N · Σᵢ` | average over the batch | the usual. |

**The one sentence you must be able to say:** *"Every question in the batch has to pick its own
duplicate out of `N` candidates; the other `N−1` duplicates act as negatives for free, so the
number of negatives is `N−1` and you get them by increasing the batch size — that's why the
batch-size ablation is the interesting one."*

Why not exhaustive comparison? Ranking 15,000 × 400 exhaustively is fine at *evaluation* time.
At *training* time, comparing all 15,000 pairs would be 225M comparisons per step. In-batch
sampling gives you `N−1` negatives per example at zero extra cost, and it is what
[DPR](https://aclanthology.org/2020.emnlp-main.550/) and every sentence-embedding model since
uses.

---

## Positive examples

> **(a, p) is a positive pair iff `p` appears in `a`'s `positive` list, i.e. the AskUbuntu
> community marked `p` as a duplicate of `a`.**

* Built **only** from the 12,745 train rows.
* **Naturally occurring.** No paraphrasing, no back-translation, no LLM-generated pairs. This
  is a hard rule: synthetic positives teach paraphrase-invariance, which is a different skill
  from recognising a duplicate in a technical forum.
* **Do not add the reverse direction** in the core project. It doubles the pair count for free,
  but it also doubles the chance that two rows in one batch share a duplicate group, which
  creates false negatives. Keep it as a one-line optional experiment.
* Expected count: **≈ 17,000–20,000 pairs.** Print the exact number.

---

## Negative examples

> A candidate `c` is a negative for anchor `a` iff `c` is **another pair's positive in the same
> mini-batch** — the standard in-batch-negative construction, obtained for free by
> `MultipleNegativesRankingLoss`.

That is the whole negative strategy. No mining, no hard negatives, no FAISS, no cross-encoder.

**The honest weakness, which you must state in the report:** "not in my `positive` list" is not
the same as "not a duplicate". [Lei et al. (2016)](https://aclanthology.org/N16-1153/) measured
this on this very corpus: *"Our manual inspection of a sample set of questions from AskUbuntu
shows that only 5% of similar pairs have been annotated by the users, with a precision of
around 79%."* So roughly 95% of true duplicate pairs are sitting in your corpus unlabelled,
and a small fraction of the labels you do have are wrong.

Consequences, stated plainly:
1. Your measured Recall@10 **underestimates** every system, because some correct answers are
   scored as errors.
2. The fine-tuned model is penalised slightly *more* than the zero-shot model, because if it
   successfully learns to find duplicates it will surface some unlabelled ones.
3. Therefore a small positive effect is a **lower bound**, and a null result needs that
   caveat before you interpret it.

You do not try to fix this. You measure, report, and discuss it. Fixing it is a different project.

---

## Training configuration — fixed, not tuned

| Setting | Value | Where it comes from |
|---|---|---|
| Batch size | **32** | default for this model size; the ablation varies it |
| Epochs | **1, 2 or 3**, chosen on dev | only hyperparameter you are allowed to select |
| Learning rate | **2e-5**, AdamW | standard BERT-family fine-tuning value |
| Warmup | 10% of steps | library default |
| Precision | fp16 on Colab T4 | optional; harmless |
| Max sequence length | 256 (text is short) | library default |
| Seeds | **13, 42, 1337** | three runs, minutes each |
| Negative strategy | in-batch | fixed |

Do not tune the learning rate. Do not try five batch sizes. Do not try schedulers. If the model
is not learning at 2e-5, the bug is in your data, not your optimiser.

**One allowed hardening (SHOULD, not MUST):** add
`batch_sampler=BatchSamplers.NO_DUPLICATES` so two pairs from the same duplicate group cannot
land in one batch. If the API fights you for more than 20 minutes, drop it — with 12,745
anchors in batches of 32 the collision rate is low.

---

## Baselines

| ID | System | Lines of code | Why it is here |
|---|---|---|---|
| **B1** | **BM25** (`rank_bm25.BM25Okapi`, default k1/b) | ~15 | The literature's most important sanity check: [BEIR](https://arxiv.org/abs/2104.08663) finds "BM25 is a robust baseline" and dense models often fail to beat it out of domain. Without B1, "my model beat random" is the only claim you can make. |
| **B2** | **`all-MiniLM-L6-v2`, zero-shot** | ~3 | **The control.** Identical to B3 in every respect except the fine-tuning. This is the comparison the research question is about. |
| **B3** | **B2 after contrastive fine-tuning** | reuse | The system under test. |

**Not included:** TF-IDF (strictly weaker than BM25 and equally easy — it is the fallback if
`rank_bm25` will not install), any second encoder, any hybrid BM25+dense system.

---

## Evaluation

Same corpus, same 400 queries (200 dev + 200 test), same code path, all three systems.

| Metric | Definition | Why this metric |
|---|---|---|
| **Recall@10** (primary) | of the duplicate questions that exist for a query, what fraction appear in the top 10 results? (averaged over queries) | This is the user-facing question: if the forum shows a poster 10 "possible duplicate" suggestions, did the real duplicate make the list? It is on a 0–1 scale, needs no relevance grading, and has headroom on a 15K-document corpus — so a real improvement can actually show up. |
| **MRR@10** (secondary) | mean over queries of `1 / rank of first correct duplicate`, counting a miss at rank > 10 as 0 | Recall@10 is blind to *where* in the top 10 the answer landed. MRR@10 is not. Two metrics, ~15 lines of numpy total. |

**Deliberately excluded:** nDCG (needs graded relevance, which this data does not have), MAP,
Recall@1/100, alignment/uniformity, bootstrap confidence intervals. Each would add code and a
paragraph of justification and none would change the conclusion.

### How uncertainty is reported (without becoming a statistics project)

1. **Three seeds (13 / 42 / 1337) for every configuration.** Report `mean ± (max − min)/2`.
   Not a confidence interval — just the spread you actually observed.
2. **Report dev and test side by side.** Two independent 200-query sets agreeing is the
   cheapest available reality check.
3. **State the resolution limit explicitly:** with 200 queries, differences below ~3 points of
   Recall@10 are not distinguishable from noise. Write that sentence in the report and then
   do not over-interpret anything smaller.
4. **No p-values, no bootstrap, no significance stars.** If you want one later, it is an
   optional extension, not a requirement.

---

## The one ablation (SHOULD HAVE)

**Batch size 16 vs 32 vs 64**, everything else identical, one seed each.

Why this one: with in-batch negatives, the batch size *is* the number of negatives per example
(15 / 31 / 63). It is a one-argument change, it costs three short runs, and it directly
demonstrates the mechanism the whole method depends on — [DPR](https://aclanthology.org/2020.emnlp-main.550/)
attributes a large part of its result to in-batch negative count. If Recall@10 rises with batch
size, you have evidence that *how many* negatives you see matters more than *which* negatives
they are, which is a genuinely interesting thing to have found in week 3.

---

## Expected outputs

### The table that defines the project

| System | Recall@10 (test) | MRR@10 (test) | Recall@10 (dev) |
|---|---|---|---|
| B1 BM25 | **?** | **?** | **?** |
| B2 MiniLM zero-shot | **?** | **?** | **?** |
| B3 MiniLM fine-tuned (3 seeds) | **?** (mean ± range) | **?** | **?** |

Question marks are intentional. Predicting the numbers before you run them is how people end
up defending a number instead of reporting one.

### Five pre-registered result patterns

Write these down before training. Whichever one you observe, you already know what to write.

| # | Pattern | Interpretation | Report headline |
|---|---|---|---|
| 1 | B3 > B2 by ≥ 3 pts; B2 > B1 | The textbook result. | "In-domain contrastive fine-tuning improves duplicate retrieval by X points, and the gain is (or is not) enough to matter." |
| 2 | B3 and B2 within 3 pts of each other | H0. **Most likely outcome.** | "Fine-tuning on 19K in-domain duplicate pairs does not measurably change retrieval quality on a model already trained on >1B pairs." |
| 3 | B3 < B2 | Overfitting to a small pair set. | Report epochs {1,2,3} on dev; if peak is at 1 epoch, say so — that is the finding. |
| 4 | B1 ≥ B2 and B1 ≥ B3 | Keyword overlap dominates. | "BM25 remains competitive with a fine-tuned dense retriever on this corpus", consistent with BEIR and Jiang et al. |
| 5 | Any number that looks impossible (Recall@10 > 95%, or BM25 < 2%) | **Your harness is broken.** | Do not write anything. Go to the debugging checkpoints. |

### Figures (2, both trivial)

1. Bar chart: three systems × Recall@10, with error bars from the 3 seeds.
2. Line chart: Recall@10 (dev) against batch size {16, 32, 64}.

### Write-up

5–7 pages: intro, related work (7 papers), method, experimental setup, results (3 tables,
2 figures), error analysis, threats to validity, conclusion. The error analysis is where a
project like this earns its mark.

---

## Threats to validity — the four you must discuss

1. **Incomplete labels.** Only ~5% of similar pairs are annotated
   ([Lei et al. 2016](https://aclanthology.org/N16-1153/)). All reported numbers are lower
   bounds; the bias works slightly against the fine-tuned model.
2. **Small test set.** 200 queries. Sub-3-point differences are noise; say so.
3. **Corpus contains the training questions.** ~96% of the corpus documents are train-split
   questions whose texts the model saw during training. The zero-shot baseline saw the same
   corpus, so the comparison is fair, but absolute numbers are optimistic. (Cheap check:
   report Recall@10 separately for gold duplicates that live in the train split vs the
   eval split — 5 lines, and it tells you whether you are measuring retrieval or memorisation.)
4. **Single domain.** AskUbuntu is one technical forum. Nothing here generalises to other
   domains, and you will not claim that it does.

---

## You are done with the design when you can answer these out loud

1. Why this dataset? *(small, pre-cleaned, fixed splits, real duplicate labels, published task)*
2. What exactly is an embedding? *(one point on a 384-d sphere; mean-pooled, L2-normalised)*
3. What makes two examples positive? *(the community marked one as a duplicate of the other)*
4. What makes them negative? *(another pair's positive in the same mini-batch)*
5. What does contrastive learning change? *(which directions in the 384-d space mean "same question")*
6. What does the loss encourage? *(rank my duplicate above the N−1 other duplicates in my batch)*
7. What is the baseline? *(BM25, plus the un-fine-tuned version of my own model)*
8. How do we know it improved? *(Recall@10 on a fixed 200-query test split, 3 seeds)*
9. What could cause the result to be wrong? *(incomplete labels, 200 queries, train-split corpus, broken harness)*

If you cannot answer one of these, re-read the relevant section rather than moving on.
