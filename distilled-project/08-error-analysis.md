# 08 — Error analysis

Aggregate metrics tell you *whether* systems differ. Error analysis tells you *why*, and it is the section that separates a competent undergraduate report from a real piece of work. Budget half a day; it is the highest-value half day in the month.

Use the run files you saved in `results/predictions/` — never re-run retrieval just to look at examples.

---

## Part A — The six things to inspect

### 1. False positives — "ranked it first, but it is not a duplicate"

**How:** for each test query, if the top-1 retrieved document is not in the gold set, record it.

**What to look for:**

| Pattern | What it means | Worth writing about? |
|---|---|---|
| High word overlap, different question ("How do I install X on Ubuntu?" vs "How do I install X on Fedora?") | The model is matching surface form, not intent | ⭐ Yes — the classic lexical-overlap failure |
| Same component/product, different bug/issue | Domain-vocabulary confusability | ⭐ Yes |
| Genuinely a duplicate, but unlabelled | **Your metric is wrong, not your model** | ⭐⭐ Yes — this is your ceiling argument |
| Completely unrelated | A real model failure | Note the count, move on |

**Concrete example of the interesting kind:**
> Query: *"Why does my Android app crash on rotation?"*
> Top-1: *"Why does my Android app crash on startup?"* (not marked duplicate)
> → Lexically near-identical, semantically different. If your fine-tuned model ranks this below the true duplicate and BM25 ranks it first, you have a crisp demonstration of what contrastive fine-tuning bought you. **That is a figure.**

### 2. False negatives — "the duplicate exists but was not retrieved"

**How:** for each test query with ≥1 gold, if no gold appears in the top-100, record it.

**What to look for:**

| Pattern | Meaning |
|---|---|
| The duplicate uses completely different vocabulary (e.g. "ANR" vs. "app not responding") | The vocabulary gap that embeddings are *supposed* to close. Count these — they are your strongest argument that the task is genuinely semantic |
| The query is very short (title only, no body) | Length effect; bucket by it |
| The information needed is in the *answers*, not the question text | Structural limitation of the dataset. [Zhang et al.](https://doi.org/10.1145/3576042) found exactly this in bug reports (information split across comments) |
| The gold is labelled but the two texts are not actually duplicates | Label noise |

### 3. Difficult examples — "where did fine-tuning change the ranking?"

**How:** for every test query, compute ΔnDCG between your best fine-tuned model and the zero-shot baseline. Sort. Inspect the top-20 **improvements** and the top-20 **regressions**.

This is the single most informative table you will produce. Improvements show you what contrastive fine-tuning learned; regressions show you what it broke. A model that improves 30 queries and breaks 30 is a very different story from one that improves 60 and breaks none, even if the mean nDCG@10 is identical.

### 4. Lexical-overlap failures

**How:** for each (query, gold duplicate) pair, compute a lexical overlap score (Jaccard on content-word unigrams, or just BM25(query, gold)). Bucket pairs into quartiles: `very low`, `low`, `high`, `very high`. Report nDCG@10 (or Recall@10) **per bucket** for BM25, zero-shot, and fine-tuned.

**The pattern to look for:**

```
                     low overlap      high overlap
BM25                     low              high
zero-shot encoder        medium           medium
fine-tuned encoder       ???              ???
```

Two possible stories, both publishable:
* Fine-tuning helps **most on low-overlap pairs** → it learned semantics beyond word matching. Consistent with the motivation.
* Fine-tuning helps **most on high-overlap pairs** → it mostly learned to sharpen lexical matching. Consistent with [Jiang et al. (2023)](https://doi.org/10.1016/j.jss.2023.111607), who found **lexical similarity matters more than semantic similarity** for duplicate bug reports. **Report this honestly if it happens** — it is a finding, not a failure.

### 5. Semantic-equivalence failures

The mirror image: pairs that a human calls duplicates but that share almost no words. Sample 20 from the low-overlap bucket and look at them. Are they (a) genuine near-duplicates with different vocabulary, or (b) actually different questions that the community wrongly merged? If (b) is common, your ceiling is partly *label* noise, not *label sparsity* — a different and more interesting conclusion.

### 6. Hard-negative failures ⭐

**How:** take the mined hard negatives for strategy N4 (model-hard) and inspect the **top-20 candidates that N5's filter removed** — i.e. the ones your filter judged most likely to be false negatives.

**What to look for:**
* How many of them would *you* call duplicates? That count, over ≥100 sampled candidates, **is your reported false-negative rate**.
* Do the removed candidates cluster by subforum? If `tex` produces far more false negatives than `gaming`, that is a domain-property finding.

**Why this is the most valuable analysis you will do:** it directly substantiates or refutes [Robinson et al. (2021)](https://openreview.net/forum?id=CR1XOQ0UTh-)'s Principle 1 in your setting, and it is the mechanism behind whichever result pattern you observed. Nobody reports this number for duplicate-question retrieval.

---

## Part B — The ceiling analysis (do this, it is quick and it matters)

With only ~1.4 labelled duplicates per query, your absolute nDCG@10 is capped by label sparsity. Quantify it:

1. Sample 100 test queries where the top-1 result is **not** in the gold set.
2. For each, judge by hand: is the top-1 nevertheless a **true** duplicate of the query?
3. Report: *"In X% of apparent top-1 failures, manual inspection found the retrieved question to be a genuine duplicate that the community never marked."*
4. If you can, get a second person to label the same 100 and report agreement (Cohen's κ). Two-author manual validation is standard in this literature and a single-author judgement is a known weakness (see the review of [arXiv:2509.02077](https://pith.science/paper/2509.02077)).

**Why it matters:** it reframes your whole result. If 35% of "failures" are actually successes, then the difference between systems is being measured on a noisy subset, and *that* is a legitimate conclusion about evaluation in this domain. [Zhang et al. (2023)](https://doi.org/10.1145/3576042) did the analogous failure analysis for bug reports and identified three distinct causes — copy the method.

---

## Part C — What belongs in the paper

A good error-analysis section is about **1.5 pages, 1 table, 1 figure, 3 concrete examples.** Structure:

1. **One sentence** stating how many errors you inspected and how you sampled them (be reproducible: "the 100 lowest-Δ queries" or "a random sample stratified by subforum").
2. **A taxonomy table**: error category → count → one-line description → example ID.
3. **The lexical-overlap bucket table** (BM25 / zero-shot / fine-tuned × four overlap quartiles). This is the analytical centrepiece.
4. **The false-negative table**: strategy → measured FN rate → nDCG@10. If these two columns are correlated — high FN rate ↔ low score — you have a mechanism and you should plot them against each other.
5. **Three verbatim examples.** One where fine-tuning clearly won, one where it clearly lost, one where everyone failed. Quote the actual text. Reviewers read the examples before they read the table.
6. **One honest limitation.** E.g.: "our manual judgements were made by a single annotator"; "our error sample covers 3 of 12 subforums".

---

## Part D — Mistakes that are *interesting* to discuss, ranked

| Rank | Mistake type | Why it is interesting |
|---|---|---|
| 1 | **Unlabelled duplicates retrieved at rank 1** | Challenges the evaluation itself; connects to the literature on incomplete duplicate labels |
| 2 | **High lexical overlap, different question** | The canonical semantic-vs-lexical failure; directly tests the premise of the whole project |
| 3 | **Mined hard negatives that are true duplicates** | The mechanism behind H0; the number nobody reports |
| 4 | **Zero vocabulary overlap, genuine duplicate** | Shows the task is genuinely semantic and worth embedding-based methods |
| 5 | **Fine-tuning helped on subforum A, hurt on subforum B** | Domain heterogeneity; connects to [MTEB](https://aclanthology.org/2023.eacl-main.148/)'s "no method dominates" |
| 6 | **Regression cases** (fine-tuned worse than zero-shot) | Prevents you from over-claiming; shows you looked |
| 7 | Very short queries | A clean, actionable, length-based finding |
| 8 | Genuinely unrelated top-1 | Least interesting — just noise. Count it and move on. |
