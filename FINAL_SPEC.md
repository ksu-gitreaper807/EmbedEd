# Final project specification — Java code clone detection

**Domain-Specific Contrastive Embeddings for Code Similarity Detection: Comparing
Negative-Sampling Strategies**

> **Status:** the specification for the project in this repository. Everything else the
> repository used to hold is archived under `junk/` (gitignored) — preserved on disk and in the
> history of branch `arena/01a0aae7-embeded`, but no longer part of the project. The literature
> archive left behind
> ([`junk/distilled/negative-pair-research/`](junk/distilled/negative-pair-research/)) is still
> cited where it carries an argument.

---

## Provenance key

Used throughout this document and [`SCOPE.md`](SCOPE.md).

| Tag | Meaning |
|---|---|
| `[spec]` | Came from your final project specification, unchanged in substance |
| `[verified]` | Read from a primary source while writing this document |
| `[check]` | From a search snippet or abstract; **open the source before quoting it** |
| `[repo]` | Carried over from the earlier specification this project grew out of |
| `[illustrative]` | A made-up number for planning. Replace it with a measurement before you rely on it |

---

## 0. What was verified, and what it changes

Four things came out of checking the specification against sources. Two of them change the plan.

### 0.1 Verified: the dataset numbers

| Quantity | BigCloneBench proper | **CodeXGLUE version (what you load)** |
|---|---|---|
| Functionalities | 43 `[verified]` | not exposed as a column `[verified]` |
| Code fragments | ~6,000,000 methods indexed `[check]` | **9,126** lines in `data.jsonl`; **8,063** unique texts `[verified — re-measured, correction 9]` |
| True clone pairs | 8,915,130 (Svajlenko's thesis) `[verified]` | — |
| False clone pairs | 288,367 `[verified]` | — |
| Pairs per split | — | **901,028 / 415,416 / 415,416** `[verified]` |
| Official metric | — | **F1** `[verified]` |
| Data format | — | `data.jsonl` (one line per function: `func`, `idx`) + `train.txt` / `valid.txt` / `test.txt` as `idx1 idx2 label` `[verified]` |

⚠ **Correction 1.** Your spec says "~8 million validated clone pairs across 43 functionalities".
That describes **BigCloneBench**, not the file you will download. The CodeXGLUE version is a
filtered subset: the CodeXGLUE paper says **9,134 Java code fragments**; the released file has
**9,126 lines, 8,063 unique texts** (see correction 9) `[verified]`, obtained by discarding
fragments with no tagged true or false clone pair. State both numbers and say clearly which one
you trained on.

⚠ **Correction 2.** The CodeXGLUE dataset **has no functionality column** `[verified]` — the
schema is `id, id1, id2, func1, func2, label`. Your generalisation test (RQ3) needs
functionality labels, which therefore have to come from outside this dataset. See §9.3. This is
the single largest unbudgeted task in the plan.

⚠ **Correction 9.** Correction 1's 9,134 is the CodeXGLUE paper's number, not the released
file's. Measured against `microsoft/CodeXGLUE` (main, 2026-09-22): `wc -l data.jsonl` =
**9,126**; 1,063 of those lines are byte-identical to another fragment → **8,063 unique
fragment texts**. The CodeXGLUE file is a re-release of the CDLH subset, and an independent
audit ("Generalizability of Code Clone Detection on CodeBERT", ICSE-SEIP 2023) already notes
"instead of 9,134 given code snippets in CDLH, CodeXGlue contains 9,126". Consequences:
- the pipeline verifies **9,126 lines / 8,063 unique fragments / 901,028 / 415,416 / 415,416
  pairs** (`EXPECTED_DATA_LINES` / `EXPECTED_FRAGMENTS` / `EXPECTED_SPLIT_ROWS`);
- fragments are the unique **texts**, keyed 0..8,062 by first appearance in data.jsonl;
- `--hf` and `--dir` both join pair rows onto that canonical list, so both modes produce
  byte-identical artifacts (the HF parquet carries text, not idx, so `--hf` fetches the
  15 MB data.jsonl once and caches it);
- all 9,126 fragments appear in ≥1 pair (no orphans), so nothing is lost by joining.

### 0.2 Verified: the generalisation test already exists in the literature

**Kitsios, Sovrano, Barr & Bacchelli, "Detecting Semantic Clones of Unseen Functionality",
ASE 2025** ([arXiv:2510.04143](https://arxiv.org/pdf/2510.04143)) is your RQ3, already done:

* Functionalities in BigCloneBench are split into train and test **uniformly at random**, so
  clone pairs of the *Fibonacci* functionality appear in both splits. Standard BCB evaluation is
  therefore a *seen-functionality* setting. `[verified]`
* Re-evaluating six state-of-the-art models on **unseen** functionality: **F1 drops by up to
  48%, average 31%**, for task-specific models. LLMs drop by only 3% on average. `[verified]`
* They also propose contrastive learning as the *fix*: F1 on clones of unseen functionality
  improves by up to 26% (average 9%). `[verified]`
* They release a functionality-balanced dataset, **BCB s′: 23 functionalities, 2,300 clone +
  2,300 non-clone pairs**, at
  [doi:10.5281/zenodo.17238379](https://doi.org/10.5281/zenodo.17238379) `[verified]`. The zip is a
  snapshot of [github.com/kitsiosk/unseen-clones](https://github.com/kitsiosk/unseen-clones);
  s′ itself is `datasets/bcb_v2_sampled_bf/data_bcb_v2_sampled_bf.pickle` — a DataFrame
  `code1, code2, label, functionality_id`, 4,600 rows (2,300/2,300), 23 functionalities
  `[verified 2026-09-22 — scripts/fetch_sprime gates on exactly this]`. The package also
  carries the SCB corpora (`datasets/scb/{Java,C}`) used for the cross-dataset protocols.
* One honest detail worth stealing: in one of their twelve experiments contrastive learning did
  not help, and their qualitative analysis found it had *"led to learning a different, more
  strict definition of a clone."* `[verified]`

**What this means for you.** "Do improvements generalise to unseen functionality?" is
**answered**. Do not re-ask it as your contribution, and do not claim it is novel — cite
Kitsios et al. and state your expected result in advance (an F1 drop of roughly 30%).

**What is still open, and is your actual contribution:**

> Does the **negative-sampling strategy** change the size of the generalisation gap?

Nobody has measured that. It is a clean, small, well-scoped question with a published baseline to
compare against. Frame the whole project around it.

### 0.3 Verified: BigCloneBench's ground truth is contested

Krinke, *"BigCloneBench Considered Harmful for Machine Learning"* (IWSC 2022) and the follow-up
*"How the Misuse of a Dataset Harmed Semantic Clone Detection"*
([arXiv:2505.04311](https://arxiv.org/html/2505.04311v1)) report:

* A manual investigation of a random sample of **406 Weak Type-3/Type-4 clone pairs** found
  **93% do not have a similar functionality and are therefore mislabelled**. WT3/T4 pairs make
  up **95%** of the dataset. `[verified]`
* At least **15%** of labelled snippets are estimated to be subjective or to contain validation
  errors. `[verified]`
* Severe imbalance: the majority of labelled methods implement *Copy File* (42,664 of 75,672);
  **over 90% of all true clone pairs belong to just 8 functionalities**, and 22 of the 43
  functionalities together account for less than 1% of them. `[verified]`
* CodeXGLUE specifically is named as containing **wrongly generated false clone pairs**, created
  by assuming (a) two methods labelled false positive for the same functionality are not clones
  of each other, and (b) two methods labelled true positive for two different functionalities are
  not clones of each other. Both assumptions are invalid. `[verified]`

This is a threat to validity, not a reason to abandon the benchmark — Kitsios et al. published at
ASE 2025 using it, and hundreds of papers do. But you must **state it in the report and measure
the part that interacts with your independent variable**. Full treatment, with mitigations, in
[`GROUND_TRUTH.md`](GROUND_TRUTH.md).

### 0.4 Verified: CodeXGLUE's own pipeline already subsamples

> *"We only use 10% training data to fine-tune and 10% valid data to evaluate."* `[verified]`

Your "manageable subset" decision has direct precedent in the benchmark's own reference
implementation. Use it as a justification, not an apology.

---

## 1. Title `[spec]`

**Domain-Specific Contrastive Embeddings for Code Similarity Detection: Comparing
Negative-Sampling Strategies**

---

## 2. Objective `[spec]`

Investigate whether contrastive fine-tuning improves a pretrained code model's ability to
represent semantic code similarity, and specifically study **how the choice of
negative-sampling strategy** during training affects the quality and generalisation of the
resulting embeddings — using code clone detection as the test domain.

---

## 3. Research questions

Your three, with their current standing in the literature. Write the standing down before you
train anything, and put this table in the report.

| # | Question `[spec]` | Standing | Where the evidence is |
|---|---|---|---|
| **RQ1** | Does contrastive fine-tuning improve embedding quality over the same model used off-the-shelf? | **Established in general**; not for this model + loss + subset | Domain adaptation helps broadly. Kitsios et al. `[verified]` show contrastive learning improves unseen-functionality F1 by up to 26%. Not a contribution on its own — it is your Condition 0 comparison |
| **RQ2** | Does the *method* used to select negatives (random / keyword-similar / semantically-similar) meaningfully change the result? | **Not established for code.** Established-ish for NLP text | [`FIVE_KEY_PAPERS.md`](junk/distilled/negative-pair-research/FIVE_KEY_PAPERS.md). The cross-paper finding is `Random ≪ {BM25, Semantic}`, with the winner between the last two depending on label completeness |
| **RQ3** | Do improvements hold up on functionalities not well-represented in training, or is it narrow overfitting? | **The base question is answered** (≈31% average F1 drop). **The interaction with negative strategy is open** | Kitsios et al. ASE 2025 `[verified]` |

**The contribution, stated precisely:**

> RQ1 is the setup, RQ3's base effect is already published, and **RQ2 × RQ3 is the gap**: does
> the negative-sampling strategy change how well a contrastively fine-tuned code model
> generalises to unseen functionality?

### 3.1 Hypotheses `[repo]`

Carried over from the earlier specification unchanged — the independent variable is the same, so
the hypotheses are too. `N1` = random, `N2` = BM25, `N3` = semantic.

* **H1 (monotone):** `N1 ≤ N2 ≤ N3` on F1, and all three beat the off-the-shelf baseline. The
  naive transfer of DPR's result.
* **H0a (flat):** all three within noise of each other.
* **H0b (inverted-U):** `N2 > N1` but `N3 < N2`. Hardness helps, then hurts. **This is the
  prediction the literature supports** — see [`FIVE_KEY_PAPERS.md`](junk/distilled/negative-pair-research/FIVE_KEY_PAPERS.md):
  STAR/ADORE state that static hard negative sampling *"improves the top-ranking performance but
  may harm the recall capability"*, and ANCE's static BM25 negatives overlap only 15% with the
  negatives the dense model actually finds hard.

All three outcomes are presentable, including the surprising ones. `[spec]`

---

## 4. Dataset `[spec]`, with verified corrections

**BigCloneBench, via CodeXGLUE:**
[`google/code_x_glue_cc_clone_detection_big_clone_bench`](https://huggingface.co/datasets/google/code_x_glue_cc_clone_detection_big_clone_bench)

| Property | Value |
|---|---|
| Origin | BigCloneBench (Svajlenko & Roy), filtered per Wang et al. (FA-AST/GNN paper) `[verified]` |
| Loading | `load_dataset("google/code_x_glue_cc_clone_detection_big_clone_bench")` |
| Fragments | **9,126 lines / 8,063 unique texts** `[verified]` — confirm with `wc -l dataset/data.jsonl` (9,126) + sha1 dedup (8,063); correction 9 |
| Pairs | train 901,028 / validation 415,416 / test 415,416 `[verified]` |
| Row format | `id`, `id1`, `id2`, `func1`, `func2`, `label` (boolean) `[verified]` |
| Functionality labels | **absent** `[verified]` |
| Official metric | F1 `[verified]` |
| Language | Java |
| Licence | C-UDA (`c-uda`) `[verified]` |

**Why this dataset** `[spec]`: real, IEEE-recognised Java clone-pair benchmark; ready-made
train/dev/test split; loads in one line.

### 4.1 What you build in `prepare_data.py`

The dataset is **pair-level**, but negative mining is an **anchor-to-corpus** operation, so you
must construct the corpus yourself. This is the pipeline's first real job.

```text
fragments  = every unique function TEXT in data.jsonl           8,063 (9,126 lines) [verified]
corpus     = fragments appearing in the TRAIN split only
positives  = (func1, func2) pairs from the train split with label == 1
negatives  = mined per strategy, from corpus, minus excluded fragments
```

Three rules, all carried over from the earlier specification `[repo]`:

1. **Mine from the train-split corpus only.** A fragment that appears in the test split must never
   be used as a training negative.
2. **Exclude every labelled clone of the anchor** before using a fragment as a negative — your
   verification step (§7.2), and the correctness check you build and test in week 1.
3. **Self-exclusion:** a fragment is never its own negative.

### 4.2 The leakage question you must measure in week 1

CodeXGLUE's splits are made at the **pair** level `[verified]`, over a shared pool of only 9,126
fragments (8,063 unique texts). It is therefore likely — and, once measured, must be measured in
**text space**, since identical text under different idx is still leakage — that the same
function appears in both the training and the test split. If so, your test F1 measures partly
*memorisation of specific functions*, not clone detection.

**Do not assume either way. Measure it on day one:**

```python
train_frag = set of idx appearing in any train pair
test_frag  = set of idx appearing in any test pair
overlap    = train_frag & test_frag
print(len(train_frag), len(test_frag), len(overlap), len(overlap) / len(test_frag))
```

Report the number in the write-up either way. If the overlap is large, say so plainly and treat
the standard-benchmark F1 as an optimistic upper bound — which is exactly what the
generalisation test (§9.3) is there to expose. `[repo — mirrors the earlier specification's leakage check]`

---

## 5. Base model

**GraphCodeBERT** `[spec]` — chosen over plain CodeBERT because it incorporates data-flow
structural information during pretraining, giving a more robust starting representation `[spec]`.

⚠ **Correction 3: the data-flow signal does not survive a default `sentence-transformers`
load.** GraphCodeBERT's full model takes *two* inputs — source tokens **and** the extracted data
flow graph (`--code_length 512 --data_flow_length 128` in Microsoft's own fine-tuning script)
`[verified]`. `sentence-transformers` has no module for that second input. You therefore have two
options, and you must pick one **before week 1 ends**:

| | Option A — GraphCodeBERT as a plain encoder | Option B — full GraphCodeBERT with data flow |
|---|---|---|
| What | `Transformer("microsoft/graphcodebert-base")` + `Pooling(768, "mean")`; source tokens only | Port the DFG extractor from `microsoft/CodeBERT` (`parser/`, needs tree-sitter + the Java grammar) and write a custom `sentence-transformers` module taking both inputs |
| Effort | ~5 lines | Days, with a real chance of not working |
| Data-flow signal | **Lost at inference time** (the pretraining still shaped the weights) | Present |
| Honest name for the model | "GraphCodeBERT used as a token-only encoder" | "GraphCodeBERT" |

**Recommendation: Option A**, with the caveat written into the report. Your stated reason for
choosing GraphCodeBERT over CodeBERT is the structural signal; under Option A you must not claim
you used it. The defensible sentence is: *"we use GraphCodeBERT as a token-only encoder; its
data-flow pretraining shapes the weights, but we do not supply data-flow graphs at inference, so
we cannot attribute any gain to structural information."*

Under Option A, `microsoft/codebert-base` is an equally defensible base model and is cheaper to
justify honestly. Keep it as the fallback — it is a one-line change.

**Sanity reference.** `mchochlov/codebert-base-cd-ft` is a real published model: CodeBERT,
768-d, mean-pooled, max_seq 128, *"fine tuned towards clone detection using contrastive learning
on parts of BigCloneBench"* `[verified]`. If your fine-tuned model does not beat it, something is
wrong with your harness, and you have a ready-made external comparison.

---

## 6. The four conditions `[spec]`

| Condition | Description |
|---|---|
| **C0 — Baseline** | Off-the-shelf GraphCodeBERT, no fine-tuning. Cosine similarity of mean-pooled embeddings |
| **C1 — Random** | Contrastively fine-tuned; negatives are randomly paired non-clone snippets |
| **C2 — BM25** | Fine-tuned; negatives are keyword-similar-but-different snippets, found via BM25