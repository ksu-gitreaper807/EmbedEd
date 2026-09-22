# Final project specification — the code track

**Domain-Specific Contrastive Embeddings for Code Similarity Detection: Comparing
Negative-Sampling Strategies**

> **Status:** the primary project in this repository. Originally the second track, running **in
> parallel** with the AskUbuntu track in [`junk/project/distilled/`](junk/project/distilled/) —
> not a replacement for it. That track is now archived under `junk/` (gitignored), preserved on
> disk and in the history of branch `arena/01a0aae7-embeded`.
> The two tracks share one independent variable (how negatives are selected) and one body of
> literature ([`junk/distilled/negative-pair-research/`](junk/distilled/negative-pair-research/));
> they differ in domain, model, and evaluation.

---

## Provenance key

Used throughout this document and [`SCOPE.md`](SCOPE.md).

| Tag | Meaning |
|---|---|
| `[spec]` | Came from your final project specification, unchanged in substance |
| `[verified]` | Read from a primary source while writing this document |
| `[check]` | From a search snippet or abstract; **open the source before quoting it** |
| `[repo]` | Carried over from this repository's existing AskUbuntu specifications |
| `[illustrative]` | A made-up number for planning. Replace it with a measurement before you rely on it |

---

## 0. What was verified, and what it changes

Four things came out of checking the specification against sources. Two of them change the plan.

### 0.1 Verified: the dataset numbers

| Quantity | BigCloneBench proper | **CodeXGLUE version (what you load)** |
|---|---|---|
| Functionalities | 43 `[verified]` | not exposed as a column `[verified]` |
| Code fragments | ~6,000,000 methods indexed `[check]` | **9,134** Java fragments `[verified]` |
| True clone pairs | 8,915,130 (Svajlenko's thesis) `[verified]` | — |
| False clone pairs | 288,367 `[verified]` | — |
| Pairs per split | — | **901,028 / 415,416 / 415,416** `[verified]` |
| Official metric | — | **F1** `[verified]` |
| Data format | — | `data.jsonl` (one line per function: `func`, `idx`) + `train.txt` / `valid.txt` / `test.txt` as `idx1 idx2 label` `[verified]` |

⚠ **Correction 1.** Your spec says "~8 million validated clone pairs across 43 functionalities".
That describes **BigCloneBench**, not the file you will download. The CodeXGLUE version is a
filtered subset: **9,134 Java code fragments**, obtained by discarding fragments with no tagged
true or false clone pair `[verified]`. State both numbers and say clearly which one you trained
on.

⚠ **Correction 2.** The CodeXGLUE dataset **has no functionality column** `[verified]` — the
schema is `id, id1, id2, func1, func2, label`. Your generalisation test (RQ3) needs
functionality labels, which therefore have to come from outside this dataset. See §9.3. This is
the single largest unbudgeted task in the plan.

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
  [doi:10.5281/zenodo.17238379](https://doi.org/10.5281/zenodo.17238379). `[verified]`
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

Carried over unchanged from the AskUbuntu track — the variable is the same, so the hypotheses are
too. `N1` = random, `N2` = BM25, `N3` = semantic.

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
| Fragments | **9,134** Java code fragments `[verified]` — confirm with `wc -l dataset/data.jsonl` |
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
fragments  = every unique function in dataset/data.jsonl            ~9,134   [verified]
corpus     = fragments appearing in the TRAIN split only
positives  = (func1, func2) pairs from the train split with label == 1
negatives  = mined per strategy, from corpus, minus excluded fragments
```

Three rules, all borrowed from the AskUbuntu track `[repo]`:

1. **Mine from the train-split corpus only.** A fragment that appears in the test split must never
   be used as a training negative.
2. **Exclude every labelled clone of the anchor** before using a fragment as a negative — your
   verification step (§7.2), and the correctness check you build and test in week 1.
3. **Self-exclusion:** a fragment is never its own negative.

### 4.2 The leakage question you must measure in week 1

CodeXGLUE's splits are made at the **pair** level `[verified]`, over a shared pool of only 9,134
fragments. It is therefore likely — but **not yet established for this dataset** — that the same
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
generalisation test (§9.3) is there to expose. `[repo — mirrors the leakage check in the AskUbuntu track]`

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
| **C2 — BM25** | Fine-tuned; negatives are keyword-similar-but-different snippets, found via BM25 retrieval |
| **C3 — Semantic** | Fine-tuned; negatives are snippets the **base model itself** finds embedding-similar but that are ground-truth non-clones |

All four use the same positive pairs. **The only thing that changes between C1, C2 and C3 is how
the negative was selected.** `[spec]` This is the whole design; protect it.

### 6.1 What each condition costs

| Condition | Extra work over C1 | Failure mode it introduces |
|---|---|---|
| C0 | none — it is your control | none |
| C1 | none — the reference run | negatives so easy the model learns almost nothing |
| C2 | build a BM25 index (`rank_bm25`) over the corpus | finds lexically similar code, which in clone detection is often *the same functionality* → false negatives |
| C3 | encode the corpus once with the untouched base model, one similarity matmul | the worst false-negative rate of the three, because what an untuned code model finds similar is heavily lexical |

⚠ **Correction 4: do not assume `C1 < C2 < C3` in difficulty — measure it.** An untuned code
encoder is largely lexical, so C3's "semantic" neighbourhoods may look a lot like C2's BM25
neighbourhoods. AugSBERT's closest analogue put BM25 at 75.08 and semantic search at 74.99 —
statistically indistinguishable `[verified]`. If C2 and C3 produce equally hard batches, your
manipulation did not take effect and the experiment produces no evidence.

**The hardness check (carried over from the AskUbuntu track, non-negotiable)** `[repo]`: after
mining, compute mean `cos(anchor, negative)` for each condition's negative set. Require
`C1 < C2 ≤ C3` with a visible gap. If it fails, stop and fix the mining — do not proceed to
training.

---

## 7. Negative-pair construction

### 7.1 The three strategies `[spec]`

| Strategy | Procedure |
|---|---|
| **Random** | Pick any non-clone fragment at random for each anchor |
| **BM25** | Build a BM25 index over the corpus (`rank_bm25`); retrieve top keyword-matching fragments per anchor; exclude true clones; use the remainder |
| **Semantic** | Embed the full corpus once using the **untouched base model**; retrieve top cosine-similarity matches per anchor; exclude true clones; use the remainder |

### 7.2 The verification step `[spec]`

> *"for both BM25 and semantic mining, explicitly filter out any snippet that is a true clone of
> the anchor before using it as a negative — this is a correctness check to build and test
> carefully in week 1-2"*

Build it as a standalone, unit-tested function with a deliberately adversarial test:

```python
def exclude_labeled_clones(anchor_id, candidates, positive_pairs) -> list:
    """Drop any candidate that is a labelled clone of anchor_id."""
```

Test it against a hand-built case where you *know* the answer — an anchor whose top-1 BM25 match
is a true clone, and assert that the clone is gone and rank 2 has been promoted.

⚠ **Correction 5: this check is necessary but not sufficient.** It removes negatives that
BigCloneBench *labelled* as clones. It cannot remove the unlabelled ones — and per §0.3 those are
numerous: the ground truth is incomplete both within and across functionalities `[verified]`. So
the residual false-negative rate is not zero, and — critically — **it grows as the negatives get
harder**, C1 < C2 < C3. That is not a bug in your pipeline; it is the mechanism you are studying.

**Therefore: measure it.** Sample 50 mined negatives from C3 and 50 from C2, and manually judge
whether each pair implements the same functionality. Two hours of work. It converts your biggest
validity threat into your strongest finding, and it is what makes an inverted-U result
*explainable* rather than merely observed. `[repo — this is the false-negative measurement from
the AskUbuntu track, and it was the highest value-per-hour item there too]`

Compare your measured rate against the nearest published figure: naive top-k hard negative
mining produces false-negative rates of **47% on StackExchange-domain data** `[check —
NV-Retriever; confirm the exact figure before quoting]`.

---

## 8. Training `[spec]`

Fine-tune each variant with `sentence-transformers`' built-in contrastive/triplet losses, on the
same subset size, the same number of epochs, the same hyperparameters. **The only thing that
differs between the three runs is the negative-selection method.**

| Parameter | Value | Note |
|---|---|---|
| Positive pairs | Same fixed sample for all conditions | Sample **once** with a fixed seed; never re-sample per condition |
| Train pairs | **[illustrative] 50,000** | Decide from a measured throughput test in week 1. CodeXGLUE's own pipeline uses 10% (≈90K) `[verified]` |
| Epochs | **[illustrative] 1–2** | |
| Batch size | **[illustrative] 16–32** | |
| Learning rate | **[illustrative] 2e-5** | Standard for BERT-scale code encoders |
| Max sequence length | **measure first** | Plot the token-length distribution of the corpus, then choose 256 or 512 and state it |
| Seeds | 3, fixed | One seed is an anecdote; three is an interval |
| Loss | One loss, identical across conditions | |

⚠ **Correction 6: with explicit `(anchor, positive, negative)` triples you are not using
`MultipleNegativesRankingLoss`.** MNRL derives its negatives from the rest of the batch, so it
cannot express "this specific hard negative". Pick one of:

* `TripletLoss` / `OnlineContrastiveLoss` with explicit triples — clean, directly expresses your
  variable, but gives one negative per anchor instead of `batch_size − 1`.
* MNRL, with hardness manipulated by **batch composition** — build each batch so the other
  anchors' positives act as the hard negatives. This is what the AskUbuntu track does `[repo]`,
  and it is why that track can hold the loss byte-identical across conditions.

Either is defensible. **Do not mix them between conditions**, and state which you chose and why.

---

## 9. Evaluation

You chose **F1 primary, MAP@R secondary**. That is the right call: F1 is the field standard
(CodeXGLUE reports F1 `[verified]`), so it is the only number you can compare to published work;
MAP@R is threshold-free, so it is the robustness check on your threshold choice.

### 9.1 Standard benchmark evaluation (primary)

| Item | Decision |
|---|---|
| Metric | **F1** (with precision and recall reported alongside) `[verified — CodeXGLUE's official metric]` |
| Threshold | Cosine threshold **selected on the validation split**, then applied unchanged to test |
| Report | The chosen threshold, per condition. Per-condition thresholds tuned on test are not comparable |
| Secondary | **MAP@R** — no threshold, so it confirms the F1 ordering is not an artefact of threshold choice |
| Splits | Use CodeXGLUE's validation and test splits as given `[verified]` |
| Runs | 3 seeds × 3 conditions, plus C0 |

**Sanity target.** CodeXGLUE's own fine-tuned CodeBERT pipeline reports an F1 around **0.95** on
this task `[check — confirm the exact number from the CodeXGLUE leaderboard before quoting it in
your report]`. If your four conditions do not land in that neighbourhood, your harness is broken,
not your hypothesis.

### 9.2 Threshold discipline

Two conditions can differ in F1 purely because one happened to sit better with respect to a fixed
threshold. Guard against it:

1. Choose the threshold **once per condition, on validation**. Report it.
2. Report **MAP@R** alongside — it needs no threshold.
3. If F1 and MAP@R disagree about the ordering, say so. That disagreement is a finding, not a
   problem to be tuned away.

### 9.3 Generalisation test (RQ3) `[spec]`

Hold out one or more entire functionality categories from training, then evaluate all four
conditions on those unseen functionalities specifically.

⚠ **Correction 7: this needs data the CodeXGLUE file does not contain** (§0.2, §4). Two routes:

| Route | What it is | Trade-off |
|---|---|---|
| **A — use Kitsios et al.'s BCB s′** `[verified]` | 23 functionalities, 2,300 clone + 2,300 non-clone pairs, released at [Zenodo](https://doi.org/10.5281/zenodo.17238379) | Functionality already balanced, already published, directly comparable to their numbers. **Recommended** |
| **B — attach functionality labels yourself** | Join CodeXGLUE fragments back to BigCloneBench proper by matching method text | More data, and a join that can silently fail. Days of work |

**Design, under Route A:**

```text
hold out k functionalities (start with k = 3, chosen from the best-represented)
train each condition on the remaining 23 − k
evaluate on:  F1_seen    (held-in functionalities, from the validation split)
              F1_unseen  (held-out functionalities)
report        Δ = F1_seen − F1_unseen       ← THIS is the dependent variable for RQ3
```

**The question is not "is F1_unseen low?"** — Kitsios et al. already showed it drops by ~31% on
average. **The question is whether Δ differs across C1 / C2 / C3.** Does harder-negative training
buy you generalisation, or does it buy you benchmark fit?

**Honest limits.** With 23 functionalities at 200 pairs each, holding out 3 leaves ~4,000 training
pairs — small enough that a single seed will be noisy. Report the generalisation numbers as
**means over the 3 seeds with the spread visible**, and treat them as secondary to the standard
benchmark. If k > 1, consider rotating the held-out set (leave-one-functionality-out) and
reporting the mean and range across folds.

### 9.4 Visualisation `[spec]`

Project embeddings from all four conditions into 2D with t-SNE/UMAP and compare cluster
tightness and separation.

⚠ **Correction 8: this is a figure, not evidence.** UMAP and t-SNE are sensitive to
hyperparameters and to the random seed, and they will happily show you a difference between two
conditions that is not there. Rules that keep it honest:

* **Fix the seed and every hyperparameter** (`n_neighbors`, `min_dist`, `random_state`) and use
  the **same sample of fragments** in all four panels.
* Project the **same points** in every panel; never re-sample per condition.
* If you colour by functionality, use the held-out functionalities from §9.3 — that is the
  visually interesting case.
* Never put the picture before the table in your argument. One picture is an illustration; the
  aggregate table is the evidence. `[repo]`

---

## 10. Expected findings, framed as hypotheses `[spec]`

| Open question | Most likely outcome, and why |
|---|---|
| Does fine-tuning help at all over the baseline? | Yes, to some degree. Already established in general — this is your setup check, not your contribution |
| Does negative strategy matter? | Yes, in the sense that `Random ≪ {BM25, Semantic}`. Which of BM25 and Semantic wins is **not** established for code `[verified across five papers]` |
| Does harder always mean better? | Probably not. H0b (inverted-U) is the literature-supported prediction — STAR/ADORE report that static hard negatives *"may harm the recall capability"* `[verified]` |
| Does the standard-benchmark improvement hold up on the generalisation test? | **It will shrink.** Plan for ~30% average F1 drop on unseen functionality `[verified — Kitsios et al.]`. The open question is whether the *strategy* changes how much |

Any outcome — including "harder negatives made it worse" — is a legitimate, presentable finding.
`[spec]`

---

## 11. Demo plan

Your three components, adapted to the constraints this repository already adopted for the
AskUbuntu demo `[repo]`.

### 11.1 Live interactive pair comparison `[spec]`

Paste two differently-written, functionally-equivalent snippets → similarity scores from baseline
vs best fine-tuned model, side by side.

⚠ **Correction 9: two hand-typed snippets prove nothing** — you can get any result you like by
choosing the snippets. The honest version:

* **Default to real test pairs.** A dropdown of held-out `(func1, func2)` pairs with their gold
  label. The app shows: `gold = clone`, `C0 cosine = 0.42 → below threshold → judged not-clone`,
  `C3 cosine = 0.81 → above threshold → judged clone`. Now the audience is looking at a number
  computed the same way the headline metric is computed.
* **Keep the free-text paste box**, clearly labelled *"your own snippets — illustrative only,
  not part of the evaluation"*.
* Show the threshold as a visible marker on the score bar.
* Support all four outcome modes, not just "ours wins" `[repo — result-agnostic design]`.

### 11.2 2D embedding-space visualisation `[spec]`

The visual centrepiece. Four panels, UMAP, fixed seed, same sample — see §9.4.

### 11.3 Results table

Four conditions × {F1, precision, recall, MAP@R} on the standard test split, plus
{F1_seen, F1_unseen, Δ} on the generalisation split. **This is the evidence**; §11.1 and §11.2 are
the illustration.

---

## 12. Honest scope and limitations to state upfront `[spec]`

1. **Java only** — BigCloneBench's language. Other languages are future work.
2. **A subset of the full dataset**, for time reasons; standard practice for a month-long project.
   CodeXGLUE's own reference pipeline does the same `[verified]`.
3. **A signal for review, not a plagiarism verdict** — a human still makes the final call.
4. **The base model's pretraining exposure is a confound.** A public benchmark this widely used
   has almost certainly been seen in some form during pretraining. The generalisation test
   addresses this; it does not eliminate it.
5. **BigCloneBench's ground truth is contested** (§0.3, and [`GROUND_TRUTH.md`](GROUND_TRUTH.md)).
   State the 93%-mislabelled finding and the imbalance explicitly, and report your own measured
   false-negative rate.
6. **Under Option A (§5), you are not using GraphCodeBERT's data-flow signal at inference time.**
   Say so.
7. **The generalisation test is small** (§9.3). Present it as secondary.

---

## 13. Week-by-week plan

Your four weeks, with the corrections folded in.

| Week | Your plan `[spec]` | Added, because of what was verified |
|---|---|---|
| **1** | Data pipeline; implement and verify all three negative-mining strategies with the true-clone-exclusion check; one training run (random negatives) end-to-end | **Measure the train/test fragment overlap** (§4.2) and the token-length distribution. **Decide Option A vs B** for GraphCodeBERT (§5). **Run a throughput test** and fix the subset size. Verify `wc -l dataset/data.jsonl` |
| **2** | BM25 and semantic training variants; begin standard-benchmark evaluation for all four conditions | **Run the hardness check** before training (§6.1). **Do the manual false-negative labelling** (§7.2) — it is the highest value-per-hour item in the project. Download BCB s′ and confirm you can load it |
| **3** | Generalisation test (held-out functionality categories); 2D visualisation; start the live demo | Build the generalisation split by **functionality**, not at random (§9.3). Fix the UMAP seed and use one shared sample (§9.4) |
| **4** | Finalise results, polish the demo, write up findings and limitations, prepare the presentation | Write the limitations section (§12) **first**, so the claims are shaped by it rather than patched afterwards |

---

## 14. Fallback if time gets tight `[spec]`

Drop to two negative strategies instead of three — e.g. random + semantic, skip BM25 — still
preserving the "does negative strategy matter" comparison.

**Endorsed, with one caveat.** Random + Semantic spans the widest difficulty range, so it is the
right pair for RQ2. The caveat: **with only two conditions you cannot detect an inverted-U.** A
peak needs three points. If C1 and C3 come out equal, you will not be able to tell
"hardness does not matter" (H0a) from "hardness helped and then hurt, and the two cancelled out"
(H0b), and those are different findings. If you do fall back to two, say in the write-up that
H0b was not testable.

---

## 15. How this track relates to the AskUbuntu track

| | AskUbuntu track | Code track (this) |
|---|---|---|
| Domain | Ubuntu duplicate questions | Java code clones |
| Corpus | ~13,000–15,000 questions | ~9,134 Java fragments |
| Model | `all-MiniLM-L6-v2` (384-d) | `microsoft/graphcodebert-base` (768-d) |
| Task framing | Retrieval (Recall@10, MRR@10, nDCG@10) | Pair classification (F1) + MAP@R |
| Negatives | In-batch, via MNRL | Explicit triples (or batch composition) |
| Extra test | none | **Generalisation to unseen functionality** |
| Shared | The independent variable; the three strategies; the hardness check; the false-negative measurement; the literature in `junk/distilled/negative-pair-research/` | |

**The code track is the stronger of the two**, because of RQ3. The AskUbuntu track answers
"does negative strategy matter?"; the code track answers "does negative strategy matter *for
generalisation*?", which has a published baseline to compare against and is not answered anywhere.

If you must prioritise one, prioritise this one. If you run both, keep `negatives.py`'s strategy
interface identical across them so a fix in one benefits the other.

---

## 16. Documents in this repository

| File | What it is |
|---|---|
| [`FINAL_SPEC.md`](FINAL_SPEC.md) | This document — the specification, with verified facts and corrections |
| [`SCOPE.md`](SCOPE.md) | MUST / SHOULD / optional / out of scope, and the stop condition |
| [`GROUND_TRUTH.md`](GROUND_TRUTH.md) | The BigCloneBench validity problem, and what this project does about it |
| [`README.md`](README.md) | Index |
