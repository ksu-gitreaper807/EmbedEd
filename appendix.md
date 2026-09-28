# Appendix — project terminology

This appendix is a plain-language glossary for **EmbedEd**. It explains the words used in the
specification, code, measurements, and experiment plan. The examples use one small Java task so
that the terms can be followed from the dataset all the way to the final metrics.

> **Important:** the numbers in the examples below are invented to make the ideas easy to see.
> They are not experimental results. The frozen project settings are listed in the final section.

## 1. The running example

Suppose the dataset contains these Java methods:

```java
// A: the anchor
static int sum(int[] values) {
    int total = 0;
    for (int value : values) total += value;
    return total;
}
```

```java
// P: a positive example
static int addAll(int[] numbers) {
    int result = 0;
    for (int number : numbers) result = result + number;
    return result;
}
```

`A` and `P` implement the same functionality, even though their method and variable names are
different. They are therefore an illustrative **clone pair**.

Here are three possible negatives:

```java
// N_random: a randomly selected, unrelated method
static void reverse(int[] values) {
    for (int left = 0, right = values.length - 1; left < right; left++, right--) {
        int temporary = values[left];
        values[left] = values[right];
        values[right] = temporary;
    }
}
```

```java
// N_bm25: lexically similar, but with a different result
static double average(int[] values) {
    int total = 0;
    for (int value : values) total += value;
    return (double) total / values.length;
}
```

```java
// N_semantic: structurally similar, but a different function
static int countPositive(int[] values) {
    int count = 0;
    for (int value : values) {
        if (value > 0) count++;
    }
    return count;
}
```

`N_bm25` and `N_semantic` are **harder negatives** than `N_random`: they look more like `A`,
but are intended to have a different label. In a real dataset, however, a supposedly negative
pair can be an unlabelled clone. That is why EmbedEd measures a **false-negative rate** instead of
assuming that every mined negative is truly different.

The basic training examples are therefore:

```text
C1: (A, P, N_random)
C2: (A, P, N_bm25)
C3: (A, P, N_semantic)
```

The anchor and positive stream is shared. Only the way the negative was selected changes.
`C0` uses `A` and `P` without training the model at all.

## 2. The project in one sentence

EmbedEd fine-tunes a pretrained Java-code encoder on labelled clone pairs, changes only the
**negative-sampling strategy**, and checks both ordinary test performance and performance on
functionality that was not seen during training.

The pipeline is:

```text
data rows
  → canonical fragments and train/validation/test splits
  → train-safe corpus
  → shared anchors and positives
  → random, BM25, or semantic negatives
  → (anchor, positive, negative) triples
  → embeddings
  → contrastive fine-tuning
  → validation threshold
  → test F1 and MAP@R
  → seen/unseen-functionality comparison
```

## 3. Dataset and code-clone terminology

### Code objects and labels

| Term | Meaning | Running example |
|---|---|---|
| **Code fragment**, **snippet**, or **method** | A piece of source code treated as one item by the dataset. In this project the items are Java methods or method-like fragments. | `A`, `P`, and each `N_*` method is one fragment. |
| **Functionality** | What a method does, independently of its spelling or exact implementation. | “Return the sum of an integer array” is the functionality of `A` and `P`. |
| **Clone** | Two code fragments that implement the same or sufficiently equivalent functionality under the dataset's clone definition. | `A` and `P` are clones in the example. |
| **Semantic clone** | A clone whose equivalence is about behaviour/functionality, not necessarily identical text. | `sum` and `addAll` use different names but produce the same result. |
| **Pair** | Two fragments considered together, usually written `(func1, func2)`. | `(A, P)` or `(A, N_bm25)`. |
| **Label** | The dataset's binary answer for a pair: `1` means clone and `0` means non-clone. | `label(A, P) = 1`; `label(A, N_bm25) = 0` in the illustrative data. |
| **Positive pair** | A pair with label `1`, used as an example the model should bring closer in embedding space. | `(A, P)`. |
| **Negative pair** | A pair with label `0`, used as an example the model should separate from the anchor. | `(A, N_random)`. |
| **Anchor** | The reference fragment from which a positive and one or more negatives are selected. | `A` is the anchor. |
| **Positive** | The known clone paired with an anchor. | `P` is the positive for `A`. |
| **Negative** | A candidate treated as not being a clone of the anchor. | `N_random`, `N_bm25`, or `N_semantic`. |
| **Triple** | The explicit training unit `(anchor, positive, negative)`. | `(A, P, N_semantic)` is one C3 triple. |

### Clone types

BigCloneBench uses the traditional clone-type vocabulary. The boundaries are not perfect, but the
terms are useful shorthand:

- **Type 1 (T1):** the same code apart from formatting, whitespace, or comments.
- **Type 2 (T2):** the same basic code with renamed identifiers or changed literals/types of
  constants.
- **Type 3 (T3):** code with added, removed, or changed statements while retaining substantial
  similarity.
- **Type 4 (T4):** different-looking implementations with equivalent behaviour.
- **WT3/T4:** “weak Type 3/Type 4”, the weak semantic-clone portion discussed in the project's
  ground-truth warnings. The project does not treat these labels as perfect human truth.

For example, `A` and `P` are at least a Type-2-style example because names and formatting change;
if the implementations were completely different but still summed the array, the example would be
closer to Type 4.

### Dataset names and files

| Term | Meaning | Running example |
|---|---|---|
| **BigCloneBench (BCB)** | The original Java code-clone benchmark from which this project gets its domain and labels. | The source of the “sum” clone judgement. |
| **CodeXGLUE** | The benchmark collection and released format used here to provide BCB through a fixed pair dataset. | The project loads the CodeXGLUE BCB clone-detection dataset. |
| **`data.jsonl`** | The canonical fragment file in the CodeXGLUE layout. Each JSON line contains a source `idx` and a function text. | One line stores `A`; another stores `P`. |
| **`idx`** | The integer identifier used to refer to a fragment in artifacts and pair files. | `idx=17` might identify `A`. |
| **Unique fragment text** | Text deduplicated by content, rather than counting duplicate source identifiers as separate methods. | Two ids containing the exact text of `A` count as one unique text. |
| **Split** | A partition of pair rows used for a different stage of the experiment. | Train pairs make triples; validation chooses a threshold; test reports final metrics. |
| **Train split** | Pair rows available for constructing training examples and mining candidates. | A labelled `(A, P)` pair may enter the shared positive stream. |
| **Validation split** (`valid` or `dev`) | Held-out rows used to choose the decision threshold. It must not be used as the final test result. | Validation decides whether a cosine of `0.81` is called a clone. |
| **Test split** | Held-out rows scored once with the already chosen threshold. | The final F1 is calculated on test pairs. |
| **Train/test fragment overlap** | The same fragment text appearing in both a training pair and a test pair. It is a possible memorisation or leakage route. | If `A` appears in both, test performance may partly reflect recognition of `A`, not general clone detection. |
| **Mining corpus** | The candidate pool from which negatives are selected. EmbedEd uses train-split fragments and removes fragments appearing in test pairs. | `A` can be an anchor, while a test-only method cannot become a training negative. |
| **BCB s′** (read “BCB s-prime”) | The external, functionality-labelled dataset released with the unseen-functionality work. It has 4,600 balanced pairs across 23 functionalities and is used for the later generalisation test. | “Sum arrays” can be one functionality held out from training and evaluated later. |

The released CodeXGLUE data used by this repository is not the full BigCloneBench collection. The
pipeline verifies **9,126 `data.jsonl` lines**, **8,063 unique fragment texts**, and pair counts of
**901,028 train / 415,416 validation / 415,416 test**.

## 4. Experimental conditions and negative mining

### Conditions and variables

| Term | Meaning | Running example |
|---|---|---|
| **Condition** | One version of the experiment with a defined training procedure. | C1, C2, and C3 differ in their negative source. |
| **C0 — baseline** | The untouched GraphCodeBERT model, evaluated without fine-tuning. It is a reference, not a fourth negative strategy. | Embed `A` and `P` with the original model. |
| **C1 — random** | A trained condition whose negatives are selected randomly, then cleaned and filtered. | Train on `(A, P, N_random)`. |
| **C2 — BM25** | A trained condition whose negatives are ranked by lexical/code-token similarity using BM25. | `N_bm25` shares words such as `int`, `value`, and `total` with `A`. |
| **C3 — semantic** | A trained condition whose negatives are ranked by cosine similarity between base-model embeddings. | `N_semantic` is near `A` in the initial embedding space. |
| **Independent variable** | The one factor intentionally changed to answer the research question. | How the negative is selected: random, BM25, or semantic. |
| **Controlled variable** | A factor held constant so it cannot explain a difference between conditions. | Same anchors, positives, `k`, loss, model family, split, batch recipe, and non-condition hyperparameters. |
| **Negative sampling/mining** | Selecting candidate non-clones to pair with each anchor. “Mining” emphasizes that candidates are ranked or filtered. | Find 20 candidates for `A` under each strategy. |
| **Candidate pool** | The fragments initially eligible to become negatives before exclusions. | Train-safe fragments, not test-only fragments. |
| **Shared anchor stream** | The one seeded list of `(anchor, positive)` pairs reused by C1, C2, and C3. | Each condition receives the same `A` and `P`; only `N_*` changes. |
| **`k` negatives** | The number of negatives paired with each anchor-positive example. | The frozen project value is `k=20`: 20 triples per anchor-positive pair. |
| **Top-k** | The first `k` items in a strategy's ranked candidate list. | The 20 highest BM25-scoring clean candidates. |
| **Overfetch** | Requesting more than `k` ranked candidates before filtering, because some will be excluded. | Fetch up to `4 × k`, remove invalid candidates, and keep 20 if possible. |
| **Padding/top-up** | Filling missing negative slots from the remaining clean corpus when a ranked list does not contain enough eligible candidates. | If only 18 BM25 candidates survive, add two deterministic clean candidates. |
| **Self-exclusion** | Never allowing an anchor to be its own negative. | `A` cannot appear as `N` for anchor `A`. |
| **True-clone exclusion** | Removing every fragment that the training labels explicitly identify as a clone of the anchor. | If `(A, P)` is a labelled clone, `P` cannot be a negative for `A`. |
| **False negative** | A mined item treated as label `0` that is actually functionally equivalent to the anchor. | `N_bm25` may secretly implement the same task as `A` even if the dataset did not label that pair. |
| **Hard negative** | A negative that is difficult for the current representation because it looks or embeds like the anchor. “Hard” describes similarity, not guaranteed correctness of its label. | `N_semantic` is harder than `N_random` if its cosine to `A` is higher. |

The exclusion check is **necessary but not sufficient**. It removes known training-label clones, but
it cannot discover every unlabelled clone. The blind manual audit is the project's measurement of
that remaining problem.

### How the three mining strategies differ

- **Random:** samples candidates without trying to make them look like the anchor. It provides the
  easiest comparison point among trained conditions.
- **BM25:** a classic lexical-information-retrieval ranking. The implementation tokenises code,
  splits names such as `addAll` into useful pieces, and ranks methods that share terms and symbols.
  BM25 is therefore **keyword/lexical similarity**, not a neural semantic judgement.
- **Semantic:** encodes the fragments once with the untouched base model and ranks them by cosine
  similarity. It is semantic only in the operational sense “nearby in this model's embedding
  space”; it is not proof that two methods have equivalent behaviour.

The indexes rank all canonical fragments and the final candidate-cleaning step restricts them to
the train-safe mining corpus. The semantic search is an exact matrix multiplication; FAISS or
another approximate index is deliberately out of scope because the corpus is small enough.

### Hardness and the G1 gate

| Term | Meaning | Running example |
|---|---|---|
| **Similarity hardness** | How close the mined negative is to the anchor under the stated similarity measure. | If `cos(A, N_semantic)=0.99` and `cos(A, N_random)=0.60`, the semantic negative is harder in that space. |
| **Hardness check** | A pre-training test that verifies that C1, C2, and C3 really produced different difficulty levels. | Check that mean `cos(anchor, negative)` increases from C1 to C2 to C3. |
| **G1** | The Phase 1 hardness gate. If the manipulation fails this gate, training is stopped rather than interpreted. | A C3 label cannot support a conclusion if C3 was accidentally random. |
| **Gate** | A pass/fail checkpoint before the next phase. | G0 checks data; G1 checks hardness; G2 reviews the main experiment; G3 reviews generalisation/demo outputs. |
| **Cosine gap** | The difference between mean anchor-negative cosine values for two conditions. | `mean(C2) − mean(C1)`. |
| **Standardised gap `d(C1→C2)`** | The C1-to-C2 gap expressed in standard-deviation units, using the pooled spread. It is scale-aware when raw cosine values occupy a narrow range. | A gap of `d=0.6` is about six-tenths of a pooled standard deviation. |
| **Corpus percentile** | The percentile position of a negative in the anchor's full corpus ranking. | The 50th percentile is roughly random; the 100th percentile is nearest-neighbour territory. |
| **D1 scale-aware rule** | The current G1 rule: `C1 < C2 ≤ C3` and `d(C1→C2) ≥ 0.5`. The older absolute `0.02` cosine margin is still recorded for history, but is not decisive. | The example passes only if its measured ordering and standardised gap meet both parts. |
| **Bootstrap over anchors** | Repeatedly resampling anchors to estimate uncertainty in the hardness gap. Negatives from one anchor are correlated, so they are not treated as independent observations. | Resample whole `A`-groups, not individual `N_*` rows. |

## 5. Models, tokens, and embeddings

### The representation path

```text
Java text → tokenizer → token ids + attention mask → GraphCodeBERT
         → mean pooling → L2 normalisation → 768-dimensional embedding
```

| Term | Meaning | Running example |
|---|---|---|
| **Base model** | The pretrained model before this project's task-specific training. | `microsoft/graphcodebert-base`. |
| **GraphCodeBERT** | The chosen pretrained code model. Its pretraining involved source tokens and data-flow information, but this repository uses the token-only loading path. | `A` becomes a 768-number vector. |
| **Option A / token-only** | The frozen implementation choice: supply source tokens only, without extracting a data-flow graph at inference. | The encoder sees `for`, `int`, `return`, and identifiers from `A`, but no DFG input. |
| **Data-flow graph (DFG)** | A representation of how values are defined and used through code. It is the structural signal in full GraphCodeBERT usage. | The value `total` is repeatedly defined/updated and then returned. |
| **Tokenizer** | Converts source text into the model's vocabulary tokens and integer ids. | `addAll` may become one or more subword tokens. |
| **Token** | A model input unit, not necessarily a whole word. Punctuation and pieces of identifiers can be tokens too. | `(`, `return`, and part of `countPositive` may each be tokenised separately. |
| **Maximum sequence length (`MAX_LEN`)** | The largest number of tokens supplied for one fragment. | The current value is 512. |
| **Truncation** | Dropping tokens beyond the maximum length. It makes long methods fit but can hide distinguishing code. | A long method's final branch may be omitted after token 512. |
| **Encoder** | The neural network that maps token sequences to contextual hidden states. | GraphCodeBERT encodes `A` and `P`. |
| **Hidden dimension / embedding dimension** | The number of values in each final vector. | The current model produces 768-dimensional embeddings. |
| **Attention mask** | A mask marking real tokens as `1` and padding as `0`, so padding does not affect pooling. | Extra padding added to a short method is ignored when averaging. |
| **Mean pooling** | Averaging the contextual token vectors for real tokens to create one fragment vector. | Average all non-padding token states for `A`. |
| **L2 normalisation** | Scaling a vector to length one. With normalised vectors, a dot product equals cosine similarity. | `e_A` and `e_P` are unit-length vectors. |
| **Embedding** | The numeric vector representing one fragment. Similarity is measured between embeddings rather than raw source text. | `e_A` is the vector for `A`; `e_P` is the vector for `P`. |
| **Cosine similarity** | The angle-based similarity between two vectors, ranging from `-1` to `1`; higher means closer under this representation. | `cos(e_A,e_P)` is compared with `cos(e_A,e_N)`. |
| **Anisotropy** | A tendency for many embeddings to point in a similar direction, compressing the useful cosine range. | A random pair can have a high cosine even when the methods are unrelated; raw cosine must be interpreted empirically. |
| **Semantic index** | The cached matrix of base-model embeddings used to rank C3 candidates. | Store vectors for the canonical fragments, then compute `E · e_A`. |

For the example, invented scores might be:

```text
cos(A, P)        = 0.92
cos(A, N_random) = 0.61
cos(A, N_bm25)   = 0.84
cos(A, N_semantic)= 0.89
```

The ordering says that the selected C3 negative is harder than the C2 negative, which is harder
than the C1 negative. It does **not** prove that `A` and `N_semantic` have different behaviour;
that is a label and audit question.

## 6. Training terminology

| Term | Meaning | Running example |
|---|---|---|
| **Fine-tuning** | Updating the pretrained model's weights on the project's task rather than training from scratch. | C1, C2, and C3 update GraphCodeBERT; C0 does not. |
| **Contrastive learning/fine-tuning** | Training that makes a positive pair more similar and a negative pair less similar in representation space. | Pull `A` toward `P` and push `A` away from `N_semantic`. |
| **Explicit-triplet loss** | The chosen objective consumes the anchor, positive, and mined negative as one explicit triple. | C2's actual `N_bm25` directly affects its loss. |
| **Triplet margin loss** | The project's loss: `max(0, margin + cos(A,N) − cos(A,P))`, averaged over a batch. | With positive `0.80`, negative `0.74`, and margin `0.10`, loss is `0.04`. |
| **Margin** | The desired minimum advantage of the positive similarity over the negative similarity. | With margin `0.10`, the model wants `cos(A,P)` at least `0.10` above `cos(A,N)`. |
| **Hinge** | The `max(0, ...)` behaviour in the loss: once the margin is satisfied, that triple contributes zero loss. | If positive `0.95` and negative `0.70`, the example has no hinge loss. |
| **Batch** | The group of triples processed before one weight update. | The frozen batch size is 8 triples; each triple carries three code sequences. |
| **Epoch** | One pass over the chosen training triples. | The frozen main recipe uses one epoch. |
| **Seed** | A fixed value controlling random initialisation, shuffling, and sampling order so a run can be repeated. | Main trained runs use seeds 13, 14, and 15. |
| **Run** | One execution of one condition with one seed. | `C2_14` means BM25 condition, seed 14. |
| **Hyperparameter** | A setting chosen before training rather than learned from each example. | Learning rate, batch size, margin, and maximum length. They are held constant across conditions. |
| **AdamW** | The optimiser used to update the model weights, combining Adam-style updates with decoupled weight decay. | Each C1/C2/C3 batch produces an AdamW update. |
| **Learning rate** | The step size of weight updates. | The frozen setting is `2e-5`; it is not tuned separately for C1, C2, or C3. |
| **Weight decay** | A regularisation term discouraging unnecessarily large weights. | The frozen setting is `0.01`. |
| **Warmup** | The initial phase in which the learning rate rises gradually before the decay schedule. | The recipe reserves 100 warmup steps. |
| **Gradient clipping** | Limiting gradient magnitude to avoid unstable updates. | The frozen gradient ceiling is `1.0`. |
| **AMP / fp16 autocast** | Automatic mixed precision; selected operations use 16-bit floating point to reduce GPU memory use. | The T4 recipe uses fp16 autocast; CPU evaluation uses float32. |
| **Checkpoint** | A saved copy of model weights and run state that can be loaded later. | `checkpoint.pt` stores the C2 seed-14 model. |
| **Resume** | Continuing an interrupted run from its matching checkpoint rather than silently starting over. | A Colab restart can resume `C2_14` if its configuration still matches. |
| **Artifact** | A generated file needed by a later stage, such as embeddings, triples, metrics, or predictions. | `triples_C3.jsonl` is a mining artifact. |
| **Artifact gate** | A check that an artifact exists and belongs to the current settings/version before reusing it. | A v5 triples file is refused by a v6 training run. |
| **Configuration fingerprint** | A stable hash of the run settings and input artifact identities. | Changing the loss or triples changes the fingerprint and invalidates the old checkpoint. |
| **Smoke run/test** | A tiny end-to-end run used to check loading, forward/backward passes, checkpointing, and metrics. It is a harness check, not a result. | Run 24 tiny triples and 64 pairs before spending GPU hours on C1–C3. |
| **T4** | The NVIDIA Tesla T4 GPU used for the measured cloud training recipe. | The T4 makes the 512-token, batch-8 experiment fit. |
| **Colab** | The cloud notebook environment used for GPU-heavy stages. | Encode the corpus in Colab, then save artifacts so the local CPU workflow can continue. |

The same encoder path is used for corpus mining, C0, and trained-model evaluation. This prevents
a difference in pooling, truncation, or normalisation from being mistaken for a negative-strategy
effect.

## 7. Evaluation and metrics

### Scores and thresholds

| Term | Meaning | Running example |
|---|---|---|
| **Pair score** | The cosine similarity between the embeddings of the two fragments in a pair. | `score(A,P)=0.92`. |
| **Decision threshold** | A cosine cutoff: scores at or above it are predicted as clone (`1`), and lower scores as non-clone (`0`). | With threshold `0.80`, `(A,P)` is predicted clone and `(A,N_random)` is predicted non-clone. |
| **Threshold selection** | Choosing the cutoff using validation data only. | Try the observed validation scores and keep the F1-maximising cutoff. |
| **Threshold policy** | The predeclared rule for choosing the cutoff. | `max_f1_on_valid`: maximise F1 on validation, then apply that threshold unchanged to test. |
| **Threshold-free** | A measure that does not need one binary cutoff. | MAP@R evaluates the ranking of scores instead of a yes/no decision. |
| **Prediction** | The model's binary decision after applying the threshold. | The model predicts whether `(A,P)` is a clone. |
| **Gold label** | The dataset's recorded answer used for scoring. It is not automatically perfect ground truth. | The stored label for `(A,N_bm25)` is `0`, subject to dataset noise. |

### Classification metrics

Let `TP` be a correctly identified clone pair, `FP` a non-clone called a clone, and `FN` a clone
called a non-clone.

- **Precision** = `TP / (TP + FP)`: when the model says “clone”, how often is it right?
- **Recall** = `TP / (TP + FN)`: how many labelled clones did it find?
- **F1** = `2 × precision × recall / (precision + recall)`: the harmonic mean of precision and
  recall. It is the primary metric in this project.

For an invented test result with `TP=8`, `FP=2`, and `FN=4`, precision is `8/10 = 0.80`, recall is
`8/12 = 0.67`, and F1 is about `0.73`. The project reports precision and recall alongside F1 so
that a single combined number is not misleading.

### Ranking metric

- **`R`** is the number of labelled positive pairs in the evaluated split.
- **Average precision at R (AP@R)** measures how highly those positives appear in the score
  ranking, considering the top `R` positions.
- **MAP@R** is the mean AP@R for the evaluated set. In this implementation it is the secondary,
  threshold-free metric.

Example: if a split has `R=2` and the two highest-scoring pairs are both labelled clones, AP@R is
`1.0`. If only one of the first two is a clone, AP@R is lower even if a later pair is positive.
This complements F1: a model can rank pairs well while a particular threshold is imperfect.

### Reporting vocabulary

| Term | Meaning |
|---|---|
| **Per-seed result** | Metrics from one exact condition/seed run, such as C2 seed 14. |
| **Mean** | Average of the three C1/C2/C3 seed results. |
| **Spread** | How much those seed results vary, reported as a standard deviation or similar summary. |
| **Sanity reference** | An external approximate result used to detect an evaluation harness problem, not a target to tune against. The project records the published CodeXGLUE fine-tuned-CodeBERT reference of roughly F1 `0.95` with its model/protocol caveat. |
| **Harness** | The complete evaluation machinery: pair order, labels, embeddings, threshold selection, and metric code. A surprising score first triggers a harness check. |
| **C0 versus trained condition** | C0 is evaluated once as an untuned reference; C1/C2/C3 are trained and run with three fixed seeds. They must not be presented as if they had identical numbers of training runs. |

## 8. Validity, audits, and generalisation

### Label quality and leakage

| Term | Meaning | Running example |
|---|---|---|
| **Ground truth** | The reference labels used to judge clone/non-clone status. In BCB they are documented but contested, especially for weak semantic clones. | The stored `0` for `(A,N_bm25)` may not settle whether the methods are functionally equivalent. |
| **Label noise** | Errors, ambiguity, or inconsistency in the labels. | A true clone left as label `0` is a false negative in the mined set. |
| **False-negative audit** | A manual check of sampled mined negatives to estimate how many are actually clones. | Review 50 C2 and 50 C3 pairs without seeing their condition, then count `clone`, `not_clone`, and `unsure`. |
| **Blind audit** | An audit where the reviewer does not know whether a pair came from C2 or C3, reducing expectation bias. | The reviewer sees code A/B, not “BM25” or “semantic”. |
| **Wilson confidence interval** | An interval for a proportion that behaves better than a simple normal interval for small samples. | Report an interval around the estimated 21/50 C2 false-negative rate. |
| **`unsure` upper bound** | A conservative audit rate that counts every `unsure` judgement as a clone. | If 15 are `clone` and 2 are `unsure` out of 50, headline rate is 30%; upper bound is 34%. |
| **Token Jaccard** | An auxiliary lexical-overlap diagnostic: intersection of token sets divided by their union. It is a clue about textual similarity, not a clone label. | `A` and `N_bm25` may have high token Jaccard because both use `total` and `int`. |
| **Difflib ratio** | A character-level similarity diagnostic used in the audit report. It is also not a behavioural equivalence test. | Two semantically equivalent methods can have a low character ratio. |
| **Near duplicate** | Text that is very similar, but not necessarily identical or behaviourally equivalent. | `average` is near `sum` lexically while returning a different value. |
| **Validity threat** | A property that could make an apparent finding misleading. | Label noise, fragment overlap, truncation, and changing more than one variable are validity threats. |

The careful claim is: “the model agrees with BigCloneBench's labels at F1 = X under this
protocol.” The project must not claim that it has perfectly detected semantic clones or separated
all hard negatives from false negatives.

### Functionality generalisation

| Term | Meaning | Running example |
|---|---|---|
| **Seen functionality** | A task/functionality represented in the training data. | The model has trained on several “sum an array” examples. |
| **Unseen functionality** | A functionality held out from training and used only for evaluation. | Hold out all “parse a date” methods, then evaluate on them. |
| **Generalisation gap (`Δ`)** | `F1_seen − F1_unseen`. A larger positive value means performance fell more on the unseen tasks. | Seen F1 `0.80`, unseen F1 `0.60` gives `Δ=0.20`. |
| **`F1_seen`** | F1 on functionalities available during training. | Score known-task pairs. |
| **`F1_unseen`** | F1 on the held-out functionalities. | Score the held-out date-parsing pairs. |
| **Holdout** | The fixed set of functionalities excluded from training. The same holdout must be used for all conditions. | C1, C2, and C3 all face the same three s′ functionalities. |
| **Strategy × generalisation interaction** | The project's central open question: whether C1/C2/C3 produce different values of `Δ`. | C2 could have the smallest gap even if it does not have the highest ordinary F1. |
| **s′ evaluation route** | The planned use of the external functionality-labelled BCB s′ package for this comparison. It is a different corpus from the main CodeXGLUE table, so absolute F1 values are not directly interchangeable. | Evaluate or train under the declared s′ protocol, then report seen/unseen results. |
| **UMAP** | A 2-D visualisation method for projecting embeddings. It can illustrate clusters but does not establish model quality. | Plot the same fragment sample for C0–C3 with fixed UMAP settings. |
| **Figure, not evidence** | The rule that UMAP is supplementary; claims must come from tables and measurements. | A pretty cluster around `sum` cannot replace F1 or MAP@R. |

### Four demo outcome modes

The planned demo compares a baseline decision with a selected trained-model decision. The four
possible `(C0 decision, trained decision)` combinations are:

1. both correct;
2. C0 wrong and the trained model right;
3. C0 right and the trained model wrong; and
4. both wrong.

The demo should use real held-out pairs with gold labels by default. Free-text input is illustrative
only and is not part of the evaluation.

## 9. Research-design terminology

| Term | Meaning |
|---|---|
| **RQ1** | Does contrastive fine-tuning improve the representation over the same model without fine-tuning? C0 is the reference comparison. |
| **RQ2** | Does the negative-selection method change the result? This is the C1/C2/C3 comparison. |
| **RQ3** | Do improvements hold on functionality not well represented in training? The base unseen-functionality effect is already in the literature; EmbedEd studies its interaction with negative strategy. |
| **H1 / monotone hypothesis** | Harder negatives improve results in order: random ≤ BM25 ≤ semantic. |
| **H0a / flat hypothesis** | The three strategies are effectively tied within experimental noise. |
| **H0b / inverted-U hypothesis** | Moderate hardness helps but extreme hardness hurts: BM25 > random, while semantic < BM25. |
| **N1, N2, N3** | Older hypothesis notation for random, BM25, and semantic negative strategies. They correspond to C1, C2, and C3. |
| **Confound** | A second change that makes it unclear what caused an observed difference. | Different positive pairs per condition would confound negative strategy with training data. |
| **Reproducibility** | The ability to rerun the same process and recover the same inputs, settings, and measurements. | Fixed splits, seeds, hashes, versioned artifacts, and `run_all.sh`. |
| **Scope** | What the project promises and what it deliberately does not attempt. | Java/BigCloneBench, one loss, four conditions, and no generative clone detector. |
| **Phase 0** | Verify data, overlap, token lengths, and GPU throughput. |
| **Phase 1** | Build the corpus/negative pipeline and pass correctness and hardness gates. |
| **Phase 2** | Run the main C0–C3 experiment, evaluation, and false-negative audit. |
| **Phase 3** | Run functionality generalisation, UMAP, and the demo. |
| **Phase 4** | Write the report, tables, limitations, and reproducibility commands. |
| **G0/G1/G2/G3** | Phase checkpoints: data verification, hardness/mining validity, main-results review, and Phase 3 review. |

## 10. One complete worked example

This miniature example combines the terminology in one place.

### Step 1 — Store a pair

The dataset stores text for `A` and `P` with a label of `1`. It may also contain `(A, N_bm25,
0)`. The two rows are **pair-level** labels; the dataset does not itself provide a complete list
of every method that is behaviourally equivalent to `A`.

### Step 2 — Build safe training inputs

The pipeline deduplicates fragment text into canonical ids, reads the train split, and creates the
train-safe **mining corpus**. It removes a candidate if it is a test-only fragment, is `A` itself,
or is a **known labelled clone** of `A`.

The remaining candidates are ranked three ways:

```text
random  → N_random, ...
BM25    → N_bm25, ...
semantic→ N_semantic, ...
```

The same positive stream is then expanded to 20 triples per anchor:

```text
C1: (A, P, N_random_1) ... (A, P, N_random_20)
C2: (A, P, N_bm25_1)   ... (A, P, N_bm25_20)
C3: (A, P, N_semantic_1)... (A, P, N_semantic_20)
```

### Step 3 — Check the manipulation before training

Using the base embeddings, suppose the mean anchor-negative cosines are:

```text
C1 = 0.61,  C2 = 0.84,  C3 = 0.89
```

The ordering `C1 < C2 ≤ C3` is present. If the standardised C1-to-C2 gap is at least `0.5`,
G1 passes under the current D1 rule. If it fails, the project fixes mining instead of training
nine runs whose conditions were not actually different.

### Step 4 — Fine-tune one triple

Suppose a current model gives:

```text
cos(A, P) = 0.80
cos(A, N_semantic) = 0.74
margin = 0.10
```

The triplet loss is:

```text
max(0, 0.10 + 0.74 - 0.80) = 0.04
```

The update tries to increase the positive score, decrease the negative score, or both. C1, C2,
and C3 use the same loss and margin; their only declared difference is which negatives they saw.

### Step 5 — Score a pair

After training, suppose `cos(A,P)=0.92` and the validation-selected threshold is `0.80`.
The model predicts “clone” because `0.92 ≥ 0.80`. A pair scoring `0.77` is predicted “not clone”.
The threshold is selected on validation, then frozen before test scoring.

### Step 6 — Calculate metrics

On a small test sample, suppose the model has 8 true positives, 2 false positives, and 4 false
negatives:

```text
precision = 8 / (8 + 2) = 0.80
recall    = 8 / (8 + 4) = 0.67
F1        ≈ 0.73
```

MAP@R separately asks whether the highest-scoring pairs are the labelled positives, without
choosing a cosine threshold.

### Step 7 — Check label quality

A blind reviewer inspects sampled `N_bm25` and `N_semantic` pairs. If the reviewer decides that
15 of 50 C2 negatives are actually functionally equivalent, the estimated C2 false-negative rate
is 30%. That does not mean the model is wrong; it means the training label supplied for those
mined examples is unreliable. The rate and its uncertainty are reported.

### Step 8 — Test generalisation

Finally, the model is evaluated on pairs from functionality held out of training. If C2 has
`F1_seen=0.80` and `F1_unseen=0.60`, its gap is:

```text
Δ = 0.80 - 0.60 = 0.20
```

The central comparison is whether this gap differs between C1, C2, and C3—not merely which one
has the largest score on one familiar benchmark split.

## 11. Frozen project recipe at a glance

These are the current implementation settings, not the invented values used in the examples:

| Item | Current value |
|---|---|
| Language/domain | Java code-clone detection |
| Dataset | BigCloneBench via CodeXGLUE |
| Base model | `microsoft/graphcodebert-base`, Option A token-only loading |
| Pooling / output | Mean pooling, L2-normalised, 768 dimensions |
| Conditions | C0 baseline, C1 random, C2 BM25, C3 semantic |
| Negative count | `k=20` for trained conditions |
| Loss | Explicit triplet cosine hinge |
| Margin | `0.10` |
| Maximum input length | 512 tokens; longer fragments are truncated |
| Main batch / epochs | 8 triples / 1 epoch |
| Main training seeds | 13, 14, 15 for C1–C3 |
| Primary metric | F1, threshold selected on validation |
| Secondary metric | MAP@R |
| Main generalisation quantities | `F1_seen`, `F1_unseen`, `Δ` |
| Main reproducibility controls | Fixed splits, shared positives, versioned artifacts, config fingerprints, and saved predictions |

For implementation details, use [`README.md`](README.md) as the index, then [`FINAL_SPEC.md`](FINAL_SPEC.md),
[`SCOPE.md`](SCOPE.md), [`PHASE2_PLAN.md`](PHASE2_PLAN.md), and [`PHASE3_PLAN.md`](PHASE3_PLAN.md).
