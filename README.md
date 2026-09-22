# Java code clone detection — BigCloneBench / GraphCodeBERT

**The project in this repository.** Domain-specific contrastive embeddings for code similarity:
compare negative-sampling strategies on Java clone pairs from BigCloneBench, fine-tuning
GraphCodeBERT as the encoder. Everything this repository used to hold around it is archived under
`junk/` — see [Repository layout](#repository-layout).

| | |
|---|---|
| Domain | Java code clone detection |
| Corpus | ~9,134 Java fragments (BigCloneBench via CodeXGLUE) |
| Base model | `microsoft/graphcodebert-base` (768-d) |
| Framing | Pair classification (F1) + MAP@R |
| Independent variable | How negatives are selected: random / BM25 / semantic |
| Extra test | **Generalisation to unseen functionality** |

**The question:** does negative-sampling strategy matter **for generalisation**? Not just F1 on a
fixed benchmark — whether the strategy changes how big the drop is on functionality the model
never trained on. There is a published baseline for the drop itself (Kitsios et al., ASE 2025:
up to 48%, average 31%), and no published answer for the strategy question.

---

## Documents

| File | What it is | Read it when |
|---|---|---|
| [`FINAL_SPEC.md`](FINAL_SPEC.md) | The specification, with verified dataset facts and eight corrections folded in | **Start here.** Before week 1 |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | The engineering plan: phases, modules, gates, run matrix, and the deck-vs-spec fixes | After the spec, before writing code |
| [`SCOPE.md`](SCOPE.md) | MUST / SHOULD / optional / out of scope, anti-scope-creep rules, stop condition, fallback | Before week 1; again in week 3 |
| [`GROUND_TRUTH.md`](GROUND_TRUTH.md) | BigCloneBench's documented ground-truth defects, and what this project does about them | Before week 1, and again before writing the limitations section |

---

## The eight corrections, in one place

Numbered as in [`FINAL_SPEC.md`](FINAL_SPEC.md).

| # | The correction |
|---|---|
| 1 | "~8 million clone pairs" describes BigCloneBench, **not** the CodeXGLUE file you download, which is 9,134 fragments and 901,028 / 415,416 / 415,416 pairs |
| 2 | The CodeXGLUE version has **no functionality column**, so the generalisation test needs data from outside it (use Kitsios et al.'s released BCB s′) |
| 3 | GraphCodeBERT's **data-flow signal does not survive a default `sentence-transformers` load** — pick a loading option and state it honestly |
| 4 | Do not assume `Random < BM25 < Semantic` in difficulty — **measure it**; an untuned code encoder is largely lexical, and the NLP analogue found BM25 ≈ semantic |
| 5 | The true-clone exclusion check is **necessary but not sufficient** — it cannot remove unlabelled clones, which are exactly what harder mining finds more of |
| 6 | Explicit `(anchor, positive, negative)` triples mean you are **not** using `MultipleNegativesRankingLoss`; pick one loss, once, and say which |
| 7 | The generalisation test as written **cannot run on the CodeXGLUE file alone** (see 2) |
| 8 | The 2D visualisation is **a figure, not evidence** — fix the seed, use one shared sample, keep it after the table |

---

## The three findings from checking the specification

These came out of verifying the spec against sources, and each one changed the plan.

1. **The generalisation test already exists.** Kitsios et al. (ASE 2025) measured an F1 drop of up
   to 48% (average 31%) for task-specific models on unseen functionality. Do not claim RQ3 is
   novel. The open question — and the contribution — is **whether the negative-sampling strategy
   changes the size of that gap**.

2. **BigCloneBench's ground truth is contested.** 93% of a sample of 406 WT3/T4 clone pairs were
   found mislabelled, and WT3/T4 is 95% of the dataset; at least 15% of labels are estimated
   subjective or erroneous. Not a reason to switch — a reason to **report differences rather than
   levels** and to measure the false-negative rate. See [`GROUND_TRUTH.md`](GROUND_TRUTH.md).

3. **CodeXGLUE's own pipeline uses 10% of the training data.** Your "manageable subset" decision
   has direct precedent in the benchmark's reference implementation. Use it as a justification.

---

## Repository layout

The repository used to hold several parallel tracks and a large body of distilled research notes.
The code clone detection project is now the whole repository; everything else has been archived.

| Path | Status |
|---|---|
| `README.md` | This file — the project index |
| [`FINAL_SPEC.md`](FINAL_SPEC.md) | The specification |
| [`SCOPE.md`](SCOPE.md) | MUST / SHOULD / out of scope, and the stop condition |
| [`GROUND_TRUTH.md`](GROUND_TRUTH.md) | BigCloneBench's ground-truth defects |
| `junk/` | **Archived, gitignored.** Not part of the project: `distilled/` and `distilled-project/` (research notes — the spec still cites `junk/distilled/negative-pair-research/`), `presentations/`, `research/`, `project/distilled/` (the archived duplicate-questions material), the `README.md` that indexed them, and `README-EmbedEd-original.md` (this repo's first stub README) |

`junk/` is ignored by git (`.gitignore`) so it stays out of commits, but it is **not deleted** —
the files remain on disk, and the same content is preserved in the history of branch
`arena/01a0aae7-embeded`. To bring any of it back under version control, delete the `junk/` line
from `.gitignore` and `git add junk/`.
