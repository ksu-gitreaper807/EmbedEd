# Code track — BigCloneBench / GraphCodeBERT

**The primary project in this repository.** It was written as the second, parallel track and ran
*alongside* the AskUbuntu track in [`junk/project/distilled/`](junk/project/distilled/), not
instead of it. That track is now archived under `junk/` — still on disk, still in git history on
branch `arena/01a0aae7-embeded`, but no longer part of this project. See
[Repository layout](#repository-layout).

| | AskUbuntu track (archived) | **Code track (this repository)** |
|---|---|---|
| Domain | Ubuntu duplicate questions | Java code clone detection |
| Corpus | ~13,000–15,000 questions | ~9,134 Java fragments |
| Base model | `all-MiniLM-L6-v2` (384-d) | `microsoft/graphcodebert-base` (768-d) |
| Framing | Retrieval (Recall@10, MRR@10, nDCG@10) | Pair classification (F1) + MAP@R |
| Extra test | — | **Generalisation to unseen functionality** |
| Shared | The independent variable; the three strategies; the hardness check; the false-negative measurement | |

**Why run this one too:** the AskUbuntu track answers *"does negative strategy matter?"*. This one
answers *"does negative strategy matter **for generalisation**?"* — which has a published baseline
to compare against (Kitsios et al., ASE 2025) and is not answered anywhere. If you must prioritise,
prioritise this one.

---

## Documents

| File | What it is | Read it when |
|---|---|---|
| [`FINAL_SPEC.md`](FINAL_SPEC.md) | The specification, with verified dataset facts and eight corrections folded in | **Start here.** Before week 1 |
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

## Shared with the other track

The independent variable, the three strategies, and the literature all live under `junk/` now.
They are archived, not deleted — the links below resolve on disk:

- [`junk/distilled/negative-pair-research/`](junk/distilled/negative-pair-research/) — start with
  [`FIVE_KEY_PAPERS.md`](junk/distilled/negative-pair-research/FIVE_KEY_PAPERS.md): the five
  papers that compare Random / BM25 / Semantic, with the merits and demerits as each paper
  discusses them. Then [`STRATEGY_EVIDENCE.md`](junk/distilled/negative-pair-research/STRATEGY_EVIDENCE.md)
  for the ten-source version.
- [`junk/distilled/embedding-library/`](junk/distilled/embedding-library/) — the `domembed` packaging
  layer. See [`CODE_TRACK.md`](junk/distilled/embedding-library/CODE_TRACK.md) for how it applies
  here.
- [`junk/distilled/adaptive-embedding/`](junk/distilled/adaptive-embedding/) — the v2
  "pick the strategy automatically" plan. See its §18 for the code-domain mapping.

**Keep `negatives.py`'s strategy interface identical across both tracks**, so a fix in one
benefits the other.

---

## Repository layout

This repository used to hold several parallel tracks and a large body of distilled research notes.
The code track has been promoted to be **the** project, and everything else has been archived.

| Path | Status |
|---|---|
| `README.md` | This file — the project index |
| [`FINAL_SPEC.md`](FINAL_SPEC.md) | The specification |
| [`SCOPE.md`](SCOPE.md) | MUST / SHOULD / out of scope, and the stop condition |
| [`GROUND_TRUTH.md`](GROUND_TRUTH.md) | BigCloneBench's ground-truth defects |
| `junk/` | **Archived, gitignored.** Every other former track and note: `distilled/`, `distilled-project/`, `presentations/`, `research/`, `project/distilled/` (the AskUbuntu track), the `README.md` that indexed them, and `README-EmbedEd-original.md` (this repo's first stub README) |

`junk/` is ignored by git (`.gitignore`) so it stays out of commits, but it is **not deleted** —
the files remain on disk, and the same content is preserved in the history of branch
`arena/01a0aae7-embeded`. To bring any of it back under version control, delete the `junk/` line
from `.gitignore` and `git add junk/`.
