# Current-context baseline

**Purpose:** an extraction of the facts the current project context treats as authoritative, for
diffing against `project-presentation-content.md` when it is supplied. Every entry below comes
from the repository as pushed at `9c4543a`, not from the presentation source files.

**Status: this file was written before the source files were available.** It is a working note for
conflict detection, not a deliverable. It will be superseded by `conflict.md` once the source
files are read.

---

## 1. Structural fact: there are two parallel tracks

The project is **not** a single AskUbuntu study. As of the most recent specification, two tracks
run **in parallel**:

| | AskUbuntu track | Code track |
|---|---|---|
| Spec | `project/distilled/` | `project/code-clone/` |
| Domain | Ubuntu duplicate questions | Java code clone detection |
| Dataset | AskUbuntu (`sentence-transformers/askubuntu`), 13,145 rows | BigCloneBench via CodeXGLUE, **9,134** Java fragments |
| Splits | train 12,745 / dev 200 / test 200 | pairs: 901,028 / 415,416 / 415,416 |
| Model | `all-MiniLM-L6-v2` (384-d) | `microsoft/graphcodebert-base` (768-d) |
| Framing | Retrieval | Pair classification |
| Metrics | Recall@10, MRR@10, nDCG@10 | **F1** (primary), **MAP@R** (secondary) |

Anything describing the project as only one of these is incomplete against current context.

---

## 2. Code-track facts (most recent specification)

* **Four conditions:** C0 off-the-shelf baseline, C1 random negatives, C2 BM25 negatives,
  C3 semantic negatives. Same positives; **only the negative-selection method changes.**
* **GraphCodeBERT is loaded as a token-only encoder** (Option A). Its data-flow signal is
  **not** supplied at inference, so no gain may be attributed to structural information.
  `microsoft/codebert-base` is the documented fallback.
* **F1 is primary** because CodeXGLUE reports F1; **MAP@R is secondary** because it is
  threshold-free. Cosine threshold is selected on **validation** and must be reported.
* **Generalisation test** holds out whole functionality categories. RQ3's base effect is
  **already published** (Kitsios et al., ASE 2025): F1 drops up to 48%, **average 31%**, for
  task-specific models. The open contribution is **whether negative strategy changes the gap**,
  measured as `Δ = F1_seen − F1_unseen`.
* **The CodeXGLUE file has no functionality column**, so the generalisation test uses Kitsios
  et al.'s released BCB s′ (23 functionalities, 2,300 + 2,300 pairs).
* **BigCloneBench's ground truth is contested.** Report differences between conditions, not F1
  levels.

---

## 3. Numbers that are verified and must not be altered

| Fact | Value | Source |
|---|---|---|
| CodeXGLUE BCB fragments | **9,134** | CodeXGLUE paper |
| CodeXGLUE BCB split sizes | **901,028 / 415,416 / 415,416** | CodeXGLUE README |
| BigCloneBench true / false clone pairs | **8,915,130 / 288,367** | Svajlenko thesis, via arXiv:2505.04311 |
| WT3/T4 audit | **93%** of 406 sampled pairs mislabelled; WT3/T4 = **95%** of dataset | arXiv:2505.04311 |
| Labelling quality | **≥15%** estimated subjective or erroneous | Krinke, IWSC 2022 |
| Functionality imbalance | **>90%** of true clone pairs in just **8** of 43 functionalities | Krinke, IWSC 2022 |
| Unseen-functionality F1 drop | up to **48%**, average **31%** (task-specific models) | Kitsios et al., ASE 2025 |
| CodeXGLUE reference pipeline | uses **10%** of training data | CodeXGLUE README |

**Marked `[check]`, not yet confirmed — do not state as fact:**
* CodeXGLUE's fine-tuned CodeBERT F1 ≈ 0.95 (harness sanity target).
* NV-Retriever false-negative rate of 47% on StackExchange-domain data.

---

## 4. Numbers that are ranges, never precise

Standing rule from earlier in this project: **when the specification states a range, never write
a precise-looking number.** Two specific figures were fabricated once and retracted:

* AskUbuntu corpus: **~13,000–15,000**, never "14,231".
* AskUbuntu duplicate pairs: **~17,000–20,000**, never "19,231".

Any precise-looking instance of these in presentation material is a conflict.

---

## 5. Literature positions in force

* `Random ≪ {BM25, Semantic}`. Which of BM25 and Semantic wins is **not established** for code.
* **H0b (inverted-U) is the literature-supported prediction**, not H1 (monotone).
  STAR/ADORE: static hard negative sampling *"improves the top-ranking performance but may harm
  the recall capability."*
* ANCE: BM25 negatives overlap only **15%** with what the dense model finds hard.
* AugSBERT: BM25 75.08 vs semantic search 74.99 — effectively tied.
* **Do not add a fused "BM25 + Semantic" condition** — AugSBERT tried it and lost on 4 of 5 tasks.

---

## 6. Product / demo facts

* `domembed` exposes five calls: `load`, `encode`, `similarity`, `search`, `info`.
* **The library is the engineering deliverable; the demo is the presentation centerpiece.**
* The demo must be **result-agnostic** — it must work under WIN / INVERTED-U / TIE / LOSS.
* The demo searches the **real evaluation corpus with gold labels**, not hand-typed strings.
* **One query is an anecdote; the aggregate table is the evidence.**
* The 2D visualisation is **a figure, not evidence** — fixed seed, shared sample, shown after the
  table.

---

## 7. Claims that must not be made

* ~~"adaptive negative sampling is novel"~~
* ~~"pilot-based strategy selection is novel"~~
* ~~"the generalisation test is novel"~~
* ~~"GraphCodeBERT's data-flow signal helps"~~ (not under Option A)
* ~~"our model detects semantic clones with F1 = X"~~ → agreement with BigCloneBench's labels
* Any claim that generalises beyond Java and BigCloneBench
