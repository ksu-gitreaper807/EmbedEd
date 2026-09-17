# Conflicts

Discrepancies between the **current project context** (the repository as of `0f6c48f`, including
`project/code-clone/FINAL_SPEC.md`, `GROUND_TRUTH.md`, `SCOPE.md` and
`distilled/negative-pair-research/`) and the supplied presentation material.

**In every case below, the current-context version was retained and the presentation follows it.**

---

## Conflict 1 — Scale of the dataset as presented

- **Current context:** BigCloneBench **proper** has 8,915,130 true clone pairs and 288,367 false
  clone pairs across 43 functionalities (Svajlenko's thesis, via arXiv:2505.04311). But the file
  the project actually loads is the **CodeXGLUE version: 9,134 Java fragments**, with
  **901,028 / 415,416 / 415,416** train / validation / test pairs. `FINAL_SPEC.md` §0.1 flags
  stating only the 8-million figure as **Correction 1** and requires both numbers be given.
- **`project-presentation-content.md`:** "BigCloneBench documents **8+ million** validated
  code-clone pairs drawn from **25,000** real open-source projects" — presented as the scale of
  the project's data, with no mention of the 9,134-fragment subset actually used.
- **Action:** Current context retained. Slide 17 (*Data Collection Methods*) states both figures
  explicitly and labels which one is loaded. The 25,000-project figure is accurate for
  BigCloneBench and was kept.

---

## Conflict 2 — Primary evaluation metric

- **Current context:** **F1 primary, MAP@R secondary.** F1 is primary because it is the metric
  CodeXGLUE officially reports, so results are comparable to published work; MAP@R is retained as a
  threshold-free robustness check. This was an explicit user decision.
- **`project-presentation-content.md`** (§5, Data Analysis Techniques): "Quantitative comparison of
  **retrieval/similarity metrics (e.g., MAP@R, precision/recall at threshold)**" — MAP@R listed
  first and F1 not mentioned.
- **Action:** Current context retained. Slide 19 (*Data Analysis Techniques*) presents F1 as
  primary with precision and recall, and MAP@R as secondary, and states the threshold-discipline
  rule (threshold chosen on validation and reported).

---

## Conflict 3 — Expected direction of the result

- **Current context:** The literature-supported prediction is **H0b — the inverted-U**: hardness
  helps up to a point and then hurts. `FINAL_SPEC.md` §3.1 and §10 state this explicitly, citing
  STAR/ADORE — static hard negatives *"improve the top-ranking performance but may harm the recall
  capability"* — and AugSBERT, where BM25 (75.08) and semantic search (74.99) are statistically
  indistinguishable. The consistent cross-paper finding is `Random ≪ {BM25, Semantic}`, with the
  winner between the last two **not established**. H1 (monotone) is described as the weakest
  hypothesis.
- **`project-presentation-content.md`** (§6, Expected Results): "**Harder negative-mining
  strategies (BM25, semantic) expected to outperform random negatives**, consistent with general
  contrastive learning literature" — presented as the expectation, with no mention of the
  inverted-U or of the false-negative mechanism that produces it.
- **Action:** Current context retained. Slide 21 (*Expected Results*) presents all three
  hypotheses (H1 / H0a / H0b), identifies the inverted-U as the prediction the literature actually
  supports, and states that BM25-vs-semantic is genuinely open. The claim that at least one
  fine-tuned strategy beats the baseline (also in the source) was kept, as it is consistent with
  current context.

---

## Conflict 4 — Novelty of the negative-strategy comparison

- **Current context:** The three-way comparison is **well established for natural-language
  retrieval** — AugSBERT (NAACL 2021), ANCE (ICLR 2021), STAR/ADORE (SIGIR 2021), SimANS
  (EMNLP 2022) and DPR (EMNLP 2020) all compare random / BM25 / semantic negatives directly (see
  `distilled/negative-pair-research/FIVE_KEY_PAPERS.md`). What is open is the comparison **for
  code**, and the interaction with generalisation. Standing project rule: never claim "never been
  done"; be conservative in novelty claims.
- **`project-presentation-content.md`** (§2 Relevance, §4 Expected Contributions): "a systematic
  comparison **not present in existing published work**" and "a comparison **not present in
  existing published literature for this domain**".
- **Action:** Current context retained. The qualified phrasing ("for this domain") is preserved;
  the unqualified claim is not used. Slides 8, 11, 13 and 17 state plainly that the comparison is
  established for text retrieval and unrun for code.

---

## Conflict 5 — Novelty of the generalisation test, and the baseline it should cite

- **Current context:** Generalisation to unseen functionality is **already established** by
  **Kitsios, Sovrano, Barr & Bacchelli, "Detecting Semantic Clones of Unseen Functionality", ASE
  2025**: F1 drops by **up to 48%, average 31%**, for task-specific models; BCB has 43
  functionalities; the authors release a functionality-balanced dataset (BCB s′: 23
  functionalities, 2,300 + 2,300 pairs). `FINAL_SPEC.md` §0.2 states the generalisation test must
  **cite** this and must **not** be claimed as novel; the contribution is whether negative strategy
  changes the size of the gap. `SCOPE.md` Part 5 lists "the generalisation test is novel" among
  claims that must not be made.
- **`project-presentation-content.md`:** attributes the generalisation concern only to
  **Sonnekalb et al., ASE 2022** (~96.5% F1 "drops significantly"), and lists "a
  **generalization-focused evaluation methodology**" as an **expected contribution** of this
  project (§4, §2).
- **Action:** Current context retained. Slide 14 cites the ~31% average drop as the measured
  baseline; slides 8 and 11 frame generalisation testing as a built-in part of the design rather
  than a novel contribution; slide 23 lists it as reinforcing standard practice. Sonnekalb et al.
  is retained in the literature table and references (it is a genuine and relevant study), but the
  quantitative baseline and the novelty framing follow current context. Kitsios et al. was added to
  the reference list.

---

## Conflict 6 — GraphCodeBERT and the data-flow signal

- **Current context:** GraphCodeBERT is loaded through `sentence-transformers` as a **token-only
  encoder** (`FINAL_SPEC.md` §5, Option A). The full model takes source tokens **and** an extracted
  data-flow graph; `sentence-transformers` has no module for the second input, so **no data-flow
  information is supplied at inference** and no gain may be attributed to structural information.
  `GROUND_TRUTH.md` and `SCOPE.md` require this caveat be stated; `microsoft/codebert-base` is the
  documented fallback.
- **`project-presentation-content.md`:** lists GraphCodeBERT as the base model and cites its
  contribution as "**structural, data-flow-aware pretraining** for code" (§2), and says the project
  "adopts the same base models (GraphCodeBERT)" for comparability — with no caveat that the
  data-flow signal is not used here.
- **Action:** Current context retained. The literature-table entry for GraphCodeBERT is kept as a
  description of that paper's contribution, but the deck never attributes any part of this
  project's method to data-flow structure. Slide 20 lists the base model as "GraphCodeBERT
  (pretrained, 768-d)" without a structural claim.

---

## Conflict 7 — Date on the EL deliverable slide

- **Current context:** Today's date is **17 September 2026**.
- **`EL-phase-1.md`** (Slide 13): "***Date on slide: Tuesday, 22 October 2024***".
- **Action:** Current context retained. Slide 35 carries "Date on slide: 17 September 2026". All
  other content of that template slide (patent / journal / conference / research proposal, and the
  "conference publication in IEEE format is the planned focus" emphasis) was reproduced unchanged.

---

## Notes on what is *not* recorded here

The following were considered and deliberately **not** recorded as conflicts, because they are
adaptations or additions rather than discrepancies:

- **Timeline.** `EL-phase-1.md` supplies a generic 13-week EL phase structure (Initiation W1–2,
  Planning W3–4, Execution W5–10, Closure W11–13); the current context has a 4-week research
  execution plan. Slide 31 keeps the EL phase structure required by the format reference and maps
  the four-week plan inside it.
- **Hybrid negative mining** as a future direction (content §6). Current context notes that a fused
  BM25 + semantic strategy lost on 4 of 5 tasks in AugSBERT. This is a caveat on a speculative
  future direction, not a conflicting claim; it appears as a note on slide 29.
- **References 7–10** (Kitsios; AugSBERT; STAR/ADORE; Krinke) were added to the reference list
  because the presentation cites them. Adding citations is not a conflict.
