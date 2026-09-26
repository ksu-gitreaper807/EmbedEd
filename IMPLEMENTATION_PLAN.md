# Implementation plan — Java code clone detection

Translates the project presentation (section 5 "Methodology" and section 9 "Expected Outcomes /
EL Deliverable") into an engineering plan. The deck's source is
[`junk/distilled-project/presentation/project-presentation-content.md`](junk/distilled-project/presentation/project-presentation-content.md);
its normative companions in this repo are [`FINAL_SPEC.md`](FINAL_SPEC.md) (what and why),
[`SCOPE.md`](SCOPE.md) (the bounds), [`GROUND_TRUTH.md`](GROUND_TRUTH.md) (the validity threats).

**Precedence: where the deck and `FINAL_SPEC.md`/`SCOPE.md` disagree, the spec wins.** The deck
predates eight of the corrections that were later verified into the spec; every place this matters
is listed in [Appendix A](#appendix-a--deck-vs-spec-fix-these-before-presenting), and the deck
should be updated before it is presented.

---

## 1. The build, in one paragraph

Five Python modules and one small demo app. A fixed pool of ~8,063 unique Java fragments (9,126-line data.jsonl) and one shared
positive set feed three negative-mining strategies (random / BM25 / semantic); two correctness
gates (true-clone exclusion unit test, hardness check) sit *before* any training; eighteen
fine-tuning runs (3 strategies × 3 seeds, main experiment) plus six smaller runs (generalisation
experiment) produce exactly three tables and one four-panel figure, which are simultaneously the
report's evidence, the demo's data, and the conference paper's results section. `run_all.sh`
regenerates every number.

## 2. What is fixed by the design (from the deck's §5, hardened by `SCOPE.md` Part 1)

| Thing | Value | Source |
|---|---|---|
| Independent variable | how the negative was selected — nothing else may differ between C1/C2/C3 | deck §5 "Design"; SCOPE P1-1 |
| Conditions | C0 baseline (no fine-tuning) + C1 random + C2 BM25 + C3 semantic | deck §5; SCOPE P1-2 |
| Positives | one fixed sample, drawn once with a fixed seed, shared by every condition | SCOPE P1-3 |
| Loss | one loss for all runs, chosen in week 1 (see §4.4 below) | SCOPE P1-4 |
| Seeds | 3, fixed | SCOPE P2-12 |
| Base model | `microsoft/graphcodebert-base` via `Transformer` + mean `Pooling` (**Option A**; data-flow caveat stated) | FINAL_SPEC §5 |
| Dataset | `google/code_x_glue_cc_clone_detection_big_clone_bench` (the 9,126-line / 8,063-unique-fragment file — *not* the 8.9 M-pair full BCB) | FINAL_SPEC §0.1, corrections 1 & 9 |
| Generalisation data | Kitsios et al.'s BCB s′ (23 functionalities, 2,300 + 2,300 pairs, [doi:10.5281/zenodo.17238379](https://doi.org/10.5281/zenodo.17238379)) | FINAL_SPEC §9.3, correction 7 |
| Compute | single free-tier Colab GPU (T4). No multi-GPU, no FAISS, no experiment trackers — the corpus is one matmul | deck §5 "Tools"; SCOPE P5 |

**Definition of done** — verbatim from `SCOPE.md` Part 7:

> C0, C1, C2 and C3 have been evaluated on the fixed CodeXGLUE test split with F1 (threshold
> chosen on validation and reported) and MAP@R over 3 seeds, and on the held-out functionality
> split with `F1_seen`, `F1_unseen` and `Δ` — and those numbers, together with the measured
> false-negative rates, are in one table.

## 3. Repository layout for the code

**Status (as of this commit): Phase 0/1 code landed and unit-tested offline** — `embeded/`
below minus `train/` `eval/` `demo/` (Phases 2–4), plus the Colab sequencer
[`notebooks/Phase0_Phase1_Colab.ipynb`](notebooks/Phase0_Phase1_Colab.ipynb) and the sharing
rules in [`COLAB.md`](COLAB.md). `python -m pytest -q embeded/tests` runs fully offline on the
synthetic mini-fixture (`embeded/fixtures/make_fixture.py`); everything touching the real
8,063-fragment corpus (9,126-line data.jsonl) download or the model runs in the notebook.
The package is named `embeded`, not `code` — a top-level `code/` package shadows the stdlib
`code` module and crashes pytest (via `pdb`) on Python 3.14.

```
embeded/
  settings.py          # all fixed constants in one place: seeds, paths, k, lr, epochs, max_len
                       #   + EMBEDED_ARTIFACTS/EMBEDED_REPORT env overrides (tests, HF-synced caches)
  data/
    prepare_data.py    # loads EITHER HF dataset or dir format; keeps data.jsonl's authoritative
                       #   idx; verifies counts; overlap & token-length stats; writes measurements
                       #   (corpus = train-minus-test §4.1 R1; positives() = seeded shared stream)
  mining/
    bm25_index.py      # rank_bm25, code tokenizer (identifier splitting), all-fragment rows
    semantic_index.py  # encode ALL fragments once with the untouched base model (Option A);
                       #   top-k = exact matmul, cached to corpus_emb.npy; `python -m` entry
    exclude.py         # exclude_labeled_clones(anchor_id, candidates, clone_sets) — §7.2
  negatives.py         # one interface, three strategies; order-preserving exclusion -> corpus
                       #   filter -> exclusion-clean top-up with pad counts reported; writes
                       #   artifacts/triples_C{1,2,3}.jsonl (one {anchor,positive,negative} per line)
  hardcheck.py         # gate G1b: mean cos(anchor, negative) per condition; evaluate() pure for
                       #   unit tests; exit code = verdict; appends to measurements.md
  fixtures/make_fixture.py  # deterministic mini dataset with the BM25 trap + overlap traps
  tests/               # 12 offline tests: adversarial exclusion, Rule 1, positives determinism,
                       #   miner invariants, semantic-vs-reference parity, gate logic
scripts/
  fetch_sprime.py      # Phase 0.7 s′ acquisition + probe + measurements block; --dry-run first
  hf_artifacts.py      # pull/push EMBEDED_ARTIFACTS + measurements via a HF dataset repo
  phase0_throughput.py # Phase 0.5: triplet mini-training bench -> replaces [illustrative] caps
  run_all.sh           # download -> prep -> mine -> pytest -> hardcheck (later phases get wired in)
train/                 # Phase 2: train.py, loss.py (one loss, chosen week 1)
eval/                  # Phase 2–3: evaluate.py (threshold-on-val F1 + MAP@R), generalize.py
                       #   (s′ holdout, Δ per condition), visualize.py (UMAP, fixed seed)
demo/                  # Phase 3: app.py, single-file Gradio (§11 constraints)
report/
  measurements.md      # GENERATED by the pipeline (G0 numbers + gate verdicts) — commit it
  limitations.md       # written FIRST, from FINAL_SPEC §12 (Phase 4)
  tables/ figures/     # generated; never hand-edited
```

`run_all.sh` is a deliverable in its own right (SCOPE P2-22) and feeds the deck's §9 journal- and
conference-submission claims (a reviewer will ask for the reproduction script).

## 4. Phased plan

### Phase 0 — Environment and data verification (days 1–2, inside week 1)

Delivers the deck's "Tools and Resources" line and every week-1 item in FINAL_SPEC §13.

1. Colab environment: `datasets`, `sentence-transformers`, `rank_bm25`, `umap-learn`,
   `scikit-learn`, `gradio`, `pytest`. Pin versions in `settings.py`; nothing heavier (SCOPE P5
   bans infra of any kind).
2. `datasets.load_dataset("google/code_x_glue_cc_clone_detection_big_clone_bench")`; verify
   **9,126 / 8,063 / 901,028 / 415,416 / 415,416** (data.jsonl lines / unique texts / pairs,
   FINAL_SPEC correction 9) against `wc -l` + sha1-dedup equivalents (SCOPE P2-2).
3. Measure **train/test fragment overlap** (FINAL_SPEC §4.2 snippet) → saved number, destined for
   the report either way.
4. Measure **token-length distribution** → choose `max_len` (256 or 512) with a stated reason
   (SCOPE P2-5).
5. **Throughput test**: one minute of training at candidate batch size; from it fix subset size
   (start [illustrative] 50k pairs; CodeXGLUE's own pipeline uses ~90k = 10% `[verified]`) and
   epochs (1–2). Both frozen afterwards.
6. Decide **Option A vs B** for GraphCodeBERT (FINAL_SPEC §5). Default: **A**, with the exact
   caveat sentence. This decision cannot slip past week 1.
7. Download BCB s′ from Zenodo and confirm it loads — this is "the single largest unbudgeted task
   in the plan" (correction 2); discovering it is broken on day 2 costs nothing, discovering it in
   week 3 costs the RQ3.

**Gate G0**: all seven numbers recorded in `report/measurements.md`. Fail ⇒ the data plan is
wrong, stop before writing mining code.

### Phase 1 — Pipeline, mining, correctness gates (week 1)

Delivers deck §5 "Data Collection Methods" — the pipeline that produces the deck's flowchart.

1. `prepare_data.py` + `corpus.py`: corpus = **train-split fragments only**; positives sampled
   once, fixed seed, written to disk and hashed so every condition provably reads the same file
   (SCOPE P1-3).
2. `exclude.py` + adversarial `test_exclude.py`: the hand-built case where an anchor's top-1 BM25
   match **is** a labelled clone; assert it is gone and rank 2 promoted (FINAL_SPEC §7.2).
3. Three miners, same `k`, same exclusion pass, one shared interface — so a fix in one is a fix
   in all three. Semantic mining encodes the corpus with the untouched base model (deck §5:
   "using the base model itself") and takes top-k minus exclusions.
4. `hardcheck.py`: mean `cos(anchor, negative)` per condition; require **C1 < C2 ≤ C3 with a
   visible gap** (SCOPE P1-6, correction 4).
5. One end-to-end **smoke run** (C1, 1 seed, tiny subset) purely to validate harness + eval path.

**Gate G1** — two hard stop-conditions, per SCOPE Part 6:
- exclusion unit test green;
- hardness check passes. *If it fails, stop and fix the mining — an experiment whose manipulation
  did not take effect produces no evidence.* If C2 ≈ C3 because an untuned code encoder is largely
  lexical (the documented risk), tighten candidate pools (e.g. BM25 rank window, semantic top-k
  depth) until the gap is real — do **not** invent a fourth strategy.

### Phase 2 — Main experiment + the two validity measurements (week 2)

Delivers the deck's §6 first two expected-result bullets and the report's main table.

1. **Loss decision, once** (correction 6): `TripletLoss`/`OnlineContrastiveLoss` on explicit
   triples (recommended — the variable *is* the specific negative) **or** MNRL with
   hardness-in-batch-composition. Record the choice + one-sentence rationale in `settings.py`.
   Never mix across conditions.
2. 9 runs: C1/C2/C3 × 3 seeds, identical everything-else; C0 evaluated once (no training).
3. `evaluate.py`: threshold chosen on **validation** per condition and **reported**; F1 primary,
   P/R alongside, MAP@R secondary (SCOPE P2-14/15).
4. **Sanity target**: CodeXGLUE's fine-tuned-CodeBERT F1 ≈ 0.95 `[check]` — if the four
   conditions are not in that neighbourhood, fix the harness before writing anything (SCOPE
   P2-16). External cross-check vs `mchochlov/codebert-base-cd-ft` if time allows (SCOPE P3-4).
5. **False-negative labelling**: sample 50 mined negatives from C2 and 50 from C3, judge
   functional equivalence by hand, ~2 h (SCOPE P2-9). Rubric + raw sheet committed. Compare to
   the nearest published figure (47% on StackExchange-domain mining `[check — NV-Retriever,
   confirm before quoting]`). This converts the GROUND_TRUTH §3 threat from an apology into a
   finding, and makes an inverted-U *explainable*.

**Gate G2**: main table populated; sanity band met. Fail ⇒ harness bug; no reporting work until
fixed.

### Phase 3 — Generalisation, figure, demo (week 3)

Delivers deck §6 third expected-result bullet ("the *size* of the drop across strategies is the
new finding") and §5's visualisation.

1. `generalize.py` (Route A, per FINAL_SPEC §9.3): hold out **k = 3 best-represented
   functionalities** of s′; each condition trains on the remaining 20; report
   **Δ = F1_seen − F1_unseen per condition, as mean over 3 seeds with spread visible**. The
   question is *not* whether F1_unseen is low (~31% average drop is published, cite Kitsios et
   al.) — it is whether **Δ differs across strategies**. Treat as secondary to the main table; if
   wildly noisy, report the noise, do not add folds (SCOPE P6-6; LOFO rotation is the first
   SHOULD if time allows).
2. `visualize.py`: four UMAP panels, fixed seed + `n_neighbors`/`min_dist`, **same fragment
   sample** in every panel, colour by functionality including the held-out ones. **A figure, not
   evidence** — it ships *after* the table, correction 8.
3. `demo/app.py`: Gradio, one file. Real held-out test pairs with gold labels as the **default**;
   per-pair display `gold / C0 cos vs threshold / C_best cos vs threshold`; threshold shown as a
   visible marker on the score bar; free-text paste kept but labelled *"illustrative only — not
   part of the evaluation"*; all four outcome modes rendered (correction 9).
4. Re-read `SCOPE.md` Part 6 tonight (week 3 is when its rules get tempted).

### Phase 4 — Write-up, IEEE paper, deck (week 4)

Delivers the deck's §9 EL deliverable (conference paper in IEEE format, journal-shaped study).

1. `limitations.md` **first**, from FINAL_SPEC §12's seven bullets — claims shaped by it, not
   patched after it.
2. The three tables (SCOPE P2-19): main (4 × {F1, P, R, MAP@R} × 3 seeds); generalisation
   (`F1_seen`/`F1_unseen`/`Δ`); measured false-negative rates (C2 vs C3). Figure §11.2 second.
3. Claims audit against SCOPE Part 5's ~~crossed-out~~ list (e.g. *"agrees with BigCloneBench's
   labels at F1 = X"*, never "detects semantic clones").
4. `run_all.sh` green from a clean checkout; numbers in the report diffed against regenerated CSVs.
5. Paper skeleton mirroring the deck: Introduction (deck §1), Related Work (§2 — add the
   Appendix-A fixes), Problem (§3), Objectives (§4), Methodology (§5 + the four rigor checks),
   Results & Expected Outcomes (§6), Threats to Validity (= `GROUND_TRUTH.md` summary + measured
   FN rates), Conclusion (§7), References IEEE style (§8, plus Kitsios et al. ASE 2025, Krinke
   2022/2025, and the five negative-sampling papers).

## 5. Experiment matrix

| | C0 baseline | C1 random | C2 BM25 | C3 semantic | Total runs |
|---|---|---|---|---|---|
| Main benchmark (CodeXGLUE subset, 3 seeds) | 1 eval, no training | 3 | 3 | 3 | 9 train + 1 eval |
| Generalisation (s′ minus 3 functionalities, 3 seeds) | 1 eval, no training | 3 | 3 | 3 | 9 train + 1 eval |

18 fine-tuning runs, all on one T4, all [illustrative]-sized (≤50k pairs, 1–2 epochs). Rough
budget: 1–2 h/run ⇒ ~20–36 GPU-hours across two weeks of Colab sessions — tight but feasible;
the Phase-0 throughput test replaces these numbers with measured ones before the matrix is
locked. Checkpoint every run to Hugging Face Hub; resume, don't rerun.

## 6. Risk table (deck §6 promises; spec says how each risk is handled)

| Risk | Likelihood | Response |
|---|---|---|
| C2 ≈ C3 hardness (lexical encoder) | medium-high (correction 4) | Gate G1; fix mining, never add strategies |
| Inverted-U (C3 < C2) | medium (STAR/ADORE report it) | This is the predicted finding. Measure FN rate, write it up, do not tune C3's lr to rescue it (SCOPE P6-2) |
| s′ acquisition/format surprises | medium (correction 2/7) | probed on day 2 (Phase 0.7); fallback k = 2; Route B explicitly *not* attempted |
| Colab time-outs / quota | high | throughput-tested run sizes; checkpoints to HF Hub; seed-cut before strategy-cut (SCOPE P6-8) |
| Whole mining pipeline behind schedule | low-medium | SCOPE P8 fallback ladder: 3 strategies → 2 (C1+C3, say inverted-U untestable) → C0+C1 only ("complete weak beats incomplete strong") |
| Benchmark memorisation inflating F1 | medium (fragment overlap) | report overlap number (Phase 0.3); if every condition beats every other by ~20 pts, re-measure overlap before celebrating (SCOPE P6-4) |

## 7. What the deck's methodology slide should gain (one bullet each)

- **Hardness gate before training** (mean cos per condition, `C1 < C2 ≤ C3`) — turns correction 4
  into visible rigor.
- **Adversarially-tested true-clone exclusion** — one line, big reviewer impression.
- **Manual false-negative audit (50 + 50)** — the project's cheapest original contribution to
  validity.
- **Train/test fragment-overlap measurement** — pre-empts the memorisation question.
- **3 seeds, one loss, shared positives** — the "identical settings" claim on the deck made
  concrete.

---

## Appendix A — deck vs spec: fix these before presenting

The deck at `junk/distilled-project/presentation/project-presentation-content.md` is consistent
with the spec on the design (4 conditions, 3 strategies, generalisation built in, same-tooling
list) but predates four verified corrections. Each is a slide edit, not a design change:

1. **Generalisation baseline citation.** The deck attributes the unseen-functionality drop to
   Sonnekalb et al. (ASE 2022, CodeBERT). Sonnekalb stays in the literature review, but the
   quantitative anchor is **Kitsios et al., ASE 2025: up to 48%, average 31% drop — and they
   released BCB s′**, which is what Phase 3 trains on. Also cite them when denying novelty of
   the generalisation *test*; the contribution is the **strategy × Δ interaction** (FINAL_SPEC
   §0.2).
2. **"8+ million pairs / 25,000 projects"** in the deck's "Importance" is true of BigCloneBench
   but not of the artifact trained on. Add the companion half-sentence: *"...and our experiments
   run on CodeXGLUE's filtered subset: 9,126 fragments (8,063 unique texts),
   901,028 / 415,416 / 415,416 pairs"*
   (correction 1).
3. **Metrics wording.** Deck §5 says "MAP@R, precision/recall at threshold"; the spec's
   primary metric is **F1 with the threshold tuned on validation and reported, per condition**,
   with MAP@R as the threshold-free robustness check (FINAL_SPEC §9.1–9.2). The deck currently
   under-claims the discipline.
4. **Method slide missing the four rigor checks** (§7 above). They are already in scope and in
   the schedule; presenting them costs nothing and answers the obvious reviewer question ("what
   makes this comparison trustworthy?").
5. **Loss framing.** The deck never names the loss; correction 6 requires picking
   explicit-triplets vs MNRL-batch-composition **once, in week 1**, and stating it. Add one
   line to the methodology slide when decided.
6. **GraphCodeBERT claim.** Do not let the "Relevance to Current Research" slide imply the
   data-flow signal is exploited: under Option A it is not present at inference (correction 3).
   Suggested deck sentence: "we use GraphCodeBERT as a token-only encoder; its data-flow
   pretraining shapes the weights, but we do not supply data-flow graphs at inference."
