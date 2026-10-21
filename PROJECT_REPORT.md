# EmbedEd — Consolidated Project Report (plain-language edition)
**Everything done so far, decisions taken (with reasons), results at each stage, and the outputs/evidence behind them.**

*Compiled 2026-10-17 from the repo's own records (`PROGRESS.md`, `PHASE2_PLAN.md` §8–§9,
`PHASE3_PLAN.md` §8–§9, `PHASE4_PLAN.md`, `sanitycheck.md` §5–§9, `report/measurements.md`,
the shipped consolidation commit `ede587a`, and the dated commit history). The numbers are
identical to the technical records; only the wording is simplified.*

---

## Glossary — the ten terms this report needs

| term | plain meaning |
|---|---|
| **clone / non-clone pair** | two code snippets that do the same job (clone) or different jobs (non-clone); the model must tell them apart |
| **training example ("triple")** | one known match (A↔B) plus one "wrong answer" snippet (C) picked to teach the model a contrast |
| **hard negative** | a wrong-answer example chosen because it *looks* very similar to A — the training strategy is exactly *how* these are picked (at random vs. by search) |
| **mislabelled example ("false negative")** | a wrong-answer example that is actually an undiscovered clone — the dataset's label is wrong, so the model is punished for correct behaviour |
| **the "do-nothing" score (0.2406)** | with 13.7% of test pairs being clones, a model that answers "non-clone" to everything scores exactly 0.2406 — our floor |
| **threshold maxed out ("pinned at 1.0")** | the model's own cut-off setting went to the extreme — proof it stopped telling anything apart |
| **the bouncer** | our filter that throws out wrong-answer examples too similar to the true match (kept the nickname; it is a Jaccard ≥ 0.40 text-overlap cut) |
| **run ID ("fingerprint")** | a short ID computed from a run's complete settings file — same ID, same experiment; we never silently change a run's identity |
| **blind audit** | a human judges 50 sampled training pairs *without knowing which experiment they came from*, and the answer sheet is saved before the check starts |
| **G1 / G1s checks** | pre-training quality gates proving the three picking strategies really produce different levels of trickiness — if they don't, the experiment measures nothing |

Everything else: **F1** = the score balancing missed clones vs. false alarms (0 = useless, 1 = perfect);
**MAP@R** = a ranking score that no "answer non-clone to everything" trick can fake;
**checkpoint/push** = saving results to our Hugging Face storage after each finished step;
**seed** = the random starting number (13/14/15); same seed + same settings = same run.

---

## 0. Executive summary — the five headline findings

| # | Finding (plain) | Where proven |
|---|---|---|
| **F1** | Picking "wrong answer" training examples by **searching for look-alikes backfires completely**: both search-based strategies (word-search C2, meaning-search C3) trained models that score **0.2478 / 0.2707 — the do-nothing score is 0.2406** and the untouched model gets 0.2572. The models answered "non-clone" to everything (their cut-offs maxed out in 10 of 12 runs each). Random picking (**C1, 0.7464**) beats them by ~3×. | `## Phase 2 — main results` in `report/measurements.md`; board push `0ce5bf4`; `sanitycheck.md` §9.1 |
| **F2** | **Why it happens — it's *which* bad examples, not *how many***: C3 and C4 carry the *same* share of mislabelled examples (30%, measured by blind audit) yet differ by 0.16 in score with opposite behaviour. The dangerous kind is a mislabelled example that is *both* a true clone *and* a near word-for-word copy; the total count is just the dose, and the training rule's required safety gap (margin 0.10) is the amplifier. | `sanitycheck.md` §9.4 FINAL (audit passed 2026-10-02) |
| **F3** | A **one-line filter (the bouncer) half-fixes it**: C4 = 0.4349 ± 0.0246, no maxed-out cut-offs, ranking score 2.7× C3's — while removing ~40% of the measured bad examples and **zero** of the checked good ones. | §9.4 + the filtered strategy in `embeded/negatives.py` (`56822ba`) |
| **F4** | On brand-new functionality (corpus s′), **every search-based strategy performs at or below the untouched model** (untuned = 0.6934; C2 0.6674, C4 0.6687, C6 0.6865 at/below it; C3 "matches" it only because its cut-off maxed out again). Only clean training helps (C1 +0.054, its filtered control C5 +0.036). **Fine-tuning on badly picked examples is worse than not fine-tuning at all.** | `## Phase 3 — generalisation` table, shipped `ede587a`; 18/18 runs |
| **F5** | The famous ~0.95 results published for this benchmark and our 0.75 are **different measuring sticks**, not a contradiction: theirs is a near-copy-heavy regime, ours scores the whole 415k-pair test set at its natural mix (where doing nothing scores 0.24). The *ordering* of strategies and the "gave up" signature reproduce on both sticks — evidence for, not against, our numbers. | Non-comparability note in the s′ table block; the same model scores 0.2572 and 0.6934 |

---

## 1. What never changed across the whole project

- **Model & training:** GraphCodeBERT; snippets trimmed to 512 tokens; batches of 8; one
  pass over up to 32,000 training examples (16,000 on the smaller corpus); the training rule
  demands a 0.10 similarity gap between true matches and near-misses.
- **Same three starting numbers (seeds 13/14/15)** for every experiment; the score's cut-off
  is always chosen on data the model *did* train on, then frozen — and **always reported**,
  because its maxing-out became our main warning light.
- **Two quality gates before any result is believed:** the trickiness checks (G1/G1s) and
  the blind 50-pair human audits.
- **Reproducibility:** every run saves a settings file with its computed ID; identical
  inputs have re-produced identical files (checked byte-for-byte at the re-mining steps).
- **No silent changes:** every design decision was written down *before* the runs it affects;
  later changes are appended as dated amendments, never edited in.

---

## 2. Stage by stage: done, decisions, results, evidence

### S1 — Phase 0: trust the data, size the job *(≈ Sep 24–26, 2026)*
**Done:** loaded the standard benchmark (BigCloneBench via CodeXGLUE) and verified its
counts; measured how much test material also appears in training (8.8% of test snippets);
measured snippet lengths; timed the hardware; downloaded the second, functionality-labelled
corpus (s′).
**Decisions:** trim to 512 tokens (covers half of all snippets fully; hardware budget);
fp16 arithmetic (full precision ran out of memory); 8 × 32,000 × 1 pass from the timing
sweep; **after a cloud machine wiped itself and ~35 minutes of work, save to Hugging Face
storage after every heavy step — a rule never broken again.**
**Results (entry checks):** all dataset counts matched the spec (9,126 snippets, 901,028
training pairs, 415,416 test pairs, 13.7% clones); s′ = 4,600 pairs over 23
functionalities — both PASS.
**Evidence:** `report/measurements.md` §Phase 0/0.5/s′; commits `7437c90`, `f01a150`,
`ff212c9`; the incident list in `PROGRESS.md` §5 (including one exposed access token —
revoked, and secrets moved to Colab's safe storage forever).

### S2 — Phase 1: pick training examples + the first quality gate *(→ Sep 27)*
**Done:** built 32,000 triples per strategy (the *same* sentence pairs and random draws for
all three — only the "wrong answer" selection differs, verified by tests); ran the
trickiness check G1; built the diagnostics that later explain the collapse; built the blind
audit tooling.
**Decisions:** **D1 (recorded before training):** the original "visible gap" rule used a
fixed 0.02 number that ignored the actual scale of the similarity space (the diagnostics
showed this) → the gate became **scale-free: the right ordering must hold AND a
standardised gap ≥ 0.5**; the old FAIL result is **kept in the record on purpose** (the
project never deletes an inconvenient measurement).
**Results:** G1 went FAIL → (new rule) → **PASS with gap 0.614**. The mislabelled-example
share was measured as rising with trickiness (~0% / 30% / 42%) — the ladder that later
explains everything.
**Evidence:** commits `502ef54`, `3a8fdf5`, `4146485`; the G1 history section in the report
(FAIL preserved); the check can be re-run offline against the saved result.

### S3 — Pre-training audit: how bad is the label noise? *(Sep 27)*
**Done:** the Phase 2 training code written and unit-tested (70 tests); the blind audit
executed **before** any training.
**Decisions:** **D2 — keep the picking as specified, measure and report the noise rate**
instead of pre-filtering. Reason: the audit did find ~30% mislabelled examples among the
meaning-search picks, but the *obvious* fix (drop near word-for-word copies) would remove
only 5–10% of them — true clones in this corpus typically differ in ~70% of their
characters. A 40%-word-overlap cut looked perfect **on the 50 audited pairs** but was
untested beyond them; cutting by the model's own similarity score would change the very
thing being studied. So: report the noise as a floor, and **reserve the filter as a
separate future experiment** — which is exactly what C4 later became.
**Results:** audited noise: C3 30%, C2 42%, C1 ≈0–2%.
**Evidence:** `PROGRESS.md` §4.2 (full D2 text); `artifacts/audit/AUDIT_JUDGEMENTS.md`;
audit tooling that refuses to score against changed/stale files.

### S4 — Phase 2 campaign: the collapse *(Sep 28 → 30)*
**Done:** 9 runs (untouched model + 3 strategies × 3 seeds) across five cloud machines,
saved to storage after each run; everything consolidated on Sep 30; the in-house
"red team" document (`sanitycheck.md`) opened to attack our own findings; closed-form
math accounts for both score floors added.
**Decisions:** cut-offs frozen to training-side data only; re-running a test run can never
overwrite a real run (own folder — added after a near-miss); a collapse-signature check
added from external review (how many test pairs get the *identical* internal score).
**Results (the main table):** untouched 0.2572 · **random 0.7464 ± 0.0176** (ranking 0.6629)
· word-search 0.2478 · meaning-search 0.2707 — **both search strategies landed on the
do-nothing score with maxed-out cut-offs in 10 of 12 runs each.** The models didn't get
worse — they stopped discriminating entirely.
**Evidence:** `report/measurements.md` `## Phase 2 — main results`; board push `0ce5bf4`;
per-run settings files on the HF snapshot; the five-machine configuration audit block
(`0c87d96`).

### S5 — Attacking our own finding *(Sep 30 → Oct 1)*
**Done:** three registered "ways this could be wrong" tested.
**Decisions/Results:** **"it's just a benchmark quirk" — REFUTED** (the collapsed models
score *below the untouched model*, which a quirk cannot explain); the training-loss wobbles
traced to fp16 arithmetic, not measurement error; the floors proven mathematically (§5.1–5.3);
**improvement is monotone in the measured noise rate**. The causal story was written down:
*noise count = the cause; the safety margin = the amplifier.*
**Evidence:** `sanitycheck.md` §5, §9.1; commits `b63f693`, `fd97aab`, `769921a`,
`e58d1d9`.

### S6 — The bouncer: strategy C4 *(Oct 1 → 2)*
**Done:** implemented the **filtered strategy** (word-search ranking, then drop anything
with ≥ 40% word overlap to the known match, plus a head-start skip); pre-registered the
whole C4 plan **before running it**; re-mined (recipe v7) with the three original
strategies verified **byte-identical** to before; trained C4 on all 6 seeds; ran its
mandatory audit.
**Decisions:** the audit gate **must pass before any C4 claim** (≤ 42% mislabelled
allowed — C2's measured rate); C4's first result was deliberately left uninterpreted until
the audit landed.
**Results:** C4 first-seed 0.4108 → **all six seeds: 0.4349 ± 0.0246**, no maxed-out
cut-offs, ranking score 2.7× the collapsed C3. **Audit PASSED: 30% mislabelled [19–44%] —
exactly C3's rate, opposite outcome.** That contrast is the paper's central mechanism
finding (F2): the filter removed the *near-copy* kind of bad example (the dangerous kind),
not the count. The 40% cut removed ~40% of C2's bad examples and **none of the audited
good ones** (good pairs top out at 34.5% overlap; true-clone pairs median 41.8% — a real
gap the cut sits inside).
**Evidence:** `56822ba`, `847a698`, `c502697`, `d9ffd7c`, `d561b25`, `912b0d0`, `e9d83fb`;
`artifacts/audit_c4/`; `sanitycheck.md` §9.4. *En-route bug:* a filter change accidentally
landed inside code shared by all strategies (caught at scale) → rule: unique anchors in
code edits + regression tests.

### S7 — Completing the picture: C5/C6 *(Oct 2 → 3)*
**Done:** the filter generalised to all three picking strategies → the full 3 × 2 picture
(recipe v8, all earlier strategies re-mined **byte-identical**); the C5/C6 plans
pre-registered with fixed decision rules; both trained to 6 seeds; a confusion-matrix
board built; the naming question settled.
**Decisions:** **naming key (after a C4↔C5 swap request — declined):** the numbers are
build order, C4 = word-search + filter, C5 = random + filter, C6 = meaning-search + filter;
documented in three places so nobody re-litigates it. **Both C5/C6 results stay
"pending audit" before any headline claim** (predictions written down first: C5 ≈ 0–2%
noise, C6 25–35%).
**Results:** the re-mine **hit both predictions**: leftover noise 0.27% for C5 (predicted
≈ 0) and 13.6% for C6 (predicted 10–25%). **C5 0.7399 ± 0.0140 — inside the random
baseline's band, confirming the filter itself doesn't help or hurt a clean supply.**
**C6 0.4722 ± 0.0282** — between the clusters, exactly as its measured 13.6% residual
noise predicts. Audits: **still open** (the one standing gate).
**Evidence:** `e8c04ac`, `1c8ccd4`, `0259ed7`, `f18c3ea`, `5161fcb`, `fb07f51`;
`sanitycheck.md` §9.5.

### S8 — Phase 3: the unseen-functionality campaign *(Oct 3 → 5)*
**Done:** after three same-day scope changes (all recorded **before** any run): six seeds
→ back to **three seeds, 18 runs** ("only 6 models over the same 3 seeds would be
enough"); the small corpus quality-checked (23 functionalities; the three
most-represented — {10, 13, 14} — held out as the "never seen" test, decided by a
written rule, before any score existed); the trickiness gate **passed (gap 0.886)**; all
**18 runs trained and scored**, saved to storage one by one; every Phase-2 model also
graded on the new corpus as a free side-reading (19 evaluations); the results table, the
dot-map figure, and the interactive demo check all produced; **the user consolidated and
shipped the whole report (Oct 5, `ede587a`).**
**Decisions:** train on the new corpus using negatives mined from the same corpus (the
only option that keeps the picking strategy as the only thing being varied); the
side-reading stays clearly secondary; "all four outcome modes" for the demo was defined
as the 2×2 of (untuned model right/wrong) × (best model right/wrong), each cell needing a
real example. The two runs trained under the briefly-larger 6-seed plan were **retrained
under the final plan** rather than given a loophole — a run's identity is its settings.
**Results (unseen functionality, means):** **C1 0.7031 (+0.054 seen-advantage)** · C5
0.7244 · C6 0.6865 · C2 0.6674 · C4 0.6687 · C3 0.7057 with cut-offs maxed again
(0.9998–0.9999). **The untouched model scores 0.6934 → no search-based strategy beats
simply not training** (F4). The strategy ordering and the "gave-up" signature reproduce
across both corpora (F5).
**Evidence:** `report/measurements.md` `## Phase 3 — generalisation` + transfer block +
census/decisions/gate sections; `report/figures/umap_panels.png` + `umap_params.json`
(plot settings fixed and saved before rendering); `demo/app.py --check` output (all four
2×2 cells populated); runs `sprime_C1..C6_13..15` on `Kusshal/Embed`.

### S9 — The tooling & reliability workstream *(continuous, Oct 3 → 5)*
**Done (each as its own tested commit; the offline test suite grew 29 → 150, all green):**
a real bug found by the notebook's own pre-training check (settings were being read once
at start-up instead of per use — broke any machine that had the extra corpora);
**storage downloads split into named slices** so a fresh machine grabs ~300 MB of what it
needs instead of ~4.5 GB (the raw corpus was also removed from the download path);
rate-limit protection (one identity check up front, patient waiting that honours the
server's own "retry after" header, and training refuses to start if saving isn't
configured — no more silently unsaved GPU-hours); **older saved models became gradeable
without retraining** (their settings IDs travel with the score as provenance — but
*continuing* an old run still requires an exact match); a latent bug fixed in the results
table that would have silently omitted the three filtered strategies; **the two modules
the plan had promised but that never existed were built**: the figure renderer (seven
dot-map panels, one shared sample, settings recorded first) and the demo app (real test
pairs, both models' cut-offs drawn on score bars, all four outcome cells required, free
text clearly labelled "illustrative only"); a notebook linter so a whole class of
command-typo bugs can't recur; and the fresh-machine boot sequence made self-diagnosing
(missing inputs now say "pull this first" instead of failing mysteriously).
**Evidence:** the commits `06171e9` → `d43002f`; your successful consolidation run and
cold-start green run.

### S10 — Closure begins *(Oct 5 → today)*
**Done:** `PHASE4_PLAN.md` — the project's own scope says "when both test surfaces and the
noise rates are measured, **stop experimenting and start writing**", so Phase 4 is exactly
that: one combined results table, the two remaining audits, nine bookkeeping repairs
(nine runs' saved score sheets disagree slightly with their raw prediction files, by at
most +0.027 — the raw files are authoritative; a re-grade fixes each), and the write-up
(limitations *before* claims; a single `run_all.sh` that reproduces every number).
`TIMELINE.md`. **`paper/paper.tex`** — a full LaTeX draft with a verified reference list
and honest "pending" markers where gates are still open.
**Evidence:** the files themselves, all pushed on `arena/01a0e714-embeded`.

---

## 3. Master decision register (with reasons)

| ID | Decision | Why (as recorded at the time) | When |
|---|---|---|---|
| Recipe | 512 tokens; 8×32,000×1; fp16 | Measured lengths + timing sweep; memory limits | Phase 0 |
| **D1** | Trickiness gate = right ordering + standardised gap ≥ 0.5; keep the old FAIL on record | The old fixed 0.02 ignored the similarity space's real scale | Sep 27 |
| **D2** | Keep example picking; **measure and report** the mislabel rate; the filter becomes its own experiment later | The obvious fix removes only 5–10% of the noise (true clones differ ~70% of characters); a 40%-overlap cut was untested beyond 50 pairs; cutting by model similarity would change the thing being studied | Sep 27 |
| Score cut-offs | Chosen on training-side data only; always reported | Prevents cheating on the test set; the cut-off value became a diagnostic | Sep 27 |
| Audits first | Blind 50-pair checks **before** training; sheets saved before checking; no headline without its audit | In this corpus, "similar" all too often means "mislabelled clone" | Sep 27, standing |
| **D3** | Train on s′ with negatives mined from s′; the other option becomes a free side-reading | The only option that keeps *picking strategy* as the only variable | Oct 3 |
| **D4** | Hold out the three best-represented functionalities, by a written counting rule | "Most data" measured, not guessed; fixed before any score | Oct 3 |
| **D5** | Phase-3 code lives in `embeded/` | Where the tested runners already are; both call sites agree | Oct 3 |
| **D6** | "All four outcome modes" = the 2×2 of (untuned right/wrong) × (best right/wrong), a real example each | The scope never defined the phrase; this reading is checkable | Oct 3 |
| Scope changes | Six seeds → symmetric entry → **three seeds, 18 runs** | Your calls; all recorded before any 16–18 run existed; the published extras stay published | Oct 3 |
| Naming | C4 = word-search + filter, C5 = random + filter, C6 = meaning-search + filter; **no swap** | Numbers are build order; documented in three places | Oct 2 |
| §9.4 verdict | *Which* bad examples (near-copies) matters, not *how many* | Same 30% rate, opposite outcomes | Oct 2 |
| Headline gates | No C5/C6 claim until their audits are scored | The standing rule, extended to the new strategies | Oct 2 |
| Run identity | Re-scope retrained 2 runs; old models gradeable but not continuable; the paper points at artifacts, never copies | A run's settings file *is* its identity | Oct 3–5 |
| **Open** | One-table layout; headline condition (C1; C5 only if its audit passes); how to frame the 0.95 gap; the δ-ladder experiment only after a scope amendment + finished draft | PHASE4 §9 | Oct 5 |

---

## 4. Master results tables

**Benchmark test set (415,416 pairs; 13.7% clones; do-nothing score 0.2406):**
| strategy | F1 | ranking score | cut-offs | measured mislabel rate |
|---|---|---|---|---|
| untouched | 0.2572 | — | 0.978 | — |
| **random (C1)** | **0.7464 ± 0.0176** | 0.6629 | normal | ≈0–2% |
| word-search (C2) | 0.2478 ± 0.0231 | — | 10/12 maxed | 42% |
| meaning-search (C3) | 0.2707 ± 0.0095 | 0.1826 | 10/12 maxed | 30% [19–44%] |
| word-search + filter (C4) | 0.4349 ± 0.0246 | 0.2760 | all normal | 30% [19–44%] |
| random + filter (C5) | 0.7399 ± 0.0140 | 0.6629 | normal | 0.27% measured (audit pending) |
| meaning-search + filter (C6) | 0.4722 ± 0.0282 | — | normal (0.63–0.82) | 13.6% measured (audit pending) |

**Unseen functionality (the three held-out functionalities; seen-score / unseen-score /
seen-advantage):**
random 0.7570 / 0.7031 / **+0.054** · random+filter 0.7606 / 0.7244 / +0.036 ·
meaning+filter 0.7169 / 0.6865 / +0.030 · word-search 0.6781 / 0.6674 / +0.011 ·
word+filter 0.6731 / 0.6687 / +0.004 · meaning-search 0.7004 / 0.7057 / −0.005 (cut-offs
maxed) · **untouched model 0.7118 / 0.6934 — the floor to beat, and only clean training
beats it.**

**Mechanism numbers:** the dangerous example kind = mislabelled *and* near word-for-word;
the 40%-overlap cut removes ~40% of word-search's noise and 0 good pairs (good pairs max
34.5% overlap < the 40% cut < true-clone median 41.8%); measured noise by strategy
≈0 / 42 / 30%, filtered leftovers 0.27% / 13.6%; trickiness gaps 0.614 (benchmark) and
0.886 (s′).

---

## 4a. Figures & graphs (from `Image_Graphs/`)

**Figure 1 — Phase 2, test-split confusion matrices, all 37 models** (untuned C0 + six
seeds × six conditions; every panel's title carries its own F1 and chosen threshold;
footer: n = 415,416 pairs, threshold chosen on validation, applied unchanged).
This is finding **F1** made visible: read each panel's *top-left/bottom-left* pair —
in every C2 (orange) and C3 (purple) panel the model predicts **non-clone for the
majority of actual clones** (the false-negative quadrant dominates, e.g. C3_13: 47.2%
of true clones caught, 34.9% of non-clones flagged), with the saturated thresholds
(`1.0000`) printed right in the titles. C1 (blues) and C5 (reds) catch ~75–78% of
clones at ~4–5% false-alarm cost; C4 (greens) sits midway (~50–57% caught); C6
(magentas) between. *(Illustrates S4/S7; see §4 table for the numbers.)*

![Phase 2 — test-split confusion matrices, 37 models](Image_Graphs/phase2_confusion_matrices%281%29.png)

**Figure 2 — s′ fragment map per strategy, seven panels** (the Phase-3 §4 figure;
held-out functionalities {10, 13, 14} starred; same seeded fragment sample in every
panel; fixed UMAP parameters recorded before rendering — SCOPE P2-18).
The untuned base encoder (C0) is one diffuse, intermingled blob; clean training (C1,
C5) shatters it into functionality-discrete islands; C2/C4 are looser; **C3 renders as
a single continuous diagonal band — the geometric face of the degenerate optimum**
its saturated thresholds record; C6 sits between band and islands. *(A figure, not
evidence — it supports no claim; it illustrates S8/F4.)*

![Seven-panel UMAP per negative-selection strategy](Image_Graphs/umap_panels.png)

## 5. Evidence index — claim → file → command

| Claim/number | Where it lives | Reproduce with |
|---|---|---|
| Dataset counts, overlap, lengths, timing | `report/measurements.md` §Phase 0/0.5 | `python -m embeded.data.prepare_data` |
| The trickiness-gate arc (FAIL kept) | G1 history in the report | `python -m embeded.hardcheck --recorded report/gate_g1/run_2026-09-26_phase01-v5.json` |
| Training-example picking + noise floors | `artifacts/mining_summary.json` + report sections | `python -m embeded.negatives`; `python -m scripts.hardness_diagnostics` |
| Benchmark table | `## Phase 2 — main results` | `python -m embeded.evaluate --results-table` |
| Audits | `artifacts/audit*/` + `AUDIT_JUDGEMENTS.md` | `python -m scripts.audit_sample make/score` |
| The bouncer verdict | `sanitycheck.md` §9.4 (log + FINAL) | the audit sheet + board pushes |
| s′ counts, decisions, trickiness gate | census/decisions artifacts + report sections | `python -m embeded.generalize --census/--mine/--hardness` |
| s′ table + side-reading | `## Phase 3 — generalisation` | `python -m embeded.generalize --results-table` |
| The dot-map figure | `report/figures/umap_{panels.png,params.json}` | `python -m embeded.visualize --panels C0..C6` |
| The demo's self-check | demo section of the report | `python -m demo.app --check` |
| Each run's identity | `runs/*/*/run_config.json` (its settings ID) | any run folder on `Kusshal/Embed` |
| Shipped state | commit `ede587a` (user's consolidation) | `git log` / the HF snapshot |
| Paper draft | `paper/paper.tex` | `pdflatex paper/paper.tex` |

**Commit chain (this branch):** Phase-2 era via PRs #6/#11 → `8214c85`…`e58d1d9` (Oct 1)
→ `847a698`…`e9d83fb` (Oct 2, §9.4) → `e8c04ac`…`f18c3ea` (Oct 2, v8/§9.5) →
`38b984e`/`5529433`/`fb07f51` → `acc634e`/`585ea55`/`e279bfd`/`06171e9` (Oct 3) →
`23f47f7`/`2f3fcea`/`213622c`/`06ae1aa`/`b919488`/`39888f7` (Oct 3–4) → `5ba985a`/
`f68aaf1`/`2ede09f`/`9cca4b2`/`91d8642`/`6f1d206`/`70cf6f5`/`cf5ee21`/`a913118`
(Oct 4–5) → **`ede587a` (user consolidation)** → `960bc5a`/`a6a0964`/`d43002f`/
`43f62f2`/`bc9e65b`/`ade0220`.

---

## 6. Incidents → the lessons they created

| What went wrong | The rule we now follow |
|---|---|
| A live access token pasted into a notebook cell | revoked; secrets only in Colab's safe storage |
| A cloud machine wiped itself mid-project | save to storage after every heavy step; every step checks its inputs |
| A re-run nearly overwrote real results | test runs get their own folder |
| Old data files under new code | version gates fail loudly, with the fix in the message |
| A code edit landed in a section shared by all strategies | unique anchors + regression tests for every edit |
| Settings read once at start-up broke test fixtures on other machines | settings are read at the moment of use (`06171e9`) |
| The storage service rate-limited us (retries made it worse) | identity check first; honour "retry after"; no training without saving configured |
| Three times, a required input had no download path | every input now has a named delivery path; notebook commands are lint-tested |
| The results table would have silently omitted three strategies | fixed + a guard that excludes odd runs *loudly* |
| Two promised modules turned out never to have been written | built; the notebook now proves each cell against real code |
| Notebooks don't refresh; sessions cache old code | checkout → reopen → restart; stale-session guards |

---

## 7. Open items (as of 2026-10-17)

1. **The two remaining audits** (C5, C6) — predictions 0–2% / 25–35%; once scored, every
   C5/C6 sentence becomes claimable.
2. **Nine score-sheet repairs** — re-grade nine benchmark runs whose saved sheets disagree
   with their raw prediction files (by at most +0.027; mechanical).
3. **The write-up** — limitations first, the one combined table, a single reproducibility
   script (PHASE4 §3/§6).
4. **The one candidate extension** — a *ladder* of filter strengths on the worst supply
   (turns finding F2 into a full dose-response curve), only after a scope amendment and a
   finished draft.
