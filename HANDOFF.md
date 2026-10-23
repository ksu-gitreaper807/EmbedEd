# HANDOFF.md — complete context transfer for a successor AI model

*Written 2026-10-23, branch `arena/01a0e714-embeded`, tip `13264f1`, offline test suite
152/152 green. This document is self-contained: a model that has never seen this session
should be able to continue the project correctly from this file plus the repo it lives in.
Where this file and another repo document disagree, the **append-only record wins**
(`sanitycheck.md` §9.5/§9.6) — this file tells you where everything lives, not just what
it says. Everything below is sourced from the repo's own records; nothing is invented.*

---

## 1. What this project is

**EmbedEd** asks one causal question about contrastive fine-tuning of code encoders:

> When you build training triplets for a code-clone/code-search encoder by mining *hard
> negatives* (look-alike wrong answers), what does the well-documented false-negative
> contamination of those mined negatives actually do to the trained model — and does
> filtering the "dangerous" kind of negative fix it?

Concretely: GraphCodeBERT is fine-tuned on identical supervised data where **only the
negative-selection strategy varies** (random vs BM25/lexical vs semantic vs each of those
plus a lexical denoising filter — "the bouncer"). The measured outcomes:

1. **The collapse.** Both search-based strategies train models that score *at or below the
   all-negative base rate* (F1 ≈ 0.25 vs 0.2406 ceiling) with validation thresholds pinned
   at 1.0 — the model stops discriminating entirely ("fp32-saturation erasure"). Random
   negatives train the same model to 0.7464. That hard negatives *backfire* this totally,
   on the standard benchmark, is the headline.
2. **The mechanism.** It is not primarily *how many* false negatives you train on but
   *which kind*: negatives that are simultaneously true clones **and** near
   word-for-word copies (Jaccard ≥ 0.40) drive the collapse; the count is the dose, and
   the loss's required 0.10 similarity margin is the amplifier.
3. **The partial fix.** A one-line lexical filter (drop candidates with token-Jaccard
   ≥ 0.40 to the positive, plus a head-skip) half-rescues the worst ranker (C4 0.4349,
   thresholds unpinned) without touching a clean supply (C5 ≈ C1).
4. **The generalisation verdict.** On unseen functionality (corpus s′), *every* poisoned
   condition performs at or below simply not fine-tuning (untuned transfer anchor 0.6934
   unseen). Only clean training beats doing nothing.
5. **The reconciliation.** The famous ~0.95 F1 published for this benchmark and this
   project's ~0.75 are different measuring sticks (near-clone-heavy regime vs corpus-wide
   natural prevalence), not a contradiction.

The benchmark is BigCloneBench via CodeXGLUE; the unseen-functionality corpus is
Kitsios et al.'s **BCB s′** (`bcb_v2_sampled_bf`, Zenodo DOI 10.5281/zenodo.17238379;
Kitsios, Sovrano, Barr, Bacchelli, ASE 2025). The intellectual frame — that FN
contamination, not hardness per se, is what poisons contrastive code search — is aligned
with RocketQA, Soft-InfoNCE, CoCoHaNeRe, and Krinke & Ragkhitwetsagul's demonstration
that 93% of a BigCloneBench semantic sample is mislabelled (all cited in
`paper/paper.tex`, which has the verified reference list).

---

## 2. The experiment design

### 2.1 Conditions (the naming key — memorise this, it was once contested)

| condition | negative supply | bouncer? | benchmark F1 (n=6) | status |
|---|---|---|---|---|
| **C0** | — (untuned GraphCodeBERT) | — | 0.2572 | anchor, never re-trained |
| **C1** | random | no | **0.7464 ± 0.0176** | clean control |
| **C2** | BM25 (lexical search) | no | 0.2478 ± 0.0231 | collapsed |
| **C3** | semantic (embedding kNN) | no | 0.2707 ± 0.0095 | collapsed |
| **C4** | **BM25 + bouncer** | yes | 0.4349 ± 0.0246 | partial rescue |
| **C5** | **random + bouncer** | yes | 0.7399 ± 0.0140 | clean control, filtered |
| **C6** | **semantic + bouncer** | yes | 0.4722 ± 0.0282 | between clusters |

**The numbers are build order, not semantics.** A C4↔C5 swap was requested on 2026-10-02
and resolved NO-SWAP (`fb07f51`, documented in three places). C4 = C2+bouncer, C5 =
C1+bouncer, C6 = C3+bouncer. Never re-litigate; point at `fb07f51`.

The design is a **ranker × bouncer factorial** on one frozen supervised backbone: same
anchors, positives, k, margin, cap, and seeds across all six conditions; *only* the
negative-selection rule differs (verified by tests).

### 2.2 The bouncer (the C4/C5/C6 filter)

Per anchor, after the ranker proposes candidates: exclude anything overlapping the
anchor's clone set or the corpus too heavily → **drop candidates with token-Jaccard
≥ 0.40 to the positive** → skip the first 10 survivors (head-skip) → pad (counted,
honouring the cut) to fill the triplet quota. The 0.40 cut was chosen in the S3-era audit
analysis: audited *clean* pairs top out at J = 0.345, true-clone pairs median 0.418 —
the cut sits inside a real gap. It removes ~40% of C2's contamination and **zero** of the
audited clean pairs.

### 2.3 Training recipe (frozen since Phase 0)

GraphCodeBERT, `MAX_LEN = 512` (p50 length 474, p90 1,535), batch 8, up to 32,000
triplets, 1 epoch, fp16, margin **0.10** (the hinge demands the positive beat the negative
by 0.10 similarity). Seeds **13/14/15** everywhere in the registered Phase-3/4 ladder
(Phase-2 extension seeds 16–18 exist and stay published as extra evaluations, never table
rows). Threshold policy: `max_f1_on_valid` — chosen on validation only, frozen, *always
reported* because a saturated threshold (pinned at 1.0) is the collapse's smoking gun.

### 2.4 Flavours

"Flavor A" rule: any re-run/new-seed campaign trains on the **same triples** (frozen
mining output), never re-mined per seed — the manipulation must stay fixed while the
seed varies. Phase 3 (s′) re-mines *within s′* because the corpus itself changes.

---

## 3. Data and scoring — the two rulers

### 3.1 The benchmark (Phase 0/1/2 surface)

BigCloneBench via CodeXGLUE, verified at G0: 9,126 fragments / 8,063 unique texts /
901,028 train pairs / 415,416 validation / 415,416 test. Test positives: 56,820/415,416
= **13.678%**, so an all-negative scorer scores **0.2406** (the ceiling) — the floor all
collapse numbers are read against. 8.8% of test fragments also appear in training
(measured, reported, not hidden). Frozen embedding matrix `corpus_emb.npy` = 8,063 × 512.
Per-run predictions live in `predictions.npz` with keys
`threshold / valid_scores / valid_labels / test_scores / test_labels`; stored metrics in
`eval_metrics.json`; a run's `run_config.json` fingerprint *is* its identity.

### 3.2 s′ — the unseen-functionality corpus (Phase 3 surface)

4,600 pairs over 23 functionalities. Holdout = the three **best-represented**
functionalities **{10, 13, 14}**, chosen by a written counting rule *before* any score
existed (decision D4 in `PROJECT_REPORT.md`'s register). 18 runs = 6 conditions × seeds
13–15, negatives mined within s′ (decision D3 — the only choice that keeps selection
strategy as the sole variable). Hardness gate G1s PASS at standardised gap **d = 0.886**
(benchmark G1: **d = 0.614**, after decision D1 rewrote the gate to be scale-free — see
§9). Scoring is **era-tolerant** (`train.load_trained_checkpoint`, `f68aaf1`): v6/v7-era
checkpoints are *scored* as-is with `transfer_provenance.json` persisted, while
*resuming* training still demands an exact fingerprint match.

---

## 4. The production pipeline — how every number was actually made

The system is **user-executed Colab VMs + repo-side deliverables landed at designed
stops**. The user runs notebook cells in Colab (GPU/CPU fleet); finished artifacts are
pushed to the Hugging Face dataset repo **`Kusshal/Embed` (dataset, `main` revision)**;
the repo (`github.com/ksu-gitreaper807/EmbedEd`, branch `arena/01a0e714-embeded`) carries
code, plans, the append-only findings record, the report, the paper, and the notebooks.
Every designed stop in a notebook (a gate, a completed campaign) is the cue to land the
corresponding repo deliverable.

`scripts/run_all.sh` is the complete single-machine reproducer (Phases 0→3; committed at
`6457b58`, `bash -n` clean):

- **Phase 0** — verify dataset counts, s′ acquisition, throughput recipe.
- **Phase 1** — index/embed corpus; mine negatives for all six strategies
  (`embeded/negatives.py`; v8 recipe; `.v7bak` backups asserted **bit-identical** on
  re-mine — two independent determinism proofs); pytest; hardness diagnostics
  (`scripts/hardness_diagnostics.py`); hardness gate G1 (`embeded/hardcheck.py`).
- **Phase 2** — smoke (own directory); C1–C6 × seeds 13–15 train+eval; C0 eval;
  `embeded.evaluate --results-table`.
- **Phase 3** — s′ census → mine (all six conditions, bouncer included) → hardness →
  smoke → 18-run train/eval loops → C0 transfer → results-table →
  `embeded.visualize --panels C0..C6` → `demo.app --check`.

Colab notebooks in `notebooks/` are the *fleet* versions of the same pipeline, sectioned
with gates and designed stops: `Phase2_Colab.ipynb` (blind-upload audit flow),
`Phase2V2_Colab.ipynb` (seeds 16–18), `Phase2C4_Colab.ipynb` / `Phase2C5C6_Colab.ipynb`
(the bouncer campaigns), `Phase3_Colab.ipynb` (s′), `Phase4_Colab.ipynb` (closure — see
§13). Supporting scripts: `scripts/audit_sample.py` (blind audits),
`scripts/audit_separability.py`, `scripts/hf_artifacts.py` (pull/push with sections),
`scripts/hardness_diagnostics.py`, `scripts/fetch_sprime.py`, `scripts/plot_*.py`,
`scripts/colab_git.py`, `scripts/colab_report.py`, `scripts/backup_hf_snapshot.py`,
`scripts/phase0_throughput.py`.

**Storage discipline:** checkpoint-push to HF after *every* heavy cell (born from a VM
recycle that ate un-checkpointed embeddings); pushes are **additive** (exclude
`sprime/*`, `*.zip`); pulls are **sectioned** — `mining` (fragments, triples_C*.jsonl,
corpus_emb, mining_summary), `runs-meta`, `runs`, `sprime`, `audit`
(`artifacts/audit*/**`), `report` — so a fresh VM downloads ~300 MB instead of ~4.5 GB.
Run pulls for Phase 3/4 fetch **only** `artifacts/runs/C[0-6]_1[345]/*` (seeds 13–15).

---

## 5. Every result, verbatim (with provenance)

Canonical store: `report/measurements.md` (per-run table generated by
`embeded.evaluate --results-table`). Secondary: `PROJECT_REPORT.md` §4, `sanitycheck.md`
§9.x, `paper/paper.tex`. *Rule: never write a number anywhere that isn't in one of these.*

### 5.1 Benchmark (n = 6 seeds; seeds 13–15 are the registered ladder, 16–18 extras)

| condition | F1 mean ± sd | MAP@R (n=6) | thresholds | pinned |
|---|---|---|---|---|
| C0 | 0.2572 | 0.1024 | 0.9779 | 0/1 |
| C1 | **0.7464 ± 0.0176** | 0.667 ± 0.033 | 0.436–0.462 | 0/6 |
| C2 | 0.2478 ± 0.0231 | 0.047 ± 0.024 | pinned ×5, 0.84–0.98 ×1 | 10/12* |
| C3 | 0.2707 ± 0.0095 | 0.077 ± 0.005 | pinned ×6 | 10/12* |
| C4 | 0.4349 ± 0.0246 | 0.2760 | 0.4965–0.5707 | 0/6 |
| C5 | 0.7399 ± 0.0140 | 0.6629 | ≈ 0.39–0.44 | 0/6 |
| C6 | 0.4722 ± 0.0282 | 0.1826 | 0.63–0.82 | 0/6 |

\* pinned counts 10/12 are over seeds 13–18 for C2/C3; collapse signature: 27–67% of the
entire test split at fp32 cosine == 1.0 in pinned runs. C2/C3 rank clones **worse than
C0** (MAP@R below the untuned baseline) — falsifier "it's a thresholding quirk" REFUTED
(`sanitycheck.md` §9.1, push `0ce5bf4`).

> **Known transcription defect (fix at the one-table rebuild):** `PROJECT_REPORT.md` §4
> and `paper/paper.tex` tab:benchmark carry **C3 MAP@R = 0.1826 — that is C6's value**;
> the correct C3 value is ≈ 0.077 (n=6 mean; 0.0776/0.0834/0.0699 on seeds 13/14/15 per
> `report/measurements.md`). Also note the stored C2_13 row (F1 0.2438) is one of the
> nine §9.2-repair suspects (npz-recomputed 0.2165) — the report table currently mixes
> stored values for those nine runs; §4's repairs fix exactly this.

### 5.2 s′ — unseen functionality (means over seeds 13–15)

| condition | F1_seen | F1_unseen | Δ (seen-advantage) |
|---|---|---|---|
| C1 | 0.7570 | **0.7031** | **+0.054** (± .022) |
| C5 | 0.7606 | 0.7244 | +0.036 |
| C6 | 0.7169 | 0.6865 | +0.030 |
| C2 | 0.6781 | 0.6674 | +0.011 |
| C4 | 0.6731 | 0.6687 | +0.004 |
| C3 | 0.7004 | 0.7057 (thr pinned 0.9998–0.9999) | −0.005 |
| **C0 (transfer anchor)** | **0.7118** | **0.6934** (thr 0.9093) | — |

**The registered finding:** no poisoned-mining condition beats the untuned encoder on
unseen functionality; only clean training (C1, and C5 pending its audit branch) converts
fine-tuning into gains. Ordering and the saturation signature replicate across both
rulers.

### 5.3 Audit rates (see §7 for what is valid)

| audit | rate | Wilson 95% | status |
|---|---|---|---|
| C1 | ≈ 0–2% | — | valid (S3-era, 2026-09-27) |
| C2 | 21/50 = **42%** | [0.294, 0.558] | **valid** (genuine sheet) |
| C3 | 15/50 = **30%** (1 unsure; upper incl. unsure 32%) | [0.191, 0.438] | **valid** (genuine sheet) |
| C4 | "30%" | — | **VOID** (§9.6) — re-audit registered |
| C5 | "30%" | — | **VOID** (§9.6) — gate open (prediction 0–2%) |
| C6 | "30%" | — | **VOID** (§9.6) — gate open (H1 prediction 25–35%) |

### 5.4 Supply measurements (v8 re-mine, free mining statistics)

C4 `jaccard_filtered` 66,958 = 10.5% of scanned (head_skipped 16,000, padded 111 =
0.35%); C5 **0.27%** (1,768/656k — predicted ≈0 ✓); C6 **13.6%** (87,024/640k —
predicted 10–25% ✓; head_skipped 15,957, padded 710 = 2.2%). C1–C4 triples bit-identical
v7→v8 (second determinism proof).

### 5.5 Gates

G1 (benchmark) FAIL under the legacy rule → PASS at **d = 0.614** under the recorded D1
rule; G1s (s′) PASS at **d = 0.886**. Registered ordering random < lexical ≤ semantic.

### 5.6 Figures (in `Image_Graphs/` and `report/figures/`)

37-model confusion board (`phase2_confusion_matrices(1).png`: C0 s13 + C1–C6 × 6 seeds,
per-panel F1 + threshold in titles) and the seven-panel UMAP (`umap_panels.png`, fixed
params recorded before rendering; C3 renders as one continuous diagonal band — the
geometric face of the degenerate optimum; C1/C5 functionality-discrete islands). The
UMAP is labelled in-figure as "a figure, not evidence".

---

## 6. How results are tested — the trust machinery

1. **Offline pytest suite** (152 green at `13264f1`; grew 29 → 150 → 152 during closure):
   every module has fixture-backed tests (offline fake embeddings, no network, no GPU);
   `test_cli_contract_matches_the_notebook` pins that notebook commands exist in the
   modules (the module bends to the notebook); the audit tests pin blindness, seeded
   reproducibility, refusal-to-clobber-labelled-work, key-vs-triples drift refusal,
   per-condition `EMBEDED_AUDIT_DIR` separation, and (new) refusal of foreign label ids +
   cross-sheet id uniqueness.
2. **Determinism proofs:** v7 and v8 re-mines asserted C1–C4 triples **bit-identical**
   to backups (`sha256` compare) — the mining pipeline is a pure function of its inputs.
3. **Gates before training:** G1/G1s hardness gates (ordering + standardised gap ≥ 0.5);
   §9.5's endpoint order is fixed: **scorer state → F1 → audited rate**.
4. **Blind audits:** 50 mined negatives per condition, distinct anchors, seeded shuffle,
   sheet written blind (no condition shown), key written alongside, human labels
   `clone | not_clone | unsure`, `score` verifies the key still matches the on-disk
   triples (stale = refuse) and reports Wilson 95% intervals (unsure counted as
   not_clone in the headline; an upper bound counts it as clone).
5. **Consistency instruments:** the §9.2 npz↔metrics mismatch detector
   (`audit_mismatches()` recomputes F1 from `predictions.npz` vs stored
   `eval_metrics.json`, tolerance 5e-4) — it found the nine repairs (§13); the
   consolidator recomputed all 19 Phase-2V2 board rows from counts (TP+FN = 56,820 etc.).
6. **Entry gates in notebooks:** §2 of every Phase-4-style notebook demands the shipped
   state (report markers present, pytest green) before closure work; §0 checks out the
   branch and warns only on true divergence (`git merge-base --is-ancestor MARKER HEAD`).
7. **Acceptance criteria that end phases:** the audit gates (§7); §4's "zero npz↔metrics
   mismatches board-wide" before any table rebuild; smoke runs in their own directory so
   a re-run can never overwrite a real run.

---

## 7. The audit program — including the §9.6 incident (read this twice)

### 7.1 Protocol (unchanged since 2026-09-27)

`python -m scripts.audit_sample make --conditions <C> --n 50 --seed 13` samples 50
(anchor, mined-negative) pairs, writes a blind `audit_pairs.md`, an empty
`audit_labels.csv`, and a secret `audit_key.csv` (per-condition directory via
`EMBEDED_AUDIT_DIR`; the shared `audit/` dir holds the scored Phase-2 C2/C3 sheet and
must never be touched again). A human labels each pair (BigCloneBench rubric: same
functionality whatever the syntax; `unsure` if it takes > 1 minute). `score` refuses a
stale key (re-mined triples) and joins labels↔key by id. Sheets are committed before
unblinding; `AUDIT_JUDGEMENTS.md` records the calls; **no headline sentence before its
audit's outcome row is appended to §9.5**.

### 7.2 Pre-registration (§9.5, written 2026-10-02, before any C5/C6 run)

- **C5 (control):** audit 0–2% → in band ⇒ C5 is a *confirmed control*, claimable beside
  C1. Out of band ⇒ C5 loses control status, addendum needed.
- **C6 (discriminator):** H1 (similarity-trigger) predicts audit 25–35% — the semantic FN
  mass passes the lexical cut untouched; H2 (dose-only) predicts C6 ≈ C3 with low
  contamination. Primary endpoint is scorer state (C6 de-saturated, F1 ≥ C4's band ⇒ H1
  survives; §9.4 extends to the semantic supply). H2 would force a §9.4 + SCOPE §6
  rewrite. **The scorer endpoint already read H1** (C6 unpinned 0.63–0.82, F1 0.4722) —
  the audit is the confirming gate and must not be adjusted post hoc.
- **C4's original gate** (§9.4): pass ≤ 42% (the C2 floor), hoped ≈ 0%.

### 7.3 What actually happened — the §9.6 void (2026-10-21)

Opening the Phase-4 gate surfaced "scored" C4/C5/C6 audits **all at exactly 30%**.
Forensics on the pulled artifacts: `audit_c4/`, `audit_c5/`, `audit_c6/` each held a
**100-row `audit_labels.csv`, byte-identical across the three dirs and identical to the
genuine `audit/` (C2/C3) labels file** (9,925 B; 36 clone / 63 not_clone / 1 unsure; same
per-pair notes; same first-20 label sequence) — against 50-pair keys. Scoring had joined
the first 50 ids (P001–P050 collided) onto each condition's key. Each bogus
`audit_result.json` recorded `n=50` but `n_labelled=100` — written at score time, proving
the wrong file was already in place when each gate "passed" (C4 on 2026-10-02, C5/C6 on
2026-10-03). Root cause: the Colab upload flow hands every sheet the same filename, and
old sheets reused ids P001–P050, so a stale Downloads copy silently "fitted" every later
sheet. Consequence: **§9.4's "C3 = C4 = 15/50" — the equal-dose coincidence at the heart
of the dose/trigger story — was one measurement joined onto two keys.** The C3-vs-C4 arm
loses its equal-dose premise until re-audited; the dose gradient (C2 42% vs C3 30%), the
J ≥ 0.40 trigger analysis, and all F1/MAP results are untouched (training never touched
audit files).

**Permanent fixes (`596f98b`):** `make` tags sheet ids with a one-way hash of the sheet's
own content (`<tag>-001…`, still blind); `score` refuses any label id not in the key;
Phase-4 §3a validates id sets and **quarantines** wrong-sheet files (renamed
`.foreign-<timestamp>`, evidence kept) *before* `make`, and writes a fresh label form
when one is missing (a fresh pull re-fetches the bogus Hub files).

**Registered repair (in §9.6, before unblinding):** re-label all three sheets from their
`audit_pairs.md` (150 pairs — this is happening *right now*, see §13), re-score with the
fixed tooling; C4 re-enters the gate beside C5/C6; new numbers replace void ones wherever
they spread (notebook Panels A/C, PHASE4_PLAN §3/§4, PROJECT_REPORT F2, paper G2 +
tab:benchmark + mechanism section); branches appended to §9.5, never edited.

---

## 8. The mechanism finding — state after §9.6

**What stands:** the trigger account — collapse is governed by the *surface* of the false
negative (a functional clone that is also a J ≥ 0.40 near-duplicate), with FN rate as
dose and the 0.10 margin as amplifier. Evidence: dose gradient C2 42% vs C3 30% with
monotone Δ-vs-C0 (+0.489 / +0.013 / −0.009 for ~0%/30%/42% floors at n=6); the J-cut
geometry (clean max 0.345 < cut 0.40 < clone median 0.418; removes ~40% of C2
contamination, 0 audited clean pairs); C6's de-saturation with 13.6% residual supply;
C5 ≈ C1 (bouncer ≈ no-op on clean supply).

**What is suspended:** every C3-vs-C4 equal-rate sentence (the "same 30%, opposite
outcome" coincidence). If the re-audit returns C4 ≈ 0% (hoped), the story *simplifies*
(the bouncer removed almost all of the danger class — rescue size tracks how much danger
class was removed); if C4 lands mid-band, the cross-ranker comparison returns but with
fresh numbers. Either way the §9.5 outcome rows are appended before any headline uses
C4/C5/C6.

---

## 9. Decision register — every consequential choice, and why

| ID | Decision | Why (as recorded at the time) |
|---|---|---|
| Recipe | 512 tokens; batch 8 × 32,000 × 1 epoch; fp16 | Measured lengths + T4 throughput sweep; fp32 OOM |
| Checkpointing | Push to HF after every heavy cell | A VM recycle ate ~35 min of un-checkpointed work, once, permanently |
| **D1** | Hardness gate = registered ordering + standardised gap ≥ 0.5; the old FAIL kept on record | The legacy fixed-0.02 margin ignored the similarity space's real scale; the project never deletes an inconvenient measurement |
| **D2** | Keep the mining as specified; *measure and report* FN rates; reserve filtering as its own experiment | The obvious fix removes only 5–10% of noise (true clones differ ~70% of characters); a J-cut validated on 50 pairs is not yet evidence; cutting by model similarity would change the thing being studied. This is exactly why C4–C6 exist |
| Thresholds | Chosen on validation only, frozen, always reported | Prevents test leakage; saturation became the collapse diagnostic |
| Audits first | Blind 50-pair audits before any headline; sheets committed before unblinding | In this corpus "similar" too often means "mislabelled clone"; standing rule |
| C4 gate | Audit ≤ 42% before any C4 comparison; first result left uninterpreted until it landed | Pre-registration §9.4; the gate was "passed" by the bogus file (§7.3) — hence the re-audit |
| Naming key | C4 = C2+bouncer, C5 = C1+bouncer, C6 = C3+bouncer; NO SWAP (`fb07f51`) | Numbers are build order; documented in three places |
| **D3** | s′ training uses negatives mined within s′ | The only choice that keeps selection strategy as the sole variable |
| **D4** | s′ holdout = three best-represented functionalities {10,13,14} by a written counting rule | "Most data" measured, not guessed; fixed before any score existed |
| Scope (Phase 3) | Six seeds → symmetric entry → **6 conditions × 3 seeds (13–15), 18 runs** | User decisions, all recorded before any affected run; the 6-seed Phase-2 inventory stays published; the two pre-re-scope runs retrained under the final fingerprint ("a run's identity is its settings") |
| Downloads | Sectioned pulls; runs pulls fetch only seeds 13–15 | Minimise VM download cost (~300 MB vs ~4.5 GB); extras never pulled |
| 429 policy | Hub 429 = rolling window: wait 10–15 min without re-running; identity probe once | Retries made it worse (measured) |
| Fleet | Trainers push run dirs only, never the report; `--include-report` and GitHub pushes on one consolidating VM; readers pull `--if-configured --light` and NEVER push (one sanctioned exception: the audit_c4 reader) | Push races are how §9.2's nine mismatches most plausibly happened |
| §9.4 verdict (as amended) | Trigger = FN surface (clone ∧ near-dup); rate = dose; margin = amplifier | §9.4 FINAL + §9.6 amendment (the equal-dose C3=C4 leg suspended) |
| **D4-1..D4-4** (Phase 4) | One table with three panels (benchmark / s′ / mechanism), non-comparability note; headline = C1 (C5 only if its audit lands); the 0.95 gap framed as pair-regime difference; the δ-ladder extension only after a SCOPE amendment and a finished draft | SCOPE Part 7's stop condition is met — closure, not new experiments; recorded in `phase4_decisions.json` (replay-safe cell) |
| Deliverables | Plain-language report + LaTeX paper with verified references + `TIMELINE.md` | The user asked for exactly these; every number in them traces to `report/measurements.md` |
| Append-only records | `sanitycheck.md` amendments (§9.5 outcome rows, §9.6) are appended, never edited; template pre-staged | The project's own no-silent-changes rule |

---

## 10. Challenges and incidents — each one changed the protocol

| Incident | Consequence / permanent rule |
|---|---|
| VM recycle lost embeddings + triples (Phase 0) | checkpoint-push after every heavy cell |
| A live access token appeared in a notebook | revoked; secrets only in Colab Secrets; training refuses to start if saving isn't configured |
| A re-run nearly clobbered finished runs | smoke/test runs get their own directory, permanently |
| A filter edit landed inside code shared by all strategies (NameError at scale) | unique anchors in code edits + regression tests ("committed-chimera lesson", `91d2f7c`) |
| Condition sets frozen at import time poisoned fixtures on any machine with C4–C6 triples | call-time `conditions()` (`06171e9`) — found by the notebook's own preflight pytest |
| HF 429 rate limits | identity probe once; honour Retry-After; patient waits; no re-runs for 10–15 min |
| §9.2: 9 of 18 stored score sheets disagree with their npz (≤ +0.0273; C1 exact) | leading hypothesis: stale-push overwrite on the shared artifacts tree; no conclusion changes (both readings in the degenerate band); Phase-4 §4 repairs via `evaluate --force`; acceptance = zero mismatches board-wide |
| The results-table regex would have silently omitted C4–C6 | fixed before §10 ran on real rows (`5ba985a`) + off-ladder seed guard |
| Two promised modules (`visualize.py`, `demo/app.py`) never existed | built to spec (P2-18, D6/P2-21), each with a `--check` that asserts the deliverable |
| Era mixing: v6/v7 checkpoints failed the s′ transfer fingerprint gate | era-tolerant *scoring* with persisted provenance; strict resume untouched (`f68aaf1`) |
| Notebook lint class: `--include` arity bugs | notebook flag linter (`70cf6f5`) |
| **§9.6: one labels file scored on three keys; C4's gate "passed" on it** | content-tagged sheet ids; score refuses foreign ids; 3a quarantines wrong-sheet files before `make`; label gates count non-empty labels, never lines; per-condition `EMBEDED_AUDIT_DIR` set before every audit call |
| Colab Run-all does not halt at failed cells | the audit gate is enforced *at the deliverable* (5b refuses to build the one table without `audit_result.json` ×3) and stops print a loud banner |
| Notebooks never auto-refresh; sessions cache old code | §0 checkout cell + restart discipline; after repo-side notebook edits the user pastes the cell text |
| An "expected HEAD prefix" marker warned forever (a marker can never equal its introducing commit) | replaced with `git merge-base --is-ancestor MARKER HEAD` — silent when the repo merely moved ahead (`c968780`) |

---

## 11. Standing operational constraints (do not violate)

1. All session work lands on **`arena/01a0e714-embeded`** and is pushed there — no other
   branch, ever.
2. Working pattern: the user executes notebook cells and pastes outputs; each designed
   stop = land the corresponding repo deliverable. The user's open Colab copy never
   auto-refreshes — after repo-side notebook edits, provide paste-ready cell text.
3. VMs push to `Kusshal/Embed@main`; **trainers push run dirs only, never the report**;
   `--include-report` + GitHub pushes happen on the one consolidating VM; identical
   secrets; identical-or-absent `EMBEDED_HF_SUBDIR`. Readers: pull
   `--if-configured --light` only, **never push**.
4. Quota: interrupt-after-push is safe; **never manually restart a finished run**;
   budget 2026-10-02 was 50 units @ 1/hr with ≈ 45 left.
5. Flavor A: new seeds train on the SAME triples; never overwrite/append into the
   original `runs/*` tree.
6. No `files.upload`/`files.download` in Phase2V2+; Phase2_Colab keeps its blind upload
   flow.
7. The 50-pair blind audit gate is **mandatory** before any C4/C5/C6 comparison — now
   covering the C4 re-audit too (§9.6).
8. Minimise VM download cost: sectioned pulls; full pulls opt-in only where checkpoints
   are actually loaded; runs pulls fetch only `artifacts/runs/C[0-6]_1[345]/*`.
9. The paper must stay **LaTeX**, honest `\pending{}` markers where gates are open, and
   only recorded numbers (or explicitly unpinned pointer language).
10. Append-only for: `sanitycheck.md` §9.5 outcome rows and amendments; audit sheets
    once labelled.

---

## 12. Repository map and canonical number stores

```
arena/01a0e714-embeded (github.com/ksu-gitreaper807/EmbedEd)
├── embeded/            the package: settings, encoder, train, evaluate, negatives,
│                       generalize (s′), hardcheck, hardness, visualize, data/, tests/
├── scripts/            audit_sample, hf_artifacts (sections), hardness_diagnostics,
│                       run_all.sh, plot_*, colab_*, fetch_sprime, backup_hf_snapshot
├── notebooks/          Phase2(_V2/_C4/_C5C6)_Colab, Phase3_Colab, Phase4_Colab (closure)
├── report/measurements.md   CANONICAL numeric record (per-run tables + audit sections)
├── sanitycheck.md      the findings cross-exam; §9 campaigns; §9.5 pre-registration +
│                       outcome TEMPLATE; §9.6 the void amendment (append-only!)
├── PROJECT_REPORT.md   plain-language narrative + master tables + figures (§4a)
├── PHASE2/3/4_PLAN.md  per-phase registered plans (PHASE4: gates, repairs, one table,
│                       decision cells, write-up checklist, execution log §10)
├── TIMELINE.md         dated reconstruction, Phase 0 → closure
├── paper/paper.tex     LaTeX draft; \pending{audit\_c4} ×4 + \pending C5/C6 ×2 await rates
├── SCOPE.md / IMPLEMENTATION_PLAN.md / FINAL_SPEC.md / GROUND_TRUTH.md / appendix.md
├── PROGRESS.md / CLOUD.md / COLAB.md / README.md
├── Image_Graphs/       user-uploaded board + UMAP (embedded in PROJECT_REPORT §4a)
└── artifacts/, report/figures/, demo/   (artifacts live mainly on the HF hub)
```

Canonical number stores, in order of authority: `report/measurements.md` →
`sanitycheck.md` §9.x → per-run `eval_metrics.json` on the hub → `PROJECT_REPORT.md` §4 →
`paper/paper.tex`. **HF hub `Kusshal/Embed@main`** holds every run dir, mining artifacts,
audit dirs, and the canonical report copy (user-pushed `ede587a`).

Recent commit chain (newest last): `43f62f2` PHASE4_PLAN → `bc9e65b` paper → `ade0220`
TIMELINE → `ceb0326` report v1 → `a9a9742` user's Image_Graphs → `0dac430` plain-language
rewrite → `6457b58` Phase-4 notebook + run_all + report §4a → `e55f443` notebook §0 →
`5893b0f` marker bump + §9.5 template → `c968780` §0 ancestry check → `c5b57c4` §3
per-dir + label counting → `e120792` 5b hard gate → `38271af` sync-cell tail →
`596f98b` id-collision fixes → `3f95a8c` §9.6 void (sanitycheck/plan/report) →
`133192f` §9.6 completion (paper + notebook) → **`13264f1` 3a quarantine-before-make +
form fallback**.

---

## 13. CURRENT STATE AND NEXT STEPS — start here

**Right now (2026-10-23):** the user is labelling the 150 re-audit pairs (C4/C5/C6 × 50)
in Colab. The §3a quarantine ran correctly (the three bogus files — P051–P100 of the
genuine C2/C3 sheet — were renamed `.foreign-20261021-070013`), the sheets regenerated
deterministically (same seed 13 + same verified triples ⇒ identical keys/pairs), and
three blank forms are waiting. Expect: user re-runs 3a (should print 50/50 ×3) → runs 3b
→ pastes **three rates with Wilson bands and unsure counts**.

**When the rates arrive, in this order:**

1. **Append §9.5 outcome rows** (template staged at the end of §9.5 — fill, never edit
   above): C4 branch = gate ≤ 42% (vs C2 floor; hoped ≈ 0%); C5 branch = 0–2% band ⇒
   confirmed control, else loses control status; C6 branch = 25–35% ⇒ H1 survives
   (consistent with the scorer endpoint), far below ⇒ H2 ⇒ §9.4 + SCOPE §6 rewrite.
2. **Fill the paper's pendings:** `\pending{audit\_c4}` ×4 (G2, tab:benchmark C4 row,
   bouncer finding, mechanism bullet) + the two original C5/C6 pendings; update the C4
   row of tab:benchmark; reconcile §6.3 (mechanism) with C4's true rate; fix the C3
   MAP@R transcription (0.1826 → ≈0.077) in paper + PROJECT_REPORT while there.
3. **Update PROJECT_REPORT** F2/F3 rows and the §4 table cells (C4/C5/C6 audited rates),
   and `PHASE4_PLAN` §3/§4 markers.
4. **§4 nine repairs** (notebook cells 4a/4b/4c): `audit_mismatches()` detects →
   `python -m embeded.evaluate --condition <c> --seed <s> --force` re-grades from
   checkpoints → acceptance = **zero** mismatches board-wide → `--results-table` rebuild
   (note the pre-repair deltas ≤ 0.0273 in the write-up; all nine are C2/C3).
5. **§5 the one table:** 5a records D4-1..D4-4 to `artifacts/phase4_decisions.json`
   (replay-safe); 5b (hard-gated on all three `audit_result.json`s) builds
   `## Phase 4 — the one table` into the report: Panel A benchmark mean±sd (seeds 13–15)
   + pinned counts + audited FN column (now quoted from the JSONs), Panel B s′ +
   C0 anchor 0.7118/0.6934, Panel C mechanism bookkeeping (as amended by §9.6),
   non-comparability note.
6. **§6 `run_all.sh` rewrite cell** (already landed at `6457b58`; verify `bash -n`), §7
   limitations-before-claims checklist, **§8 sync**: `hf_push('--include-report')` from
   the consolidating VM, GitHub `git push`, PHASE4_PLAN §10 execution-log row.
7. **Then the write-up is unblocked** (D4-3: limitations first; claims ledger). The only
   sanctioned extension is the **δ-ladder** (bouncer threshold δ ∈ {0, 0.05, 0.10, 0.20}
   × 3 seeds on C3's supply → FN-weighted loss → codebert-base on the winner) — and only
   after a SCOPE amendment **and** a finished draft (D4-4). Not sanctioned: leave-one-out,
   second language, symmetric pairs, reranker, ANCE.

**Known small defects to fix opportunistically:** the C3 MAP@R transcription (above);
unescaped `##` inside the paper appendix's `\texttt{}`; a garbled Moreira bibitem in
`paper/paper.tex`; the Hub's `report/measurements.md` currently carries the placeholder
one table (self-healing — §5b/§8 overwrite it).

---

## 14. Agent-specific gotchas (for an AI working in this repo)

- **Always `cd /home/user/EmbedEd`** inside bash commands (default cwd is `/home/user`,
  which is not a git repo — a whole `&&` chain dies there "silently" with exit 128). Use
  `set -o pipefail &&` everywhere; `git fetch origin arena/01a0e714-embeded && git reset
  --hard FETCH_HEAD` at turn start (the session is rollback-prone; heal = fetch + reset +
  `git checkout <sha> -- <files>`); `git log --oneline -1` before edits.
- **`.venv` is snapshot-excluded** and vanishes between sessions — guard:
  `ls .venv/bin/python || (python3 -m venv .venv && .venv/bin/pip install …)` before any
  pytest; the install set that works is in that guard's history (torch, transformers,
  numpy, pandas, scikit-learn, scipy, rank_bm25, pytest, …).
- **Patch notebooks/files via python heredocs, never by hand**: verify each anchor with
  `count()==1` against actual file bytes first; `ast.parse` every replacement cell
  (strip lines starting with `%`/`!` magics first); write at the end. **Never nest
  triple-quoted strings inside a heredoc's outer string** — build embedded multi-line
  messages with `chr(10).join([...])` (this bit three times: nested `'''`, an
  `f'''…'''` gate message, and a bad assert that let a commit message overclaim its
  contents — always `git show --stat` after).
- **Numbers discipline:** only recorded values or explicit "unpinned" pointers in the
  paper; verify asserts against the file, not memory; docs-only commits may skip pytest
  but never *claim* green without output.
- **huggingface.co is unreachable from the sandbox** (VM-only); api.github.com and
  gh work. The repo's offline suite (152) is the contract — notebooks bend to modules
  (`test_cli_contract_matches_the_notebook`).
- **Colab realities:** Run-all does not halt at failed cells (hence hard gates at the
  deliverable); the open notebook copy lags the repo (paste cells, don't assume
  refresh); label-gates must count *non-empty* labels; every audit call needs
  `EMBEDED_AUDIT_DIR` set per condition.
- **User-facing rhythm:** designed stops in notebooks are the cue to land repo
  deliverables; when the user pastes an error, reproduce it in the repo copy first
  (often the open copy is stale — check which).

---

## 15. Glossary

| term | meaning |
|---|---|
| triple | anchor + positive + mined negative; one training example |
| false negative (FN) | a mined "negative" that is actually a clone of the anchor — dataset label noise, the study's subject |
| bouncer | the Jaccard ≥ 0.40 lexical denoising filter (+ clone/corpus exclusion, head-skip, counted pad) |
| pinned threshold | validation-chosen cut-off saturated at 1.0 — the collapse signature |
| ceiling | 0.2406 = best possible F1 for an all-negative scorer at 13.678% prevalence |
| MAP@R | mean average precision at R; ranking metric an all-negative scorer cannot fake |
| Wilson band | 95% interval for an audit proportion |
| fingerprint | hash of a run's full settings; identity, resume gate, provenance |
| s′ | Kitsios et al.'s functionality-labelled BCB variant; the unseen-functionality ruler |
| era-tolerant | scoring old-era checkpoints as-is with provenance (resume stays strict) |
| writer/reader VM | consolidating VM (may push report + GitHub) vs monitoring VM (pull-only) |
| §9.5 / §9.6 | the pre-registered C5/C6 gate record / the 2026-10-21 void amendment |

*End of handoff. The next expected event in the project's life is the user pasting three
audit rates.*
