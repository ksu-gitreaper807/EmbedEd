# Phase 4 — Closure: the audits, the one table, and the write-up

**Status:** planned; the consolidation it builds on is **done** (`report/measurements.md` as
shipped with the Phase 3 campaign, repo commit `ede587a`). No Phase 4 experiment has run.
**Purpose:** SCOPE Part 7's stop condition is now satisfied — C0–C3 (and the bouncer extension
C4–C6) are evaluated on the fixed CodeXGLUE test split with F1 + MAP@R over seeds, *and* on the
held-out functionality split with `F1_seen` / `F1_unseen` / `Δ` — so the scope's own instruction
applies: **stop running experiments and start writing.** Phase 4 is therefore not a new
experiment phase. It is the registered closure: score the two pending audit gates, repair the
nine known data-integrity defects, assemble the numbers into the one table Part 7 asks for,
write the report (limitations before claims, per SCOPE item 20), and finish `run_all.sh`
(item 22). Optional extensions are named, priced, and then **gated behind a SCOPE amendment**
(§8, D4-4) — Part 6 rule 7 is explicit that finishing early means improving the report, not
starting a half-finished denoising experiment.

Standing rules carry over unchanged: thresholds from the seen slice only; Δ reported as mean ±
spread over seeds, no p-values; `HARDNESS_D_MIN` is never lowered; every headline comparison
involving C5/C6 stays gated on its 50-pair blind audit; decisions are recorded before the runs
they affect (PHASE3_PLAN §8 discipline, same in this file's §9 log).

---

## 1. Starting point — what Phase 3 leaves behind

All numbers live in `report/measurements.md` (consolidated `ede587a`); this is the claim
inventory Phase 4 writes up.

**Benchmark table (CodeXGLUE test, n = 415,416 pairs, positive rate 13.7%, ceiling 0.2406):**
C1 0.7464 ± 0.0176 (2.90× C0; MAP@R 0.6629); C2 0.2478 ± 0.0231; C3 0.2707 ± 0.0095
(MAP@R 0.1826; 10/12 thresholds pinned at 1.0); C4 0.4349 ± 0.0246 (MAP@R 0.2760; 6/6
unpinned, 0.4965–0.5707); C5 0.7399 ± 0.0140 (≈ C1 band — control confirmed); C6 0.4722 ±
0.0282 (unpinned, between the clusters). C0 0.2572 (threshold 0.9779). Three seeds each.

**Generalisation table (s′, holdout {10, 13, 14}, seeds 13–15):** mean `F1_seen / F1_unseen /
Δ` — C1 0.7570 / 0.7031 / +0.0539 (sd 0.0222); C2 0.6781 / 0.6674 / +0.0107; C3 0.7004 /
0.7057 / −0.0053 (thresholds pinned 0.9998–0.9999); C4 0.6731 / 0.6687 / +0.0044; C5 0.7606 /
0.7244 / +0.0362; C6 0.7169 / 0.6865 / +0.0304. The untuned base model scores 0.6934 unseen
(C0 transfer, threshold 0.9093): **no poisoned-mining condition beats not fine-tuning at all**
on unseen functionality; C1 and C5 are the only conditions that edge past it.

**Mechanism (§9.4 FINAL):** collapse is not determined by the FN *rate* (C3 and C4 share 30%)
but by the FN *surface* (trigger = an FN that is simultaneously a functional clone and a
J ≥ 0.40 near-duplicate); rate is the dose, the 0.10 margin is the amplifier, and the failure
mode is fp32-saturation erasure (thresholds pinned 1.0, base-rate F1). The v8 re-measurement
put the poison supply where predicted: C5 0.27% (audit prediction ≈0), C6 13.6% of mined
candidates (prediction 10–25%).

**Measured FN rates (audited):** C4 15/50 = 30% [0.191, 0.438]; C2 42%; C1 ≈ 0–2%. The
J ≥ 0.40 cut removes ~40% of C2's contamination in-sample while keeping every audited clean
pair (clean max 0.345 vs clone median 0.418).

**Open items inherited (the reason this phase exists):**
1. `audit_c5` / `audit_c6` — the 50-pair blind gates, unscored (§4 here).
2. Nine C2/C3 runs with stored-metrics vs `predictions.npz` mismatches, up to +0.0273 (§5).
3. The write-up itself, `run_all.sh` (still commented), and the literal one-table deliverable.

---

## 2. Entry criteria — all satisfied at the time of writing

1. **Phase 3 consolidation shipped**: 18/18 s′ runs trained and on the hub; the Δ table, the
   19 transfer readings, the §4 figure (`umap_panels.png` + `umap_params.json` under P2-18
   parameters), and the demo check (`demo.app --check`, all four D6 cells populated) are in
   the shipped report.
2. **Every notebook cell has a module behind it** (`embeded/generalize.py`,
   `embeded/visualize.py`, `demo/app.py` — the last two landed 2026-10-05 after the sequencer
   exposed them as missing), and the notebook green-runs from a cold VM.
3. **The Phase-2 main table exists** and its G2 review is recorded; s′ is fetched with the
   gate recorded; `SPRIME_SEEDS = (13, 14, 15)` matches Phase 2's registered inventory.
4. **Suite green** (150 tests) — Phase 4 adds no module without offline tests in the same
   commit (the Phase 3 lesson, applied).
5. **SCOPE Part 7's stop condition is met** — say so in the report's first paragraph. Phase 4
   runs **no training**; the only GPU work is the nine §5 re-evaluations and (if amended in)
   the §8 extension.

---

## 3. The one table (SCOPE Part 7, made literal)

Part 7: *"those numbers, together with the measured false-negative rates, are in one table."*
Today they live in two tables whose absolute F1 is **not** comparable (13.7%-positive
base-rate surface vs balanced slices; the same base model scores 0.2572 and 0.6934). The
deliverable is a single **report section** — "## Phase 4 — the one table" — with:

- **Panel A — benchmark:** condition × (F1 ± sd, best threshold, MAP@R) × seeds, C0 first,
  ceiling printed under the table.
- **Panel B — generalisation:** condition × (F1_seen, F1_unseen, Δ ± sd) × seeds, with the
  untuned C0 transfer row as the within-corpus anchor (0.6934 unseen) and the pinned-threshold
  column beside it (the collapse signature travels).
- **Panel C — measured FN rates:** audited rate + Wilson interval per condition (C1 ≈ 0–2%,
  C2 42%, C3 30%, C4 30%), the J-cut in-sample effect, and the v8 supply measurements for
  C5 (0.27%) / C6 (13.6%) marked *predicted, audit pending §4*.
- A written **non-comparability note** (D3=B; "the corpus changed under the metric") and the
  one-sentence cross-corpus finding: the ordering and the degenerate signatures replicate on
  both rulers.

**Decision D4-1 (record before building):** the exact panel layout, and whether C4–C6 appear
inline or as a clearly-flagged extension block (Part 7 names only C0–C3; the bouncer trio
joined by the recorded 2026-10-03 addendum — keep it visible, not smuggled in).

---

## 4. The audit gates — `audit_c5` and `audit_c6` (blocking)

Protocol: the Phase-2 50-pair blind audit, unchanged (`scripts/audit_sample.py`; reader VM is
fine — never push from a reader). Pre-registered predictions and endpoint order live in
`sanitycheck.md` §9.5: scorer state first, then F1, then the audited contamination rate.

| gate | prediction under H1 | outcome handling (pre-registered) |
|---|---|---|
| `audit_c5` | 0–2% (the bouncer on a clean supply) | rate inside band → C5 = confirmed control, headlineable beside C1 |
| `audit_c6` | 25–35% under H1 | H1 survives → §9.4's dose/similarity story extends to the semantic supply; H2 (rate ≪ band) → **§9.4 and SCOPE §6's rewrite**: the similarity-amplifier claim was BM25-specific |

Rules: the audit sheet is committed before unblinding (sheet first, judgements after);
`AUDIT_JUDGEMENTS.md` records every call; **no C5/C6 sentence enters the write-up's claims
ledger until its gate is scored** — this is the standing gate from Phase 2, and Phase 4
finally pays it.

---

## 5. Data integrity — the nine §9.2 repairs (before any table is frozen)

Nine C2/C3 runs have stored metrics ≠ recomputed-from-`predictions.npz` (up to +0.0273; the
board and the audits read the npz). Procedure, in one commit per batch:

1. Per-file HF SHAs of the nine `predictions.npz` (provenance of what is being replaced).
2. `python -m embeded.evaluate --condition <c> --seed <s> --force` — re-score from the
   checkpoint. The era-tolerant scoring load (Phase 3, `load_trained_checkpoint`) makes this
   work from the v8 checkout with the recorded fingerprint travelling in the output; **the
   strict resume gate is untouched.**
3. Push the repaired run dirs; regenerate the main table, the board and both audits' inputs;
   re-run the mismatch check repo-wide.
4. **Acceptance: zero npz↔metrics mismatches across all runs**; the report states the
   pre-repair deltas (≤ 0.0273, all C2/C3 — none cross a condition boundary) so the repair is
   visible, not silent.

---

## 6. The write-up (`report/measurements.md` → the report)

- **Limitations BEFORE claims** (SCOPE item 20), and specifically before the audit outcomes are
  known where possible. The list must include: WT3/T4 label noise (93% mislabelling estimate;
  the mined-hard region is the dirtiest), the imbalanced-test base-rate trap (why C2's 0.2478
  is "predict nothing", read via the ceiling and MAP@R), the two-ruler non-comparability,
  single language / single encoder / three seeds / no p-values, the G1s gate as the
  manipulation check, per-seed fingerprinting as the determinism story, the §5 repairs and
  their pre-repair deltas, and the published-CodeBERT ≈ 0.95 sanity-check tension (item 16):
  our 0.7464 measures a different pair regime (corpus-wide 13.7% positives vs the
  near-clone-heavy published splits) — state the reconciliation, do not chase the 0.95.
- **Claims ledger**: every sentence the report headlines maps to (table panel, run dirs,
  gate status). C5/C6 rows stay "pending audit" until §4 closes; the §9.4 mechanism sentence
  is gated on `audit_c6`'s outcome branch.
- **`run_all.sh`** (item 22): uncomment and order the real commands (fetch → prepare → mine →
  train C1–C6 → evaluate → generalize mine/train/table → visualize → demo check → hf sync),
  each with its artifact gate, so every report number has a reproducing command. Tested
  headless (`--help`-level smoke in the suite).
- **Demo**: already shipped and checked in Phase 3 (§12); the report links it. Nothing new.

---

## 7. Budget

≈ 45 units of the 2026-10-02 quota remained at the last count; consolidation consumed almost
none of it (one T4 day). Phase 4 costs: audits ≈ free (reader VM, CPU); §5 repairs ≈ 1–2
GPU-h (nine evaluations, no training); write-up ≈ 0 GPU. The reserve exists for one amended
extension (§8, D4-4) — and only for one that survived the SCOPE amendment.

## 8. Risks and controls

| risk | control |
|---|---|
| `audit_c6` overturns §9.4's amplifier claim mid-write-up | the outcome branches are pre-registered in §9.5; write §6's limitations first so the rewrite is additive, not foundational |
| §5 repairs move a number someone already quoted | pre-repair deltas are stated in the report; all downstream surfaces regenerate in one commit (table, board, audit inputs) |
| Hindsight: writing limitations after framing the claims | item 20 ordering is an entry condition for §6, checked in this file's log |
| Rule-7 temptation: start the denoising δ-ladder "while writing" | D4-4 requires a SCO.md-recorded amendment AND a completed draft before any extension run |
| Quota interruption mid-repair | per-run pushes after each §5 evaluation; repairs are idempotent (`--force` re-runs cleanly) |

## 9. Open decisions — record before running

- **D4-1** the one-table layout (§3): panels, extension-block placement for C4–C6.
- **D4-2** the demo/report headline condition: C1 by default; C5 only if `audit_c5` lands in
  band (§4) — decided once, not per-paragraph.
- **D4-3** the 0.95 sanity-check framing (item 16): reconciliation text, approved as a
  limitation, not treated as a harness defect to chase.
- **D4-4** the optional extension, if any, **after** the write-up draft exists and only with a
  SCOPE amendment. The menu (SCOPE Part 4 + Part 3), with the honest recommendation:
  1. *Denoising δ-ladder* — RocketQA-style `cos(anchor, positive) − δ` cut, δ ∈ {0.0, 0.05,
     0.10, 0.20} × 3 seeds on C3's supply: turns §9.4's dose/similarity story into a measured
     curve. SCOPE itself calls this "the natural next experiment"; the bouncer (J ≥ 0.40 hard
     cut) is its δ = ∞/binary special case, so the ladder interpolates between C3 and C4.
  2. *FN-weighted loss* on C3 (order-statistic weights, per the code-search FN literature) —
     a loss-shape change, distinct from the forbidden P6-2 LR rescue, but it must be recorded
     as such or not run.
  3. *Second encoder* (`codebert-base`) on the winning condition only — ranking-robustness.
  4. Not recommended now: leave-one-functionality-out rotation (10× the s′ compute for tighter
     Δ error bars — a paper-revision item), second language, symmetric pairs, reranker,
     ANCE refresh, full dataflow GraphCodeBERT.

## 10. Execution log

| date | what | files | deviations / notes |
|---|---|---|---|
| 2026-10-05 | Plan written (this file), after the Phase-3 consolidation shipped (`ede587a`) and the sequencer green-ran end to end. No Phase-4 experiment has run; entry criteria §2 verified. | `PHASE4_PLAN.md` | Written ahead of any Phase-4 work, in the same plan-as-record shape as PHASE2/PHASE3. The stop condition (Part 7) is acknowledged as met: §3–§6 are closure, not new experiments. |
