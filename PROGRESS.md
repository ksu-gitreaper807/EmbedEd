# PROGRESS.md — what has been done, measured, inferred and decided

Running log for the Phase 0 / Phase 1 work. **Facts (measured numbers) and inferences
(our reading of them) are kept in separate sections on purpose.** Numbers are copied from
`report/measurements.md` and the Colab outputs of 2026-09-26; the report file is the source
of truth if the two ever disagree.

*Last updated 2026-09-26 · branch `arena/01a0dc7d-embeded` · PR
[#6](https://github.com/ksu-gitreaper807/EmbedEd/pull/6) (open, unmerged) · tip `4146485`.*

---

## 0. Status at a glance

| Step (IMPLEMENTATION_PLAN) | Status | Evidence |
|---|---|---|
| 0.2 load + verify counts (**Gate G0**) | ✅ PASS | 9,126 / 8,063 / 901,028 / 415,416 / 415,416 |
| 0.3 train/test fragment overlap | ✅ measured | 76 / 860 test fragments (8.8 %) |
| 0.4 token lengths → `MAX_LEN` | ✅ frozen | p50 474, p90 1,535 → `MAX_LEN = 512` |
| 0.5 throughput → `BATCH` / `TRAIN_PAIRS_CAP` / `EPOCHS` | ✅ frozen | 512:8 → 8 / 32,000 / 1 |
| 0.7 s′ acquisition (**s′ gate**) | ✅ PASS | 4,600 pairs, 23 functionalities |
| 1.0 all-fragment embeddings | ✅ ran once (245 s, T4) — **artifact lost with the VM, must be regenerated** | `corpus_emb.npy` 8,063 × 768 |
| 1.1 mining C1 / C2 / C3 | ✅ ran once — **artifacts lost, regenerate** | 32,000 triples per condition |
| 1.2 tests | ✅ | 36 passed at `4146485` (29 at the time of the mining run) |
| 1.3 **Gate G1** hardness check | ❌ **FAIL** as defined | C1→C2 gap +0.0146 < margin 0.02 |
| 1.4 hardness diagnostics + label-noise floors | ✅ measured | §2.6 |
| 1.5 smoke run | ⏸ not run (harmless, but pointless before G1 is settled) | — |
| 2.x **training** | ⛔ **blocked** by G1 (SCOPE P6-3) | — |
| 2.5 50 + 50 false-negative audit | 🛠 tooling ready, **pulled forward to before training** | `scripts/audit_sample.py` |

Two decisions are open and must be recorded before any training (§4.2): the **gate rule** and
**what to do about near-duplicate negatives**.

---

## 1. Timeline (commits on PR #6)

| Commit | What |
|---|---|
| `7437c90` | Phase 0: `MAX_LEN = 512` from measured token lengths; throughput test rewritten to fit the T4 (fp16 AMP, OOM-tolerant) |
| `f01a150` | Phase 0.5: sweep results; `BATCH = 8`, `TRAIN_PAIRS_CAP = 32_000`, `EPOCHS = 1` frozen |
| `502ef54` | Gate G1 FAIL recorded as-is; `scripts/hardness_diagnostics.py`; `hardcheck` keeps a run history instead of overwriting |
| `3a8fdf5` | Diagnostics: label-noise floors (valid-split labels, token-Jaccard near-duplicates) |
| `ff212c9` | Resume protocol: every stage artifact-gated; notebook checkpoints to the HF dataset repo after each heavy cell |
| `4146485` | Blind 50 + 50 audit tooling (`make` / `score`, Wilson CIs, report section) |

Colab sessions: one full T4 session (Phase 0 → G1 → diagnostics), one VM recycle that lost
the un-checkpointed embeddings/triples, one CPU-only reconnect (Phase 0 redone in 3 min 42 s,
first successful HF checkpoint push).

---

## 2. Measured numbers (facts)

### 2.1 Dataset (Phase 0.2–0.4)

| quantity | value |
|---|---|
| `data.jsonl` lines / unique fragment texts | 9,126 / 8,063 (1,063 byte-identical duplicates) |
| pairs train / valid / test | 901,028 / 415,416 / 415,416 |
| fragments appearing in train pairs / test pairs | 6,527 / 860 |
| test fragments also in a train pair | 76 (8.84 %) |
| GraphCodeBERT tokens per fragment p50 / p90 / p95 / p99 / max | 474 / 1,535 / 2,233 / 5,944 / 36,823 |
| mining corpus (train fragments − test fragments) | 6,451 |

### 2.2 Throughput on the Colab T4 (Phase 0.5, fp16 autocast, 512 tokens)

| max_len:batch | triples/s | peak GB | cap for a 40-min epoch |
|---|---|---|---|
| 512:4 | 11.9 | 4.37 | 28,648 |
| 512:8 | **13.5** | **6.71** | 32,367 |
| 512:16 | 14.1 | 11.39 | 33,951 |
| 512:32 | OOM | — | — |

Plain fp32 at the original candidates (256:32) OOMed outright — the fp16 recipe is now shared
with `train.py`.

### 2.3 s′ (Phase 0.7)

Zenodo replication package of *Detecting Clones of Unseen Functionality* (ASE '25), 138 MB:
`datasets/bcb_v2_sampled_bf/data_bcb_v2_sampled_bf.pickle` — 4,600 pairs, labels 2,300 / 2,300,
`functionality_id` with 23 unique values (+ 4 duplicate copies elsewhere in the package). **PASS.**

### 2.4 Mining run (Phase 1.0–1.1, T4)

- `corpus_emb.npy`: 8,063 × 768, mean-pooled, `max_len = 512`, 245 s on the T4.
- Anchor stream: 1,600 (anchor, positive) pairs (= `TRAIN_PAIRS_CAP // k`), identical for all
  conditions; `k = 20` → 32,000 triples per condition; `anchors_short = 0`; `padded = 7` for
  C2 and C3 (7 of 32,000 slots topped up deterministically after exclusion).
- BM25 mining is ~15–30 min of single-threaded CPU; everything else is seconds to minutes.

### 2.5 Gate G1 (1,000 sampled anchors × 20 = 20,000 per condition)

| | mean cos(anchor, negative) | sd |
|---|---|---|
| C1 random | 0.96068 | 0.0257 |
| C2 bm25 | 0.97526 | 0.0216 |
| C3 semantic | 0.98845 | 0.0072 |

Ordering `C1 < C2 ≤ C3` holds. `C2 − C1 = +0.0146 < HARDNESS_MARGIN = 0.02` → **FAIL**.
Recorded in `report/measurements.md` under *Gate G1 — hardness check* (run history; later
runs are appended, never overwrite this one).

### 2.6 Diagnostics (all 1,600 anchors × 20 = 32,000 per condition)

Scale of the space: **random corpus pair cos = 0.9603**; **cos(anchor, labelled positive) =
0.9651** (sd 0.023).

| | C1 | C2 | C3 |
|---|---|---|---|
| mean cos / sd | 0.9606 / 0.026 | 0.9755 / 0.021 | 0.9883 / 0.008 |
| gap vs C1 | — | +0.0150 | +0.0278 |
| standardised gap *d* vs C1 | — | **0.63** | 1.46 |
| position in the C1→C3 range | 0 | 0.54 | 1 |
| percentile of negative in the anchor's corpus ranking, mean / **median** | 50.0 / **50.1** | 79.6 / **91.9** | 99.8 / **99.8** |
| negatives inside the anchor's top-20 / top-100 (of 6,451) | 0.3 % / 1.5 % | 15.4 % / 28.3 % | 94.8 % / 99.98 % |
| negatives closer to the anchor than its labelled positive | 39.6 % | 71.5 % | 99.2 % |

Negative-set overlap per anchor: C1∩C2 0.3 %, C1∩C3 0.3 %, **C2∩C3 15.8 %**.

Label-noise floors (proxies, not the audit):

| | C1 | C2 | C3 | labelled positives |
|---|---|---|---|---|
| negatives that are **train**-labelled clones (exclusion self-check) | 0 | 0 | 0 | — |
| negatives that the **valid** split labels as clones | 0.02 % | 0.04 % | 0.05 % | — |
| median token-set Jaccard with the anchor | 0.21 | 0.37 | 0.40 | 0.28 |
| Jaccard ≥ 0.5 | 0.9 % | **28.0 %** | **31.4 %** | 5.1 % |
| Jaccard ≥ 0.75 | 0.03 % | 6.6 % | 6.9 % | 0.2 % |

Eyeballed examples (first negative per condition): anchor `copy(pathFileIn, pathFileOut)` →
C3 #1 `copyFileAscii(src, dest)` with the same body, C2 #1 `copyFileToDir`; anchor
`newGuidSeed(secure)` → C2/C3 #1 `getRandomGuid` / `getRandomGUID`. The anchors' *labelled*
positives were `choosePivotVertex()` and `encryptPassword()`.

---

## 3. Inferences (our reading — each tagged with the facts it rests on)

**I1 — The untuned encoder's space is extremely anisotropic.** A random pair of corpus
methods already scores 0.960; the model's own nearest neighbours reach 0.988. The entire
usable range of the cosine ruler is ≈ 0.028 wide. *(§2.6 scale row.)*

**I2 — Off-the-shelf GraphCodeBERT carries almost no signal for BigCloneBench's labels.**
Labelled clones sit at 0.965 vs 0.960 for random pairs (d ≈ 0.2), and 40 % of *random*
fragments are closer to an anchor than its labelled clone. This is consistent with the
dataset's make-up — ~95 % of its clone pairs are weak Type-3 / Type-4 (see
`GROUND_TRUTH.md`) — and it means the C0 baseline should be expected near chance, and that
"cos(anchor, negative) > cos(anchor, positive)" is **not** a false-negative indicator in this
space. *(§2.6.)*

**I3 — The G1 FAIL is a scale artefact of an absolute margin, not a failed manipulation.**
`HARDNESS_MARGIN = 0.02` was written into `settings.py` before the spread of the space was
known (SCOPE only says "a visible gap"); it demands that BM25 negatives cover ~75 % of the
0.028-wide ruler. In scale-free terms the manipulation is large: the median BM25 negative
sits in the top 8 % of the corpus by the model's own ranking, 28 % are in the anchor's
top-100, and the gap is d = 0.63 (~70 standard errors). *(§2.5, §2.6.)* The gate did its job
by forcing this to be examined; the verdict as defined still stands until the rule is
changed explicitly (§4.2).

**I4 — C2 and C3 are genuinely different manipulations.** Only 16 % of their negatives
coincide per anchor, and C3 is a tight cluster (sd 0.008, 95 % inside the anchor's top-20)
while C2 is broad (mean percentile 80, median 92 — a long tail of poor lexical matches).
*(§2.6 overlap and sd rows.)*

**I5 — Correction 5 has materialised, and it is not a labelled-clone leak.** The exclusion
removes every train-labelled clone (0 %), and the valid split adds almost nothing (0.0x %).
The issue is *unlabelled* pairs: 28–31 % of hard negatives share ≥ 50 % of their token set
with the anchor, against 5 % for BCB's own clones — the corpus is full of textual near-copies
that were never labelled for that pair. It affects C2 and C3 about equally, so it does not
bias the C2-vs-C3 comparison, but it caps how "hard negatives" may be interpreted. *(§2.6
label-noise table, examples.)*

**I6 — Token Jaccard is a floor, not a filter.** The near-verbatim `copyFileAscii` copy
scored only 0.57 (different lengths) while a related-but-different `getRandomGuid` scored
0.71, so a Jaccard threshold would both miss and over-catch. The measurement the plan
prescribes — the manual 50 + 50 audit — is the right instrument, and it needs only the
triples, hence pulling it ahead of training. *(§2.6 examples.)*

**I7 — `MAX_LEN = 512` truncates about half the fragments** (p50 = 474, p90 = 1,535). Accepted
because the alternative (1,024+) does not fit a 40-min-per-run T4 budget; consequence to
report: clones whose distinguishing code lies past token 512 are invisible to every condition
equally. *(§2.1, §2.2.)*

**I8 — Batch 8 over 16.** 512:16 is only 4 % faster but peaks at 11.4 of 15 GB — one long
batch from OOM; 512:8 is within 10 % of the best at 6.7 GB. Rule recorded in
`scripts/phase0_throughput.py::suggest`. *(§2.2.)*

**I9 — Operational: the environment is recoverable only if the expensive artifacts are
checkpointed.** Model weights and the dataset re-hydrate from the Hub in ~1 min; the things
that cost time are the GPU encode (4 min), BM25 mining (15–30 min) and every measured
number — those now go to the HF dataset repo after each heavy cell, and every stage skips
itself when its artifacts are present (`COLAB.md`, *Resume protocol*).

---

## 4. Decisions

### 4.1 Frozen (in `embeded/settings.py`, `VERSION = "phase01-v5"`)

| Decision | Value | Basis |
|---|---|---|
| `MAX_LEN` | 512 | §2.1 token lengths + T4 budget (I7) |
| precision | fp16 autocast, grad checkpointing off | fp32 OOMs at the original candidates |
| `BATCH` / `TRAIN_PAIRS_CAP` / `EPOCHS` | 8 / 32,000 / 1 | §2.2 sweep (I8); 1,600 anchors × k = 20 |
| `K_NEGATIVES` | 20, same for all conditions | SCOPE P2-6 |
| anchor stream, exclusion, k | identical across C1/C2/C3 (tested) | `test_mining_invariants` |
| s′ source | ASE '25 replication package, `bcb_v2_sampled_bf` | §2.3 |
| G1 record | FAIL kept verbatim, run history from now on | SCOPE P6-3 transparency |

### 4.2 Open — owner's call, to be recorded before training

**D1 — the gate rule.** SCOPE P1-6 requires `C1 < C2 ≤ C3` "with a visible gap"; the 0.02
absolute margin is an implementation choice that ignores the scale of the space (I1, I3).
Options: (a) keep 0.02 and change the mining until it passes (sanctioned by the plan, but
pushes the *lexical* condition to look *semantic*, and the ceiling of the ruler makes it
unlikely); (b) re-operationalise "visible gap" scale-free — **recommended: standardised gap
`d(C1→C2) ≥ 0.5`, ordering kept, percentiles reported alongside**, committed once with the
rationale and the FAIL left in the history. Current numbers pass (b) with d = 0.63.

**D2 — near-duplicate negatives (I5, I6).** Options: (a) keep the mining as specified and
*measure* the false-negative rate (the plan's default); (b) extend the exclusion to drop
near-verbatim candidates from all conditions. **Recommended: (a) now — run the blind 50 + 50
audit before training; revisit (b) only if the audit shows a large rate (≳ 30 %) in C3.**

Rules in force while these are open: no Phase 2 training (SCOPE P6-3), no fourth strategy
(SCOPE P6-1), no quiet edits to `HARDNESS_MARGIN`, all three conditions re-mined together
after any mining change.

---

## 5. Incidents and lessons

| Incident | Consequence | Fix / lesson |
|---|---|---|
| A live HF token was pasted into a notebook cell | token exposed | revoked and recreated; tokens live only in Colab secrets |
| fp32 throughput test OOMed at 256:32 on the T4 | Phase 0.5 blocked | fp16 autocast + OOM-tolerant sweep; 512:32 still OOMs in fp16 — never suggest it |
| Colab VM recycled with nothing checkpointed | embeddings + triples lost (~35 min of compute), Phase 0 redone | HF checkpoint now configured (`Kusshal/Embed`, first push done); stages artifact-gated; heavy cells push on completion |
| Reconnect landed on a **CPU** runtime (`torch 2.11.0+cpu`) | cell 13 cannot run sensibly | check `cuda=True` in cell 2 before any GPU step; switch runtime type first |
| `report/measurements.md` is generated *and* committed | `git pull` conflicts in the Colab clone | `git checkout -- report/measurements.md` before pulling; gate sections append rather than overwrite |
| Cell 0's `rm -rf` on a live VM | would delete the clone's report edits | only run cell 0 on a fresh VM |

---

## 6. Where everything lives

| What | Where |
|---|---|
| Code, frozen constants, this log | GitHub, branch `arena/01a0dc7d-embeded`, PR #6 |
| Measured numbers | `report/measurements.md` (Phase 0, 0.5, s′, G1 history, diagnostics when re-run) |
| Generated artifacts checkpoint | HF dataset repo `Kusshal/Embed` — currently Phase 0 artifacts + report (10 files); embeddings / triples to be added on the next GPU run |
| Diagnostics | `scripts/hardness_diagnostics.py` → `hardness_diagnostics.json` + report section |
| Audit | `scripts/audit_sample.py` → `artifacts/audit/{audit_pairs.md, audit_labels.csv, audit_key.csv, audit_result.json}` |
| Notebook | `notebooks/Phase0_Phase1_Colab.ipynb` (sequencer only; all logic in `embeded/`) |

---

## 7. Next steps

1. Get a T4 (`Runtime ▸ Change runtime type`), confirm `cuda=True` in cell 2.
2. Cells 0 → 1 → 2 → 5 (`[cache]`) → 11 (`36 passed`) → 13 → 14 → 15 (FAIL again, recorded as run 2; diagnostics + checkpoint run automatically).
3. `python -m scripts.audit_sample make`; label the 100 pairs blind (~2 h, no GPU needed); `python -m scripts.audit_sample score`.
4. Record D1 and D2 (settings + `measurements.md`), re-run `hardcheck` under the chosen rule.
5. Smoke run (cell 17), then Phase 2: 3 conditions × 3 seeds ≈ 6 GPU-hours.
