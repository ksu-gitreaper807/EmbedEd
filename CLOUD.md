# CLOUD.md — GPU consolidation runbook (Colab free T4)

Goal: **your local machine never touches a GPU** (no torch, no CUDA, ever). All
torch work lives in Colab sessions; everything that is CPU/network work stays
local for free. And because Colab free-tier quota is billed by **wall-clock
minutes a GPU runtime is connected — not by GPU usage** — the second goal is:
**keep the T4 mounted only while a GPU stage is actually running, and in as few
sessions as possible.**

Rules that follow from that:

1. CPU stages run locally (dataset download, data prep, tests, scoring, figures).
2. GPU stages are batched into one kind of session each; a T4 session is opened,
   filled with GPU work to the session cap, and closed.
3. **Encode once, reuse everywhere.** The base-model corpus embedding matrix
   (8,063 × 768 ≈ 25 MB) is the one expensive GPU artifact; mining, the C0
   baseline, the smoke run, and every eval pass hang off the cached copy.
4. **Checkpoint to Hugging Face Hub, resume, don't rerun** (IMPLEMENTATION_PLAN §5). Every
   stage is idempotent and artifact-gated: if its output exists and matches
   `settings.VERSION`, it is skipped.

---

## 1. Stage map — what runs where

| Stage | Phase | Compute | Where | Time est. | Artifacts out (local `EMBEDED_ARTIFACTS/`, optionally synced to HF Hub) |
|---|---|---|---|---|---|
| Data prep: download, G0 counts, overlap, token lengths | 0.2–0.4 | network + CPU (tokenizer) | **local** | 10–20 min + download | `fragments.jsonl`, `pairs_{train,valid,test}.tsv`, `manifest.json`, `overlap.json`, `token_lengths.json`, `report/measurements.md` |
| s′ acquisition + probe | 0.7 | network + CPU | **local** | 5 min + download | `sprime/` (parsed pairs) |
| Unit tests (exclusion, pipeline, hardcheck) | 0 | CPU | **local** | seconds | — |
| Throughput test → freezes `MAX_LEN`/`CAP`/`BATCH`/`EPOCHS` | 0.5 | GPU (T4) | cloud | ~5 min | printed settings (commit to `settings.py`) |
| Corpus encode (base model, ONCE) | 1 | GPU (T4) | cloud | ~15–30 min | `corpus_emb.npy`, `corpus_emb.sha1` |
| Mining C1/C2/C3 (random / BM25 / semantic top-k) | 1 | CPU (matmul on cached embeddings) | cloud (or local) | ~5 min | `triples_C{1,2,3}.jsonl`, BM25 index |
| **Gate G1: hardness check** | 1 | CPU (reads cached embeddings) | cloud | ~1–2 min | `gate_runs.json` + a run appended to the G1 section of `report/measurements.md` — **FAIL ⇒ stop, fix mining** |
| Smoke run (15 steps) | 1.5 | GPU (trivial) | cloud | ~3 min | log only |
| **Training: 9 main runs** (`embeded.train`) | 2 | GPU (T4) | cloud, one cell per run | ~40 min each (measured 13.5 triples/s) | `runs/{condition}_{seed}/` `checkpoint.pt` + `run_config.json` + `train_log.jsonl` + `metrics.json` |
| Eval passes: threshold on validation, then test once (`embeded.evaluate`) | 2 | GPU (T4) | cloud, batched | ~10–20 min per model | `runs/{condition}_{seed}/` `eval_metrics.json` + `predictions.npz` + `eval_config.json` |
| Main-results table (F1, P/R, MAP@R, mean/spread) | 2 | CPU (numpy) | **local** | seconds | `## Phase 2 — main results` in `report/measurements.md` |
| Generalisation on s′ (`embeded.generalize`) | 3 | GPU (T4) | cloud | ~15 min per model | Δ per condition (Phase 3) |
| False-negative labelling (50+50 judged by hand) | 2 | human + CPU | **local** | ~2 h | rubric + sheet (committed) |
| UMAP figure, tables, paper | 3–4 | CPU | **local** | — | `report/`, paper |

**Local pip set (no torch):** `datasets transformers rank-bm25 umap-learn scikit-learn numpy requests pytest`.
`transformers` tokenizes fine without torch; `prepare_data.py` falls back to a
word-count approximation for token lengths if the tokenizer is unavailable and
flags it — on a laptop the real tokenizer works.

---

## 2. The GPU budget

From the Phase 0.5 throughput test (run it first — it replaces all
[illustrative] numbers, SCOPE P2-13):

| Work | GPU estimate |
|---|---|
| Session 1: throughput + corpus encode + G1 + smoke | **~0.75 h** |
| 18 fine-tuning runs (9 main @ 32k triples ≈ 40 min each — measured 13.5 triples/s at 512:8 fp16; 9 smaller on s′) | **~6 h main + ~2–3 h s′** |
| 6 eval encodes (+ 2 s′ evals, smaller) | **~2–4 h** |
| **Total** | **~11–14 GPU-h**, i.e. **1–2 Colab days** at the ~12 h/day quota |

Calendar at the free-tier limits (12 h/session hard cap, ~12 h/day quota,
~90 min idle disconnect):

| Day | Session | Content | Notes |
|---|---|---|---|
| 1 | S1 (~2 h wall) | setup + 0.5 throughput → commit settings; corpus encode; mining; **G1**; smoke | G1 FAIL ⇒ stop here; nothing else is worth running |
| 2 | S2 | main runs 1–6 | stop with ≥1 h under the cap |
| 3 | S3 | main runs 7–9 + generalisation runs 1–3 | generalisation runs are smaller (s′ = 2,300 pairs) |
| 4 | S4 | generalisation runs 4–9 | |
| 5 | S5 | 6 eval encodes + download score arrays | then close the GPU forever — scoring/tables/figure are local CPU |

A **running cell counts as activity**, so a 1.5-h training cell does not
idle-disconnect while it runs. What kills sessions is an *idle* runtime and the
daily quota — both are covered by the resume protocol below. Worst case: a
session dies mid-run ⇒ that run (≤2 h) is lost and reruns from scratch next
day; everything before it is in the last HF checkpoint you pushed.

---

## 3. Local work (zero GPU)

```bash
# once, on the laptop — no torch needed anywhere in this list
pip install datasets transformers rank-bm25 umap-learn scikit-learn numpy requests pytest

# Phase 0.2–0.4: download + G0 counts + overlap + token lengths (~10–20 min + download)
python -m embeded.data.prepare_data --hf --verify-spec        # must print "all counts match ✅"
python -m pytest -q embeded/tests                              # seconds, fully offline

# Phase 0.7: s′
python -m scripts.fetch_sprime --dry-run && python -m scripts.fetch_sprime
```

Then upload the small artifacts **once** to your HF dataset repo so Colab never
redoes prep: `fragments.jsonl` (~3 MB), `pairs_*.tsv` (~15 MB),
`manifest.json`, `overlap.json`, `token_lengths.json`.

```bash
python -m scripts.hf_artifacts push --include-report
```

(Or, if you'd rather not keep a local copy at all, run the same `prepare_data`
cell inside a **CPU** Colab runtime — zero quota cost — before ever attaching a
T4, then push from there.)

Token-length p99 from `token_lengths.json` decides `MAX_LEN`
(256 if p99 ≤ 250, else 512 — **a 2× GPU cost**, so read it before committing).

After the cloud eval session, scoring is also local:

```bash
python -m embeded.eval.evaluate --from-scores   # Phase 2 code; CPU numpy, seconds
```

produces the three tables (`report/tables/`). The whole week-3/4 deliverable —
tables, UMAP figure, paper numbers — is then local CPU work.

---

## 4. Colab session recipes

Every session starts identically (cells 1–2 of the notebook): clone the repo,
load Colab secrets into `EMBEDED_HF_REPO_ID` / `HF_TOKEN` if present, set
`EMBEDED_ARTIFACTS` / `EMBEDED_REPORT`, pin env from `settings.PINNED`, then
pull any saved checkpoint with `python -m scripts.hf_artifacts pull --if-configured`.
Then, **only** the GPU stages that are missing:

**S1 — setup + mining** (GPU ~0.75 h)
```bash
python -m scripts.phase0_throughput --candidates 512:4 512:8 512:16 512:32 --minutes 0.5   # fp16 AMP; OOM candidates are skipped
#   → commit the printed MAX_LEN / TRAIN_PAIRS_CAP / BATCH / EPOCHS to settings.py FIRST
python -m embeded.mining.semantic_index --batch 32        # corpus encode, cached; skips if present
python -m embeded.negatives --strategies random,bm25,semantic --k 20
python -m embeded.hardcheck                               # exit 0 = G1 PASS; 1 = FAIL → STOP
python -m scripts.hf_artifacts push --if-configured --include-report
# (smoke run: notebook cell, 3 min)
```

**S2–S4 — training** (the bulk; one cell per run so a death costs one run)
```bash
# idempotent: skips a run whose metrics.json matches the current config
# fingerprint, resumes an interrupted one from its checkpoint
for c in C1 C2 C3; do for s in 13 14 15; do
  python -m embeded.train    --condition $c --seed $s
  python -m embeded.evaluate --condition $c --seed $s
done; done
python -m embeded.evaluate --condition C0        # untuned baseline, once
python -m embeded.evaluate --results-table       # rebuild the main-results section
# generalisation on s′ (Phase 3 — NOT implemented yet; `embeded.generalize`):
# for c in C1 C2 C3; do for s in 13 14 15; do
#   python -m embeded.generalize --condition $c --seed $s --holdout-k 3
# done; done
```
Ordering: main table first (it's the deliverable); generalisation second.
If a session runs low, stop between runs — the next session skips finished
ones.

**S5 — nothing separate to run.** `embeded.evaluate` encodes the split
fragments, selects the validation threshold and scores test in one pass per
run, so it runs inside the S2–S4 loop (and `--condition C0` for the untuned
baseline). It writes `predictions.npz`, so the table can be rebuilt later
without a GPU:
```bash
python -m embeded.evaluate --results-table               # local CPU, seconds
```
Then **disconnect the T4**. Everything after the runs is local CPU (§3).

---

## 5. Keeping the GPU hours down (levers, ranked)

1. **`MAX_LEN`** — 2× cost between 256 and 512 on every GPU stage, but the Phase 0.4
   measurement settled it: p50 = 474 tokens, p99 = 5,944 → **512** (256 would truncate the
   median fragment). Not a lever any more; the recipe that makes 512 fit is fp16 AMP
   (`settings.AMP_DTYPE`; plain fp32 OOMs the T4 even at 256×32).
2. **`TRAIN_PAIRS_CAP` from the measured throughput test** — frozen at **32k**
   (13.5 triples/s × 40 min at 512:8; `report/measurements.md` Phase 0.5). One run ≈ 40 min,
   so a session death costs at most that; never raise it without a re-measured reason, and
   never above ~90k (CodeXGLUE's own 10% slice).
3. **`EPOCHS = 1`** (already frozen) — do not "try 2" without budget for the
   doubled runs.
4. **Encode once** — the corpus matrix is hashed against `settings.VERSION`;
   bumping `VERSION` deliberately invalidates caches, accidentally never.
5. **Artifact-gate everything** — a finished stage/run is never rerun; the
   session recipes above are idempotent by design.
6. **Generalisation runs stay small** (s′ = 2,300 pairs ⇒ smaller cap) —
   "six smaller runs" is in the spec (Phase 3.1); don't normalize them to the
   main cap.
7. **Quota exhaustion ladder** (SCOPE P6-8): cut **seeds** before cutting
   **strategies** (3 seeds → 2 before C3 goes away); the fallback ladder is
   3 strategies → 2 → C0+C1 ("complete weak beats incomplete strong").

---

## 6. Resume protocol (why a dead session costs ≤ 1 run)

- All state lives under local `EMBEDED_ARTIFACTS/` and is checkpointed to the
  configured HF dataset repo when you run `python -m scripts.hf_artifacts push`.
  That tree is shared, versioned by `settings.VERSION`, and per-run outputs live
  under `runs/{condition}_{seed}/`.
- A stage is *done* iff its output artifact exists **and** the manifest's
  version matches; otherwise it regenerates just that stage.
- Per-run checkpoint: `runs/{c}_{s}/` contains the fine-tuned model after each
  epoch + `train.log`; a run that died mid-epoch restarts that one run
  (EPOCHS = 1 ⇒ ≤2 h lost, nothing else).
- The `scripts/gpu_session` orchestrator (built with Phase 2's `train.py`,
  per plan) automates exactly this table — one command, `--resume`, per-stage
  skip. Until it lands, the per-cell recipes in §4 are the protocol: each cell
  is safe to rerun, and the notebook's startup/final sync cells move checkpoints
  between Colab and HF Hub.
