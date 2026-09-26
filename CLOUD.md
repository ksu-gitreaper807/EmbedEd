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
| **Gate G1: hardness check** | 1 | GPU (small encode) | cloud | ~5–10 min | `hardcheck.json` — **FAIL ⇒ stop, fix mining** |
| Smoke run (15 steps) | 1.5 | GPU (trivial) | cloud | ~3 min | log only |
| **Training: 9 main + 9 generalisation runs** | 2–3 | GPU (T4) | cloud, batched | 1–2 h each (measured) | `runs/{condition}_{seed}/` checkpoint + `train.log` + `metrics.json` |
| Eval passes: encode test/s′ fragments per fine-tuned model | 2–3 | GPU (T4) | cloud, batched | ~15–25 min per model × 6 | `scores/{condition}_{set}.npy` + `threshold.json` |
| Scoring: threshold-on-val, F1, P/R, MAP@R, Δ per condition | 2–3 | CPU (numpy) | **local** | seconds | `report/tables/*.csv` |
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
| 18 fine-tuning runs (9 main @ ≤50k pairs, 9 smaller on s′) | **~12–36 h** (1–2 h/run measured) |
| 6 eval encodes (+ 2 s′ evals, smaller) | **~2–4 h** |
| **Total** | **~15–40 GPU-h**, i.e. **3–5 Colab days** at the ~12 h/day quota |

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
python -m scripts.phase0_throughput --candidates 256:32 256:16 512:16 --minutes 0.5
#   → commit the printed MAX_LEN / TRAIN_PAIRS_CAP / BATCH / EPOCHS to settings.py FIRST
python -m embeded.mining.semantic_index --batch 32        # corpus encode, cached; skips if present
python -m embeded.negatives --strategies random,bm25,semantic --k 20
python -m embeded.hardcheck                               # exit 0 = G1 PASS; 1 = FAIL → STOP
python -m scripts.hf_artifacts push --if-configured --include-report
# (smoke run: notebook cell, 3 min)
```

**S2–S4 — training** (the bulk; one cell per run so a death costs one run)
```bash
# Phase 2 code (lands with train.py); idempotent — skips if runs/C1_s13/metrics.json exists
for c in C1 C2 C3; do for s in 0 1 2; do
  python -m embeded.train.train --condition $c --seed-offset $s
done; done
# generalisation (Phase 3, smaller cap):
for c in C1 C2 C3; do for s in 0 1 2; do
  python -m embeded.train.train --condition $c --seed-offset $s --sprime --holdout-k 3
done; done
```
Ordering: main table first (it's the deliverable); generalisation second.
If a session runs low, stop between runs — the next session skips finished
ones.

**S5 — eval** (GPU ~2–4 h)
```bash
# Phase 2–3 code; one forward pass per (model, fragment-set); C0 reuses the
# cached base-model embeddings — no GPU for it
python -m embeded.eval.evaluate --encode-only            # writes scores/*.npy to EMBEDED_ARTIFACTS
```
Then **disconnect the T4**. Everything after S5 is local CPU (§3).

---

## 5. Keeping the GPU hours down (levers, ranked)

1. **`MAX_LEN` 256 vs 512** — 2× cost on every GPU stage. Freeze it from the
   Phase 0.4 p99; stay at 256 if p99 ≤ 250.
2. **`TRAIN_PAIRS_CAP` from the measured throughput test** — size 1 epoch to
   land in ~1–1.5 h (headroom for the 12-h cap); never above the 50k
   [illustrative] without a measured reason.
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
