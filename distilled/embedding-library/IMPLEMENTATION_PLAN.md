# Implementation plan — `domembed`

**Total additional effort: ~5.5 days**, most of it in the gaps while models train. None of it
needs GPU time, and none of it needs the fine-tuned model to exist.

That last sentence is the key scheduling fact, so it is worth being explicit about it:

> **Build the library against `all-MiniLM-L6-v2` first.** `DomainEmbedder.load()` accepts the
> generic baseline, so every line of `domembed` can be written, tested and demoed *before* any
> fine-tuning finishes. When the trained checkpoint lands, you change one string —
> `"./runs/quad_agree_hard/final"` — and the product is finished.

---

## 1. Schedule

Four phases, mapped onto the existing four-week research plan
([`project/distilled/IMPLEMENTATION_PLAN.md`](../../project/distilled/IMPLEMENTATION_PLAN.md)).
**Do not start Phase 1 before the research harness produces its first baseline numbers**
(Step 4 of that plan) — until BM25 and the zero-shot encoder have survived the debugging
checkpoints, the research may still change shape.

| Phase | When | Work | Effort | Depends on |
|---|---|---|---|---|
| **0** | Week 2, alongside training | Packaging skeleton + `similarity.py` | 0.5 day | nothing |
| **1** | Week 2–3, alongside training | `embedder.py` — load, encode, similarity, search, info | 1.5 days | Phase 0 |
| **1.5** | Week 2–3, alongside training | **Demo shell** against the generic model | **0.75 day** | Phase 1 |
| **2** | Week 3, alongside training | Aggregate panel + all four result-mode captions; CLI, benchmark, tests, README | 1.5 days | Phase 1.5 |
| **3** | Week 4, after the winning model exists | Export script, real weights, real numbers, delete three captions, rehearse | 1.5 days | a trained checkpoint |

Phases 0–2 run **in parallel with the research** and cost it nothing: they use no GPU and no
model other than the one already downloaded. Phase 3 is the only part that waits.

**Phase 1.5 is the important addition.** The demo is the presentation centerpiece, so it must not
be built in the last two days. The shell is built in week 2 against the generic model — corpus,
index, two-column layout, `gold_rank` — with numbers that mean nothing yet. Then week 4 is only
"drop in the real models and delete three captions" instead of "build the most important thing
under time pressure".

---

## 2. Phase 0 — skeleton (half a day)

```text
domembed/
├── __init__.py        # exports + __version__ = "0.1.0"
├── errors.py          # 5 exceptions
├── similarity.py      # cosine(), top_k()
└── pyproject.toml     # hatchling, extras demo/dev
tests/
└── test_similarity.py
```

### `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "domembed"
version = "0.1.0"
description = "Domain-specific sentence embeddings, packaged."
readme = "README.md"
requires-python = ">=3.9"
license = { text = "Apache-2.0" }
dependencies = ["sentence-transformers>=3.0", "numpy>=1.24"]

[project.optional-dependencies]
demo = ["streamlit>=1.30"]
dev  = ["pytest>=8", "ruff>=0.6"]

[project.scripts]
domembed = "domembed.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["domembed"]

[tool.pytest.ini_options]
markers = ["model: tests that need a real sentence-transformers model"]
```

Setuptools would work equally well; hatchling is chosen because the config above is the whole
config. `requires-python = ">=3.9"` plus `from __future__ import annotations` keeps it usable on
an older lab machine.

**Checkpoint 0.1** — `pip install -e .` succeeds in a fresh virtualenv.
**Checkpoint 0.2** — `import domembed; domembed.__version__` works.
**Checkpoint 0.3** — `pytest tests/test_similarity.py` passes in under one second with no model
download.

Write `similarity.py` first and completely. It is 45 lines with no dependencies inside the
package, and it is the part most likely to harbour a subtle bug (sign errors, unnormalised
vectors, `argsort` on a score array).

```python
def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity. a: (n,d) or (d,), b: (m,d) or (d,)."""
    a = np.atleast_2d(a); b = np.atleast_2d(b)
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return (a @ b.T)                      # (n, m)


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, in descending order."""
    k = min(k, scores.shape[-1])
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]
```

Note the `np.clip` on the norms: it is what makes `cosine` safe on a zero vector instead of
producing `nan`. It costs nothing and it is the kind of detail that turns a confusing bug into
`0.0`.

---

## 3. Phase 1 — the embedder (1.5 days)

Work strictly in this order; each step is verifiable before the next begins.

### Step 1.1 — `load()` and device resolution (1 hour)

```python
def load(cls, source, device=None, **kw):
    path = Path(source)
    if not path.exists():
        try:
            st = SentenceTransformer(str(source), device=..., **kw)
        except Exception as e:
            raise ModelNotFoundError(f"Could not resolve {source!r} as a local "
                                     f"directory or a Hugging Face model id. {e}") from e
    else:
        st = SentenceTransformer(str(path), device=..., **kw)
    info = read_model_info(path if path.exists() else None)
    return cls(st, info, model_id=str(source))
```

Device resolution: `device or ("cuda" if torch.cuda.is_available() else "mps" if
torch.backends.mps.is_available() else "cpu")`.

**Checkpoint 1.1** — `DomainEmbedder.load("sentence-transformers/all-MiniLM-L6-v2")` prints a
clear error, not a stack trace, when given a nonsense id.

### Step 1.2 — `encode()` (1.5 hours)

Input normalisation, validation, delegation, shape restoration. Then the truncation warning:
count how many inputs exceed `max_seq_length` in tokens and warn once.

**Checkpoint 1.2** — `encode("a").shape == (384,)`, `encode(["a","b"]).shape == (2, 384)`, and
`np.linalg.norm(v) ≈ 1.0`.

### Step 1.3 — `similarity()` and `search()` (2 hours)

**Checkpoint 1.3** — `similarity(t, t)` is `1.0 ± 1e-5`; `search()` on five documents returns
five results sorted descending; each score equals `similarity(query, doc)` to `1e-5`.

That last check is the important one: it proves `search` and `similarity` use the same
mathematics, so the demo cannot contradict the numbers.

### Step 1.4 — `metadata.py` + `info()` (2 hours)

Define the `model_info.json` schema (§6), write `read_model_info()` with the
missing-file-is-normal path, and render `ModelInfo.__str__`.

**Checkpoint 1.4** — `info()` works on a model with no `model_info.json` and prints
`Not reported` for the metadata fields, with real values for `dim`, `max_seq_length` and
`device`.

### Step 1.5 — `DocumentIndex` (1 hour)

**Checkpoint 1.5** — `index.search(q, 5)` returns the same results as
`model.search(q, documents, 5)`, to `1e-5`.

### Step 1.6 — `errors.py` wiring and the edge-case matrix (1 hour)

Walk the 20-row table in [`API_DESIGN.md`](API_DESIGN.md) §7 and make each row behave as
specified.

---

## 4. Phase 2 — surface (1.5 days)

| Step | Work | Effort |
|---|---|---|
| 2.1 | `cli.py`: `info`, `encode`, `similarity`, `search`, `bench` | 3 h |
| 2.2 | `benchmark.py` + the `bench` command | 1.5 h |
| 2.3 | Test suite: 5 named tests + property tests | 3 h |
| 2.4 | `examples/quickstart.py` | 1 h |
| 2.5 | README | 2 h |
| 2.6 | `ruff` clean; read the whole library top to bottom once | 1 h |

### 4.1 CLI details

argparse with subparsers. Two rules:

1. **Every command has `--json`.** It is what makes the CLI scriptable and it costs three lines
   per command.
2. **Pretty output by default.** The CLI exists to be looked at during a presentation, so
   `search` prints an aligned table, not a Python repr.

```bash
domembed search "how do I configure ssh keys?" questions.txt --top-k 5
```

```text
  #   score   document
  1   0.847   Setting up ssh public/private keys for login
  2   0.791   Passwordless ssh login between two machines
  3   0.402   How do I generate a new ssh key pair
  4   0.203   Installing nvidia drivers on a fresh install
  5   0.118   How do I change my screen resolution from the terminal
```

Exit codes: `0` success, `2` invalid usage (argparse default), `1` runtime error with the
message on stderr and no traceback.

### 4.2 Benchmark details

Measure what the demo actually feels:

```text
device                        cpu
dimension                     384
encode, single text           14.2 ms
encode, batch of 32           103 ms   (311 texts/s)
index build, ~14,000 docs      41.3 s   (one-time)
index.search, per query       11.8 ms
peak RSS                      742 MB   (torch, not domembed)
```

Every number above is a **placeholder**. The rule: *run `domembed bench` on the presentation
machine and paste the real output.* Do not write numbers you have not measured, and do not
measure on the Colab T4 and then present it as laptop performance.

Method: 3 warm-up encodes (first call includes lazy initialisation), 20 timed repeats, report
the mean. Report the machine (CPU model, RAM, OS) next to the table or it means nothing.

---

## 5. Phase 3 — connecting to the research (2 days)

### Step 3.1 — `scripts/export_model.py` (2 hours)

```python
"""Add model_info.json to a trained checkpoint. Does not touch the weights."""
# usage: python scripts/export_model.py runs/quad_agree_hard/seed13/final \
#            --strategy "disagreement quadrant AGREE-HARD (BM25 ∩ cosine, k=64)" \
#            --metrics results/metrics.json
```

Writes:

```json
{
  "name": "domembed-askubuntu-c1-agree-hard",
  "base_model": "sentence-transformers/all-MiniLM-L6-v2",
  "domain": "AskUbuntu — Ubuntu/Linux technical support questions",
  "objective": "MultipleNegativesRankingLoss (in-batch InfoNCE, scale 20)",
  "negative_strategy": "disagreement quadrant AGREE-HARD (BM25 ∩ cosine, k=64)",
  "training_config": { "epochs": 2, "batch_size": 32, "lr": 2e-5, "seed": 13,
                       "train_pairs": 19231 },
  "evaluation": { "recall@10": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0,
                  "_source": "results/metrics.json" },
  "provenance": "runs/quad_agree_hard/seed13/final"
}
```

The `evaluation` values are read from `results/metrics.json`. If that file is missing, the
script writes `"evaluation": null` and prints a warning — it does **not** ask you to type a
number, and it does not write a zero that could later be mistaken for a result.

Optionally `--push-to-hub org/name`, which uploads the directory plus a generated model card.

**Checkpoint 3.1** — `DomainEmbedder.load("runs/<cond>/<seed>/final").info()` prints the strategy
and the evaluation numbers that appear in `results/main_table.md`. Compare them by hand, once.

### Step 3.2 — ship every condition (30 min)

Run the export script on **each** condition (N1/N2/N3, or the four quadrants). Each is ~90 MB,
so keep them local unless you want them on the Hub. The demo needs at least two.

### Step 3.3 — Streamlit demo (3 hours to finish; shell built in week 2)

See [`DEMO.md`](DEMO.md) for the design. One file, `demo/app.py`, ~120 lines.

**Built in two passes, deliberately:**

| Pass | When | Against | Produces |
|---|---|---|---|
| Shell | Week 2 (Phase 1.5) | `all-MiniLM-L6-v2` only | Corpus + index + multi-column layout + `gold_rank` + ★ markers. Numbers meaningless; plumbing proven. |
| Fill | Week 4 (Phase 3) | The real conditions | Real models, real qrels, real metrics, all four result-mode captions trimmed to one. |

The week-2 pass is what makes this safe. If the research runs late, you still have a working
demo with a worse model in it, rather than a beautiful app with nothing to show.

`gold_rank()` must handle a miss explicitly — return `None` and render `> k`, never silently
drop the row.

### Step 3.4 — rehearsal (1 hour)

Run the full demo twice, on the presentation machine, with a stopwatch. Then read §7.

---

## 6. Testing

Three files, ~14 tests, ~120 lines.

### `tests/test_similarity.py` — no model, runs in milliseconds

| Test | Asserts |
|---|---|
| `test_cosine_self` | `cosine(v, v) == 1.0 ± 1e-6` |
| `test_cosine_orthogonal` | orthogonal vectors → `0.0` |
| `test_cosine_opposite` | `cosine(v, -v) == -1.0` |
| `test_cosine_zero_vector` | no `nan`, returns `0.0` |
| `test_top_k_order` | descending order |
| `test_top_k_larger_than_input` | returns all, sorted, no error |
| `test_top_k_ties` | returns exactly `k` items |

### `tests/test_embedder.py` — needs a model; skipped if unavailable

```python
pytestmark = pytest.mark.model

@pytest.fixture(scope="session")
def model():
    try:
        return DomainEmbedder.load("sentence-transformers/all-MiniLM-L6-v2",
                                   local_files_only=True)
    except Exception:
        pytest.skip("all-MiniLM-L6-v2 not in the local HF cache")
```

Why `local_files_only=True`: a test suite that downloads 90 MB on every run is a test suite
people stop running. The model is already in the cache from the research pipeline, so this works
offline and takes ~10 seconds.

| Test | Asserts |
|---|---|
| `test_model_loading` | loads; `dim == 384`; `device` is a non-empty string |
| `test_single_encode` | shape `(384,)`, dtype `float32`, norm `1.0 ± 1e-5` |
| `test_batch_encode` | shape `(n, 384)`; rows match single encodes to `1e-5` |
| `test_similarity` | `similarity(t, t) == 1.0 ± 1e-5`; symmetry; a related pair scores above an unrelated pair |
| `test_search` | returns `min(top_k, n)` results, descending, each score equals `similarity(q, doc)` |

Plus the property tests from [`API_DESIGN.md`](API_DESIGN.md) §7:

| Test | Property |
|---|---|
| `test_batch_invariance` | `encode(["a","b"])[0] ≈ encode("a")` — catches padding bugs |
| `test_index_matches_search` | `index.search(q, 5)` equals `model.search(q, docs, 5)` |
| `test_empty_input_raises` | `EmptyInputError` on `""`, `"   "`, `[]` |
| `test_invalid_top_k_raises` | `InvalidInputError` on `0` and `-1` |
| `test_missing_model_raises` | `ModelNotFoundError` with a readable message |
| `test_info_without_metadata` | works on a model with no `model_info.json` |

### `tests/test_cli.py` — subprocess, no model

`--help` exits 0 for every subcommand; `search` with `--top-k 0` exits non-zero; `info` on a
missing model exits 1 with the message on stderr.

### What is deliberately not tested

The *quality* of the embeddings. That is the research evaluation's job, and duplicating
Recall@10 inside the library's test suite would create two sources of truth for one number.

---

## 7. Definition of done

Not "it runs" — these statements, each checkable. The first four are the ones that matter:

1. **The Streamlit demo answers a typed query in under 2 seconds on the presentation machine**,
   with the gold duplicate starred and its rank shown per column.
2. **The same query run through ≥2 conditions produces visibly different rankings** — the
   research variable is on screen, not just asserted.
3. **With the TIE caption loaded, the app still runs and still says something true** — the demo
   does not depend on a winning result.
4. **A screenshot of the demo is above the fold in the README.**
5. `pip install -e .` works in a clean virtualenv.
6. `pytest` is green, and `pytest -m "not model"` finishes in under 5 seconds.
7. `domembed` is importable from any directory, not just the repo root.
8. `model.info()` prints the domain, the base model and the negative-pair strategy.
9. The evaluation numbers in the README match `results/main_table.md`.
10. A group member who did not write the code completes UC1–UC5 from the README alone.
11. The whole library plus demo is under 620 lines
    (`wc -l domembed/*.py demo/app.py`). If it says 1,400, something crept in.

**Stop condition** — if the research runs late, cut in this order:

```text
1. CLI (keep `info` only if anything)          ← cut first
2. benchmark module (use a pasted table)
3. examples/quickstart.py
4. property tests beyond the five named ones
5. DocumentIndex — ONLY if you also trim the corpus to ~2,000 docs
──────────────────── stop here ────────────────────
6. the aggregate panel      ← painful; it is both evidence and fallback
7. the gold ★ marking       ← do not; this is what makes the demo evidence
8. export_model.py          ← do not; this is the research connection
9. the Streamlit demo       ← do not; it is the centerpiece
```

Never cut `model_info.json`: provenance is the only part of the library that is not available
from `sentence-transformers` already. Full reasoning in [`DEMO.md`](DEMO.md) §8.

---

## 8. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| torch install breaks on the presentation laptop | medium | Rehearse on that exact machine in week 3, not the morning of |
| Demo stalls encoding ~14,000 documents | **high** | `DocumentIndex` + `st.cache_resource`; pre-build the index with `scripts/build_demo_index.py` |
| **Demo is built too late to be rehearsed** | **high** | Phase 1.5 builds the shell in week 2; week 4 is only "swap models, delete captions" |
| **Result is a tie or a loss and the demo only handles winning** | **real** | All four result-mode captions written in week 3 ([`DEMO.md`](DEMO.md) §1.2) |
| **Only one condition shipped, so Visual C is impossible** | medium | Export every condition in Step 3.2; three columns is the minimum for Visual C |
| Fine-tuned model turns out worse than the baseline | real | The library and demo both work regardless; the honest story is specified in the README's honesty rules |
| Scope creep ("let's add FAISS / a server / reranking") | **high** | §7 stop condition; the non-goals table in [`PRODUCT_SPEC.md`](PRODUCT_SPEC.md) §5 |
| Library steals time from the research | medium | Phases 0–2 need no GPU and no trained model; hard rule: **no library work before the research baselines are validated in week 2** |
| Model weights cannot be published (licence) | low | Ship as a local directory; `domembed` supports both |
| Streamlit version drift breaks `app.py` | low | Pin `streamlit>=1.30,<2`; screenshot the working demo as a fallback |
