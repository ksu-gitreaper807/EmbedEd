# domembed

**Domain-specific sentence embeddings, in three lines of Python.**

```python
from domembed import DomainEmbedder

model = DomainEmbedder.load("your-name/domembed-askubuntu")
results = model.search("how do I configure ssh keys?", documents, top_k=5)
```

---

## The demo

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│  domembed · AskUbuntu duplicate-question search                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  Query  [ how do I configure ssh keys?                              ]  [🔍]  │
│  Top-k  ────●────────  5      ☑ compare with all-MiniLM-L6-v2 (generic)      │
│  Conditions  ☑ N1 random   ☑ N2 lexical   ☑ N3 model-hard                    │
├──────────────────────────────────────────────────────────────────────────────┤
│   ★ = a duplicate that the AskUbuntu community actually marked               │
│                                                                              │
│   all-MiniLM-L6-v2 (generic)      domembed · N1 random    domembed · N3 hard │
│   ──────────────────────────      ─────────────────────   ─────────────────  │
│   ① 0.79  Setting up ssh         ① 0.83  Setting up ssh  ① ★ 0.88  Generate  │
│   ② 0.74  Enable remote desktop  ② 0.80  Passwordless    ② 0.85  Setting up  │
│   ③ 0.71  What is ssh?           ③ ★ 0.77  Generate ssh  ③ 0.82  Passwordless│
│   ④ 0.70  Installing openssh     ④ 0.71  What is ssh?    ④ 0.74  What is ssh?│
│   ⑤ ★ 0.68  Generate ssh key     ⑤ 0.69  Enable remote   ⑤ 0.70  Enable      │
│                                                                              │
│   Known duplicate "Generate ssh key pair for github"                         │
│        generic → rank 5        N1 → rank 3        N3 → rank 1                │
│                              ▲ +4 ranks, from changing the negatives         │
└──────────────────────────────────────────────────────────────────────────────┘
```

*Layout mockup — the real screenshot goes here after rehearsal. Scores are placeholders.*

Three columns. Identical model, identical data, identical loss, identical number of steps. **The
only thing that differs is which pairs counted as negatives** — and the duplicate the community
marked moves from rank 5 to rank 1.

```bash
pip install -e ".[demo]"
streamlit run demo/app.py
```

The demo searches the **real evaluation corpus with its gold labels**, so "rank 1" is not a
vibe — it is the same quantity `Recall@10` is computed from. See [`DEMO.md`](DEMO.md).

---

## The result

> **Not yet measured.** This section is filled from `results/main_table.md` once the runs are
> done. It is never filled by hand, and it is never filled with a claim the numbers do not
> support.

| System | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|
| BM25 | **?** | **?** | **?** |
| `all-MiniLM-L6-v2` (no fine-tuning) | **?** | **?** | **?** |
| `domembed` · N1 random negatives | **?** | **?** | **?** |
| `domembed` · N2 lexical negatives | **?** | **?** | **?** |
| `domembed` · N3 model-hard negatives | **?** | **?** | **?** |

200 held-out test queries, mean of 3 seeds, 95% bootstrap intervals reported in
[`results/contrast_table.md`](../../project/distilled/PROJECT_SPEC.md).

**One query in the demo is an anecdote. This table is the evidence.** If the two disagree, the
table is what we believe, and the README says so.

---

## Installation

```bash
git clone <this-repo>
cd domembed
pip install -e .
```

Runtime dependencies: `sentence-transformers`, `numpy`, `torch` (pulled in by
`sentence-transformers`). That is it — no vector database, no serving stack, no ONNX toolchain.

```bash
pip install -e ".[demo]"   # adds streamlit, for the demo app
pip install -e ".[dev]"    # adds pytest + ruff
```

> **We do not publish to PyPI as part of the project.** `pip install -e .` is the supported
> install. Publishing is a ~30-minute after-the-fact step described in
> [`ROADMAP.md`](ROADMAP.md), not a deliverable.

---

## Quick start

```python
from domembed import DomainEmbedder

model = DomainEmbedder.load("your-name/domembed-askubuntu")

# 1. embed
vec = model.encode("How do I configure SSH key authentication?")
print(vec.shape, vec.dtype)
# (384,) float32

# 2. compare
score = model.similarity(
    "How do I configure SSH key authentication?",
    "Setting up ssh public/private keys for login",
)
print(f"{score:.3f}")
# 0.812   ← illustrative output shape; the value depends on the model

# 3. search
documents = [
    "Setting up ssh public/private keys for login",
    "How do I change my screen resolution from the terminal?",
    "Passwordless ssh login between two machines",
    "Installing nvidia drivers on a fresh install",
]
for r in model.search("how do I configure ssh keys?", documents, top_k=3):
    print(f"{r.score:.3f}  {r.text}")

# 0.847  Setting up ssh public/private keys for login
# 0.791  Passwordless ssh login between two machines
# 0.203  Installing nvidia drivers on a fresh install
#        ↑ illustrative ordering; real values depend on the model
```

`search()` returns `SearchResult` named tuples, so all of these work:

```python
r = model.search(q, documents, top_k=1)[0]
r.text, r.score, r.index        # attribute access
text, score, index = r          # tuple unpacking
[r._asdict() for r in results]  # JSON-ready list of dicts
```

---

## API

| Call | Returns |
|---|---|
| `DomainEmbedder.load(source, device=None)` | A `DomainEmbedder`. `source` is a local directory **or** a Hugging Face id. |
| `.encode(text_or_texts)` | `np.ndarray` — `(384,)` for one string, `(n, 384)` for a list. `float32`, L2-normalised. |
| `.similarity(a, b)` | `float` for two strings; `np.ndarray` for two equal-length lists. |
| `.search(query, documents, top_k=5)` | `list[SearchResult]`, descending by score. |
| `.index(documents)` | A `DocumentIndex` — encode once, query many times. |
| `.info()` | `ModelInfo` — provenance and measured evaluation. |

Five methods is the whole product surface. Full signatures, edge cases and error types in
[`API_DESIGN.md`](API_DESIGN.md).

---

## Model provenance

```python
print(model.info())
```

```text
Model:       domembed-askubuntu-n3-model-hard
Base model:  sentence-transformers/all-MiniLM-L6-v2
Domain:      AskUbuntu — Ubuntu/Linux technical support questions
Dimension:   384    Max seq: 256 tokens
Device:      cpu

Trained with
  Objective:  MultipleNegativesRankingLoss (in-batch InfoNCE, scale 20)
  Negatives:  top-64 by cosine of the zero-shot encoder        ← the research variable
  Data:       ~19,000 duplicate pairs, AskUbuntu train split
  Schedule:   2 epochs, batch 32, lr 2e-5, AdamW, seed 13

Evaluation (200 test queries, Recall@10)
  This model:  0.???  [filled from results/main_table.md — not typed by hand]
  Baseline:    0.???  (all-MiniLM-L6-v2, no fine-tuning)

Provenance:  runs/n3_model_hard/seed13/final
```

The evaluation block is **generated, never hand-written.** `scripts/export_model.py` reads the
real numbers out of `results/metrics.json`. If no evaluation has been run, it prints
`Not evaluated` — which is the correct thing to print rather than a flattering guess.

---

## Research connection

This library exists because of one research question:

> When contrastively fine-tuning a small pretrained encoder for duplicate-question retrieval,
> does **how the negative pairs are constructed** determine whether fine-tuning helps?

| Research artefact | Product artefact |
|---|---|
| `runs/<condition>/<seed>/final` (a saved `SentenceTransformer`) | loadable by `DomainEmbedder.load(path)` with no conversion |
| `negatives.py` — the negative-pair strategy | recorded in `model_info.json`, printed by `info()` |
| `results/metrics.json` — Recall@10 / MRR@10 / nDCG@10 | copied into `model_info.json` by the export script |
| The generic baseline `all-MiniLM-L6-v2` | loadable through the identical API, for the comparison |

**We ship every condition, not just the winner.** Whether the research ends up with three
hardness conditions (random / lexical / model-hard) or four disagreement quadrants, each is a
separate model directory that `domembed` can load. The demo puts them side by side: *same
dataset, same loss, same batch size, same number of steps — only the negatives differ, and the
search results change.*

### Honesty rules for this README

These are requirements, not aspirations:

1. **No claim without a measured number.** The results table is filled from
   `results/main_table.md`. Until it is measured, it says *not yet measured*.
2. **If the fine-tuned model is not better, the README says so.** "We fine-tuned a domain
   embedding model and packaged it as a library" is a true and complete contribution even when
   Recall@10 does not improve. "Our model is better" is not.
3. **The demo is an anecdote; the table is the evidence.** A demo query is one example. The
   README must not generalise from it.
4. **Which model ships is decided by the evaluation, in advance.** If two conditions are within
   the bootstrap interval of each other, ship the simpler one and say the choice was a tie.

---

## Performance

Measured with `domembed bench` on the presentation machine — see
[`DEMO.md`](DEMO.md) for the exact procedure. Template until then:

| Operation | Measured | Notes |
|---|---|---|
| Model load (cached, CPU) | *to be measured* | |
| `encode` single text | *to be measured* | |
| `encode` batch of 32 | *to be measured* | |
| `index()` over ~14,000 documents | *to be measured* | one-time cost |
| `index.search()` per query | *to be measured* | what the demo actually feels |
| Peak RSS | *to be measured* | dominated by torch, not by us |

---

## Limitations

Stated plainly, because a product that hides its limits is harder to defend than one that names
them:

* **One domain.** Trained on AskUbuntu. It will not improve text from other domains, and may
  make it worse. Do not use it as a general embedding model.
* **`search()` re-encodes the documents every call.** Fine for hundreds of documents; use
  `index()` for thousands. `search()` logs a hint when you pass more than 5,000.
* **No vector database.** There is no persistence, no incremental updates, no metadata filtering
  and no ANN index. The corpus is ~14,000 documents and a single matrix multiply; a database
  would be pure overhead.
* **Not a serving stack.** No HTTP server, no batching queue, no GPU sharing.
* **Not thread-safe for concurrent `encode()` calls.** The underlying
  `sentence-transformers` model is not documented as thread-safe.

---

## Problem and approach

A general-purpose embedding model (`all-MiniLM-L6-v2`) is trained to make *English sentences that
mean the same thing* end up near each other. In a technical support forum, "same thing" is
stricter and weirder than that:

| Query | Generic model thinks | The domain truth |
|---|---|---|
| `how to install vlc on 12.04` | ≈ any VLC question | version-specific — a different answer |
| `can't mount ntfs partition` | ≈ any disk question | often a genuine duplicate of `ntfs drive won't mount` |
| `ssh key authentication setup` | ≈ security questions in general | a specific, recurring question |

Generic embeddings blur these distinctions because nothing in their training said they matter.
Contrastive fine-tuning on naturally occurring duplicate pairs can — **if** the negative pairs
are chosen well, which is what the research half of this project measures.

```text
AskUbuntu duplicate pairs
        ↓
contrastive fine-tuning (MultipleNegativesRankingLoss)
        ↓
        └── negative-pair strategy ← the research variable
        ↓
fine-tuned 384-d encoder
        ↓
domembed  →  encode()  →  similarity()  →  search()
```

## Features

| | Capability | Call |
|---|---|---|
| ✅ | Encode one string or a list | `model.encode(text_or_texts) -> np.ndarray` |
| ✅ | Cosine similarity between texts | `model.similarity(a, b) -> float` |
| ✅ | Semantic search over a document list | `model.search(query, documents, top_k=5)` |
| ✅ | Reusable index for larger corpora | `index = model.index(documents); index.search(q, k)` |
| ✅ | Model provenance printed cleanly | `model.info()` |
| ✅ | **Streamlit demo with the multi-model comparison** | `streamlit run demo/app.py` |
| ✅ | Command-line interface | `domembed search "..." docs.txt --top-k 5` |
| ✅ | Latency / throughput measurement | `domembed bench` |
| ✅ | Loads **any** model, not just ours | `DomainEmbedder.load("all-MiniLM-L6-v2")` |

That last row is deliberate and matters for the presentation: because `domembed` can load the
generic baseline too, the before/after comparison runs through **one code path**, with no
opportunity for the demo to be accidentally rigged.

## Command-line interface

```bash
domembed info "your-name/domembed-askubuntu"

domembed encode "how do I configure ssh keys?" --model ./my-model

domembed similarity "how do I install vlc" "installing vlc media player"

domembed search "how do I configure ssh keys?" questions.txt --top-k 5

domembed bench --model ./my-model --batch-size 32
```

`search` accepts `--json` for machine-readable output and `-` for stdin. Five commands, argparse
only, no CLI framework dependency.

## Architecture

```text
                   embedder.py ─── the only class users touch
                       │
        ┌──────────────┼──────────────┬──────────────┐
        ↓              ↓              ↓              ↓
  similarity.py   metadata.py     errors.py    benchmark.py
   cosine, top-k   ModelInfo      exceptions    timing
        │
        └── sentence-transformers  (loading, tokenising, pooling, device)
                └── transformers → torch
```

Seven small modules, ~430 lines total; ~120 more for the demo. Full rationale in
[`ARCHITECTURE.md`](ARCHITECTURE.md); full signatures in [`API_DESIGN.md`](API_DESIGN.md).

> **Status: specification.** The library is written in week 3–4, against the base model first and
> the fine-tuned weights second. See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

---

## Licence and attribution

Library code: **Apache 2.0** (matching `sentence-transformers` and `all-MiniLM-L6-v2`).

Before publishing model weights, check the AskUbuntu/Stack Exchange content licence (CC BY-SA).
If that is unclear, keep the model in a private Hugging Face repository or ship it as a local
directory only — `domembed` supports both, so nothing in the product breaks.

## Acknowledgements

Built on [`sentence-transformers`](https://www.sbert.net/) and
[`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2); trained on
the AskUbuntu duplicate-question corpus of
[Lei et al. (NAACL 2016)](https://aclanthology.org/N16-1153/).
