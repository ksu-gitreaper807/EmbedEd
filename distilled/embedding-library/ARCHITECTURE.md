# Architecture — `domembed`

Seven modules, ~430 lines, one external dependency that matters.

---

## 1. The stack

```text
  ┌───────────────────────────────────────────────────────────────┐
  │  user code / CLI / Streamlit demo                             │
  └───────────────────────────┬───────────────────────────────────┘
                              │  encode · similarity · search · info
  ┌───────────────────────────▼───────────────────────────────────┐
  │  embedder.py        DomainEmbedder + DocumentIndex            │  ~170 lines
  │  ─────────────────────────────────────────────────────────    │
  │  • resolves local path vs Hub id                              │
  │  • owns the SentenceTransformer instance                      │
  │  • orchestrates: text → vectors → scores → results            │
  └───┬──────────────┬──────────────────┬─────────────────┬───────┘
      │              │                  │                 │
  ┌───▼────┐  ┌──────▼──────┐  ┌────────▼───────┐  ┌──────▼──────┐
  │similarity│ │ metadata.py │  │   errors.py    │  │benchmark.py │
  │  .py     │ │  ModelInfo  │  │  5 exceptions  │  │   timing    │
  │          │ │  load/save  │  │                │  │             │
  │ cosine   │ │ model_info  │  │                │  │ latency,    │
  │ top_k    │ │   .json     │  │                │  │ throughput, │
  │          │ │             │  │                │  │ peak RSS    │
  │ ~45 ln   │ │  ~60 lines  │  │   ~25 lines    │  │  ~60 lines  │
  └──────────┘ └─────────────┘  └────────────────┘  └─────────────┘
        │
  ┌─────▼─────────────────────────────────────────────────────────┐
  │  sentence-transformers   loading, tokenising, pooling, device │
  │  ├── transformers        BertModel forward pass               │
  │  └── torch               tensors, autograd (unused at infer)  │
  └───────────────────────────────────────────────────────────────┘
```

Plus two **consumers**, both of which talk only to the public API and neither of which the
library knows about:

| Consumer | ~Lines | Status |
|---|---|---|
| `cli.py` | 90 | Sits beside `embedder.py`; argparse and formatting only. **Cuttable.** |
| `demo/app.py` | 120 | The Streamlit app. **The presentation centerpiece — not cuttable.** |

The demo additionally reads `data/corpus.jsonl`, `data/qrels.json` and `results/metrics.json`
**directly from the research repo**. It does not go through `domembed` for those, because
corpora, gold labels and evaluation metrics are research concerns and must not leak into the
library. The library never learns that qrels exist.

---

## 2. Module responsibilities

| Module | Owns | Does **not** own |
|---|---|---|
| `__init__.py` | Public exports, `__version__`, `py.typed` | Any logic |
| `embedder.py` | `DomainEmbedder`, `DocumentIndex`, `SearchResult`. Resolving `source`; holding the `SentenceTransformer`; calling into `similarity` and `metadata`; raising typed errors. | Any tensor mathematics; any JSON parsing |
| `similarity.py` | `cosine(a, b)` and `top_k(scores, k)` — pure numpy, no model, no I/O. | Anything model-shaped |
| `metadata.py` | The `ModelInfo` dataclass and reading/writing `model_info.json`. | Printing policy (that is `ModelInfo.__str__`, but the *decision* of what to print lives in the spec) |
| `errors.py` | The five exception types. | — |
| `benchmark.py` | Timing and memory measurement. | Interpreting the numbers |
| `cli.py` | Argument parsing, output formatting, exit codes. | Any logic that is not available from the Python API |

The test that the split is right: **`similarity.py` has no imports from the rest of the
package.** It is two numpy functions. That makes the mathematical core testable in milliseconds
with no model download, which is most of the value of having a test suite at all.

---

## 3. Why this structure and not the one suggested in the brief

The brief proposed `model.py, similarity.py, search.py, models.py, io.py, cli.py`. Three of
those modules are dropped and one is renamed, for reasons that should be checkable.

| Proposed | Verdict | Reason |
|---|---|---|
| `model.py` | **Renamed** to `embedder.py` | `model.py` in a package that also has a *model on disk* and a `ModelInfo` type is three meanings for one word. `embedder.py` says what the class is. |
| `similarity.py` | **Kept** | The only pure-mathematics module. Testable without a model. |
| `search.py` | **Merged into `similarity.py`** | Top-k selection over a score vector is four lines of `argpartition`. A separate 30-line module for it splits one idea across two files and forces an import cycle decision (`search` needs `cosine`, `embedder` needs both). One module named `similarity` containing "compare vectors" and "pick the best ones" is honest about how small this is. |
| `models.py` | **Dropped** | A model registry makes sense when you ship several models and need aliases, versioning and download policies. We ship one model per domain and load them by name. Adding a registry now is building for a future that may not happen — and when it does happen, it will be `metadata.py` that grows. |
| `io.py` | **Dropped** | The only I/O in the library is reading one JSON file, which `metadata.py` does in eight lines. An `io.py` module implies a file-format abstraction layer. There is no layer. |
| `cli.py` | **Kept** | Genuinely a separate concern: argparse and formatting. |
| — | **`metadata.py` added** | Provenance is the entire reason this library exists rather than just using `sentence-transformers` directly. It deserves a home. |
| — | **`errors.py` added** | Five exception types need somewhere to live that is not the top of `embedder.py`. |
| — | **`benchmark.py` added** | The brief requires a performance section; keeping the timing loop out of the class keeps the class clean. |

---

## 4. Data flow

### 4.1 `encode`

```text
str | Sequence[str]
    ↓  embedder.encode: normalise the input shape to a list, validate types
    ↓  self._st.encode(list, batch_size, normalize_embeddings=True, convert_to_numpy=True)
    ↓   (transformers tokenises → BERT forward → mean pooling → L2 norm)
np.ndarray  (384,) or (n, 384)
```

The only logic we add is input validation and shape restoration.

### 4.2 `similarity`

```text
a, b
    ↓  embedder.encode([a, b])            one batch, two texts
    ↓  similarity.cosine(v_a, v_b)        defensive norms, then dot
float | np.ndarray
```

### 4.3 `search`

```text
query, documents, top_k
    ↓  embedder.encode([query] + list(documents))
    ↓  similarities = doc_vectors @ query_vector          one matmul
    ↓  idx = similarity.top_k(similarities, top_k)        argpartition, not argsort
    ↓  [SearchResult(text=documents[i], score=float(s), index=i) for i in idx]
list[SearchResult]
```

`argpartition` is `O(n)` rather than `O(n log n)`. At 14,000 documents this saves microseconds
and is therefore **not a performance feature** — it is used because it is the correct tool, and
it is mentioned here so nobody later "optimises" it into a full `argsort`.

### 4.4 `index` / `DocumentIndex.search`

```text
documents ──encode once──→ vectors (n, 384)   stored in the DocumentIndex
query     ──encode───────→ q (384,)
                            ↓  vectors @ q
                            ↓  top_k
                        list[SearchResult]
```

The only difference from `search()` is where the document vectors come from. Same
`similarity.top_k`, same `SearchResult`, same ordering guarantees — which is why it is one
method on the embedder returning a small object, not a parallel code path.

### 4.5 `info`

```text
load()
    ↓  metadata.read_model_info(source)   →  model_info.json if present, else defaults
    ↓  measure dim / max_seq_length from the live model
    ↓  ModelInfo(...)
print(model.info())
```

---

## 5. The export seam — where research meets product

**This is the most important diagram in the document.** It is the single point of contact
between the two halves of the project.

```text
  research repo                              product
  ─────────────                              ───────
  train.py
     ↓
  runs/<condition>/<seed>/final/     ← a saved SentenceTransformer, untouched
     │
     ├── config.json, model.safetensors, tokenizer…, modules.json
     │
     └── scripts/export_model.py  ──writes──▶ model_info.json
              reads:                              ├── name
              • the condition name                ├── base_model
              • negatives.py --strategy           ├── domain
              • training config (epochs, lr, …)   ├── objective
              • results/metrics.json              ├── negative_strategy   ← the research variable
                                                  ├── training_config
                                                  ├── evaluation          ← copied, never typed
                                                  └── provenance
                                                       ↓
                                          DomainEmbedder.load("./runs/.../final")
```

Rules for the seam:

1. **The research repo never imports `domembed`.** Training code has no knowledge of the
   product. If it did, a broken library would block a training run.
2. **`export_model.py` never edits the weights.** It only adds `model_info.json` (and optionally
   pushes to the Hub). The checkpoint stays a plain `SentenceTransformer` directory, which is
   what makes `DomainEmbedder.load(path)` work with zero conversion.
3. **Evaluation numbers are copied, never retyped.** `export_model.py` reads
   `results/metrics.json`. A human typing `0.83` into a JSON file is how a README ends up
   claiming something the experiment did not show.
4. **The seam runs one way.** Nothing flows back from the product into the research repo.

That is the entire integration surface: **one 40-line script and one JSON file.**

---

## 6. Dependency policy

| Dependency | Status | Reason |
|---|---|---|
| `sentence-transformers` | **required** | Loading, tokenising, pooling, device handling, Hub integration. This is ~90% of the heavy lifting and none of it should be reimplemented. |
| `numpy` | **required** | Vectors, cosine, top-k. Ships with `sentence-transformers`. |
| `torch` | **transitive** | Comes with `sentence-transformers`. Never imported directly by `domembed`. |
| `streamlit` | **extra** `[demo]` | Only the demo app. Keeps the core install lean. |
| `pytest`, `ruff` | **extra** `[dev]` | — |
| `psutil` | **optional at runtime** | Used by `benchmark()` if present; `peak_rss_mb` is `None` otherwise. |

**Everything else is refused.** No FAISS (a matmul over 14K vectors is milliseconds), no
`transformers` import (go through `sentence-transformers`), no `click`/`typer` (argparse is in
the stdlib and the CLI has five commands), no `pydantic` (a dataclass and a named tuple are
enough), no `scikit-learn` (we need cosine, not a modelling toolkit).

Cross-checking against the libraries surveyed for §9 of the brief:

| Existing library | What we reuse | What we deliberately do not |
|---|---|---|
| **sentence-transformers** | Nearly everything: loading, `encode`, batching, normalisation, device, Hub | Nothing |
| **transformers / torch** | Indirectly, through `sentence-transformers` | We never import them directly |
| **numpy** | Array maths | — |
| **fastembed / model2vec** | *Idea* only: that a focused, dependency-light embedding API is good design | Not used — they target ONNX/static models and would require converting our checkpoint. Possible v0.2. |
| **FAISS** | Nothing | Its value starts in the millions of vectors; ours is ~14,000 |
| **Chroma / Qdrant / LanceDB** | Nothing — but they define our non-goal | Persistence, filtering, incremental indexing. We are stateless by design. |
| **gensim** | Nothing | Static word vectors; a different generation of technology |
| **scikit-learn** | Nothing | `cosine_similarity` is one line; not worth a 30 MB dependency |

The honest summary: **`domembed` is a metadata-carrying convenience wrapper with good error
messages.** Its contribution is curation and provenance, not computation. The architecture is
designed so that this remains true and visible.

---

## 7. State and ownership

| Object | Owns | Lifetime |
|---|---|---|
| `DomainEmbedder` | one `SentenceTransformer`, one device, one `ModelInfo` | Created by `load()`; the model stays loaded. Loading is the expensive part and must happen once. |
| `DocumentIndex` | a `list[str]` and an `(n, dim)` array | Independent of the embedder after construction, but holds no reference to it — it cannot re-encode, so it can never silently go stale. |
| `SearchResult` | three immutable values | Value object; safe to share |
| `ModelInfo` | immutable dataclass | Snapshot at load time |

Nothing is cached globally. Nothing is lazily initialised behind a property. Every expensive
operation is a named method call, so a user reading the code can see exactly where the cost is.

---

## 8. Where the source lives

The specifications in this folder describe a package that will live in the research repository:

```text
EmbedEd/                              ← the research repo
├── src/
│   ├── prepare_data.py  negatives.py  train.py  evaluate.py  analyze.py  utils.py
├── runs/<condition>/<seed>/final/     ← trained models
├── scripts/export_model.py            ← the seam
├── domembed/                          ← the library (this spec)
│   ├── __init__.py  embedder.py  similarity.py
│   ├── metadata.py  errors.py  benchmark.py  cli.py
│   ├── py.typed
├── tests/
├── demo/app.py
├── examples/quickstart.py
├── pyproject.toml
└── README.md
```

One repository, one `pip install -e .`, one model directory. The library is a sibling of the
training code, not a separate project — which is exactly what keeps it from becoming a second
giant project.
