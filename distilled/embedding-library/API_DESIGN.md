# API design — `domembed`

The complete public surface. If a name is not in this document, it is private (prefixed with `_`)
and may change without notice.

Design rule applied throughout: **three verbs, plain types, no configuration required.**
Everything else — device selection, normalisation, batching, truncation — has a default that is
right for the target user and is overridable by the one who cares.

---

## 1. The public surface

```python
from domembed import (
    DomainEmbedder,   # the only class most users touch
    SearchResult,     # named tuple returned by search()
    ModelInfo,        # dataclass returned by info()
    DocumentIndex,    # optional: encode a corpus once, query it many times
    benchmark,        # timing helper
    __version__,
)
```

Five names and a version string. That is the whole library.

> ### The demo-critical subset
>
> The demo is the presentation centerpiece, so it defines what the API must provide. Everything
> below exists to serve these five calls — if a method is not reachable from this snippet, it is
> support, not product:
>
> ```python
> model   = DomainEmbedder.load(path_or_hub_id)      # must accept the generic baseline too
> vec     = model.encode("how do I configure ssh keys?")   # shown once, in passing
> score   = model.similarity(q, "Generate ssh key pair…")  # shown once, in passing
> results = model.search(q, corpus, top_k=10)              # ─ or index.search(q, 10)
> info    = model.info()                                   # fills the provenance panel
> ```
>
> | Call | Demo role | Cuttable? |
> |---|---|---|
> | `load` (both source forms) | Runs both models through one code path — the comparison's honesty | **No** |
> | `encode` | Establishes "three lines" in 15 seconds | No (it is 20 lines) |
> | `similarity` | Same | No (it is 10 lines) |
> | `search` / `index().search()` | The columns | **No** |
> | `info` | The provenance panel; where the negative-pair strategy appears | **No** |
> | `benchmark` | The performance panel | Yes — a pasted table works |
> | CLI | Cachet | Yes |
>
> **Smallest implementation that supports the demo: ~300 lines of library** (`embedder.py` ~170,
> `similarity.py` ~45, `metadata.py` ~60, `errors.py` ~25) **plus ~120 lines of `demo/app.py`.**
> The other ~170 lines (CLI, benchmark) are the first things cut. See [`DEMO.md`](DEMO.md) §4 and
> §8.

---

## 2. `DomainEmbedder`

### 2.1 `DomainEmbedder.load()` — constructor

```python
@classmethod
def load(
    cls,
    source: str | os.PathLike,
    device: str | None = None,
    *,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> "DomainEmbedder"
```

| Parameter | Default | Semantics |
|---|---|---|
| `source` | required | A local directory **or** a Hugging Face model id (`"org/name"`, `"all-MiniLM-L6-v2"`). Detection: if `Path(source).exists()` treat as local; otherwise pass through to `SentenceTransformer`, which resolves it on the Hub. |
| `device` | `None` | `None` → auto: `cuda` if available, else `mps` on Apple Silicon, else `cpu`. Explicit values: `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`. |
| `trust_remote_code` | `False` | Passed through. Left off by default. |
| `local_files_only` | `False` | `True` → never hit the network. Used by the test suite and for offline demos. |
| `revision` | `None` | Pin a git revision on the Hub. Pin it before recording any demo. |
| `cache_dir` | `None` | Override the Hugging Face cache location. |

**Why `load()` and not `__init__` or `from_pretrained()`:**

* `__init__` implies you can construct one cheaply and configure it. You cannot — it downloads
  90 MB and spins up torch.
* `from_pretrained()` is the Hugging Face convention, and it is the right call **if** you are a
  Hugging Face library. We are a product wrapper whose users may not know what "pretrained"
  means. `load()` reads correctly to someone who has never fine-tuned anything.
* `load()` also leaves room for `load_many()` later without breaking anything.

```python
model = DomainEmbedder.load("your-name/domembed-askubuntu")   # Hub
model = DomainEmbedder.load("./runs/quad_agree_hard/final")   # local checkpoint
model = DomainEmbedder.load("sentence-transformers/all-MiniLM-L6-v2")  # the baseline
```

**We do not implement downloading.** `SentenceTransformer(source)` already accepts a local path
or a Hub id and handles caching for both. `load()` adds three things on top: device
auto-detection, a friendly error when neither resolution works, and reading `model_info.json`.

### 2.2 `encode()`

```python
def encode(
    self,
    texts: str | Sequence[str],
    *,
    batch_size: int = 32,
    normalize: bool = True,
    show_progress: bool = False,
) -> np.ndarray
```

| Aspect | Behaviour |
|---|---|
| `str` input | Returns shape `(dim,)`, e.g. `(384,)`. |
| `Sequence[str]` input | Returns shape `(n, dim)`. |
| Empty list | Raises `EmptyInputError`. |
| Empty / whitespace-only string | Raises `EmptyInputError`. Encodes to a valid but meaningless vector, so failing loudly is correct. |
| Non-string element | Raises `TypeError` naming the offending index. |
| `normalize=True` | L2-normalise. **Default** — the model was trained with normalised embeddings and evaluated with cosine, so anything else is a silent mismatch with the research. |
| dtype | Always `float32`. |
| Text longer than `max_seq_length` | Truncated by the tokenizer, and a `logger.warning` fires **once per call** saying how many inputs were affected and what the limit is. |
| Batching | Handled by `sentence-transformers`. Batching is an implementation detail that must not change the result — asserted by a property test. |

**On `encode_batch()`:** the brief asks whether it is needed. **It is not.** A method whose only
difference is accepting a list creates two names for one operation and forces users to decide
which to call. `encode()` is polymorphic: one string in, one vector out; list in, matrix out.
That is the pattern `sentence-transformers`, `fastembed` and `model2vec` all converge on, and
copying it means users who already know one of those libraries already know ours.

### 2.3 `similarity()`

```python
@overload
def similarity(self, a: str, b: str) -> float: ...

@overload
def similarity(self, a: Sequence[str], b: Sequence[str]) -> np.ndarray: ...

def similarity(self, a, b) -> float | np.ndarray
```

* Two strings → `float` in `[-1, 1]`.
* Two equal-length sequences → `np.ndarray` of shape `(n,)`, compared **element-wise**
  (`a[i]` against `b[i]`), not as a cross product.
* Mismatched lengths → `ValueError` with both lengths in the message.
* Empty input → `EmptyInputError`.
* Always cosine similarity, computed with norms taken defensively (so it is correct even if
  someone passes `normalize=False` embeddings through a lower-level path).

No `metric=` parameter. Cosine is what the model was trained and evaluated with; offering dot
product or Euclidean would invite a comparison the research does not support.

### 2.4 `search()`

```python
def search(
    self,
    query: str,
    documents: Sequence[str],
    top_k: int = 5,
) -> list[SearchResult]
```

| Aspect | Behaviour |
|---|---|
| Return | `list[SearchResult]`, length `min(top_k, len(documents))`, sorted by descending score. |
| `top_k` | Must be a positive integer. `top_k > len(documents)` is **not** an error — returns all documents, sorted. `top_k <= 0` raises `ValueError`. |
| Empty `documents` | Raises `EmptyInputError`. Returning `[]` would hide a caller bug. |
| Duplicate documents | Both are returned, with their own indices. No deduplication — dedup is a policy decision and policies belong to the caller. |
| Cost | Encodes the query **and every document** on each call. O(n) forward passes. |
| Large `documents` | If `len(documents) > 5000`, `logger.info` suggests `model.index()`. A hint, not an error — and not an exception, since a slow correct answer beats a fast failure. |

`search()` does **not** exclude the query itself from the results. If the query string is in the
documents, it will be returned with score ≈ 1.0. This is the right default for a library (a
search function that silently hides rows is a debugging nightmare), and it is deliberately
different from the *evaluation* code, which masks the query's own id because there it would
inflate every metric to 100%. Documented explicitly, because the two behaviours opposite each
other is exactly the kind of thing that looks like a bug later.

### 2.5 `index()`

```python
def index(self, documents: Sequence[str], *, batch_size: int = 256,
          show_progress: bool = False) -> DocumentIndex
```

Encodes the corpus once and returns an object that answers queries by encoding only the query.

```python
index = model.index(corpus)                    # ~14K docs, one pass
results = index.search("printer not detected", top_k=5)
```

| Aspect | Behaviour |
|---|---|
| `len(index)` | Number of documents. |
| `index.search(query, top_k=5)` | Same return type as `DomainEmbedder.search()`. |
| Cost per query | One forward pass for the query + one matrix multiply. |
| Mutability | Immutable. Build a new one to change the corpus. |

Why this exists when `search()` already works: encoding ~14,000 documents takes tens of seconds
on a laptop CPU. A live demo that stalls for a minute is not a demo. `index()` is the difference
between "works" and "works on stage".

### 2.6 `info()`

```python
def info(self) -> ModelInfo
```

Returns a `ModelInfo` dataclass whose `__str__` renders the block shown in the README, and whose
`to_dict()` is JSON-serialisable for `--json` output.

| Field | Source | Fallback if unknown |
|---|---|---|
| `name` | `model_info.json` → else directory name | directory name |
| `base_model` | `model_info.json` | `"Not reported"` |
| `domain` | `model_info.json` | `"Not reported"` |
| `dim` | measured from the model (real) | — |
| `max_seq_length` | measured from the model (real) | — |
| `device` | measured at runtime | — |
| `objective` | `model_info.json` | `"Not reported"` |
| `negative_strategy` | `model_info.json` | `"Not reported"` |
| `training_config` | `model_info.json` | `{}` |
| `evaluation` | `model_info.json` | `None` → prints `Not evaluated` |
| `provenance` | `model_info.json` | `str(source)` |

**Never fabricate.** `"Not reported"` is a legitimate value for every metadata field, and the
rule from the research brief carries over into the product: if it was not measured, say so.

### 2.7 Properties

| Property | Type | Notes |
|---|---|---|
| `model.dim` | `int` | 384 |
| `model.device` | `str` | `"cpu"`, `"cuda:0"`, … |
| `model.max_seq_length` | `int` | 256 |
| `model.model_id` | `str` | Whatever was passed to `load()` |
| `model.base` | `SentenceTransformer` | Escape hatch. Documented as: *you should not need this; it is here so the library is never a dead end.* |

---

## 3. `SearchResult`

```python
class SearchResult(NamedTuple):
    text: str
    score: float
    index: int
```

A `NamedTuple`, not a dict and not a dataclass, because it gives three things for one line of
code:

```python
r.text                  # attribute access
text, score, idx = r     # tuple unpacking
r._asdict()             # {"text": ..., "score": ..., "index": ...} — JSON-ready
```

Sorting is by score descending. Ties keep the original document order (stable sort).

---

## 4. `DocumentIndex`

```python
class DocumentIndex:
    def __len__(self) -> int
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]
```

Holds `texts: list[str]` and `vectors: np.ndarray` of shape `(n, dim)`. Nothing else. No
persistence methods in v0.1 — if you want to save it, `np.save` the matrix and `json.dump` the
texts; that is two lines and teaching the library to do it buys nothing.

---

## 5. `benchmark()`

```python
def benchmark(
    model: DomainEmbedder,
    *,
    texts: Sequence[str] | None = None,
    batch_size: int = 32,
    warmup: int = 3,
    repeats: int = 20,
) -> dict
```

Measures and returns:

```python
{
    "device": "cpu",
    "dim": 384,
    "single_latency_ms": ...,          # mean over `repeats` single-string encodes
    "batch_32_latency_ms": ...,
    "throughput_texts_per_sec": ...,
    "peak_rss_mb": ...,                # via resource / psutil if available, else None
}
```

Does **not** measure model load time — by the time you hold a `DomainEmbedder`, loading has
already happened. Load time is measured by the CLI (`domembed bench --include-load`) with a
`time.perf_counter()` around `load()`.

`psutil` is not a dependency; if it is absent, `peak_rss_mb` is `None` and the report says so
rather than printing a misleading zero.

---

## 6. Errors

Five exception types, all subclassing `DomembedError`, which subclasses `Exception`.

| Exception | Raised when | Message tells the user |
|---|---|---|
| `DomembedError` | never directly | base class |
| `ModelNotFoundError` | `load()` cannot resolve `source` as a local path **or** a Hub id | both things it tried, and the two valid forms |
| `EmptyInputError` | empty list, empty string, whitespace-only string, empty `documents` | which argument was empty |
| `InvalidInputError` | non-string element in a list, `top_k <= 0`, mismatched lengths | the offending value and its index |
| `MetadataError` | `model_info.json` exists but is malformed | the path and the field that failed to parse |

**A missing `model_info.json` is not an error.** It is the normal case for any model we did not
train ourselves — and since loading the generic baseline must work, it *has* to be the normal
case. `MetadataError` is only for a file that exists and is broken.

---

## 7. Edge-case matrix

Every row is a test.

| Input | Behaviour |
|---|---|
| `encode("text")` | `(384,)` float32, norm ≈ 1.0 |
| `encode(["a", "b"])` | `(2, 384)` |
| `encode([])` | `EmptyInputError` |
| `encode("")` | `EmptyInputError` |
| `encode("   ")` | `EmptyInputError` |
| `encode(["a", 5])` | `TypeError` naming index 1 |
| `encode(text_500_tokens)` | truncated + one warning |
| `encode(["a"]) == encode("a")` | equal within `1e-6` |
| `similarity(t, t)` | `1.0` within `1e-5` |
| `similarity(a, b) == similarity(b, a)` | equal within `1e-5` |
| `similarity(["a","b"], ["c"])` | `InvalidInputError` (lengths 2 vs 1) |
| `search(q, docs, top_k=5)` where `len(docs)=3` | 3 results |
| `search(q, docs, top_k=0)` | `InvalidInputError` |
| `search(q, [], top_k=5)` | `EmptyInputError` |
| `search(q, docs)` where `q in docs` | the query is returned at score ≈ 1.0 |
| `search` scores | equal to `similarity(q, doc)` within `1e-5` |
| `search` ordering | monotonically non-increasing |
| `load("./nonexistent")` | `ModelNotFoundError` |
| `load("org/does-not-exist-xyz")` | `ModelNotFoundError` |
| `info()` on a model with no `model_info.json` | works; metadata fields read `Not reported` |

---

## 8. Type hints and typing policy

* Full annotations on every public function. `from __future__ import annotations` at the top of
  every module so `str | None` works on Python 3.9.
* `Sequence[str]` rather than `list[str]` for inputs — accept tuples, numpy arrays of strings,
  whatever the caller has.
* `np.ndarray` for outputs, never `Any`.
* `float | np.ndarray` overloads on `similarity()` only. Everywhere else the return type is
  unconditional.
* No `typing.Protocol`, no generics, no `TypeVar`. If the type annotations start needing a
  paragraph of explanation, the API is too clever.
* `py.typed` marker included so consumers get hints too.

---

## 9. Return-type policy

| Question | Decision | Reason |
|---|---|---|
| numpy or torch? | **numpy, always.** | Users who know `numpy` vastly outnumber users who know torch; numpy prints better; numpy does not carry a device the user has to reason about. |
| `float32` or `float64`? | `float32`. | Matches the model. `float64` would double memory for no accuracy the model has. |
| `(384,)` or `(1, 384)`? | `(384,)` for a single string. | Matches `sentence-transformers`. Users expect `encode(one)` to give one vector. |
| `list[SearchResult]` or `list[dict]`? | `SearchResult` named tuples. | Attribute access reads better in demos, `r._asdict()` gives the dict when needed. |
| Tensor option? | Not in v0.1. | Revisit only if a real use case appears. |

---

## 10. Alternatives considered and rejected

| Alternative | Why rejected |
|---|---|
| Functional API: `domembed.embed(model, text)` | Module-level functions with a model argument is the `scikit-learn` convention, and it is right there because the model is a fitted estimator. Here the model *is* the product. Methods on the object read better and make the model's lifetime explicit. |
| `Embedder.from_pretrained()` | Correct Hugging Face convention, but `load()` is clearer for someone who has never fine-tuned anything. Not worth the extra syllable. |
| Separate `encode_batch()` | Two names for one operation. Rejected in §2.2. |
| `search()` returning `(scores, indices)` tuples | Scores without text forces the caller to index back into their own list — one more thing to get wrong in a demo. |
| `rank()` instead of `search()` | `rank` is the `sentence-transformers` `CrossEncoder` name and implies reranking. `search` is what a user calls it. |
| `metric=` parameter on `similarity()` | Offering dot product / Euclidean invites a comparison the research does not support. |
| Async API | No async in the underlying library. Pretending otherwise would be a lie. |
| Context manager (`with DomainEmbedder.load(...) as m:`) | Tempting for GPU cleanup, but it makes every example three lines longer and the gain is one `torch.cuda.empty_cache()` nobody in the target audience needs. |
| Global default model (`domembed.set_default_model(...)`) | Hidden global state. The one thing guaranteed to produce a confusing bug report. |

---

## 11. Stability promise

v0.1 is `0.1.x`. Within `0.1.x`: no public name is removed or repurposed; new keyword arguments
may be added with defaults; the numpy return type and the `SearchResult` field order are fixed.
Field *additions* to `ModelInfo` are allowed. Anything not in this document may change.
