# Product specification — `domembed`

The smallest product that makes the research tangible, and the list of things it deliberately
does not do.

---

## 1. One-paragraph definition

`domembed` is a Python package that loads a domain-specific sentence embedding model and exposes
it through three operations — `encode`, `similarity`, `search` — plus model provenance
(`info()`), a command-line interface, and **a Streamlit demo that is the presentation
centerpiece**. It wraps `sentence-transformers` rather than replacing it, and its distinguishing
feature is that it carries the *research metadata* (domain, negative-pair strategy, training
config, measured evaluation) alongside the weights, and can load the generic baseline model
through the identical interface so the before/after comparison is honest by construction.

### Two domains

`domembed` now serves **two parallel research tracks**, and is specified to be neutral between
them:

| | AskUbuntu track | Code track |
|---|---|---|
| Domain | Ubuntu duplicate questions | Java code clone detection |
| Model | `all-MiniLM-L6-v2` (384-d) | `microsoft/graphcodebert-base` (768-d) |
| Decision rule | rank of the gold duplicate | cosine similarity against a threshold |
| Specification | [`project/distilled/`](../../project/distilled/) | [`project/code-clone/`](../../project/code-clone/) |

The contract below is written for both. Where they differ, the difference is noted inline and
collected in **[`CODE_TRACK.md`](CODE_TRACK.md)**.

### Priority doctrine

> **The library is the engineering deliverable. The demo is the presentation centerpiece.**
>
> They are not in competition, but when they conflict, the demo wins the tie. The library's
> purpose in this project is to make the demo *honest* — to prove the comparison runs through
> one code path and the numbers are generated rather than typed. The demo's purpose is to make
> the research land in ten seconds.
>
> Practical consequence: the demo is designed first, around the research result, and the library
> is the minimum that makes the demo defensible. See [`DEMO.md`](DEMO.md).

---

## 2. Target users

Three personas, in priority order. The library is built for the first and demoed to the second.

### Primary — "the developer with a pile of domain text"

A second-year student or junior developer who has a few thousand pieces of domain text
(forum questions, bug reports, product docs) and wants semantic search over them without
learning what a contrastive loss is.

* Can write Python, has used `pip`, has used `numpy`.
* Has **not** read a paper. Will not read the docs beyond the README quick-start.
* Will copy the first code block they see and expect it to run.
* Will pass a list of 20,000 documents to `search()` and wonder why it is slow.

**Design consequences:** three methods with obvious names; zero required configuration;
everything returns plain Python or numpy types; helpful errors instead of stack traces; a
`logger.info` hint when they do something quadratic.

### Secondary — the presentation audience

Examining the project as a piece of engineering. They will not install it. They need to see, in
under two minutes, that this is a *product* and not a notebook.

**Design consequences:** `info()` output that looks designed; a CLI that works live in a
terminal; a Streamlit app with two columns; a README that reads like a real package.

### Tertiary — the group members, in six months' time

Whoever picks this up next semester to add a second domain.

**Design consequences:** `model_info.json` is a documented schema, not an ad-hoc dict; the
export script is the single seam between research and product; no hidden state.

**Explicitly not a target user:** anyone running production retrieval at scale. That is what
Qdrant, Chroma and LanceDB are for, and competing with them is an explicit non-goal.

---

## 3. Use cases

Each of these must work at v0.1. They are ordered by how often they will actually be used.

### UC1 — Semantic search over a document list *(the demo)*

```python
model = DomainEmbedder.load("your-name/domembed-askubuntu")
for r in model.search("how do I configure ssh keys?", questions, top_k=5):
    print(f"{r.score:.3f}  {r.text}")
```

### UC2 — Is this question a duplicate of one we already have? *(the real task)*

```python
if model.similarity(new_question, existing_question) > 0.85:
    flag_as_possible_duplicate()
```

This is the task the model was trained for, and the one the evaluation measures. Worth saying
out loud during the demo.

### UC3 — Compare our model against the generic one *(the research connection)*

```python
ours   = DomainEmbedder.load("./runs/quad_agree_hard/final")
theirs = DomainEmbedder.load("sentence-transformers/all-MiniLM-L6-v2")

for name, m in [("ours", ours), ("generic", theirs)]:
    print(name, [r.text[:40] for r in m.search(query, corpus, top_k=3)])
```

Same API, same corpus, same code path. This is the honest version of the comparison and it is
why "load any model" is a MUST, not a nicety.

### UC4 — Embed a corpus once, query it many times

```python
index = model.index(corpus)          # ~14K documents, encoded once
index.search("printer not detected", top_k=5)
```

Required for the live demo to feel instant on the real corpus.

### UC5 — "What model is this?"

```python
print(model.info())
```

Used in the demo, in the report's reproducibility section, and by future-you.

### UC6 — **See the research result, live** *(the centerpiece)*

```python
results = {name: m.search(query, corpus, top_k=10) for name, m in models.items()}
gold_rank(results["generic"], qid)   # 5
gold_rank(results["n3_hard"], qid)   # 1
```

Where the duplicate that the AskUbuntu community marked sits, under each model, on the real
evaluation corpus. This is the use case every design decision serves. Full spec in
[`DEMO.md`](DEMO.md).

### UC7 — Script it

```bash
domembed search "how do I configure ssh keys?" questions.txt --top-k 5 --json > out.json
```

### UC8 — Show that it is fast enough to use

```bash
domembed bench
```

---

## 4. Core functionality

| # | Function | Contract |
|---|---|---|
| F1 | **Load** | Accept a local directory or a Hugging Face model id; return a ready `DomainEmbedder`. |
| F2 | **Encode** | One string or a list → L2-normalised `float32` numpy array(s). |
| F3 | **Similarity** | Two texts (or two equal-length lists) → cosine similarity, `float` or `ndarray`. |
| F4 | **Search** | Query + document list + `top_k` → results sorted by descending score. |
| F5 | **Index** | Encode a corpus once; query it repeatedly with only the query being encoded. |
| F6 | **Info** | Print/return the model's domain, base, dimension, training config and measured evaluation. |
| F7 | **Demo** | A Streamlit page: query → ranked columns per model, with the gold duplicate starred and its rank shown per column. **The presentation centerpiece — designed first, and result-agnostic.** |
| F8 | **CLI** | `encode`, `similarity`, `search`, `info`, `bench`. *Cuttable.* |
| F9 | **Benchmark** | Load time, encode latency, throughput, peak memory. *Cuttable.* |

Anything not in this table is not in the product.

**F1–F6 plus F7 are the product.** F8 and F9 are polish and are the first things cut — see
[`DEMO.md`](DEMO.md) §8.

---

## 5. Non-goals

Stated so they can be pointed at when scope pressure arrives.

| We are not building | Why | What to do instead |
|---|---|---|
| A new embedding framework | `sentence-transformers` already is one | Wrap it |
| A new architecture or loss | That is the *research* half, and it is already specified | Consume the trained model |
| A vector database | 14K documents is a 23 MB matrix | `model.index()` + numpy |
| ANN search (FAISS/HNSW) | Brute force over 14K × 384 is milliseconds | Revisit above ~500K docs |
| A model server / HTTP API | Infrastructure work with no research content | The CLI and the Streamlit demo |
| Distributed or multi-GPU inference | Explicitly out of scope | `device=` on one machine |
| A training API | `train.py` in the research repo does this | The export script |
| Multi-domain model routing | One domain, one model | v0.2, if a second domain ships |
| Hybrid BM25 + dense retrieval | Adds a second index and a tuning parameter | v0.2 |
| ONNX / quantised inference | Needs a toolchain and a correctness check | v0.2 (`fastembed` or `optimum`) |
| A general NLP toolkit | — | Use `spaCy`, `gensim`, `scikit-learn` |
| PyPI publication | Not required, adds a release process | `pip install -e .` |
| A web frontend | The Streamlit app is the frontend | — |
| A demo that only shows our model winning | The result is not known yet, and may be a tie or a loss | The four result modes in [`DEMO.md`](DEMO.md) §1.2 |

---

## 6. Product identity (§19 of the brief)

### Candidate names

Availability checked against the PyPI JSON API on **2026-09-17**. Re-check before publishing —
names get taken.

| Name | Import | Read as | PyPI | Verdict |
|---|---|---|---|---|
| **`domembed`** | `domembed` | "dom-embed" | **free** | **Recommended.** Short, typo-resistant, unambiguous, says what it is. |
| `nichevec` | `nichevec` | "niche-vec" | **free** | Runner-up. More brandable, less self-describing. |
| `domvec` | `domvec` | "dom-vec" | **free** | Terser; reads like a vector-graphics library. |
| `domain-embed` | `domain_embed` | — | **free** | The brief's placeholder. Correct but generic; the hyphen/underscore mismatch trips people up. |
| `duqa` | `duqa` | "du-qa" | **free** | Ties it to duplicate-QA. Too narrow if a second domain ships. |
| `askembed` | `askembed` | — | **free** | Locks the name to AskUbuntu. |
| `domain-embedder` | `domain_embedder` | — | **free** | Fine, verbose. |
| `indomain` | `indomain` | "in-domain" | *not checked* | Reads like a DNS or email tool. |

**Chosen: `domembed`**, with the class `DomainEmbedder` (the brief's name, kept — it is clear and
it is what people will look for).

* **One-line description:** Domain-specific sentence embeddings, packaged.
* **Tagline:** *Three lines to semantic search in your domain.*
* **README opening paragraph:** "General-purpose embedding models are trained to judge whether
  two English sentences mean the same thing. In a technical domain, that is not the question.
  `domembed` ships a small sentence-embedding model that was contrastively fine-tuned on
  naturally occurring duplicate questions from one domain, behind an API you can use without
  knowing what contrastive learning is."
* **PyPI availability:** free as of 2026-09-17 — verified by requesting
  `https://pypi.org/pypi/<name>/json` for each candidate (a `Not Found` response means unused;
  the same check against `sentence-transformers` returns metadata, confirming the method).

---

## 7. MUST / SHOULD / OPTIONAL

### MUST HAVE — the product is not demoable without these

| # | Item | Notes |
|---|---|---|
| M1 | `DomainEmbedder.load()` — local path **and** Hugging Face id | 5-line dispatch; `SentenceTransformer` does the work |
| M2 | `encode()` — one string **and** a list | Always `np.ndarray`, always `float32`, always L2-normalised |
| M3 | `similarity()` — text-to-text cosine | Returns `float` for strings |
| M4 | `search()` — query + documents + `top_k` | Returns `list[SearchResult]`, descending |
| M5 | `info()` and `__repr__` | Reads `model_info.json`; degrades gracefully when absent |
| M6 | `model_info.json` schema + `scripts/export_model.py` | **The seam between research and product** |
| M7 | Error handling for the five documented edge cases | Empty input, missing model, bad `top_k`, wrong types, overlong text |
| M8 | `pyproject.toml` such that `pip install -e .` works | hatchling backend, extras for `demo` and `dev` |
| M9 | README with the demo screenshot above the fold and an honest results section | Numbers generated, never typed |
| M10 | **The Streamlit demo: query → ranked columns per model, gold duplicate starred, rank shown per column** | **The presentation centerpiece. ~120 lines. Designed first; result-agnostic.** |
| M11 | **All four result-mode captions written** (WIN / INVERTED-U / TIE / LOSS) | So a flat or negative result still has a demo |
| M12 | `model.index()` / `DocumentIndex` | Required for a responsive demo on the full corpus — cut only if the corpus is trimmed |
| M13 | Test suite: the five named tests | See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §5 |

### SHOULD HAVE — do these if the research stays on schedule

| # | Item | Notes |
|---|---|---|
| S1 | The aggregate panel (Tab 2: metrics table + `info()` + benchmark) | The evidence that licenses the anecdote, and the fallback if the live query misbehaves |
| S2 | CLI — `info` and `search` at minimum | `info` is the one command that looks good on a slide |
| S3 | `benchmark()` + `domembed bench` | Feeds the performance panel |
| S4 | Property tests (symmetry, self-similarity, batch invariance) | ~15 lines, catches real bugs |
| S5 | Both loading paths demonstrated in the README | |
| S6 | `logger` warnings rather than silent truncation | |
| S7 | `examples/quickstart.py` | Largely superseded by the demo |

### OPTIONAL — only after everything above works

| # | Item |
|---|---|
| O1 | FAISS backend behind `index(backend="faiss")` |
| O2 | ONNX / `fastembed` CPU path, removing the torch dependency |
| O3 | Gradio variant of the demo |
| O4 | `similarity_matrix(a_list, b_list)` |
| O5 | On-disk index caching (`index.save()` / `DocumentIndex.load()`) |
| O6 | PyPI publication (incl. a TestPyPI rehearsal) |
| O7 | A second domain model, to prove the pipeline generalises |
| O8 | Hybrid BM25 + dense search |

### The cut order, when time runs short

Cut from the bottom of this list: O-items → S7 → S3 → S2 → S4 → M12 (only with a trimmed
corpus). **Stop there.** Everything above that line is what makes the demo work. Full reasoning
in [`DEMO.md`](DEMO.md) §8.

---

## 8. Success criteria

Checkable at the end of the project. Each is a yes/no question.

| # | Criterion | Check |
|---|---|---|
| 1 | **Installable** | In a fresh virtualenv, `pip install -e .` succeeds and `import domembed` works. |
| 2 | **Three-line promise kept** | Load → encode → search works from a cold start in under 10 lines of user code. |
| 3 | **Runs on the generic model too** | `DomainEmbedder.load("sentence-transformers/all-MiniLM-L6-v2").search(...)` works unchanged. |
| 4 | **The demo works live** | A query typed by an audience member returns ranked results with the gold duplicate starred, in under 2 seconds. |
| 5 | **The demo shows the research variable** | The same query run through ≥2 negative-pair conditions produces visibly different rankings. |
| 6 | **The demo survives a bad result** | With the TIE caption loaded, the app still runs, still looks finished, and still says something true. |
| 7 | **Provenance is visible** | `model.info()` names the domain, the base model and the negative-pair strategy. |
| 8 | **The numbers are real** | Every metric in the README and in Tab 2 traces to `results/metrics.json`. |
| 9 | **Honest by construction** | If the fine-tuned model loses to the baseline, the README, `info()` and the demo all still work and all say so. |
| 10 | **Someone else can use it** | A group member who did not write the code completes UC1–UC5 from the README alone, in under 15 minutes, asking at most one question. |
| 11 | **Tests pass** | `pytest` green, and the fast subset runs in under 5 seconds without a model download. |
| 12 | **Small enough to read** | A newcomer can read the whole library in one sitting. Target: ~490 lines of library + demo. |

Criteria 4, 5 and 6 are the new centre of gravity. Everything else is a proxy for them.

Criterion 4, 5 and 6 are the new centre of gravity. Everything else is a proxy for them.

---

## 9. Scope guard

Three questions to ask before adding anything:

1. **Does it make the research result easier to see?** If not, it is decoration.
2. **Does it require new machine-learning work?** If yes, it belongs in the research repo.
3. **Can a user discover it without documentation?** If no, it is probably too clever.

If the answer to all three is "no", it does not go in v0.1.

One addition, now that the demo leads:

4. **Does it make the demo more likely to work on stage?** If no, and it competes with demo
   time, cut it. The demo failing in front of an audience costs more than any feature earns.
