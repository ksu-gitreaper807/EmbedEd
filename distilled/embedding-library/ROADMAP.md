# Roadmap — `domembed`

Three columns: what ships, what might, what must not sneak into the one-month scope.

---

## v0.1 — the project version

**Everything in this column is the deliverable.** Anything not listed here is not in the project.

| Area | Ships |
|---|---|
| Loading | `DomainEmbedder.load()` — local directory and Hugging Face id |
| Encoding | `encode()` for a string or a list; `float32`, L2-normalised |
| Comparison | `similarity()` — text to text, cosine |
| Retrieval | `search(query, documents, top_k)` → `list[SearchResult]` |
| | `index(documents)` → `DocumentIndex` for repeated queries |
| Metadata | `info()` / `ModelInfo` / `model_info.json` schema |
| | `scripts/export_model.py` — the research-to-product seam |
| CLI | `encode`, `similarity`, `search`, `info`, `bench` |
| Measurement | `benchmark()` — latency, throughput, peak RSS |
| Packaging | `pyproject.toml` (hatchling), `pip install -e .`, extras `demo` and `dev` |
| Docs | README (with generated, honest numbers), `examples/quickstart.py` |
| Tests | ~14 tests, including the property tests; fast subset under 5 s |
| Demo | **Streamlit app: query → ranked columns per model, gold duplicate starred, rank shown per column, plus the aggregate panel. The presentation centerpiece — designed first, and built to all four result modes (win / inverted-U / tie / loss).** |
| Quality | `ruff` clean; ≤ 500 lines across 7 modules |

**Version number: 0.1.0.** The `0.x` is deliberate — it is the honest signal that the API may
still change and that this is a research artefact, not a maintained product.

---

## v0.2 — possible improvements, only after v0.1 is finished

Each has an explicit trigger. **No item starts just because it sounds good.**

| Item | Trigger | Effort |
|---|---|---|
| **FAISS backend** — `index(backend="faiss")` | Corpus above ~500K documents, or a real user complains about latency | 0.5 day |
| **ONNX / `fastembed` inference path** | Someone needs to run this without torch (a Raspberry Pi, a serverless function, a 200 MB Docker budget) | 1 day + a numerical-equivalence test |
| **`DocumentIndex.save()` / `.load()`** | The demo index takes long enough to build that it annoys the *developer*, not just the audience | 2 h |
| **`similarity_matrix(a_list, b_list)`** | A clustering or paraphrase-mining use case actually appears | 1 h |
| **A second domain model** | The pipeline is proven and there is a second labelled corpus with natural pairs | the research cost is the real cost, not the library cost |
| **Hybrid BM25 + dense search** | The evaluation shows BM25 beating every dense system — in which case the honest product is a hybrid | 1 day |
| **Gradio variant of the demo** | A shareable public link is needed | 1 h |
| **PyPI publication** | Someone outside the group wants to install it | 30 min (see below) |
| **Model comparison helper** — `compare(models, query, docs)` | Writing the two-column comparison by hand gets tedious | 1 h |
| **Reranking with a cross-encoder** | Retrieval quality matters more than latency for a real user | 1 day + evaluation |

---

## Future ideas — explicitly not scheduled

Recorded so they stop being re-proposed in every meeting.

| Idea | Why it is parked |
|---|---|
| Multi-domain model routing (`domembed.load(domain="medical")`) | Needs several domain models first |
| A fine-tuning API inside the library | That is `train.py`; duplicating it creates two sources of truth |
| A REST service / Docker image | Infrastructure work with no research content |
| Quantised (int8) embeddings | Needs an accuracy measurement to justify it |
| Streaming / incremental indexing | That is what vector databases are for |
| Multi-vector (ColBERT-style) retrieval | A completely different retrieval paradigm |
| Matryoshka / truncated-dimension support | Interesting, needs an evaluation of the truncation trade-off |
| Cross-lingual domain embeddings | Needs multilingual data and a different evaluation |
| An evaluation harness (MTEB-style) inside the library | The research repo already has `evaluate.py` |
| Automatic domain detection | A solution looking for a problem |
| **Automatic negative-strategy selection** | **Promoted out of this list — it is a full v2, specified separately in [`distilled/adaptive-embedding/`](../adaptive-embedding/)** |

---

## Anti-roadmap — things that must not enter v0.1

The scope guard from [`PRODUCT_SPEC.md`](PRODUCT_SPEC.md) §9, restated as a checklist. If one of
these appears in a pull request, the answer is no until the research is finished *and* there is
spare time:

```text
[ ] FAISS / ANN indexing
[ ] ONNX or quantised inference
[ ] A REST API or web service
[ ] A training or fine-tuning API
[ ] A vector database or persistence layer
[ ] Multi-GPU or distributed inference
[ ] A second domain
[ ] Hybrid retrieval
[ ] Reranking
[ ] PyPI publication
[ ] Anything with "agent" in it
[ ] Anything that needs a hyperparameter search
[ ] Adaptive / automatic negative-strategy selection  ← v2, see ../adaptive-embedding/
```

---

## v2 — the adaptive extension (planned, not scheduled)

A separate document, because it is a different question from everything above. Where v0.1 asks
*"how do we package one model?"*, v2 asks *"given a new dataset, which negative strategy should
the model be trained with?"*

→ **[`distilled/adaptive-embedding/`](../adaptive-embedding/)**

Start it only after the current experiment is written up. Two reasons it is genuinely separate
work rather than a late feature: it depends on having the finished N1/N2/N3 results to validate
against (its central check is "do cheap pilots predict full runs?"), and its honest novelty
assessment is that pilot-based selection is standard model selection — a systems contribution,
not a new algorithm.

---

## The code track (parallel, second domain)

A second research track runs alongside the AskUbuntu work: **BigCloneBench / GraphCodeBERT / Java
clone detection**.

→ **[`project/code-clone/`](../../project/code-clone/)** for the research specification
→ **[`CODE_TRACK.md`](CODE_TRACK.md)** for what it changes in `domembed`

The short version: **nothing in the contract changes.** Same five calls, same `model_info.json`,
same export seam. What differs is three configuration values (768-d, cased, cosine threshold
instead of a rank), five extra metadata fields, and a re-designed demo centrepiece — pairwise
scores against a drawn threshold, rather than a rank jump.

If both tracks finish, `domembed` ships **two** domain models through one interface. That is a
better demonstration of the design than a single model would be, and it costs nothing beyond a
second `model_info.json`.

---

## Publishing to PyPI (documented, not scheduled)

Not a deliverable. Written down so it is a 30-minute task later rather than an unknown.

1. Confirm the name is free: `curl -s https://pypi.org/pypi/domembed/json` → `Not Found`.
   *(Verified free on 2026-09-17. Re-check — names get taken.)*
2. Rehearse on TestPyPI: `python -m build` then
   `twine upload -r testpypi dist/*`, then install from TestPyPI in a clean venv.
3. Upload for real: `twine upload dist/*`.
4. Tag the release: `git tag v0.1.0 && git push --tags`.

Two reasons it is not scheduled: it creates an expectation of maintenance, and a `0.1.0`
published package with one domain model is not obviously more impressive in a presentation than
a clean repository with `pip install -e .` in the README. The second is also undoable.

---

## How the roadmap interacts with the research

```text
Week 1   research: data + baselines           library: nothing
Week 2   research: negatives + first runs     library: Phases 0–1.5 (skeleton, embedder,
                                                       DEMO SHELL against the generic model)
Week 3   research: all conditions, statistics library: Phase 2 (aggregate panel, all four
                                                       result-mode captions, CLI, tests, README)
Week 4   research: write-up                   library: Phase 3 (export, real models, real
                                                       numbers, delete three captions, rehearse)
```

Week 2 now includes the demo shell. That is the change: the centerpiece is built when there is
time to rehearse it, not in the last two days.

The library is **downstream** of the research and **parallel** to it in time. If the research
slips, the library slips — never the other way round. The hard rule from
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md):

> **No library work before the research baselines are validated.** A beautiful wrapper around a
> model that was never properly evaluated is the failure mode this whole document is written to
> avoid.
