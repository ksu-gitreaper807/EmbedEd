# 06 — Implementation architecture

Principle: **use established libraries for everything except the parts that are your contribution.** Your contribution is the negative-sampling strategies and the experiment design — not a training loop, not an nDCG implementation, not a BM25 index.

---

## Part A — Folder layout

```
project/
│
├── dataset/
│   ├── raw/                    # whatever ir_datasets / BEIR caches; never edited
│   └── processed/
│       ├── corpus.jsonl        # {doc_id, subforum, title, body, text, timestamp}
│       ├── queries.jsonl       # {query_id, subforum, text, gold_ids:[...]}
│       ├── qrels.tsv           # query_id  doc_id  relevance
│       ├── groups.json         # duplicate-group connected components
│       └── splits.json         # {train:[qid...], val:[...], test:[...]}
│
├── src/
│   ├── preprocessing/
│   │   ├── load_cqa.py         # ir_datasets -> corpus/queries/qrels
│   │   ├── clean.py            # strip HTML, code blocks, URLs; build `text`
│   │   └── build_groups.py     # duplicate graph -> connected components
│   ├── pair_generation/
│   │   ├── split.py            # group-constrained split + leakage assertion
│   │   ├── positives.py        # (anchor, positive) from duplicate groups
│   │   └── negatives.py        # N1 random / N3 bm25 / N4 mined / N5 denoised
│   ├── models/
│   │   ├── encoder.py          # SentenceTransformer wrapper (prefix handling!)
│   │   └── index.py            # FAISS flat-IP build + search
│   ├── training/
│   │   ├── train.py            # sentence-transformers Trainer, MNRL
│   │   └── configs/            # one YAML/JSON per configuration (N1..N5, seeds)
│   ├── evaluation/
│   │   ├── metrics.py          # nDCG@10, Recall@k, MRR@10, MAP (via ranx/pytrec_eval)
│   │   ├── run_retrieval.py    # encode -> index -> top-100 -> qrels -> metrics
│   │   └── diagnostics.py      # alignment/uniformity, false-negative rate
│   └── analysis/
│       ├── error_analysis.py   # FP/FN buckets, lexical-overlap bins
│       └── plots.py
│
├── experiments/
│   ├── baseline/               # BM25, TF-IDF, zero-shot encoders, hybrid
│   ├── contrastive/            # N1..N5 runs
│   └── hard_negative/          # A2: 0/1/3/5 negatives
│
├── results/
│   ├── metrics/                # one JSON per run: {config, seed, split, metrics}
│   ├── predictions/            # TREC-format run files (keep these — cheap, useful)
│   └── plots/
│
├── paper/
│   ├── main.tex  or  report.md
│   └── figures/
│
├── tests/
│   ├── test_split_leakage.py   # FAILS if train/test IDs overlap
│   ├── test_metrics.py         # nDCG on a hand-computable toy example
│   └── test_pairs.py           # no positive appears in its own negative set
│
├── requirements.txt
├── README.md                   # how to reproduce, end to end
└── run_all.sh                  # one command: raw data -> all tables
```

---

## Part B — What each component does

### `src/preprocessing/`

| Module | Responsibility | Notes |
|---|---|---|
| `load_cqa.py` | Pull `beir/cqadupstack/<subforum>` via `ir_datasets` (or the BEIR `DataLoader`); write `corpus.jsonl`, `queries.jsonl`, `qrels.tsv` | Start with **3 subforums** (`android`, `programmers`, `unix`) so iteration is fast; extend to 12 later |
| `clean.py` | Strip HTML tags, fenced code blocks, inline code, URLs; collapse whitespace; build `text = title + ". " + body` | Keep the raw text too — you will want it for error analysis |
| `build_groups.py` | Treat duplicate marks as undirected edges; compute **connected components**; write `groups.json` and attach `group_id` to every query and document | This is what makes the split safe |

**Exploratory analysis you must do before anything else** (30 minutes, saves days):
number of queries and docs per subforum; distribution of group sizes; number of queries with ≥1 gold; **distribution of gold-per-query** (this is your ceiling argument); length distribution in tokens (justifies the 256-token budget).

### `src/pair_generation/`

| Module | Responsibility |
|---|---|
| `split.py` | Assign whole **groups** to train/val/test (70/10/20 by query count). Print and **assert** that `train_doc_ids ∩ test_gold_ids = ∅` and `train_group_ids ∩ test_group_ids = ∅`. Also emit a subforum-holdout split |
| `positives.py` | For each anchor in train, emit `(anchor, positive)` for every other member of its group. Cap pairs per group (e.g. 5) to stop large groups dominating |
| `negatives.py` | The **contribution**. Five functions, one per strategy, all returning `{anchor, positive, negatives:[...]}` |

`negatives.py` in detail:

```python
# N1 random      -> random.sample(pool \ group(a), k)
# N2 in-batch    -> []            # MNRL uses other batch positives; nothing to add
# N3 lexical     -> rank_bm25 top-k, excluding group(a)
# N4 model-hard  -> faiss top-k by CURRENT model embedding, excluding group(a)
# N5 denoised    -> N4, then drop c where sim(a,c) > sim(a,p) - DELTA
#                   (or use sentence_transformers.util.mine_hard_negatives with
#                    absolute_margin / relative_margin / max_score / range_min)
```

**Design rule:** all five consume the same `(anchor, positive)` list, so the only thing that varies is the negative column. That is what makes it a controlled experiment.

### `src/models/`

| Module | Responsibility |
|---|---|
| `encoder.py` | Thin wrapper over `SentenceTransformer`. **Handles the query prefix**: `bge-small-en-v1.5` needs `"Represent this sentence for searching relevant passages: "` prepended to *queries*. Encode with `normalize_embeddings=True`, `batch_size=256` |
| `index.py` | `faiss.IndexFlatIP(384)` over the **fixed** document pool. Exact search — with ≤90K docs per subforum there is no reason to approximate |

### `src/training/`

* Use the `sentence-transformers` `SentenceTransformerTrainer` — do **not** hand-roll a training loop.
* One config file per run: `{"strategy": "N5", "seed": 42, "batch_size": 64, "lr": 2e-5, "epochs": 3, "num_hard_negatives": 3, "max_seq_length": 256}`.
* `batch_sampler = BatchSamplers.NO_DUPLICATES` ([docs](https://sbert.net/): MNRL benefits from no duplicate samples in a batch).
* Save the best checkpoint by **validation** nDCG@10, evaluated every N steps.
* Log to a JSON file per run, not just to stdout.

### `src/evaluation/`

* **Do not implement nDCG yourself.** Use `ranx` or `pytrec_eval`. Write one unit test against a toy example you can compute by hand.
* `run_retrieval.py` must be **system-agnostic**: given a callable `encode_fn` and a system name, it produces a run file. BM25, TF-IDF and every neural system go through the *same* function. This single design choice prevents most baseline-fairness bugs.
* Keep TREC-format run files in `results/predictions/` — they make error analysis trivial later.

### `src/analysis/`

Bucketing and plots; see [`08-error-analysis.md`](08-error-analysis.md).

---

## Part C — Libraries: what is actually necessary

| Library | Necessary? | Why |
|---|---|---|
| **Python 3.10+** | ✅ | |
| **PyTorch** | ✅ | Comes with `sentence-transformers` |
| **sentence-transformers** | ✅ **Core.** Provides the model, mean pooling, `MultipleNegativesRankingLoss`, `TripletLoss`, `mine_hard_negatives()`, `SentenceTransformerTrainer`, evaluators | |
| **Hugging Face `transformers`** | ✅ (transitive) | Tokenizer and base model |
| **Hugging Face `datasets`** | ⚠️ Optional | Only if you add the Quora-duplicates check |
| **ir_datasets** (or the BEIR repo) | ✅ **Core.** One-line access to CQADupStack queries, docs and qrels in a standard format | |
| **NumPy** | ✅ | |
| **pandas** | ⚠️ Helpful | Tables and EDA. Not required in the hot path |
| **scikit-learn** | ⚠️ Helpful | TF-IDF vectoriser for baseline B1; train/test utilities |
| **rank_bm25** (or `pyserini`) | ✅ **Core.** BM25 baseline B0 and lexical hard negatives N3 | |
| **ranx** or **pytrec_eval** | ✅ **Core.** nDCG@10, Recall@k, MRR, MAP — do not implement these | |
| **faiss-cpu** | ✅ **Core.** Retrieval index and hard-negative mining. `faiss-gpu` only if you hit a wall (you will not at this scale) | |
| **matplotlib** | ✅ | 3–4 figures |
| **PyTorch Metric Learning** | ❌ Not needed | |
| **Weights & Biases / TensorBoard** | ❌ Optional | A JSON log file is enough and has fewer moving parts |
| **UMAP** | ❌ Optional | Only for the "nice to have" embedding plot |

**Minimal install:**

```bash
pip install sentence-transformers ir_datasets rank_bm25 ranx faiss-cpu pandas scikit-learn matplotlib
```

---

## Part D — Reproducibility rules

1. `requirements.txt` with pinned versions, or a `conda env export`. Record `torch`, `transformers`, `sentence-transformers` versions in the paper.
2. `seed_everything(seed)` at the top of every entry point — Python `random`, `numpy`, `torch`, `torch.cuda`, and the data loader generator.
3. Every run writes `results/metrics/<system>__<strategy>__seed<seed>.json` containing the full config, the seed, and all metrics.
4. `run_all.sh` reproduces every table from `dataset/raw/` in one command.
5. `tests/` with the three tests listed in the layout. **The leakage test is not optional** — it is the difference between a project and a cautionary tale.
6. Commit the split file (`splits.json`). A result is only interpretable relative to a specific split.

---

## Part E — Where the research actually lives in this codebase

| File | Engineering | Research |
|---|---|---|
| `load_cqa.py`, `clean.py`, `index.py`, `metrics.py`, `run_retrieval.py` | ✅ 100% | — |
| `build_groups.py`, `split.py` | ✅ mostly | the group-constrained split rule is a methodological choice you defend |
| `positives.py` | ✅ | the "no synthetic positives" rule |
| **`negatives.py`** | ✅ some | **⭐ the entire independent variable** |
| `diagnostics.py` (false-negative rate) | ✅ some | **⭐ your most original measurement** |
| `analysis/error_analysis.py` | ✅ some | ⭐ interpretation |
| `train.py` | ✅ 100% | — |

If `negatives.py` and `diagnostics.py` are the only files you truly understand deeply, you have understood your project.
