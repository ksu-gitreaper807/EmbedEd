# `domembed` on the code track

How the v0.1 contract in this folder applies to the second, parallel project:
**BigCloneBench / GraphCodeBERT / Java clone detection**
([`project/code-clone/`](../../project/code-clone/)).

> **Short version: the contract does not change.** `domembed` is a wrapper around an embedding
> model; it was never specific to English questions. Five calls, one metadata file, one export
> seam. What changes is three configuration values, one demo redesign, and five extra fields in
> `model_info.json`.

---

## 1. What stays exactly the same

| Item | Status |
|---|---|
| The public API — `load` / `encode` / `similarity` / `search` / `info` | **Unchanged.** Pairwise comparison already works: `similarity(a, b)` is the clone score |
| `model_info.json` as the provenance carrier | **Unchanged** |
| `scripts/export_model.py` as the single seam to the research code | **Unchanged** |
| Loading the baseline through the identical interface | **Unchanged** — and still the whole point |
| No FAISS, no vector DB | **Unchanged.** 9,134 fragments is one matmul, same as ~14K questions |
| "The library is the engineering deliverable; the demo is the presentation centerpiece" | **Unchanged** |
| The demo must be result-agnostic | **Unchanged** — four outcome modes, see §4 |
| "One query is an anecdote; the aggregate table is the evidence" | **Unchanged**, and it matters more here (§4.3) |

---

## 2. What changes in configuration

| | AskUbuntu track | Code track |
|---|---|---|
| Base model | `all-MiniLM-L6-v2` | `microsoft/graphcodebert-base` |
| Dimension | 384 | **768** |
| Max sequence length | 256 | **256 or 512** — measure the token-length distribution, then fix it and record the choice |
| Casing | lowercased upstream | **cased** — GraphCodeBERT uses a RoBERTa tokenizer; `do_lower_case=False`. Do not lowercase Java identifiers |
| Pooling | as shipped | **mean pooling**, added explicitly (the base checkpoint ships no pooling module) |
| Domain string | `askubuntu` | `bigclonebench-java` |
| Decision rule | rank of the gold duplicate | **cosine threshold**, chosen on validation, stored in `model_info.json` |

Two notes:

* **The data-flow caveat travels with the model.** If the code track uses Option A
  (`FINAL_SPEC.md` §5 — GraphCodeBERT as a token-only encoder), `model_info.json` must say so.
  Otherwise the package quietly misrepresents the model it ships. See §3.
* **`search()` is still useful**, and more so than on the AskUbuntu track. "Find the most similar
  functions in this corpus" is both a plausible product feature and exactly the operation the
  semantic negative-mining step performs. One matmul, no index.

---

## 3. Extra fields in `model_info.json`

Everything the AskUbuntu version records, plus five that this track's claims depend on:

```json
{
  "domain": "bigclonebench-java",
  "base_model": "microsoft/graphcodebert-base",
  "dataflow_at_inference": false,
  "negative_strategy": "semantic",
  "decision_threshold": 0.71,
  "threshold_selected_on": "validation",
  "evaluation": {
    "f1_test": 0.94,
    "precision_test": 0.93,
    "recall_test": 0.95,
    "map_at_r_test": 0.88,
    "f1_seen": 0.95,
    "f1_unseen": 0.66,
    "generalisation_gap": 0.29,
    "measured_false_negative_rate": 0.31
  },
  "caveats": [
    "GraphCodeBERT used as a token-only encoder; no data-flow graphs at inference",
    "F1 is agreement with BigCloneBench labels, which are estimated ~15% contested",
    "generalisation numbers from 23 functionalities; treat as secondary"
  ]
}
```

**`dataflow_at_inference` and the `caveats` array are the important additions.** They exist so the
package cannot be demonstrated without the limitations travelling with it — which is the honest
version of "the library's purpose is to make the demo honest".

All numbers above are **illustrative placeholders**. `export_model.py` must refuse to write a file
containing them.

---

## 4. The demo, re-specified

The AskUbuntu demo's centrepiece was *"the known duplicate jumps from rank 5 to rank 1"*. The
code track needs a different centrepiece, because the task is pair classification, not ranking.

### 4.1 The new centrepiece: two scores, one threshold, one gold label

```
┌────────────────────────────────────────────────────────────────┐
│  TEST PAIR #4182            gold label:  CLONE                 │
├────────────────────────────────────────────────────────────────┤
│  C0  off-the-shelf          0.42  ──────▮────────────  not clone│
│  C3  semantic negatives     0.81  ─────────────▮─────  CLONE ✓  │
│                                          ↑ threshold 0.71      │
├────────────────────────────────────────────────────────────────┤
│  [ previous ]  [ next ]   [ random clone ]  [ random non-clone ]│
└────────────────────────────────────────────────────────────────┘
```

Three things make this honest where a paste-box alone would not:

1. **It defaults to real held-out test pairs** with their gold label, drawn from the same
   distribution the headline table summarises. The audience is looking at a number computed the
   same way the metric is computed.
2. **The threshold is drawn.** Without it the two scores are not a decision, just two numbers.
3. **The free-text paste box stays**, but is labelled *"your own snippets — illustrative, not part
   of the evaluation"*. It is for the "wow" moment, not for evidence.

### 4.2 Panel order

| # | Panel | Status |
|---|---|---|
| 1 | Results table — four conditions × {F1, P, R, MAP@R} + the generalisation columns | **Evidence** |
| 2 | Pairwise comparison (§4.1) | Anecdote, clearly labelled as one |
| 3 | UMAP, four panels, fixed seed, same sample | Illustration |

The table comes first. The picture comes last. This is the reverse of the instinct, and it is the
rule from [`DEMO.md`](DEMO.md) §1.1 applied to a new domain.

### 4.3 Result-agnostic: four modes, not one

| Result | Headline the app renders |
|---|---|
| Fine-tuning helps, semantic wins | "Harder negatives helped: F1 0.88 → 0.94" |
| Fine-tuning helps, strategy makes no difference | "Fine-tuning helped (+X F1). Negative strategy did not change the outcome — that is the finding" |
| Inverted-U (BM25 best) | "Hardest negatives were worse: F1 peaks at BM25. Measured false-negative rate: X%" |
| Everything ties, or fine-tuning does not help | "Fine-tuning did not improve over the off-the-shelf model on this benchmark" |

The app is built once and the caption changes. **Never build the demo around the result you
hope for.**

### 4.4 The visualisation caveat, enforced in code

UMAP and t-SNE will show a difference between two conditions whether or not one exists. So:

* `random_state` fixed, `n_neighbors` and `min_dist` fixed, identical across all four panels.
* **One shared sample of fragments**, drawn once, projected four times. Never re-sample per
  condition.
* If functionality labels are available (they are, on the generalisation subset), colour by
  functionality and **show the held-out ones** — that is the visually interesting case, and it is
  the picture that corresponds to RQ3.
* The panel carries the caption *"illustrative; cluster shape depends on UMAP hyperparameters"*.

---

## 5. What is *not* re-specified

Deliberately left alone, because the code track changes the domain, not the product:

* The package layout, the five-call API, the CLI verbs, `DocumentIndex`, the Streamlit app
  skeleton, the export script, and the test plan in
  [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).
* The hard rule: **no library work before the research baselines are validated.** On this track
  that means no `domembed` code until C0's F1 on the fixed test split matches the published
  neighbourhood.

---

## 6. Sequencing against both tracks

```text
Week 1   code track: data, overlap measurement, mining   library: nothing
Week 2   code track: C1/C2/C3, hardness + FN checks      library: demo SHELL against the
                                                                  off-the-shelf base model
Week 3   code track: generalisation test, UMAP           library: aggregate panel, four result
                                                                  modes, CLI, tests
Week 4   code track: write-up                            library: export, real numbers,
                                                                  delete three captions, rehearse
```

If the two tracks both reach week 4 with models, `domembed` ships **two** domain models through
one interface — which is a better demonstration of the design than one model would be, and costs
nothing extra beyond a second `model_info.json`.
