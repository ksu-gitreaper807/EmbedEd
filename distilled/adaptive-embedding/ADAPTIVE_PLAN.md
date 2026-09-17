# Adaptive `domembed` — full plan (Version 2 / future extension)

**Status: proposed future work. Not part of the current four-week project.**

> **On numbers.** Every number in this document is **illustrative** unless it is explicitly marked
> as measured. The current research (N1/N2/N3 on AskUbuntu) has not produced results yet, so no
> real score appears anywhere below. Numbers used in worked examples are tagged
> `[ILLUSTRATIVE]`.

---

## Contents

| § | Section |
|---|---|
| 1 | What adaptive `domembed` actually does |
| 2 | The complete pipeline (current research vs future extension) |
| 3 | How N1 / N2 / N3 become the basis of the adaptive system |
| 4 | The automatic-selection part |
| 5 | Domain profiling |
| 6 | The pilot-training process |
| 7 | The final training stage |
| 8 | Architecture |
| 9 | User-facing API |
| 10 | End-to-end example: AskUbuntu |
| 11 | Second example: domain flexibility (hypothetical) |
| 12 | What is novel and what is not |
| 13 | Scope control: MUST / SHOULD / FUTURE |
| 14 | The research / product boundary |
| 15 | Implementation milestones |
| 16 | Risks and mitigations |
| 17 | Recommended architecture — the summary |
| 18 | **Appendix — mapping this plan onto the code track** (BigCloneBench / GraphCodeBERT) |

---

## 1. What adaptive `domembed` actually does

### 1.1 Plain language first

The current project trains **three versions** of the same small encoder. The three versions are
identical in every way except one: **which questions were used as negative examples during
training.**

* **Random** — any other question in the batch counts as a negative. Easy, cheap, possibly too
  easy to teach anything.
* **BM25 / lexical** — negatives are questions with similar *words*. Harder.
* **Semantic** — negatives are questions the base model already thinks are *similar in meaning*.
  Hardest — and therefore most likely to accidentally include genuine duplicates that the
  dataset failed to label.

The research measures which of the three produces the better model. Right now, **a human decides
by reading a results table.**

Adaptive `domembed` automates that decision for a **new** dataset. Instead of asking the user
"which negative strategy do you want?", it tries all three cheaply, measures them, picks the
winner, and then spends the real training budget on that one.

### 1.2 A concrete example

Suppose someone hands the library a **StackOverflow duplicate-question dataset**.

```text
  User provides
  ─────────────
  StackOverflow duplicate questions
  (pairs of questions the community marked as duplicates)
        ↓
  domembed analyses the domain
        ↓   "31,000 pairs, short texts, high lexical overlap between duplicates,
             the base model already separates duplicates fairly well"
        ↓
  domembed tries all three strategies  —  cheaply
        ↓
        ├── train briefly with Random negatives    → score it
        ├── train briefly with BM25 negatives      → score it
        └── train briefly with Semantic negatives  → score it
        ↓
  BM25 scored highest
        ↓
  domembed uses BM25 for the FULL training run
        ↓
  final StackOverflow-domain encoder
        ↓
  saved, exported, loadable by domembed
```

The user wrote no training loop, picked no strategy, and read no table. They gave it a dataset
and got back a model — plus a saved report explaining what was tried and why the winner won.

### 1.3 Why this is worth building at all

Not because it is clever. Because it makes the research result **reusable by someone who has not
read the research**. The current project produces a number ("on AskUbuntu, N2 beat N3"). The
adaptive product turns that into a procedure that works on a dataset nobody has studied.

---

## 2. The complete pipeline

```text
 ═══ INPUT ═══════════════════════════════════════════════════════════════════════
   A new domain dataset with naturally occurring positive pairs
 ═════════════════════════════════════════════════════════════════════════════════
                                      ↓
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 1 · DATASET PREPARATION                                                  │
 │ Build corpus, queries, qrels, train pairs, duplicate groups. Leakage check.    │
 │ STATUS: exists — src/prepare_data.py                                           │
 └────────────────────────────────────────────────────────────────────────────────┘
                                      ↓
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 2 · DOMAIN PROFILING                                        ★ NEW (v2)   │
 │ Compute ~8 descriptive statistics. Describe the domain; run guard-rail checks. │
 │ STATUS: new, ~90 lines — see §5                                                │
 └────────────────────────────────────────────────────────────────────────────────┘
                                      ↓
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 3 · CANDIDATE STRATEGIES                                                 │
 │ N1 random · N2 BM25/lexical · N3 model-mined/semantic  — and nothing else.     │
 │ STATUS: exists as an experiment; needs a small registry wrapper — see §8       │
 └────────────────────────────────────────────────────────────────────────────────┘
                                      ↓
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 4 · PILOT TRAINING                                          ★ NEW (v2)   │
 │ 25% of pairs · 1 epoch · 1 shared seed · three short runs.                     │
 │ STATUS: new, ~120 lines over existing train.py — see §6                        │
 └────────────────────────────────────────────────────────────────────────────────┘
                                      ↓
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 5 · VALIDATION                                                           │
 │ Recall@10 / MRR@10 / nDCG@10 on the full dev split.                            │
 │ STATUS: exists — src/evaluate.py                                               │
 └────────────────────────────────────────────────────────────────────────────────┘
                                      ↓
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 6 · STRATEGY SELECTOR                                       ★ NEW (v2)   │
 │ argmax, with a margin-based tie-break toward the cheaper strategy.             │
 │ STATUS: new, ~40 lines — see §4                                                │
 └────────────────────────────────────────────────────────────────────────────────┘
                                      ↓
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 7 · FULL TRAINING                                                        │
 │ All pairs · epochs chosen on dev · 3 seeds · the ONE selected strategy.        │
 │ STATUS: exists — src/train.py                                                  │
 └────────────────────────────────────────────────────────────────────────────────┘
                                      ↓
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 8 · MODEL EXPORT                                                         │
 │ Checkpoint + model_info.json (+ the pilot report) → hub or local dir.          │
 │ STATUS: exists — scripts/export_model.py                                       │
 └────────────────────────────────────────────────────────────────────────────────┘
                                      ↓
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 9 · domembed INFERENCE API                                               │
 │ encode · similarity · search · info                                            │
 │ STATUS: exists — v0.1 library                                                  │
 └────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Where the current research sits

The current project occupies **Stages 1, 3, 5, 7, 8** — but it runs them **three times** (once
per strategy), three times each (three seeds), and its *output is a comparison table*, not a
decision.

```text
  CURRENT RESEARCH (4 weeks)              FUTURE PRODUCT EXTENSION (v2)
  ──────────────────────────              ─────────────────────────────
  Stage 1  prepare AskUbuntu      ──┐     Stage 1  prepare ANY domain
                                    │
  Stage 3  three strategies       ←─┼──→  Stage 3  the SAME three strategies
                                    │                    (reused, not extended)
  Stage 7  three FULL trainings      │     Stage 4  three CHEAP pilots
           × 3 seeds                 │     Stage 6  automatic selector
                                    │     Stage 7  ONE full training
  Stage 5  evaluate all of them   ←─┼──→  Stage 5  evaluate the pilots
                                    │
  Stage 8  export (for the demo)     │     Stage 8  export + pilot report
                                    │
           OUTPUT:                  │              OUTPUT:
           a results table          │              a trained model
           + an answer to           │              + a logged rationale
           "does it matter?"        │
                                    └── shared code, shared strategies,
                                        shared evaluation — different purpose
```

**The research asks "does it matter?" The product asks "which one, for this dataset?"** Both use
the same three strategies and the same metrics. The product is not a new experiment; it is the
research's machinery pointed at a decision instead of a table.

---

## 3. How the three existing models become the basis of the adaptive system

### 3.1 What the current research produces

```text
                    Base MiniLM  (sentence-transformers/all-MiniLM-L6-v2)
                                 │
                                 │  contrastive fine-tuning
                                 │  same loss, same data, same batch size
                                 │
        ┌────────────────────────┼────────────────────────┐
        ↓                        ↓                        ↓
   Random negatives        BM25 negatives          Semantic negatives
   (any other pair         (top-64 by BM25)       (top-64 by cosine of the
    in the batch)                                   zero-shot encoder)
        ↓                        ↓                        ↓
   ╔═════════╗             ╔═════════╗             ╔═════════╗
   ║ Model N1║             ║ Model N2║             ║ Model N3║
   ╚═════════╝             ╚═════════╝             ╚═════════╝
        └──────────────── three SEPARATE models ────────────┘
```

Three separate 384-d encoders. **Nothing is merged, averaged, ensembled or distilled.** Each is a
complete model in its own directory, independently loadable by `domembed`.

### 3.2 The same three, reframed

The adaptive system does not add a fourth strategy and does not blend the three. It reframes
them:

| In the research | In the adaptive product |
|---|---|
| Three experimental conditions | Three **candidate training paths** |
| All three trained to completion | All three trained **briefly** (pilot) |
| Compared in a table for a human | Compared by an `argmax` in code |
| The comparison *is* the contribution | The comparison is a **means to a decision** |
| Result: "N2 > N3 on this corpus" | Result: "N2 is used, N1 and N3 are discarded" |

```text
   NEW DOMAIN
       ↓
   ┌───────────────────────────────────────────────────┐
   │  PILOT (cheap)                                    │
   │                                                   │
   │   N1  ──train 25%──→  eval  ──→  Recall@10 = 0.54 │
   │   N2  ──train 25%──→  eval  ──→  Recall@10 = 0.61 │  ← best
   │   N3  ──train 25%──→  eval  ──→  Recall@10 = 0.57 │
   └───────────────────────────────────────────────────┘
                    [ILLUSTRATIVE NUMBERS]
       ↓
   SELECT: N2 (BM25)
       ↓
   ┌───────────────────────────────────────────────────┐
   │  FULL TRAINING — only N2, all data, 3 seeds       │
   │  the N1 and N3 pilot checkpoints are thrown away  │
   └───────────────────────────────────────────────────┘
       ↓
   one final domain encoder
```

### 3.3 Why not merge them instead?

Worth stating explicitly, because "adaptive" invites the idea of a mixture:

* **Ensembling three encoders** means three forward passes at inference and a 1,152-d vector.
  It also destroys the experiment: you could no longer say which strategy produced the behaviour.
* **Averaging the weights** (model soup) is not meaningful here — the three models were trained
  on the same data with different batch compositions, not different initialisations, and weight
  averaging has no theoretical justification in this setting.
* **Picking one** is the only option that keeps the research claim interpretable: the final model
  is attributable to a named strategy.

So: **three candidate paths, one survivor.** The other two are deleted after selection — but the
*pilot report* is kept, because it is the evidence for the choice.

---

## 4. The automatic-selection part

### 4.1 No LLM required

The word "adaptive" suggests something that reasons. Nothing here does. The selector is a
for-loop and a comparison. That is a deliberate design decision, not a compromise: for a
three-way choice measured on a validation set, anything more sophisticated would be harder to
debug, harder to explain in a viva, and no more accurate.

### 4.2 The rule, in symbols

```text
    s*  =  arg max  Recall@10_dev ( s )
           s ∈ S

    where  S = { Random, BM25, Semantic }
```

### 4.3 The rule, in plain English

> Train a small model using each of the three strategies. Score all three on the validation
> questions. **Take the strategy whose model got the highest score.**

`arg max` is mathematical shorthand for "the thing that produced the biggest number". Nothing
more is happening.

### 4.4 A concrete example

```text
   PILOT RESULTS (200 dev queries)                        [ILLUSTRATIVE]

   Strategy     Recall@10    MRR@10     nDCG@10
   ─────────────────────────────────────────────
   Random         0.54        0.41        0.46
   BM25           0.61        0.48        0.53      ← highest
   Semantic       0.57        0.44        0.49

   SELECTOR:  argmax = BM25
```

**What happens next:**

1. The BM25 pilot checkpoint is discarded (it was trained on 25% of the data for one epoch — it
   is not a good model, it was only a measuring instrument).
2. A **fresh** model is initialised from `all-MiniLM-L6-v2`.
3. It is trained on **all** the pairs, for the full number of epochs, with three seeds, using
   **BM25 neighbourhoods** for batch composition.
4. The best of the three seeds (or all three, if they agree) is exported.
5. The pilot table above is saved next to the model as `pilot_report.json` and its summary is
   copied into `model_info.json`, so `domembed`'s `info()` can show *why* this strategy was used.

### 4.5 The one refinement: a margin for noise

On 200 dev queries, the project's own spec notes that **differences below roughly 3 points are
noise**. A selector that confidently picks a 0.6-point winner is pretending to a precision it
does not have. So:

```python
def select(scores: dict[str, float], costs: dict[str, int], margin: float = 0.02):
    """scores: strategy -> Recall@10 on dev.  costs: strategy -> relative mining cost."""
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    best, best_score = ordered[0]
    runner_up_score  = ordered[1][1]

    if best_score - runner_up_score < margin:
        # Statistically indistinguishable. Prefer the cheaper / simpler one.
        tied = [s for s, v in scores.items() if best_score - v < margin]
        return SelectionReport(
            strategy=min(tied, key=lambda s: costs[s]),
            scores=scores, margin=margin,
            confident=False,
            reason=f"top two within {margin}; chose the cheapest of {tied}",
        )
    return SelectionReport(strategy=best, scores=scores, margin=margin,
                           confident=True, reason="clear winner")
```

Two behaviours, both honest:

| Situation | Output | `confident` |
|---|---|---|
| BM25 = 0.61, Semantic = 0.57 | **BM25** | `True` |
| BM25 = 0.61, Semantic = 0.60 | **Random or BM25** (cheapest within margin) | `False` |

When `confident=False`, the report says so out loud. A selector that admits "I cannot tell" is
more useful than one that guesses — and "use the cheap one when you cannot tell" is a policy a
reviewer can accept immediately.

### 4.6 Which metric should the selector use?

The brief suggests `MRR@10`. **We recommend `Recall@10`,** for two reasons:

1. It is the research's **primary** metric, so the pilot and the report speak the same language.
2. On 200 queries, `MRR@10` depends on the exact rank of the first correct hit, which gives it a
   heavier tail and therefore **higher variance** than `Recall@10`. When the whole point is to
   make a decision on a noisy signal, the lower-variance metric is the better instrument.

Record all three anyway. **If `Recall@10` and `MRR@10` disagree about the winner, that is a
signal, not a nuisance** — log it and set `confident=False`. Disagreement between metrics is the
cheapest available detector of "this pilot result is noise".

---

## 5. Domain profiling

### 5.1 What it is for — and what it is not

**Profiling does not select the strategy.** This is important enough to state twice:

```text
   Domain profiling  ──→  CHARACTERISES the domain
                          (provenance, guard rails, hypotheses)

   Pilot validation  ──→  DECIDES which strategy to use
                          (the only selection mechanism)
```

A tempting failure mode is to build a "smart" rule like *"if lexical overlap is high, use
semantic negatives"*. Do not. With one or two domains there is no evidence for any such rule,
and a hand-written heuristic that looks authoritative is worse than no heuristic at all, because
it will be believed.

Profiling earns its place for three honest reasons:

| Role | Why it matters |
|---|---|
| **Provenance** | The profile goes into `model_info.json`, so the exported model records what it was trained on — same principle as recording the negative strategy |
| **Guard rails** | Cheap sanity checks that catch bad datasets *before* spending GPU time (e.g. too few positive pairs to train on) |
| **Hypotheses for the report** | Gives the write-up something concrete to discuss, and accumulates data for the future question "does the best strategy correlate with any domain property?" |

### 5.2 The eight statistics

All are cheap: a few passes over the corpus plus one encoding pass with the base model, which the
pipeline needs anyway.

| Statistic | What it means | Why it might matter |
|---|---|---|
| `n_documents` | Corpus size | Determines whether mining needs an index. Below ~50K, one matrix multiply suffices |
| `n_positive_pairs` | Training pairs available | **Guard rail:** below ~2,000 pairs there is probably not enough signal to fine-tune at all |
| `duplicate_density` | Mean number of known duplicates per anchor | Low density means the positive labels are sparse — which raises the risk that mined "negatives" are unlabelled duplicates |
| `mean_token_length`, `pct_over_max_seq` | Text length vs the 256-token limit | Long texts get truncated; heavy truncation means the encoder never sees the discriminative part |
| `vocabulary_size`, `type_token_ratio` | Distinct tokens / total tokens | A small, repetitive vocabulary suggests lexical signals (BM25) carry little information |
| `lexical_signal_ratio` | Mean word-overlap of duplicate pairs ÷ mean word-overlap of random pairs | Near 1.0 means word matching cannot distinguish duplicates from non-duplicates — so **BM25 negatives may be uninformative** |
| `semantic_signal_ratio` | Mean base-model cosine of duplicate pairs ÷ mean cosine of random pairs | Low means the base model does not yet understand this domain — so **semantic mining will mine noise** |
| `neighbour_similarity` | Mean cosine between an anchor and its top-5 non-gold neighbours | **A crude false-negative proxy.** High means the model's nearest unlabelled neighbours look like duplicates. It is a *proxy*, not a measurement — only the manual annotation in the research measures the real rate |

### 5.3 What the profile produces

```text
  DOMAIN PROFILE — sentence-transformers/askubuntu            [ILLUSTRATIVE]

  n_documents                 14,100
  n_positive_pairs            19,000
  duplicate_density           1.5
  mean_token_length           41 tokens
  pct_over_max_seq            0.4%
  vocabulary_size             28,400
  type_token_ratio            0.031
  lexical_signal_ratio        2.8          ← duplicates share notably more words than chance
  semantic_signal_ratio       1.6          ← base model already separates duplicates somewhat
  neighbour_similarity        0.71         ← nearest unlabelled neighbours look quite similar

  GUARD RAILS
    ✓ enough positive pairs
    ✓ texts within the token limit
    ⚠ neighbour_similarity is high — semantic mining may surface unlabelled duplicates
```

That last warning is the kind of thing profiling is genuinely good at: it is a **flag for a human
to read**, derived from a measurable quantity, and it makes no claim to know the answer.

### 5.4 The hypothesis to record (not assert)

> *Hypothesis, to be tested with real domains:* when `semantic_signal_ratio` is low, the semantic
> strategy should underperform, because the base model's nearest neighbours are noise rather than
> hard negatives. When `neighbour_similarity` is very high, the semantic strategy should
> underperform for the opposite reason — its neighbours are too likely to be unlabelled
> duplicates.

With one domain, this is a hypothesis. With five domains, it becomes an experiment. With one
hundred, it might become a learned selector — which is firmly in the FUTURE column (§13).

---

## 6. The pilot-training process

### 6.1 The shape of a pilot

```text
   FULL DATASET (~19,000 pairs)
        ↓  take 25%
   PILOT SUBSET (~4,800 pairs)
        ↓
   ┌─────────────────────────────────────────────────────┐
   │  train with N1 neighbourhoods · 1 epoch · seed 13   │──→ eval on dev
   │  train with N2 neighbourhoods · 1 epoch · seed 13   │──→ eval on dev
   │  train with N3 neighbourhoods · 1 epoch · seed 13   │──→ eval on dev
   └─────────────────────────────────────────────────────┘
        ↓
   three numbers  →  selector  →  one strategy
```

### 6.2 The settings

| Setting | Pilot | Full run | Why |
|---|---|---|---|
| Training pairs | **25%** (~4,800) | 100% | The pilot is a measuring instrument, not a product |
| Epochs | **1** | chosen on dev from {1, 2, 3} | One epoch is enough to separate the strategies; more just costs time |
| Seeds | **1** (seed 13) | 3 (13, 42, 1337) | One seed is noisy — which is exactly why §6.4 exists |
| Neighbourhood `k` | 64 | 64 | Unchanged — the strategies must be identical to the research ones |
| Batch size | 32 | 32 | Unchanged |
| Learning rate | 2e-5 | 2e-5 | Unchanged |
| **Dev queries** | **all 200** | all 200 | **Do not shrink the evaluation** — see below |
| Mining | same code | same code | Unchanged |

**Shrink the training, never the evaluation.** Encoding the corpus and scoring 200 queries against
~14,000 documents is a single matrix multiply — milliseconds. Cutting dev from 200 queries to 50
would triple the noise on the very number the decision rests on, to save essentially nothing.
This is the most common way a pilot design goes wrong.

### 6.3 Why one seed is defensible — and what it costs

One seed per strategy means the pilot comparison is **paired but not replicated**: all three
strategies see the same data order and the same initialisation, which removes one source of
difference, but there is no way to estimate run-to-run variance from the pilot alone.

That is acceptable **only because the pilot is not the final word**. The final model is still
trained with three seeds, and the system never reports the pilot scores as results. If a second
pilot seed is affordable, run it — but the default is one, and the `SelectionReport` records the
seed so the choice is reproducible.

### 6.4 How to stop the pilot becoming another huge experiment

Hard caps, set once and not revisited:

```text
  Rule 1 · Three strategies. Not four, not ten.
  Rule 2 · One epoch, one seed, 25% of data. No exceptions "just this once".
  Rule 3 · No hyperparameter tuning inside the pilot. The pilot inherits the
           research's fixed settings (lr 2e-5, batch 32, k=64).
  Rule 4 · No re-running the pilot until the answer looks right. If you find
           yourself doing this, the effect is within noise — say so and move on.
  Rule 5 · Everything is logged to pilot_report.json, including the margin and
           whether the selector was confident.
```

Rule 4 is the important one. It is the same discipline as pre-registering the research's three
contrasts: you do not get to keep sampling until you like the result.

### 6.5 What the pilot actually costs — honestly

Estimated on a single T4, MiniLM, ~19,000 pairs. **These are planning estimates, not
measurements** — replace them with `domembed bench` output once the pilot runner exists.

| Stage | Pilot (×3) | Full sweep (3 strategies × 3 seeds) |
|---|---|---|
| Mining | seconds | seconds |
| Training | ~1 min each → **~3 min** | ~5 min per run → **~45–60 min** |
| Corpus encoding for eval | ~20 s each → **~1 min** | shared/cacheable |
| Scoring | milliseconds | milliseconds |
| **Total** | **~5 minutes** | **~50–70 minutes** |

Then the adaptive pipeline still runs one full training (3 seeds, ~15 min).

So the honest accounting is:

```text
   full 3×3 research sweep     ~50–70 min   →  a comparison table
   adaptive (pilots + 1 run)   ~20 min      →  a trained model
```

**That is a saving of tens of minutes, not hours or days.** At this scale — a 23M-parameter model
and a 14K-document corpus — compute is not scarce, and the pilot should not be justified as an
efficiency breakthrough. Its real value is **automation and reproducibility**: the user does not
have to run three experiments and interpret a table, and every selection leaves an auditable
record. Any claim of large compute savings would be false at this scale, and a reviewer would
spot it immediately.

### 6.6 The check that makes the whole thing credible

```text
   On AskUbuntu we will ALREADY have full runs for N1, N2 and N3 (× 3 seeds).
   So the first job of the pilot runner is to reproduce their ranking cheaply.

   ┌───────────────────────────────────────────────────────────────┐
   │  Strategy   pilot Recall@10   full Recall@10   rank agreement │
   │  ────────────────────────────────────────────────────────────│
   │  Random          0.54              0.58              ✓        │
   │  BM25            0.61              0.63              ✓        │
   │  Semantic        0.57              0.60              ✓        │
   └───────────────────────────────────────────────────────────────┘
                                          [ILLUSTRATIVE NUMBERS]

   If the pilot's ORDER matches the full run's ORDER → the selector is credible.
   If it does not                                   → report that, and ship the
                                                       selector with a caveat.
```

This costs one extra table and no new training runs, because the full results already exist from
the research. **It is the single most valuable experiment in v2** and it is Phase 4.

---

## 7. The final training stage

```text
   selected strategy (e.g. N2 · BM25)
        ↓
   fresh initialisation from all-MiniLM-L6-v2
        ↓
   ALL training pairs (~19,000)
        ↓
   full contrastive fine-tuning
        · MultipleNegativesRankingLoss, scale 20
        · batch 32, k = 64 neighbourhoods
        · epochs chosen on dev, lr 2e-5
        · 3 seeds: 13, 42, 1337
        ↓
   report mean ± range over seeds
        ↓
   export:  checkpoint + model_info.json + pilot_report.json
        ↓
   DomainEmbedder.load("./models/mydomain")
        ↓
   encode · similarity · search · info
```

Three details that matter:

1. **Fresh initialisation.** The pilot checkpoint is never continued from. It was trained on a
   quarter of the data for one epoch; it is a measurement device, not a head start.
2. **The pilot report travels with the model.** `model_info.json` gains two fields:
   `strategy_selected` and `pilot_report`. So `domembed`'s `info()` can answer *"which negative
   strategy does this model use, and how was that decided?"* — the research question, visible
   inside the product.
3. **The pilot is not evidence about the final model.** The final model's numbers come from the
   real evaluation on the test split, exactly as in the current research. The pilot scores appear
   only in the provenance panel, clearly labelled as pilot scores.

---

## 8. Architecture

### 8.1 Where the new code lives

The v0.1 spec has an explicit non-goal: *"A training API — `train.py` in the research repo does
this."* Adding a trainer to the inference library would break that. But the brief asks for
`from domembed import AutoDomainEmbedder`.

**Resolution:** the adaptive code lives in a submodule, `domembed.adaptive`, behind an optional
extra. A lazy top-level re-export gives the requested import path without dragging training
dependencies into the inference install.

```python
# domembed/__init__.py
def __getattr__(name):
    if name == "AutoDomainEmbedder":
        try:
            from domembed.adaptive import AutoDomainEmbedder
            return AutoDomainEmbedder
        except ImportError as e:
            raise ImportError(
                "AutoDomainEmbedder needs the training extra:  pip install -e '.[fit]'"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

So `from domembed import AutoDomainEmbedder` works as the brief specifies, and `pip install
domembed` stays light for someone who only wants to *use* a model.

### 8.2 The components

| Component | New? | ~Lines |
|---|---|---|
| `domembed/adaptive/profile.py` | new | 90 |
| `domembed/adaptive/strategies.py` | new | 70 |
| `domembed/adaptive/pilot.py` | new | 120 |
| `domembed/adaptive/selector.py` | new | 60 (includes `SelectionReport`) |
| `domembed/adaptive/pipeline.py` | new | 110 |
| `src/prepare_data.py` | **reused** | — |
| `src/negatives.py` | **reused** (has `build_neighbourhoods` + `NeighbourhoodBatchSampler`) | — |
| `src/train.py`, `src/evaluate.py`, `src/utils.py` | **reused** | — |
| `domembed/embedder.py` and friends | **reused** (v0.1) | — |
| `domembed/cli.py` | extended with `fit`, `profile` | +60 |

**~450 new lines.** The research repo already contains the expensive parts; the adaptive layer is
glue around them.

### 8.3 What each component does

#### `profile.py` — `DomainProfile`

| | |
|---|---|
| **Does** | Computes the eight statistics from §5 and runs the guard-rail checks |
| **Input** | `corpus.jsonl`, `train_pairs.tsv`, `qrels.json`, the base model |
| **Output** | A `DomainProfile` dataclass → JSON |
| **Why** | Provenance, guard rails, hypotheses. **Does not select.** |

```python
profile = profile_domain(corpus, pairs, qrels, model)
print(profile.guard_rails)     # ["ok: enough pairs", "warn: high neighbour_similarity"]
```

#### `strategies.py` — the strategy registry

| | |
|---|---|
| **Does** | Names the three strategies and maps each to its mining function and relative cost. **Delegates to the research repo's `negatives.py` — duplicates none of it.** |
| **Input** | Strategy name |
| **Output** | A `Strategy` (name, mining function, cost, description) |
| **Why** | Turns three hardcoded experimental branches into a list you can loop over. This is what makes the system extensible without adding strategies. |

```python
STRATEGIES = {
    "random":   Strategy("random",   mine=None,                        cost=0,
                         desc="any other pair in the batch"),
    "bm25":     Strategy("bm25",     mine=mine_bm25_neighbourhoods,    cost=1,
                         desc="top-64 by BM25 among training anchors"),
    "semantic": Strategy("semantic", mine=mine_cosine_neighbourhoods,  cost=2,
                         desc="top-64 by cosine of the zero-shot encoder"),
}
```

`mine_bm25_neighbourhoods` and `mine_cosine_neighbourhoods` are thin adapters over
`src/negatives.py::build_neighbourhoods`, which already implements both.

#### `pilot.py` — `run_pilots()`

| | |
|---|---|
| **Does** | For each strategy: subset 25% of pairs, build neighbourhoods, train 1 epoch at seed 13, evaluate on the **full** dev split. Returns one `PilotResult` per strategy. |
| **Input** | Pairs, corpus, qrels, strategy list, `pilot_fraction` |
| **Output** | `dict[str, PilotResult]` with all three metrics, the seed and the wall time |
| **Why** | This is the only genuinely new *experimental* machinery. Everything else is orchestration. |

```python
pilots = run_pilots(pairs, corpus, qrels, strategies=["random", "bm25", "semantic"])
for name, p in pilots.items():
    print(f"{name:10s} Recall@10 = {p.recall_at_10:.3f}   ({p.wall_time_s:.0f}s)")
```

#### `selector.py` — `select()`

| | |
|---|---|
| **Does** | `argmax` over the pilot scores with the margin tie-break from §4.5. Checks metric agreement. Builds a serialisable `SelectionReport`. |
| **Input** | `dict[str, float]` scores, costs, `margin` |
| **Output** | `SelectionReport(strategy, scores, margin, confident, reason)` |
| **Why** | Separates *the decision* from *the measurement*. ~40 lines, unit-testable with a dictionary — no model required. |

#### `pipeline.py` — `AutoDomainEmbedder`

| | |
|---|---|
| **Does** | Orchestrates stages 1–8: prepare → profile → pilot → select → full train → export. Returns a `DomainEmbedder`. |
| **Input** | A dataset id or a prepared data directory, plus an output path |
| **Output** | A trained, exported model **and** a `DomainEmbedder` ready to use |
| **Why** | The only component a user sees. Everything above is internal. |

### 8.4 Why these filenames and not the suggested ones

The brief suggested `analyzer.py, negatives.py, pilot.py, selector.py, trainer.py, model.py,
search.py, cli.py`. Deviations, all deliberate:

| Suggested | Decision | Reason |
|---|---|---|
| `negatives.py` | **Not duplicated** | The research repo already has `src/negatives.py` with `build_neighbourhoods` and `NeighbourhoodBatchSampler`. A second one would create two sources of truth for the exact thing the experiment manipulates. `strategies.py` wraps it instead. |
| `trainer.py` | **Not created** | `src/train.py` exists. `pilot.py` calls it with different arguments; it does not reimplement training. |
| `model.py`, `search.py` | **Not created** | Already `domembed/embedder.py` and `similarity.py` in v0.1. |
| `analyzer.py` | **Renamed** `profile.py` | "Analyzer" implies it draws conclusions. It does not — see §5.1. |
| `pilot.py`, `selector.py`, `cli.py` | **Kept** | Good names, kept as-is. |

---

## 9. The user-facing API

### 9.1 Training and inference are different operations

```python
# TRAINING — slow. Analyses, pilots, selects, trains, exports to disk.
from domembed import AutoDomainEmbedder

model = AutoDomainEmbedder.fit(
    dataset="sentence-transformers/askubuntu",
    output_dir="./models/askubuntu",
)
```

```python
# INFERENCE — fast. Loads what fit() produced.
from domembed import DomainEmbedder

model = DomainEmbedder.load("./models/askubuntu")
```

`fit()` **returns a `DomainEmbedder`**, so the two converge on one class immediately after
training. There is no second model type to learn:

```python
model = AutoDomainEmbedder.fit(dataset="...", output_dir="./models/mydomain")

# model is a DomainEmbedder — the same object the v0.1 library documents
model.encode("Why is my WiFi not working?")
model.similarity(text_a, text_b)
model.search(query, documents, top_k=5)
```

### 9.2 What `fit()` prints

```text
  domembed adaptive fit · sentence-transformers/askubuntu

  [1/6] preparing dataset .................. 14,100 docs · 19,000 pairs
  [2/6] profiling domain ................... lexical_signal 2.8 · semantic_signal 1.6
        ⚠ neighbour_similarity 0.71 is high: semantic mining may surface
          unlabelled duplicates
  [3/6] running pilots (25% · 1 epoch · seed 13)
        random      Recall@10 = 0.54   (48s)
        bm25        Recall@10 = 0.61   (61s)
        semantic    Recall@10 = 0.57   (73s)
  [4/6] selecting .......................... BM25  (margin 0.04 over runner-up)
  [5/6] full training (3 seeds · 2 epochs) . done
  [6/6] exporting .......................... ./models/askubuntu
                                                          [ILLUSTRATIVE NUMBERS]
```

### 9.3 Parameters, kept few

```python
AutoDomainEmbedder.fit(
    dataset,                    # HF dataset id, or a prepared data directory
    output_dir,                 # where the model + reports go
    strategies="all",           # or ["random", "bm25"] to skip one
    pilot_fraction=0.25,        # fraction of pairs used for pilots
    pilot_epochs=1,
    pilot_seed=13,
    selection_metric="recall@10",
    margin=0.02,                # tie-break threshold
    seeds=(13, 42, 1337),       # seeds for the FULL run only
    device=None,                # auto
)
```

Every parameter has a default that matches the research configuration. The point of the library
is that none of them need to be touched.

### 9.4 Profiling without training

```python
from domembed.adaptive import profile_domain

profile = profile_domain(dataset="sentence-transformers/askubuntu")
print(profile.to_dict())
```

Useful on its own: it is the cheapest way to find out whether a new dataset is even suitable,
before spending any GPU time.

### 9.5 CLI

```bash
domembed profile sentence-transformers/askubuntu

domembed fit sentence-transformers/askubuntu \
    --output ./models/askubuntu \
    --pilot-fraction 0.25 \
    --json-run-report ./runs/askubuntu-adaptive.json
```

---

## 10. End-to-end example: AskUbuntu

> **All numbers in this section are illustrative.** The research has not been run yet.

### 10.1 Input

```text
  sentence-transformers/askubuntu
  AskUbuntu (Stack Exchange) duplicate questions
  13,145 rows → train 12,745 · dev 200 · test 200
  Each row: a question, plus 1–3 questions the community marked as duplicates
```

### 10.2 Analyse the domain

```text
  ┌──────────────────────────────────────────────────────────────────┐
  │  n_documents                 14,100                              │
  │  n_positive_pairs            19,000                              │
  │  duplicate_density           1.5                                 │
  │  mean_token_length           41 tokens                           │
  │  lexical_signal_ratio        2.8                                 │
  │  semantic_signal_ratio       1.6                                 │
  │  neighbour_similarity        0.71                                │
  │                                                                  │
  │  ⚠ high neighbour_similarity → semantic mining may surface        │
  │    unlabelled duplicates. (Consistent with Lei et al. (2016):    │
  │    only ~5% of similar pairs in this corpus are annotated.)      │
  └──────────────────────────────────────────────────────────────────┘
                                                      [ILLUSTRATIVE]
```

The warning is exactly the hypothesis the current research is designed to test: the semantic
strategy's hardest negatives may be genuine duplicates nobody labelled. **Profiling raises the
warning; only the pilot decides.**

### 10.3 Candidate strategies

```text
  N1  Random      — any other pair in the batch
  N2  BM25        — top-64 by BM25 among training anchors
  N3  Semantic    — top-64 by cosine of the zero-shot encoder
```

No others. The brief is explicit about not adding strategies, and there is no reason to.

### 10.4 Pilot

```text
  ┌──────────────────────────────────────────────────────────────────┐
  │  25% of pairs (~4,800) · 1 epoch · seed 13 · dev = all 200       │
  │                                                                  │
  │   Strategy     Recall@10    MRR@10    nDCG@10    wall time       │
  │   ──────────────────────────────────────────────────────────     │
  │   Random         0.54        0.41       0.46        48 s         │
  │   BM25           0.61        0.48       0.53        61 s         │
  │   Semantic       0.57        0.44       0.49        73 s         │
  │                    ↑                                             │
  │                 best, and 0.04 clear of the runner-up            │
  └──────────────────────────────────────────────────────────────────┘
                                                      [ILLUSTRATIVE]
```

Note the pattern: **BM25 > Semantic > Random.** The semantic strategy beats random but loses to
BM25 — which is what an inverted-U would look like, and what the false-negative argument
predicts. This is precisely the shape the current research is looking for. (It is also exactly
the pattern one should *not* assume in advance — if the research finds a different order, the
illustrative numbers here are simply wrong and will be replaced.)

### 10.5 Selection

```text
  argmax {Random: 0.54, BM25: 0.61, Semantic: 0.57}  =  BM25
  margin over runner-up = 0.04  >  0.02 threshold  →  confident = True
```

### 10.6 Full training

```text
  fresh all-MiniLM-L6-v2
  all 19,000 pairs · 2 epochs (chosen on dev) · lr 2e-5 · batch 32 · k = 64
  seeds 13, 42, 1337
  BM25 neighbourhoods for batch composition
        ↓
  exported to ./models/askubuntu with model_info.json + pilot_report.json
```

### 10.7 The result, in use

```python
from domembed import DomainEmbedder

model = DomainEmbedder.load("./models/askubuntu")

print(model.info())
```

```text
  Model:       domembed-askubuntu-adaptive
  Base model:  sentence-transformers/all-MiniLM-L6-v2
  Domain:      AskUbuntu — Ubuntu/Linux technical support
  Dimension:   384    Max seq: 256

  Strategy selected automatically
    Candidates:   random (0.54) · bm25 (0.61) · semantic (0.57)
    Selected:     bm25
    Margin:       +0.04 over runner-up   [confident]
    Pilot config: 25% of pairs · 1 epoch · seed 13 · dev = 200 queries
                                                              [ILLUSTRATIVE]

  Evaluation (200 test queries)
    Recall@10 = 0.???   ← filled from results/metrics.json, never typed
```

```python
model.search("Ubuntu WiFi stopped working after update", corpus, top_k=3)
```

```text
  0.88  Wireless not working after upgrading to a new kernel
  0.84  Wifi disconnected after system update
  0.79  How do I reinstall the wifi driver?
                                                              [ILLUSTRATIVE]
```

---

## 11. Second example: what "domain-flexible" would mean

> **HYPOTHETICAL USE CASE.** This domain has **not** been tested. Nothing here is a claim about
> results — it exists only to show what "domain-flexible" means mechanically.

### Future example — StackOverflow duplicate questions

```text
  INPUT    StackOverflow duplicate-question pairs (~100K pairs, longer texts,
           code snippets, many distinct tags)

  PROFILE  mean_token_length           higher than AskUbuntu (code blocks)
           vocabulary_size             much larger (many languages, libraries)
           lexical_signal_ratio        1.4   ← lower: duplicates share fewer words
           semantic_signal_ratio       1.9   ← higher: base model separates them well

  HYPOTHESIS (not a decision)
           Lower lexical signal → BM25 neighbourhoods may be less informative.
           Higher semantic signal → semantic mining may be less contaminated.

  PILOT    random   →  ?.??
           bm25     →  ?.??
           semantic →  ?.??

  SELECT   whichever wins — or, if within the margin, whichever is cheaper

  OUTPUT   a StackOverflow-domain encoder, exported identically
                                                              [ILLUSTRATIVE]
```

The point of the example is the **mechanism is unchanged**: same three strategies, same pilot
shape, same selector, same export. Only the numbers differ. That is what "domain-flexible" buys —
not a claim that the same strategy wins everywhere.

### Other hypothetical domains

| Domain | Positive pairs would come from | Extra caution |
|---|---|---|
| **Bug reports** | Duplicates marked in an issue tracker | Very high lexical overlap between distinct bugs — semantic mining is risky |
| **Medical text** | Requires a curated corpus with real pair labels | Do not invent clinical claims from embeddings; evaluation needs domain input |
| **Legal documents** | Cited / linked cases | Long documents — the 256-token limit would need revisiting first |

In every case the framework is the same and **the evidence requirement is the same**: you need
naturally occurring positive pairs and a held-out split. Without those, the pilot has nothing to
measure and the selector has nothing to select.

---

## 12. What is novel and what is not

This section exists because the honest answer is mostly "not novel", and saying so is worth more
than a claim that collapses under one question.

### 12.1 The three contributions, stated separately

| | Contribution | Status |
|---|---|---|
| **Current research** | Studying how different negative-construction strategies affect contrastive domain adaptation, on one domain, with one controlled variable and a measured false-negative rate | A small, well-controlled empirical study |
| **Current product (v0.1)** | Packaging the resulting encoder as an installable library with a provenance record | Engineering, not research |
| **Future adaptive extension (v2)** | Automatically selecting a negative-construction strategy for a new domain using cheap pilots | **Mostly engineering. See 12.2.** |

### 12.2 Automatic strategy selection is not novel

Three separate things are being combined, and **each is already established**:

| Component | Already known as | Reference |
|---|---|---|
| "Pick the configuration with the best validation score" | **Model selection / hyperparameter optimisation.** Textbook. | Any ML textbook |
| "Use a short cheap run to stand in for a full one" | **Multi-fidelity HPO.** Successive halving and Hyperband do exactly this, with a principled early-stopping schedule. | [Li et al., Hyperband, arXiv:1603.06560](https://arxiv.org/abs/1603.06560) (ICLR 2017 / JMLR 2018) |
| "Adapt the negative sampling" | **Adaptive / dynamic negative sampling** is an established term in recommender systems. | [Shi et al., DNS, WWW 2023](https://doi.org/10.1145/3543507.3583849); [Lai et al., AHNS, AAAI 2024](https://arxiv.org/abs/2401.05191) |

The recsys work deserves a precise distinction, because it is the closest prior art:

* **DNS and AHNS adapt hardness *during* training, per example, using the model's current scores.**
  AHNS's motivation is that fixed-hardness samplers suffer both a false-positive problem and a
  false-negative problem.
* **Our selector picks one fixed strategy *before* training, using a validation pilot.**

Different axis, different mechanism, different cost profile. But "adaptive negative sampling" is
not an empty space, and anyone claiming novelty here has to say clearly what they mean by
"adaptive".

There is also a useful *connection*: AHNS's "false negative problem" — hard negatives that are
actually positives — is the same phenomenon our N3 condition is expected to exhibit, in text, for
the reason [Lei et al. (2016)](https://aclanthology.org/N16-1153/) measured (only ~5% of similar
pairs annotated). Finding the same failure mode named in an adjacent field is reassuring about
the research hypothesis and deflating about its novelty. **Both are worth knowing.**

### 12.3 What is genuinely open — and unproven

| Question | Status | What would settle it |
|---|---|---|
| Do cheap pilots predict full-run rankings **for this specific choice**? | **Unknown.** This is the real question. | The Phase 4 fidelity check (§6.6), on every domain available |
| Does the best strategy **transfer** across domains? | **Unproven.** One domain proves nothing. | The same pipeline on 3+ domains |
| Do domain statistics **predict** the best strategy? | **Unproven and under-powered.** | Many domains; a correlation study |
| Is a pilot-selected strategy **better than just always using BM25**? | **Unknown — and this is the uncomfortable one.** | Compare adaptive selection against the fixed-default baseline (§16) |

That last row matters. If BM25 wins on every domain tried, then the adaptive system is a
complicated way of choosing BM25, and the honest report says so.

### 12.4 Required literature work before any novelty claim

Do not write the word "novel" until these searches have been run and read:

| # | Search | Why |
|---|---|---|
| 1 | Multi-fidelity HPO: Hyperband, successive halving, ASHA, BOHB | The pilot is an instance of this. If a published method already covers it, cite it rather than reinvent it |
| 2 | **Low-fidelity bias** — when do short runs misrank configurations? | Directly threatens the whole design. Multi-fidelity work is explicit that cheap evaluations are *biased*, not just noisy |
| 3 | Learning-curve extrapolation / performance prediction from short runs | The mature version of "can a short run predict a long one" |
| 4 | "Adaptive negative sampling" in **both** recsys and IR/NLP | Establish the exact sense in which each is "adaptive" |
| 5 | AutoML applied to contrastive learning or hard-negative hyperparameters | Might already exist as a published pipeline |
| 6 | Any paper that selects a negative-sampling strategy **per dataset** | The closest possible prior art; if it exists, this becomes a replication |

Search 2 is the one that could invalidate the design, so run it first.

### 12.5 The conservative claim

If the pilots do turn out to predict the full runs, the defensible sentence is:

> *"We implemented and evaluated an inexpensive pilot-based procedure for selecting a
> negative-construction strategy for contrastive domain adaptation, and measured the extent to
> which short runs predict full runs on [N] domains."*

That is a **systems-and-measurement contribution**, not an algorithmic one. It is a perfectly
respectable undergraduate contribution. Claiming more would not make it bigger; it would just
make it wrong.

---

## 13. Scope control

### MUST HAVE — the smallest functioning adaptive system

| # | Item | Notes |
|---|---|---|
| M1 | `strategies.py` — registry over the existing three | Wraps `src/negatives.py`. Adds no strategies |
| M2 | `profile.py` — the eight statistics + guard rails | Provenance and warnings only |
| M3 | `pilot.py` — 25% / 1 epoch / 1 seed, all 200 dev queries | Fixed caps, no tuning |
| M4 | `selector.py` — `argmax` + margin tie-break + `SelectionReport` | ~40 lines, unit-testable |
| M5 | `pipeline.py` — `AutoDomainEmbedder.fit()` | Orchestration, returns a `DomainEmbedder` |
| M6 | Export integration — `strategy_selected` and `pilot_report` into `model_info.json` | So `info()` shows the rationale |
| M7 | **The pilot-fidelity check** (§6.6) against the known AskUbuntu full runs | **Not optional** — without it the selector is unvalidated |
| M8 | Tests for `selector.py` with hardcoded score dictionaries | No model needed; milliseconds |

### SHOULD HAVE

| # | Item |
|---|---|
| S1 | CLI: `domembed fit`, `domembed profile` |
| S2 | `pilot_report.json` written for every run, including wall-clock times |
| S3 | Profile stored in `model_info.json` |
| S4 | A **second real domain**, to make the transfer question testable at all |
| S5 | A `profile-only` mode that costs no GPU time |
| S6 | Metric-agreement check (Recall@10 vs MRR@10) feeding `confident` |

### FUTURE — do not attempt in this project

This is a **do-not-do list**. Each item is a trap, not a goal.

| # | Item | Why it is out |
|---|---|---|
| F1 | Learned meta-selector across many domains | Needs dozens of domains; we have one |
| F2 | Reinforcement learning for strategy selection | Disproportionate to a three-way choice |
| F3 | LLM-based reasoning about which strategy to use | An LLM has no access to pilot measurements, which are the only real evidence |
| F4 | More than three negative strategies | The brief forbids it and there is no reason to |
| F5 | Neural architecture search / encoder selection | Different problem, huge search space |
| F6 | Multi-domain meta-training | Needs the domains first |
| F7 | Continuous / online adaptation during training | That is AHNS-family work; a different axis (§12.2) |
| F8 | Bayesian optimisation over pilot configurations | Overkill for 3 arms; Hyperband-style scheduling is the mature answer and belongs in F1's budget, not here |
| F9 | Automatic hyperparameter tuning of lr / epochs / batch | Confounds the variable the research isolates |
| F10 | AutoML platform features (schedulers, dashboards, distributed runs) | Infrastructure, not research |

---

## 14. The research / product boundary

```text
                                  EmbedEd
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                RESEARCH                          PRODUCT
                    │                                 │
        N1 / N2 / N3 comparison              auto strategy selection
        (3 conditions × 3 seeds)             (pilot → argmax → full run)
                    │                                 │
        metrics: Recall@10 / MRR@10 / nDCG@10   training + export API
                    │                                 │
        false-negative measurement             model_info.json provenance
        (50 queries × top-5, by hand)                  │
                    │                                 │
                    └────────────────┬────────────────┘
                                     │
                              trained Model
                                     │
                                     ▼
                                 domembed
                        encode · similarity · search · info
```

### Why the boundary matters scientifically

| Reason | Explanation |
|---|---|
| **The product must not contaminate the experiment** | If training code imports the product, a bug in the library becomes a bug in the results. The v0.1 spec already enforces this: *"the research repo never imports `domembed`."* The same rule applies to `domembed.adaptive`, which is why it lives behind an extra. |
| **Selection is not evaluation** | The pilot chooses a strategy. It does **not** produce the reported numbers. The reported numbers come from the test split, via the same `evaluate.py` the research uses. Conflating them is how a pilot score ends up quoted as a result. |
| **The research question stays answerable** | "Does the negative strategy matter?" has to be asked with all three strategies trained to completion. An adaptive pipeline that only trains the winner can never answer it. Keeping the research path intact is what makes the product's premise meaningful. |
| **One direction of dependency** | Research → product, never back. `export_model.py` reads `results/metrics.json`; nothing in `src/` reads anything from `domembed/`. |

In v2 the only new crossing is **`pilot.py` calling `src/train.py` and `src/evaluate.py`** — still
inside the research half. The product half (`domembed/embedder.py` and friends) is not involved
until the finished model is loaded.

---

## 15. Implementation milestones

Difficulty is relative: ★ easy, ★★ moderate, ★★★ the part that will take real thought.

### Phase 1 — Refactor current `domembed` (prerequisite)

| | |
|---|---|
| **Files** | `domembed/embedder.py`, `pyproject.toml` (add the `[fit]` extra) |
| **Output** | `domembed` installable with and without training deps; `AutoDomainEmbedder` re-export stub in place |
| **Difficulty** | ★ |
| **Depends on** | v0.1 library complete |

### Phase 2 — Add the strategy interface

| | |
|---|---|
| **Files** | `domembed/adaptive/strategies.py` |
| **Output** | `STRATEGIES` registry; `for s in STRATEGIES: mine(s)` produces the same neighbourhoods the research produces — verified by comparing against `data/nb_*.json` from the research runs |
| **Difficulty** | ★★ |
| **Depends on** | Phase 1; `src/negatives.py` |
| **Check** | Neighbourhoods are **identical** to the research's. This is not a place to be approximately right. |

### Phase 3 — Add pilot training

| | |
|---|---|
| **Files** | `domembed/adaptive/pilot.py` |
| **Output** | `run_pilots()` returns three `PilotResult`s with all metrics, seed and wall time |
| **Difficulty** | ★★ |
| **Depends on** | Phase 2, `src/train.py`, `src/evaluate.py` |
| **Check** | Three pilots complete in the estimated ~5 minutes; dev split is the full 200 queries |

### Phase 4 — Add the selector **and run the fidelity check**

| | |
|---|---|
| **Files** | `domembed/adaptive/selector.py`, `domembed/adaptive/profile.py` |
| **Output** | `select()` with margin tie-break; **the pilot-vs-full-run ranking table (§6.6)** |
| **Difficulty** | ★★★ — the check is easy to run and hard to interpret honestly |
| **Depends on** | Phase 3 **and the finished research results** |
| **Check** | Pilot ranking vs full-run ranking on AskUbuntu, written down **before** any claim is made about the selector working |

> Phase 4 is the gate. If the fidelity check fails, stop and report that, rather than proceeding
> to Phase 5 with an unvalidated selector.

### Phase 5 — Full training + export

| | |
|---|---|
| **Files** | `domembed/adaptive/pipeline.py`, `scripts/export_model.py` (extended) |
| **Output** | `AutoDomainEmbedder.fit()` end to end; `model_info.json` carries `strategy_selected` and `pilot_report` |
| **Difficulty** | ★★ |
| **Depends on** | Phase 4 |

### Phase 6 — CLI and demo

| | |
|---|---|
| **Files** | `domembed/cli.py`, demo page |
| **Output** | `domembed fit` and `domembed profile`; a demo panel showing the pilot table next to the final model |
| **Difficulty** | ★ |
| **Depends on** | Phase 5 |

### Dependency chain

```text
  Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4 ──→ Phase 5 ──→ Phase 6
                                        ▲
                                        │
                          the current research must be
                          FINISHED here (full N1/N2/N3 results)
```

Nothing in Phases 1–3 requires the research to be complete — they can be built against the base
model. **Phase 4 cannot start early, and must not be faked.**

---

## 16. Risks and mitigations

| # | Risk | Why it is real | Mitigation |
|---|---|---|---|
| 1 | **Pilot results are noisy** | One seed, 25% of data, 200 dev queries. The research's own spec says sub-3-point differences are noise at full scale; a pilot is noisier still | Margin tie-break (§4.5); metric-agreement check; log the seed; never treat a small margin as a finding |
| 2 | **Validation performance may not generalise to test** | Selecting on dev and reporting on dev is selection bias | Select on dev, **report on test** — exactly as the research does. The pilot scores appear only in the provenance panel, never in the results table |
| 3 | **Semantic mining creates false negatives** | It is the expected failure mode of N3, and the reason the inverted-U is predicted | This is the research's central measurement. The pilot inherits it: if semantic mining is contaminated, the pilot will simply rank it low. Record `neighbour_similarity` as the cheap proxy |
| 4 | **Strategy and hardness are not perfectly separable** | "Harder" and "more likely to be an unlabelled duplicate" are nearly the same property in this corpus | Already stated as a threat to validity in the research. The adaptive system cannot fix it; it can only avoid selecting the strategy that triggers it |
| 5 | **One domain cannot establish universality** | AskUbuntu is one corpus with one labelling convention | Say it. Every claim is scoped to the domains tested. A second domain (S4) is the only real mitigation |
| 6 | **Pilot overhead** | Three extra training runs before the real one | ~5 minutes at this scale (§6.5). Measurable and logged — if it grows, the pilot fraction is the dial |
| 7 | **The selector may just be choosing the strongest baseline** | If BM25 always wins, "adaptive selection" is a complicated constant function | **Run the fixed-default baseline**: always-BM25 vs adaptive. Report both. If they tie, say so — that is a finding, not a failure |
| 8 | **Low-fidelity bias: short runs misrank** | Multi-fidelity HPO literature is explicit that cheap evaluations are *biased*, not merely noisy | The Phase 4 fidelity check is the entire mitigation. If pilots misrank at 25%, try 50% before abandoning |
| 9 | **Profile statistics get treated as predictors** | A tempting, unfounded shortcut | §5.1: the profile is characterisation only. `select()` takes scores, never profile features |
| 10 | **Scope creep into AutoML** | Every item in §13 F1–F10 is a plausible-sounding "improvement" | The FUTURE list is a do-not-do list, and rule 4 in §6.4 forbids tuning the pilot |

Risk 7 deserves emphasis: **the adaptive system's most likely honest result is "the strategies
differ less than the noise, so it does not matter much which you pick."** That is a legitimate
outcome and the plan is written so it can be reported rather than hidden.

---

## 17. Recommended architecture

```text
  ═══════════════════════════════════════════════════════════════════════

    NEW DOMAIN  (a dataset with naturally occurring positive pairs)
        │
        ▼
    ┌───────────────────────────────────────────────────────────────┐
    │  src/prepare_data.py            [EXISTS]                      │
    │  corpus · qrels · train pairs · duplicate groups              │
    └───────────────────────────────┬───────────────────────────────┘
                                    ▼
    ┌───────────────────────────────────────────────────────────────┐
    │  domembed.adaptive.profile        [NEW · 90 lines]            │
    │  8 statistics · guard rails · hypotheses. SELECTS NOTHING.    │
    └───────────────────────────────┬───────────────────────────────┘
                                    ▼
    ┌───────────────────────────────────────────────────────────────┐
    │  domembed.adaptive.strategies     [NEW · 70 lines]            │
    │  random · bm25 · semantic   ──wraps──▶  src/negatives.py      │
    │                                        [EXISTS — untouched]   │
    └───────────────────────────────┬───────────────────────────────┘
                                    ▼
    ┌───────────────────────────────────────────────────────────────┐
    │  domembed.adaptive.pilot          [NEW · 120 lines]           │
    │  25% pairs · 1 epoch · 1 seed · calls src/train.py            │
    └───────────────────────────────┬───────────────────────────────┘
                                    ▼
    ┌───────────────────────────────────────────────────────────────┐
    │  src/evaluate.py                [EXISTS]                      │
    │  Recall@10 / MRR@10 / nDCG@10 on the full 200-query dev split │
    └───────────────────────────────┬───────────────────────────────┘
                                    ▼
    ┌───────────────────────────────────────────────────────────────┐
    │  domembed.adaptive.selector       [NEW · 60 lines]            │
    │  argmax  +  margin tie-break  +  SelectionReport              │
    └───────────────────────────────┬───────────────────────────────┘
                                    ▼
                            selected strategy
                                    ▼
    ┌───────────────────────────────────────────────────────────────┐
    │  src/train.py  (full)           [EXISTS]                      │
    │  all pairs · 3 seeds · the one selected strategy              │
    └───────────────────────────────┬───────────────────────────────┘
                                    ▼
    ┌───────────────────────────────────────────────────────────────┐
    │  scripts/export_model.py        [EXISTS · extended]           │
    │  model_info.json += strategy_selected, pilot_report, profile  │
    └───────────────────────────────┬───────────────────────────────┘
                                    ▼
    ┌───────────────────────────────────────────────────────────────┐
    │  domembed  (v0.1)               [EXISTS]                      │
    │  encode · similarity · search · info                          │
    └───────────────────────────────────────────────────────────────┘

  ═══════════════════════════════════════════════════════════════════════
```

### What we build now

The **current research**: three full fine-tuning runs of `all-MiniLM-L6-v2` on AskUbuntu,
identical except for how negatives were constructed, evaluated with Recall@10 / MRR@10 / nDCG@10
over three seeds, with a hand-measured false-negative rate. Plus **v0.1 of `domembed`**: a small
library that loads any of those models and exposes `encode` / `similarity` / `search` / `info`,
with a Streamlit demo comparing them side by side.

### What the current research proves

Whether, and how much, the negative-construction strategy matters **on this corpus with this
encoder**. Nothing more. Not that BM25 is best in general; not that the finding transfers.

### What the adaptive library would add

About **450 lines** that turn "here is a comparison" into "give me a dataset, get back a model":
a strategy registry over the same three strategies, a domain profile for provenance and guard
rails, a cheap three-way pilot, and an `argmax` with an honest tie-break.

It adds **no new strategy, no new loss, no new architecture, and no claim to novelty.** What it
could add — if the fidelity check passes — is a measured answer to a question nobody has answered
for this setting: *do short runs tell you which negatives to use?*

---

### The three questions, one last time

| | Question |
|---|---|
| **CURRENT project** | *"What happens when we change the negative strategy?"* |
| **FUTURE domembed** | *"Given a new domain, which negative strategy should we use?"* |

And the bridge:

> **Run small controlled pilots and select the strategy that performs best on validation.**

---

## 18. Appendix — mapping this plan onto the code track

Added because a second research track now runs in parallel:
**BigCloneBench / GraphCodeBERT / Java clone detection**
([`project/code-clone/`](../../project/code-clone/)). Everything above was written for AskUbuntu;
this section states what carries over and what does not, without rewriting §§1–17.

### 18.1 "Domain" means something different, and that matters

On the AskUbuntu track, "a new domain" meant a new pile of English text. On the code track it can
mean three different things, and the system has to know which one it is being asked about:

| Sense of "domain" | Example | Is adaptive selection the right tool? |
|---|---|---|
| **Functionality category** | "detect clones of *Sort Array* having trained on other functionalities" | No — this is the *generalisation test*, an evaluation, not a training-time choice |
| **Programming language** | Java → Python | **Yes.** Closest analogue to the NLP case, and the honest target |
| **Codebase / repository** | one company's Java | Yes, but the interesting variable is size, not semantics |

Consequence: do not promise "adaptive selection across functionalities". That conflates a
training-time decision with an evaluation protocol, and it would be claiming to solve the thing
Kitsios et al. (ASE 2025) measured as an open problem. The right framing is: **given a new
corpus of code, which negative strategy should we train with?**

### 18.2 The selection metric changes

| | AskUbuntu track | Code track |
|---|---|---|
| Selection metric | `argmax Recall@10_dev` | `argmax F1_dev`, threshold selected on dev per strategy |
| Tie-break | margin 0.02, then cheapest | same, unchanged |
| Costs | random 0 / bm25 1 / semantic 2 | **unchanged** |

One honest complication: on the code track the metric people actually care about is
**generalisation** (`Δ = F1_seen − F1_unseen`), and `Δ` cannot be estimated from a plain dev split
— it needs held-out *functionalities*. A cheap pilot therefore cannot directly optimise the thing
the project is about. Options, in order of honesty:

1. Select on `F1_dev` and state plainly that the pilot optimises benchmark fit, not
   generalisation.
2. Add one held-out functionality to the pilot. Costs almost nothing at pilot scale (25% of pairs,
   1 epoch) and buys a noisy but real `Δ` estimate. **Recommended if the generalisation split
   exists.**

### 18.3 The profile statistics need rewriting for code

The eight statistics in §5 are English-text statistics. For code, the informative ones are
different:

| Statistic | Why it matters for code |
|---|---|
| Token-length distribution | Java methods are long; drives max_seq_length and cost |
| Lexical near-duplicate rate | Code corpora are full of copy-paste; determines how much BM25 mining degenerates into finding T1/T2 clones |
| Function concentration | BigCloneBench is dominated by ~8 functionalities; a few methods dominate the pairs |
| **Estimated label completeness** | **The dominant variable.** Everything in [`GROUND_TRUTH.md`](../../project/code-clone/GROUND_TRUTH.md) says the false-negative burden is what decides BM25 vs semantic |
| Comment-to-code ratio | Comments make BM25 behave like natural-language retrieval |
| Identifier tokenisation quality | Whether `camelCase` splits cleanly affects BM25 far more than it affects dense retrieval |

Note the fourth one: **label completeness is the variable the literature says decides the answer**
(§5 of [`STRATEGY_EVIDENCE.md`](../negative-pair-research/STRATEGY_EVIDENCE.md)), and it is also
the one that is expensive to estimate without human labelling. That is the honest weak point of
the adaptive idea on this track, and it should be stated as such.

### 18.4 What carries over unchanged

* The pipeline stages (§2), the pilot design (§6: 25% of pairs, 1 epoch, 1 seed, k=64), the
  "shrink the training, never the evaluation" rule, and the five hard rules.
* The five modules (`profile.py`, `strategies.py`, `pilot.py`, `selector.py`, `pipeline.py`) and
  the ~450-line budget. None of the architecture is language-specific.
* The rejection of ensembling, weight averaging, and model soups (§3.3).
* The `argmax` + margin tie-break + `confident=False` logic (§4).
* The novelty audit (§12): pilot-based selection is standard model selection. **Nothing about
  code changes that.**

### 18.5 What gets worse

| Risk | Why it is worse on the code track |
|---|---|
| **Risk 7 — "if BM25 always wins, adaptive selection is a complicated constant function"** | More acute. If lexical near-duplicates dominate, BM25 and semantic mining may be nearly the same operation, and all three strategies collapse to one answer |
| **Low-fidelity bias** (§12.4, search 2) | Untested at any scale, on either track. Still the gate, still must be run first |
| **False-negative contamination** | Larger on code (copy-paste is pervasive), and it is *the* mechanism under study — so a pilot that mis-estimates it will mis-select |
| **Metric instability** | F1 with a threshold is noisier at small scale than Recall@10, so pilots need a wider margin before declaring a winner |

### 18.6 Recommendation

Do not build anything for the code track until the code track's own C0/C1/C2/C3 results exist.
The adaptive layer's only real validation is "do cheap pilots predict full runs?", and that
question cannot be answered once, let alone twice.

If both tracks eventually produce results, the **cross-track check** becomes the most interesting
experiment in the whole plan, and it is free: run the pilot on both domains and ask whether the
strategy ranking transfers. Two domains is the minimum for claiming anything about adaptivity at
all — with one domain, "adaptive" is unfalsifiable.
