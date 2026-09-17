# Adaptive `domembed` — the 5-minute version

> **What this is:** a plan for a **Version 2 / future extension** of `domembed`, to be built
> *after* the current four-week research experiment.
>
> **What it is not:** part of the current project. Nothing here should be started before the
> N1/N2/N3 comparison is finished and written up.
>
> **Numbers:** every number in this folder is **illustrative** unless explicitly marked as
> measured. The current research has not produced results yet.

---

## 1. The idea in one paragraph

The current project asks: *"what happens if we change how negative pairs are chosen?"* The future
library asks: *"given a new domain, which way of choosing negatives should we use?"* The bridge
is deliberately unglamorous: **train three small cheap models instead of three full ones, measure
them on a validation set, and use whichever won.** No LLM, no reinforcement learning, no neural
meta-learner — a for-loop over three strategies and an `argmax`.

## 2. The one question this must answer first

> **Does a cheap pilot predict the expensive result?**

On AskUbuntu we will already have full runs for all three strategies (that is the current
research). So the very first thing adaptive `domembed` must do is prove that its cheap pilots
would have picked the same winner the full runs did.

```text
                    pilot says         full run says         agree?
   Random            0.54              0.58                    —
   BM25              0.61              0.63                   ✓  winner matches
   Semantic          0.57              0.60                    —
                                                    [ILLUSTRATIVE NUMBERS]
```

If the pilot's ranking matches the full run's ranking, the selector is credible. **If it does
not, the honest conclusion is that the pilot does not work** — and that is a finding worth
reporting rather than hiding. This check costs almost nothing, because the full runs already
exist. It is Phase 4 of the plan and it is the most important experiment in v2.

## 3. The pipeline

```text
  ┌───────────────────────────────────────────────────────────────────────────┐
  │  INPUT DOMAIN                                                             │
  │  a dataset with naturally occurring positive pairs                         │
  │  e.g. AskUbuntu duplicate questions   (future: StackOverflow, bug reports) │
  └────────────────────────────────────┬──────────────────────────────────────┘
                                       ↓
  ┌────────────────────────────────────▼──────────────────────────────────────┐
  │  1. DATASET PREPARATION                        [exists — src/prepare_data] │
  │     corpus · queries · qrels · train pairs · duplicate groups              │
  └────────────────────────────────────┬──────────────────────────────────────┘
                                       ↓
  ┌────────────────────────────────────▼──────────────────────────────────────┐
  │  2. DOMAIN PROFILING                                          [NEW, small] │
  │     sizes, lengths, lexical overlap, semantic separation                   │
  │     → describes the domain. DOES NOT choose the strategy.                  │
  └────────────────────────────────────┬──────────────────────────────────────┘
                                       ↓
  ┌────────────────────────────────────▼──────────────────────────────────────┐
  │  3. CANDIDATE STRATEGIES                            [exists — 3 of them]   │
  │     N1 random  ·  N2 BM25/lexical  ·  N3 model-mined/semantic              │
  └────────────────────────────────────┬──────────────────────────────────────┘
                                       ↓
  ┌────────────────────────────────────▼──────────────────────────────────────┐
  │  4. PILOT TRAINING                                           [NEW, small]  │
  │     25% of pairs · 1 epoch · 1 seed · three separate short runs            │
  └────────────────────────────────────┬──────────────────────────────────────┘
                                       ↓
  ┌────────────────────────────────────▼──────────────────────────────────────┐
  │  5. VALIDATION                                  [exists — src/evaluate.py] │
  │     Recall@10 / MRR@10 / nDCG@10 on the full 200-query dev split           │
  └────────────────────────────────────┬──────────────────────────────────────┘
                                       ↓
  ┌────────────────────────────────────▼──────────────────────────────────────┐
  │  6. STRATEGY SELECTOR                                       [NEW, ~40 ln]  │
  │     pick the best; if the top two are within noise, pick the cheapest      │
  └────────────────────────────────────┬──────────────────────────────────────┘
                                       ↓
  ┌────────────────────────────────────▼──────────────────────────────────────┐
  │  7. FULL TRAINING                                    [exists — src/train]  │
  │     all pairs · epochs from dev · 3 seeds · the ONE selected strategy      │
  └────────────────────────────────────┬──────────────────────────────────────┘
                                       ↓
  ┌────────────────────────────────────▼──────────────────────────────────────┐
  │  8. MODEL EXPORT                          [exists — scripts/export_model]  │
  │     checkpoint + model_info.json (now also recording the pilot report)     │
  └────────────────────────────────────┬──────────────────────────────────────┘
                                       ↓
  ┌────────────────────────────────────▼──────────────────────────────────────┐
  │  9. domembed INFERENCE API                        [exists — v0.1 library]  │
  │     encode · similarity · search · info                                    │
  └───────────────────────────────────────────────────────────────────────────┘

  [exists] = already built or specified for the current project
  [NEW]    = the adaptive extension
```

Only three stages are genuinely new: **profiling, piloting, selection.** Everything else is
reused.

## 4. The three models are candidate paths, not a mixture

```text
                     Base MiniLM (all-MiniLM-L6-v2)
                              │
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                     ↓
   N1 random             N2 BM25              N3 model-mined
   negatives             negatives             negatives
        ↓                     ↓                     ↓
    Model N1             Model N2              Model N3
        └──────────── three separate models ────────┘
                              │
                    nothing is merged or averaged
```

The current research produces three **separate** fine-tuned encoders and compares them. The
adaptive system reuses exactly that machinery, but at pilot scale and for a different purpose:
instead of *reporting* which is best, it *uses* the winner and throws the other two away.

```text
   NEW DOMAIN
       ↓
   run three cheap pilots
       ↓
   ┌───────────────────────────────┐
   │  N1 → Recall@10 = 0.54        │
   │  N2 → Recall@10 = 0.61   ←────┼── highest
   │  N3 → Recall@10 = 0.57        │
   └───────────────────────────────┘
       ↓
   SELECT N2 (BM25)
       ↓
   train ONE model on all the data with N2
       ↓
   discard the pilots
```

## 5. The "AI" is an `argmax`

The selection rule, in symbols:

```text
    s*  =  argmax over s in {Random, BM25, Semantic}  of  Recall@10_dev(s)
```

In plain English: **train a small model with each strategy, measure each one on the validation
questions, and take the strategy whose model scored highest.** That is the whole "intelligence".

The only refinement is a tie-break, because on 200 dev queries differences below about 3 points
are noise:

```python
best   = max(scores, key=scores.get)
second = sorted(scores.values())[-2]
if scores[best] - second < MARGIN:      # MARGIN = 0.02
    best = cheapest strategy within MARGIN of the top
```

So the selector's most common sensible output is sometimes *"these two are indistinguishable —
use the cheap one."* That is a feature, not a limitation: random negatives cost no mining at all.

## 6. How to use it

```python
from domembed import AutoDomainEmbedder

# Training: analyses the domain, runs pilots, selects, trains, exports.
model = AutoDomainEmbedder.fit(
    dataset="sentence-transformers/askubuntu",
    output_dir="./models/askubuntu",
)

# Inference: exactly the same object the v0.1 library already defines.
model.encode("Ubuntu WiFi stopped working after update")
model.similarity(text_a, text_b)
model.search(query, documents, top_k=5)
```

Training and inference stay separate operations:

```python
AutoDomainEmbedder.fit(...)          # slow — produces a model on disk
DomainEmbedder.load("my-domain")     # fast — loads it
```

`fit()` **returns a `DomainEmbedder`**, so there is still only one inference class in the
system. `AutoDomainEmbedder` is a factory, not a second model type.

## 7. Is any of this novel?

**No — and the plan says so plainly.** Choosing a configuration by validation score is standard
model selection; using a cheap short run to stand in for a full one is standard multi-fidelity
hyperparameter optimisation ([Hyperband](https://arxiv.org/abs/1603.06560), Li et al.). Adaptive
negative sampling already exists as a term in the recommender-systems literature
([DNS](https://doi.org/10.1145/3543507.3583849), Shi et al., WWW 2023;
[AHNS](https://arxiv.org/abs/2401.05191), Lai et al., AAAI 2024) — though those adapt hardness
*during* training, per example, whereas this selects one fixed strategy *before* training.

What might be worth something, and what would need real literature work to claim:

| Claim | Status |
|---|---|
| "We built a system that picks a negative strategy by validation" | **Engineering. Not novel.** |
| "Cheap pilots reliably predict full-run rankings *for this choice*" | **Unknown.** Needs the Phase 4 check and a literature search on low-fidelity bias |
| "Domain statistics predict the best strategy" | **Unproven.** One domain cannot establish this |
| "The best strategy transfers across domains" | **Unproven.** Needs several domains |

Full honesty discussion, with the specific searches to run before writing the word "novel", is in
[`ADAPTIVE_PLAN.md`](ADAPTIVE_PLAN.md) §12.

## 8. Scope

| | |
|---|---|
| **MUST HAVE** | Strategy registry over the existing 3 · domain profile · pilot runner · selector with tie-break · full training + export · the pilot-fidelity check against the known AskUbuntu results |
| **SHOULD HAVE** | CLI (`domembed fit`, `domembed profile`) · saved pilot report JSON · profile recorded in `model_info.json` · a second real domain |
| **FUTURE — not now** | Learned meta-selector · RL · LLM reasoning about strategies · more than 3 strategies · NAS · online adaptation · multi-domain meta-training |

The FUTURE list is a **do-not-do list**, not a backlog.

---

## 9. The bottom line

| | Question |
|---|---|
| **Current research** | *"What happens when we change the negative strategy?"* |
| **Current product (v0.1)** | *"How do we package the resulting model so it can be used?"* |
| **Future adaptive extension (v2)** | *"Given a new domain, which negative strategy should we use?"* |

And the bridge between the first and the last is one sentence:

> **Run small controlled pilots and select the strategy that performs best on validation.**

The full plan — 17 sections, component-by-component architecture, the pilot budget, risks and
milestones — is in [`ADAPTIVE_PLAN.md`](ADAPTIVE_PLAN.md).
