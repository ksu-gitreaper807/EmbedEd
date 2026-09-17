# DEMO.md — the demo is the presentation centerpiece

> **Priority doctrine (reverses the earlier draft):**
> **The library is the engineering deliverable. The demo is the presentation centerpiece.**
>
> An audience will not remember `pip install -e .`. They will remember one screen where the
> known duplicate of a question jumps from rank 5 to rank 1. So the demo is designed **first**,
> around the research result, and the library is the minimum that makes the demo honest.

Everything in this document is designed backwards from one question:

> **What would make a room understand, in ten seconds of looking, that how we chose negative
> pairs changed what the model knows?**

---

## 0. The seven questions, answered

| # | Question | Answer, short form | Where |
|---|---|---|---|
| 1 | What does the user interact with? | One page: a query box, a top-k slider, a compare toggle, and **two or more ranked result columns with the known duplicate highlighted**. Nothing else. | §2 |
| 2 | What visual comparison demonstrates the research? | Not two lists of plausible text — **the rank of the known gold duplicate, side by side**, first generic-vs-ours, then across the negative-pair conditions. Three visuals, in a fixed order. | §3 |
| 3 | What does `domembed` expose? | Five methods the demo touches: `load`, `encode`, `similarity`, `search`/`index`, `info`. Everything else is support. | §4 |
| 4 | What is the smallest implementation? | ~300 lines of library + ~120 lines of Streamlit. The demo-critical path is 5 methods. | §4 |
| 5 | What goes in the GitHub README? | Demo screenshot **above the fold**, then the result, then install, then API. Not the reverse. | §7 |
| 6 | What is shown in 3–5 minutes? | A timed beat sheet: problem → result → product → demo → the research variable → close. | §6 |
| 7 | What can be cut? | CLI, benchmark module, `DocumentIndex` (if the corpus is trimmed), all optional extras. **Not** `info()`, **not** the side-by-side, **not** the aggregate panel. | §8 |

---

## 1. Design around the research result first

### 1.1 The demo runs against the evaluation corpus, not a toy list

This is the single most important design decision in this document.

The demo does **not** search five hand-typed strings. It searches the real ~14,000-document
AskUbuntu corpus and, critically, **the app knows the gold labels** (`data/qrels.json`, already
built by `prepare_data.py`).

That one fact changes the demo's epistemic status completely:

| Demo searches… | The audience sees… | What it proves |
|---|---|---|
| five hand-written sentences | two plausible orderings | nothing — you could get any ordering you like by choosing the sentences |
| the eval corpus, with gold labels | *"the known duplicate is at rank 5 under the generic model and rank 1 under ours"* | a number, computed the same way the headline metric is computed |

The demo becomes a **live, single-query instance of the evaluation itself** — the same corpus,
the same gold labels, the same ranking code as `Recall@10`. It is an anecdote, but it is an
anecdote *drawn from the same distribution the table summarises*, and you can say so.

Implementation: `demo/app.py` loads `data/corpus.jsonl`, `data/qrels.json` and the precomputed
index; for each query it looks up `qrels[qid]`, finds the rank of the gold document in each
model's result list, and renders that rank as the headline number.

### 1.2 The design must survive every possible result

The research result is not known yet. The demo therefore cannot be designed around "our model
wins" — it must be **result-agnostic**: the same app, the same code, four different captions.

| Outcome | Recall@10 pattern | What the app leads with | The caption | What you must NOT say |
|---|---|---|---|---|
| **WIN** — hardness helps monotonically | `N1 < N2 < N3`, all > baseline | Visual A then C | "The harder the negatives, the higher the known duplicate ranks — and the aggregate agrees." | "Our method is better" (say: "scores higher here") |
| **INVERTED-U** — the predicted outcome | `N2 > N1` and `N2 > N3` | **Visual C** — the curve turning over is the story | "Harder helps, then hurts. The most similar neighbours are unlabelled duplicates, so training against them pushes away real answers." | "Hard negatives are best" — the data says otherwise |
| **TIE** — all within noise | intervals overlap | Visual B, then A with a shrug | "On this corpus, with three seeds, we cannot distinguish the conditions. That is a real result about a technique everyone assumes matters." | any claim of superiority |
| **LOSS** — BM25 wins | lexical beats every dense system | Visual B, honestly | "Keyword overlap is the signal here. The library still works; the finding is that fine-tuning did not help." | anything implying the model improved |

**The TIE and LOSS rows are why the app needs the aggregate panel built in from the start.** If
you only build the flattering path and the result comes out flat, you have no demo. Build all
four captions in week 3 and delete three in week 4.

### 1.3 The narrative spine

```text
PROBLEM    a generic embedding cannot tell these two Ubuntu questions apart
   ↓
RESEARCH   can contrastive fine-tuning fix that — and do the negatives decide whether it does?
   ↓
RESULT     [whatever the evidence supports — stated as a number with an interval]
   ↓
PRODUCT    the same model, behind a three-line API, installable with pip
   ↓
DEMO       one query, two columns, and the known duplicate changing rank
```

Note the order: **result before product, product before demo.** Presenting the demo first makes
it a magic trick; presenting the result first makes the demo an illustration of a claim you have
already made and already bounded.

---

## 2. What the user interacts with

A single Streamlit page, two tabs, six widgets. Nothing hidden in a sidebar, nothing behind a
settings panel.

### Tab 1 — "Search" (the money shot)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│  domembed · AskUbuntu duplicate-question search                              │
│  trained on ~19,000 duplicate pairs · evaluated on 200 held-out queries        │
├──────────────────────────────────────────────────────────────────────────────┤
│  Query  [ how do I configure ssh keys?                              ]  [🔍]  │
│                                                                              │
│  Top-k  ────●────────  5      ☑ compare with all-MiniLM-L6-v2 (generic)      │
│  Conditions  ☑ N1 random  ☑ N2 lexical  ☑ N3 model-hard                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ★ = a duplicate that the AskUbuntu community actually marked               │
│                                                                              │
│   all-MiniLM-L6-v2 (generic)      domembed · N1 random    domembed · N3 hard │
│   ──────────────────────────      ─────────────────────   ─────────────────  │
│   ① 0.79  Setting up ssh         ① 0.83  Setting up ssh  ① ★ 0.88  Generate  │
│   ② 0.74  Enable remote desktop  ② 0.80  Passwordless   ② 0.85  Setting up   │
│   ③ 0.71  What is ssh?           ③ ★ 0.77  Generate ssh  ③ 0.82  Passwordless│
│   ④ 0.70  Installing openssh     ④ 0.71  What is ssh?    ④ 0.74  What is ssh?│
│   ⑤ ★ 0.68  Generate ssh key     ⑤ 0.69  Enable remote   ⑤ 0.70  Enable      │
│                                                                              │
│   Known duplicate "Generate ssh key pair for github"                         │
│                                                                              │
│        generic → rank 5        N1 → rank 3        N3 → rank 1                │
│                        └──────────────▼──────────────┘                       │
│                              ▲ +4 ranks, from changing the negatives         │
├──────────────────────────────────────────────────────────────────────────────┤
│  ℹ One query is an anecdote. Recall@10 over 200 queries is the evidence →    │
│    [Tab 2: The experiment]                                                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

*Layout mockup. Scores are placeholders — fill this slide from a real rehearsal screenshot.*

| Widget | Behaviour | Why it exists |
|---|---|---|
| **Query box** | Prefilled with the chosen query; fully editable | Prefilled = reliable; editable = you can take an audience query without code changes |
| **Top-k slider** (1–20, default 5) | Controls list length | Lets you widen to 10 and point at Recall@10 directly |
| **"Compare with generic" toggle** (default on) | Adds/removes the generic column | Off = a clean single-model screenshot for the README |
| **Condition checkboxes** | Shows one column per fine-tuned condition | **This is the research variable, made visible.** Requires ≥2 exported models |
| **★ marker** | Marks rows that `qrels` says are gold duplicates | The entire evidentiary value of the demo |
| **Rank-movement footer** | `generic → rank 5 · N1 → rank 3 · N3 → rank 1` | The one number the audience will actually remember |

Everything else is deliberately absent: no file upload, no model-name text field, no temperature
sliders, no chunking options. Every extra control is a control that can be wrong in front of
people.

### Tab 2 — "The experiment" (the evidence, and the fallback)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│  Recall@10 · 200 test queries · mean of 3 seeds · 95% bootstrap interval     │
│                                                                              │
│  BM25                       ████████░░░░░░░░░░░░  0.???                      │
│  all-MiniLM-L6-v2 (generic) ██████████░░░░░░░░░░  0.???                      │
│  domembed · N1 random       ████████████░░░░░░░░  0.???                      │
│  domembed · N2 lexical      ██████████████░░░░░░  0.???                      │
│  domembed · N3 model-hard   ██████████████░░░░░░  0.???                      │
│                                                                              │
│  Contrast          Δ Recall@10     95% CI          Reading                   │
│  N1 − generic      +0.???          [?, ?]          does fine-tuning help?    │
│  N2 − N1           +0.???          [?, ?]          does lexical hardness?    │
│  N3 − N2           +0.???          [?, ?]          does semantic hardness?   │
│                                                                              │
│  ── model provenance ──────────────────────────────────────────────────────  │
│  $ domembed info ./runs/n3_model_hard/final                                  │
│    Negatives:  top-64 by cosine of the zero-shot encoder                     │
│    Objective:  MultipleNegativesRankingLoss (scale 20)                       │
│    Schedule:   2 epochs · batch 32 · lr 2e-5 · seed 13                       │
│                                                                              │
│  ── performance (this laptop) ─────────────────────────────────────────────   │
│    single encode   ?? ms · batch-32  ?? ms · index search  ?? ms             │
└──────────────────────────────────────────────────────────────────────────────┘
```

Tab 2 exists for three reasons: it is the evidence that licenses the anecdote, it is the screen
you stay on if the live query misbehaves, and it is where `info()` makes the negative-pair
strategy visible inside the product.

---

## 3. The three visuals, in order

The order is fixed and it is the argument.

### Visual A — Generic vs. ours (30 seconds)

*The claim:* domain fine-tuning changes what "duplicate" means.

```
generic → rank 5          ours → rank 1              ▲ +4
```

Keep it to ten seconds of talking. Everyone already believes this part; it is the setup, and
lingering on it makes the interesting result look like more of the same.

### Visual B — The aggregate (20 seconds)

*The claim:* the anecdote generalises, and here is the uncertainty.

Shown **before** Visual A in the presentation proper (see the beat sheet) but labelled Visual B
here because the *app* presents it in tab 2. The bar chart must carry the bootstrap intervals —
a bar chart without intervals invites the question "is that difference real?", and the answer is
on the same slide.

### Visual C — The research variable (60 seconds) — the money shot

*The claim:* everything is identical except which pairs counted as negatives, and the ranking
changes anyway.

```
Same query · same corpus · same ~19,000 pairs · same loss · same batch size
same epochs · same seed

                    only the negatives differ

   N1 random        N2 lexical        N3 model-hard
   ─────────        ──────────        ─────────────
   rank 4           rank 2            rank 1
   "any 31 other    "31 lexically     "31 semantically
    pairs in the     similar pairs"    similar pairs"
    batch"

   ...and if N3 < N2, the column that should be best is not — because the most
   similar questions in this corpus are unlabelled duplicates.
```

**This is the visual that demonstrates the research.** Visual A would look the same for any
domain-fine-tuning project ever done. Visual C is *this* project's independent variable, made
visible. If you only have time to build one thing well, build this.

If the research ends up with the four disagreement quadrants (C1) instead of N1/N2/N3, this
becomes four columns with the same layout — which is why the condition-checkbox widget is
generic over the number of exported models rather than hardcoding three.

---

## 4. What `domembed` must expose, and the smallest implementation

### The demo-critical API — five methods

```python
model   = DomainEmbedder.load("./runs/n3_model_hard/final")   # also loads the generic baseline
vec     = model.encode("how do I configure ssh keys?")        # shown once, in passing
score   = model.similarity(q, "Generate ssh key pair…")       # shown once, in passing
results = model.search(q, corpus, top_k=5)                    # ─ or index.search(q, 5)
info    = model.info()                                        # fills the provenance panel
```

That is the entire surface the demo touches. `encode` and `similarity` each appear once, for
fifteen seconds, to establish that the API is three lines long; the demo proper is `search` and
`info`.

### Smallest implementation

| Component | Lines | Required for the demo? |
|---|---|---|
| `embedder.py` — `load`, `encode`, `similarity`, `search`, `info`, `index` | ~170 | **Yes — all of it** |
| `similarity.py` — `cosine`, `top_k` | ~45 | Yes |
| `metadata.py` — `ModelInfo`, `model_info.json` | ~60 | Yes (the provenance panel) |
| `errors.py` — 5 exceptions | ~25 | Yes (cheap; collapses to 2 if pressed) |
| `demo/app.py` — the Streamlit page | ~120 | **Yes — it is the centerpiece** |
| `scripts/export_model.py` — the research seam | ~40 | Yes |
| `pyproject.toml` | ~30 | Yes |
| **Demo-critical subtotal** | **~490** | |
| `cli.py` — 5 commands | ~90 | No |
| `benchmark.py` | ~60 | No (a hardcoded measured table works) |
| tests | ~120 | Only the five named tests |

**~490 lines gets you the demo and the library.** The other 270 buy polish and are the cut list
in §8.

### Demo-specific code that is not in the library

Deliberately kept out of `domembed` — the library does not know about qrels, corpora or
evaluations:

```python
# demo/app.py (excerpt)
qrels = json.load(open("data/qrels.json"))

def gold_rank(results, qid):
    gold = set(qrels[qid])
    for i, r in enumerate(results, start=1):
        if r.index in gold:
            return i
    return None          # not in top-k → say so, do not pretend

# → renders "generic → rank 5   N1 → rank 3   N3 → rank 1"
```

`gold_rank` returning `None` when the duplicate is not in the top-k is handled explicitly: the
app prints `> k` rather than silently dropping the row. Showing a miss is more credible than
hiding one.

---

## 5. Building it (where this lands in the schedule)

Because the demo is the centerpiece, it must not be built in the last two days.

| When | What | Depends on |
|---|---|---|
| **Week 2** | Demo **shell** against the generic model: corpus + index + two-column layout + `gold_rank`. Numbers are meaningless but the plumbing is proven. | `similarity.py`, `embedder.py` |
| **Week 3** | All four result-mode captions written. Aggregate panel wired to a placeholder `results/metrics.json`. Rehearse with fake numbers. | Phase 2 of the library |
| **Week 4** | `export_model.py` on every condition → real models → real numbers. Delete three of the four captions. Rehearse with real numbers. | the trained checkpoints |

The week-2 shell is what makes this safe: if the research runs late, you still have a working
demo with a worse model in it, rather than a beautiful app with nothing to show.

---

## 6. The 3–5 minute presentation

Timings total 4:30, leaving 30 seconds for one question. Rehearse to 4:00.

### 0:00–0:30 · The problem, in one concrete example

> "Two questions from an Ubuntu forum. A general-purpose embedding model scores them 0.79 —
> nearly identical. They are not the same question: one is about SSH keys, one is about remote
> desktop. Generic embeddings were trained to know when two *English sentences* mean the same
> thing. In a technical domain, that's the wrong question."

*Slide: the two questions and the score. No code, no architecture. One idea.*

### 0:30–0:55 · The research design, in four lines

> "So we fine-tune a small encoder on real duplicate pairs from AskUbuntu. The one thing we
> vary is **which pairs count as negatives**. Same data, same loss, same batch size, same
> epochs. Three conditions: random, lexically similar, semantically similar."

*Slide: a single diagram — one variable, everything else greyed out and labelled "held
constant".*

### 0:55–1:25 · **The result** — before any product

> "Here is Recall@10 on 200 held-out queries, mean of three seeds, with bootstrap intervals."

*Slide: Visual B. Say the number. Say the interval. **Say the honest result, whatever it is** —
including "we cannot distinguish these conditions".*

This beat comes before the product on purpose. If you show the demo first, the audience spends
the rest of the talk wondering whether you cherry-picked it. State the claim, bound it, then
illustrate it.

### 1:25–1:55 · The product

```bash
pip install -e .
```

```python
from domembed import DomainEmbedder
ours   = DomainEmbedder.load("./runs/n3_model_hard/final")
theirs = DomainEmbedder.load("sentence-transformers/all-MiniLM-L6-v2")
```

```python
print(ours.info())
```

```text
Negatives:  top-64 by cosine of the zero-shot encoder
Objective:  MultipleNegativesRankingLoss (scale 20)
Schedule:   2 epochs · batch 32 · lr 2e-5 · seed 13
Evaluation: Recall@10 = 0.??? (baseline 0.???)
```

> "Two models, one API — including the off-the-shelf one we started from. And the provenance
> travels with the weights: which negatives were used, and the measured score. Those numbers are
> generated from the evaluation file, not typed in."

*Have the cells pre-run. Never make an audience watch a 90 MB download.*

### 1:55–2:30 · **Visual A** — generic vs. ours, live

Type the query (or click Search on the prefilled one).

> "Left column: the generic model puts the known duplicate at rank 5. Right column: ours puts
> it at rank 1. Same corpus, same code path. The star is a duplicate the AskUbuntu community
> actually marked — that's from the dataset, not my judgement."

Then, without waiting for applause:

> "One query is an anecdote. That table a minute ago is the evidence."

### 2:30–3:40 · **Visual C** — the money shot

Enable all three conditions.

> "Now the actual experiment. Four columns: the generic model, and three versions of our model.
> **Identical in every respect except which pairs counted as negatives.**"

Walk the columns left to right. Then land the result:

* If hardness helped monotonically: *"harder negatives, higher rank — and the aggregate agrees."*
* If the curve turned over: *"and here's the interesting one. The hardest negatives made it
  **worse**. Why? Because in this corpus, only about 5% of similar pairs are labelled — so the
  most similar questions are often genuine duplicates nobody marked. Train against them and you
  push away real answers."*

That explanation is the intellectual payload of the whole project. Say it slowly.

### 3:40–4:15 · Audience query + the rest of the library

> "Give me a question."

Type it. Whatever comes out, narrate it honestly — including a tie or a miss.

Then, quickly:

```bash
domembed search "how do I configure ssh keys?" questions.txt --top-k 3
domembed bench
```

> "It's a real package: pip-installable, CLI, tests, and here's the latency on this laptop —
> and most of that memory footprint is torch, not us."

### 4:15–4:30 · Close

> "One model, trained once. The research asks whether the negatives matter; the library is how
> you'd actually use the model if they do."

---

## 7. What goes in the GitHub README

**Order matters.** The README's job is to convert a visitor in fifteen seconds, and a visitor
who scrolls past the demo screenshot has already decided whether this is a real product.

| # | Section | Notes |
|---|---|---|
| 1 | **One-line pitch** | "Domain-specific sentence embeddings, packaged." |
| 2 | **The demo screenshot** (or GIF) — **above the fold** | Real output from the rehearsal. The rank-movement footer must be legible in the image. |
| 3 | **The result, in one sentence + the table** | With intervals. `?` until measured. Never a claim without a number. |
| 4 | **Install** — three lines | `git clone`, `cd`, `pip install -e .` |
| 5 | **Quick start** — one code block, ≤ 10 lines | Load → encode → search |
| 6 | **Run the demo** — one command | `streamlit run demo/app.py` |
| 7 | **API** — a five-row table | `load` / `encode` / `similarity` / `search` / `info` |
| 8 | **Model provenance** — `info()` output block | Where the negative-pair strategy becomes visible |
| 9 | **How this connects to the research** | Five bullets, one diagram |
| 10 | **Performance** | Measured on the presentation machine |
| 11 | **Limitations** | One domain, re-encodes on `search()`, no vector DB, not a server |
| 12 | **Licence / attribution** | Apache-2.0 code; check the CC BY-SA corpus before publishing weights |

**Deliberately moved below the fold** relative to the earlier draft: the problem essay, the
architecture diagram, the full feature table, the non-goals. They matter to a contributor, not
to a visitor.

One rule enforced throughout: **every number in the README is generated or measured.** The
results table comes from `results/metrics.json`, the performance table from `domembed bench`, the
screenshot from the rehearsal.

---

## 8. What can be cut — and what cannot

### Cut these first (the product still looks complete)

| Cut | Saving | Why it is safe |
|---|---|---|
| `cli.py` — all five commands | ~90 lines, 3 h | The Streamlit demo and the README quick-start both show the API. A CLI is cachet, not evidence. *(Keep `info` only, if you keep anything — it is the one command that looks good on a slide.)* |
| `benchmark.py` | ~60 lines, 1.5 h | Replace with a table you measured by hand once and pasted into the README |
| `DocumentIndex` | ~40 lines, 1 h | Only if you also trim the demo corpus to ~2,000 documents. **Do not cut it while the corpus is full** — a 40-second stall kills the demo |
| Tests beyond the five named ones | ~1.5 h | Keep `test_model_loading`, `test_single_encode`, `test_batch_encode`, `test_similarity`, `test_search` |
| `examples/quickstart.py` | ~1 h | The demo supersedes it |
| `py.typed`, `ruff` config | 30 min | Polish |
| The `[demo]` / `[dev]` extras | 15 min | One flat dependency list is fine |
| Condition columns beyond two | — | Two columns (generic + best condition) tell the story; Visual C needs three to be persuasive, but two is survivable |

### Cut these only if the research is on fire

| Cut | Consequence |
|---|---|
| The aggregate panel (Tab 2) | Loses the evidence that licenses the anecdote, and loses your TIE/LOSS fallback. **Very painful.** |
| The gold ★ marking | Downgrades the demo from evidence to vibes. **Do not.** |
| `export_model.py` | Then `info()` prints nothing, and the research connection evaporates. **Do not.** |

### Never cut

1. **`load()` accepting both a local path and the generic baseline** — the comparison is the
   entire argument, and it only works if both models go through one code path.
2. **`info()` / `model_info.json`** — the only part of the product that `sentence-transformers`
   does not already provide.
3. **The side-by-side comparison** — one column is a search box; two columns are a result.
4. **The honest results table** — the thing that makes the demo credible rather than slick.
5. **`pip install -e .` working** — the claim "this is a package" is falsified instantly if it
   is not.

---

## 9. Fallbacks and the night before

| Failure | Fallback |
|---|---|
| Streamlit will not start | Terminal two-column script, same content, no browser dependency |
| Encoding too slow | Pre-built index (`scripts/build_demo_index.py`); if still slow, trim to 2,000 documents **and say so on the slide** |
| No network | Everything cached; verified with Wi-Fi off the night before |
| torch will not load on the laptop | Stay on Tab 2 (static) and walk the app's code |
| The live query comes out flat or wrong | **Show it.** "And there it's a tie, which is what the aggregate predicts" is the best available answer |
| Total failure | Tab 2 is a static screenshot on a slide. The talk survives without the live app. |

**Rule: the demo is never the only evidence on screen.** Tab 2 must also exist as a slide that
needs no computer.

### Rehearsal checklist

```text
[ ] Demo shell runs against the real corpus with gold ★ markers
[ ] All four result-mode captions written (WIN / INVERTED-U / TIE / LOSS)
[ ] Wi-Fi off: still works end to end
[ ] First query returns in < 2 s (pre-built index, st.cache_resource)
[ ] Chosen query rehearsed; one unscripted audience query tried
[ ] Screenshots for the README and for fallback.png, from the real app
[ ] Tab 2 exists as a slide deck image
[ ] Stopwatch: under 5:00, twice
[ ] Numbers in Tab 2 match results/main_table.md exactly
```
