# Demo — the 2–3 minute live presentation

The goal is one sentence in the audience's head:

> "They trained a domain-specific embedding model, measured it properly, and shipped it as a
> library you can `pip install`."

Everything below is designed against that sentence and nothing else.

---

## 1. The principle that governs this demo

> **The live demo is an anecdote. The table is the evidence.**

A single query where our model ranks the right duplicate first proves nothing on its own — with
200 test queries and a corpus of 14K, you can always find one. So the demo never carries the
argument by itself. The structure is always:

```text
aggregate result  (Recall@10 table, 3 seeds, bootstrap interval)
        ↓
then and only then
        ↓
one live query that makes the aggregate concrete
```

If the two disagree — the table says "no difference" but the demo query looks great — **say so on
the slide**. "This query is a nice example, but the aggregate says the effect is within noise,
and the aggregate is what I believe" is the single most impressive sentence available to an
undergraduate presenter, because it is the one a professional would say.

### Choosing the demo query

| Rule | Why |
|---|---|
| Pick it in advance from the evaluation output | Reproducible, and you know it works |
| Prefer the query with the **largest per-query improvement** of the fine-tuned model over the baseline | It is the honest best case, and it is the same query the error analysis already found |
| **Disclose** that it was selected this way | One clause: "I picked the query with the biggest gap — here is the aggregate, which is the honest number" |
| Have a second query typed by the audience, and accept whatever happens | Shows it is not rigged. If it comes out flat, say "and there it's a tie, which is what the aggregate predicts" |
| Never use a query from the **test** split in a way that implies it is held-out | It is held out for the *model*, but the audience will read it as held out for the *demo*. Say "evaluation query" if asked |

---

## 2. Setup, the night before

Nothing in the demo should depend on the network.

| # | Item | Command |
|---|---|---|
| 1 | Install with extras | `pip install -e ".[demo]"` |
| 2 | Warm the HF cache | `python -c "from domembed import DomainEmbedder; DomainEmbedder.load('sentence-transformers/all-MiniLM-L6-v2')"` |
| 3 | Export the trained model | `python scripts/export_model.py runs/<winner>/final --strategy "…" --metrics results/metrics.json` |
| 4 | Pre-build the demo index | `python scripts/build_demo_index.py --corpus data/corpus.jsonl --out demo/cache/` |
| 5 | Run the benchmark on **this** machine | `domembed bench --model ./runs/<winner>/final` and paste the real output into the README |
| 6 | Rehearse, twice, with a stopwatch | |
| 7 | Screenshot the working demo | `demo/fallback.png` — carried on the slide deck |
| 8 | Disable Wi-Fi and re-run once | Catches anything that secretly hits the network |

Item 8 is not paranoia. It is the difference between "the demo failed" and "the demo worked".

---

## 3. The script

Timings are for a 3-minute slot. Rehearse to 2:30 so there is room for a question.

### Beat 0 — 0:00–0:15 · The problem, in one sentence

> "A general embedding model is trained to know when two English sentences mean the same thing.
> On an Ubuntu forum, that's the wrong question — 'install vlc on 12.04' and 'install vlc on
> 14.04' look nearly identical to it, but they have different answers."

*Slide: two questions, a high similarity score from the generic model, and the word "duplicate?"*

### Beat 1 — 0:15–0:35 · The research, in three lines

> "So we fine-tuned a small encoder on real duplicate pairs from AskUbuntu. The thing we varied
> was **which pairs count as negatives** — same data, same loss, same batch size, same number of
> steps. That's the whole experiment."

*Slide: the main results table — Recall@10 for BM25, zero-shot MiniLM, and each negative
condition, with the bootstrap intervals.* **Land the aggregate number here**, before touching
the demo. Say the honest result, whatever it is.

### Beat 2 — 0:35–0:55 · Install and import

```bash
pip install -e .
```

```python
from domembed import DomainEmbedder

ours   = DomainEmbedder.load("./runs/quad_agree_hard/final")
theirs = DomainEmbedder.load("sentence-transformers/all-MiniLM-L6-v2")
```

> "Two models, one API. The second one is the off-the-shelf model we started from."

*This is the beat where the product becomes visible.* Have both cells already run — do not make
the audience watch a 90 MB download.

### Beat 3 — 0:55–1:15 · What the model knows about itself

```python
print(ours.info())
```

```text
Model:       domembed-askubuntu-c1-agree-hard
Base model:  sentence-transformers/all-MiniLM-L6-v2
Domain:      AskUbuntu — Ubuntu/Linux technical support questions
Dimension:   384    Max seq: 256 tokens
Device:      cpu

Trained with
  Objective:  MultipleNegativesRankingLoss (in-batch InfoNCE, scale 20)
  Negatives:  disagreement quadrant AGREE-HARD  (BM25 ∩ cosine, k=64)
  Data:       19,231 duplicate pairs, AskUbuntu train split
  Schedule:   2 epochs, batch 32, lr 2e-5, AdamW, seed 13

Evaluation (200 test queries, Recall@10)
  This model:  0.???
  Baseline:    0.???  (all-MiniLM-L6-v2, no fine-tuning)

Provenance:  runs/quad_agree_hard/seed13/final
```

> "The provenance travels with the weights — which negatives were used, and the measured score.
> Those numbers are generated from the evaluation file, not typed in."

*This is the beat that separates "we trained a model" from "we built a product". Do not skip it.*

### Beat 4 — 1:15–1:40 · Encode and similarity

```python
vec = ours.encode("How do I configure SSH key authentication?")
print(vec.shape, vec.dtype)
# (384,) float32

ours.similarity(
    "How do I configure SSH key authentication?",
    "Setting up ssh public/private keys for login",
)
# 0.812
```

> "One string in, 384 numbers out. Two strings in, one number out. That's the whole API."

Keep this beat short. It is the setup for search, and everyone in the room can already guess
what it does.

### Beat 5 — 1:40–2:15 · The money beat: search, side by side

Open the Streamlit app (or run the two-column snippet):

```
┌──────────────────────────────────────────────────────────────────────────┐
│  domembed — AskUbuntu semantic search                                    │
├──────────────────────────────────────────────────────────────────────────┤
│  Query                                                                   │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ how do I configure ssh keys?                                       │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│  Top-k  ──●───────  5        [ Search ]        ☑ show generic model      │
├───────────────────────────────┬──────────────────────────────────────────┤
│  domembed (fine-tuned)        │  all-MiniLM-L6-v2 (generic)              │
│  ─────────────────────        │  ──────────────────────────              │
│  0.85  Setting up ssh         │  0.79  Setting up ssh                    │
│        public/private keys    │        public/private keys               │
│  0.81  Passwordless ssh       │  0.74  How to enable remote desktop      │
│        login between hosts    │  0.71  What is ssh and how does it work  │
│  0.77  Generate ssh key pair  │  0.70  Installing openssh server         │
│        for github             │  0.68  Ubuntu firewall configuration     │
│  0.31  Enable remote desktop  │  0.66  Setting up a vpn connection       │
│  0.28  Ubuntu firewall setup  │  0.61  Generate ssh key pair for github  │
└───────────────────────────────┴──────────────────────────────────────────┘
   ↑ the real duplicate sits at rank 5 in the generic column
```

*The scores above are illustrative placeholders. Fill the slide with a real screenshot from the
rehearsal.*

Say the one thing that matters:

> "Same corpus, same code path, same query. The only difference is which negatives we trained
> against — and the duplicate that was at rank 5 is now at rank 3."

Then, immediately:

> "One query is an anecdote. The Recall@10 table on the previous slide is the evidence, and that
> is the number I'd defend."

### Beat 6 — 2:15–2:35 · It is a real library

```bash
domembed search "how do I configure ssh keys?" questions.txt --top-k 3
domembed info ./runs/quad_agree_hard/final
domembed bench
```

> "It installs with pip, it has a CLI, it has tests, and here's the latency on this laptop."

*Slide: the benchmark table, measured on this machine. Say the memory number out loud — "742 MB,
almost all of it torch" — because volunteering a limitation is more convincing than being
caught by it.*

### Beat 7 — 2:35–3:00 · Audience query and close

Take one query from the room. Type it. Whatever comes out, narrate it honestly.

Close with:

> "One model, trained once. The research asks whether the negatives matter; the library is how
> you'd actually use the model if they do."

---

## 4. The Streamlit app

`demo/app.py`, ~60 lines. Installed via `pip install -e ".[demo]"`, run with
`streamlit run demo/app.py`.

```python
import streamlit as st
from domembed import DomainEmbedder

st.set_page_config(page_title="domembed", layout="wide")
st.title("domembed — AskUbuntu semantic search")

@st.cache_resource                       # encode the corpus ONCE, not per query
def load_index(path):
    model = DomainEmbedder.load(path)
    return model, model.index(CORPUS, show_progress=True)

query = st.text_input("Query", "how do I configure ssh keys?")
top_k = st.slider("Top-k", 1, 20, 5)
compare = st.checkbox("show generic model", value=True)

if st.button("Search") and query:
    col1, col2 = st.columns(2)
    for col, path, title in [(col1, OURS, "domembed (fine-tuned)")] + \
                           ([(col2, GENERIC, "all-MiniLM-L6-v2 (generic)")] if compare else []):
        model, index = load_index(path)
        with col:
            st.subheader(title)
            for r in index.search(query, top_k=top_k):
                st.write(f"**{r.score:.2f}**  {r.text[:70]}")
```

Two details that make or break it:

| Detail | Why |
|---|---|
| `@st.cache_resource` on `load_index` | Without it, Streamlit re-encodes 14K documents on every interaction. The demo dies. |
| Pre-built index via `scripts/build_demo_index.py` | Encodes the corpus once, to a `.npy`, ahead of time. First-run latency drops from ~40 s to ~1 s. |

Cache directory is gitignored (`*.npy`) and rebuilt by the script — a 23 MB generated file has
no business in git.

---

## 5. Fallbacks

Things will go wrong. Decide now what happens.

| Failure | Fallback |
|---|---|
| Streamlit will not start | Run the two-column snippet in a plain terminal; it is the same content |
| Encoding is too slow | Pre-built index; if still slow, cut the corpus to a 2,000-document slice and say so |
| No network | Everything is cached; verified the night before with Wi-Fi off |
| Laptop refuses to load torch | Show `demo/fallback.png` and walk the code instead |
| The query returns something embarrassing | Show it anyway and explain it. A presenter who explains a bad result is more credible than one with a suspiciously perfect demo |
| Everything fails | The README's quick-start block, on a slide, plus the results table. The project does not depend on the live demo |

**Rule:** never let the demo be the only evidence on screen. The results table must be on a
slide that does not need a computer to render.

---

## 6. What NOT to say

| Don't say | Say instead |
|---|---|
| "Our model is better than the generic model" | "Our model scores X on Recall@10 against Y for the baseline, with a 95% interval of [a, b]" |
| "State of the art" | "Better than the baseline we trained it from, on this corpus" |
| "It understands Ubuntu" | "It ranks domain duplicates higher on this test set" |
| "Production-ready" | "It's a 0.1 — one domain, no persistence, no serving" |
| "This query proves it" | "This query is the clearest example; the table is the evidence" |
| "Negative-pair strategy X is best" | "X scored highest here; the interval overlaps Y, so I wouldn't claim more than that" |
| "You can search millions of documents" | "You can search the 14K corpus in milliseconds; beyond that you'd want FAISS" |

---

## 7. Rehearsal checklist

Tick these the day before:

```text
[ ] `pip install -e ".[demo]"` clean on the presentation machine
[ ] Wi-Fi off: demo still runs end to end
[ ] Demo index pre-built; first query returns in < 2 s
[ ] `domembed bench` output pasted into the README, measured on this machine
[ ] Results slide matches `results/main_table.md` exactly
[ ] `model.info()` prints the real evaluation numbers, not "Not evaluated"
[ ] Two models load (ours + generic)
[ ] Chosen demo query rehearsed; the second, unscripted query tried at least once
[ ] `demo/fallback.png` exists and is on the slide deck
[ ] Stopwatch run: under 3:00, twice
```
