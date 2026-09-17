# BigCloneBench's ground truth — the threat you must state, and what you do about it

> Read this **before week 1**, not in week 4. It changes two design decisions (the
> false-negative measurement in [`FINAL_SPEC.md`](FINAL_SPEC.md) §7.2, and how you phrase every
> claim in the report).

---

## 0. The one-paragraph version

BigCloneBench is the standard benchmark for this task and you should use it. It also has
**documented, empirically measured ground-truth defects**: a manual audit of 406 sampled
Weak Type-3/Type-4 clone pairs found 93% were mislabelled, and WT3/T4 pairs are 95% of the
dataset `[verified]`. None of this invalidates your experiment, because you are not claiming
"model X detects clones with F1 = Y" — you are claiming **"changing the negative-sampling
strategy changes the outcome, by this much, and here is the measured mechanism"**, which is a
*comparison* and survives label noise far better than an absolute number does. But one of those
defects — incomplete labels — is not merely background noise for your project. **It is
correlated with your independent variable.** That is the part you have to handle.

---

## 1. What the sources actually say

Primary sources, both `[verified]`:

* **Krinke, "BigCloneBench Considered Harmful for Machine Learning"**, IWSC 2022 —
  [UCL copy](http://www0.cs.ucl.ac.uk/staff/j.krinke/publications/iwsc22.pdf)
* **"How the Misuse of a Dataset Harmed Semantic Clone Detection"** (2025) —
  [arXiv:2505.04311](https://arxiv.org/html/2505.04311v1)

### 1.1 The six findings

| # | Finding | Verbatim / numbers |
|---|---|---|
| 1 | Cross-functionality pairs are not necessarily non-clones | *"BigCloneBench does not contain information about pairs for different functionalities."* Example: ≥77 methods labelled true positive for *Copy Directory* invoke a `copyFile` method, so labelling them non-clones of *Copy File* is wrong |
| 2 | There are true but **unlabelled** clone pairs across functionalities | *"BigCloneBench does contain true but unlabelled clone pairs for different functionalities."* |
| 3 | The ground truth is incomplete even **within** a functionality | Two methods both labelled false positive for a functionality may still be clones of each other |
| 4 | Labelling quality is limited | *"at least 15% are estimated to be subjective or have validation errors"* |
| 5 | The WT3/T4 clone ground truth is flawed | 93% of a random sample of **406** WT3/T4 pairs *"do not have a similar functionality and are therefore mislabelled"*. WT3/T4 = **95%** of the dataset |
| 6 | Severe imbalance and bias | *"the majority of labelled methods in the released dataset are for the functionality 'Copy File', 42,664 out of 75,672"*; **over 90% of true clone pairs belong to just 8 functionalities**; 22 of 43 functionalities together are under 1% of them; 70% of false clone pairs are *Copy File*, and over 98% belong to just 8 functionalities |

### 1.2 CodeXGLUE is named specifically

The 2025 paper examined the CodeXGLUE subset directly and concluded it *"expanded the ground
truth provided by BigCloneBench without considering the issues we observed"*, confirming
*"the presence of wrongly generated false clone pairs caused by assuming (A) two methods
labelled as false positive for the same functionality are not clones of each other, and
(B) two methods labelled as true positive for two different functionalities are not clones of
each other"* `[verified]`.

**So: the file you are downloading is the version this criticism applies to most directly.** Not
a reason to panic — a reason to be precise.

---

## 2. What each finding does to this project

| Finding | Effect on your project | Severity |
|---|---|---|
| 1 & 2 — unlabelled cross-functionality clones | **Directly interacts with your independent variable.** See §3 | **High** |
| 3 — incomplete within functionality | Same mechanism, smaller magnitude | Medium |
| 4 — ~15% labelling noise | Raises the noise floor for every condition equally. Costs you sensitivity but not validity | Medium |
| 5 — 93% of WT3/T4 mislabelled | Means your absolute F1 is not a measure of real-world clone detection. It **is** still a valid basis for comparing four conditions | Medium |
| 6 — imbalance | Headline F1 is dominated by ~8 functionalities. The generalisation test is what exposes this | Medium |

### 2.1 Why a comparison survives what an absolute number does not

If 15% of labels are wrong, and the error rate is roughly the same in every condition's
evaluation, then:

```text
F1(C3) − F1(C1)   still estimates the effect of the negative strategy, biased toward zero
F1(C3)            does NOT estimate "how good is this model at finding clones"
```

Label noise that is *independent* of your manipulation attenuates your effect size; it does not
invent one. So you should:

* **Report the differences** (C2−C1, C3−C2, C1−C0) as the headline, not the levels.
* **Never write** "our model detects clones with 94% F1" without the caveat.
* **Always write** "changing the negative strategy moved F1 by X points, under labels that are
  themselves estimated to be ~15% subjective or erroneous".

That is an honest sentence and it is still a real result.

---

## 3. The one that matters: unlabelled clones are correlated with your variable

This is the crux, and it is worth being slow about it.

Your three strategies mine progressively more similar fragments:

```text
C1 Random     →  a random fragment.                P(unlabelled clone) ≈ base rate
C2 BM25       →  a lexically similar fragment.     P(unlabelled clone)  HIGHER
C3 Semantic   →  what the base model finds similar. P(unlabelled clone) HIGHEST
```

Because similar code is disproportionately *same-functionality* code, and because
BigCloneBench's clone labels are incomplete (findings 2 and 3), **the harder your negatives get,
the more of them are genuine clones that the dataset forgot to label.**

So the inverted-U result — C2 > C1 but C3 < C2 — has two competing explanations:

| | Explanation | Implication |
|---|---|---|
| **(a)** | Hardness *itself* becomes counterproductive past a point (the Robinson et al. / SimANS story) | Theoretically interesting; transfers |
| **(b)** | C3's negative set is simply **more contaminated with false negatives**, and you are training the model to push apart code that is genuinely equivalent | A measurement artefact of BigCloneBench; does not transfer |

**You cannot distinguish (a) from (b) without measuring the false-negative rate.** That is why
the manual labelling in [`FINAL_SPEC.md`](FINAL_SPEC.md) §7.2 is a MUST and not a SHOULD. It is
the difference between an observation and a finding.

### 3.1 The exact measurement

```text
Sample 50 mined negatives from C2 and 50 from C3, stratified across the difficulty
rank (for example, ranks 1–5, 6–20, 21–50).

For each (anchor, negative) pair, judge one question by eye:
    "Do these two methods implement the same functionality?"
    Answer: YES / NO / UNDECIDABLE

Report, per strategy and per rank band:
    FN rate = (YES + UNDECIDABLE) / N
```

Two hours of work. Compare against the nearest published figure for naive top-k hard-negative
mining: **47% false-negative rate on StackExchange-domain data** `[check — NV-Retriever; confirm
before quoting]`.

### 3.2 What each outcome means

| Measured FN rate | Interpretation | What to write |
|---|---|---|
| C2 ≈ C3 (both low) | Contamination is not the driver. If C3 < C2, explanation **(a)** — hardness itself — is supported | "The inverted-U is a property of hardness, not of label noise; we measured the false-negative rate and it does not differ across strategies" |
| C3 ≫ C2 | Explanation **(b)** — your result is partly a BigCloneBench artefact | "Harder negatives are more contaminated with unlabelled clones; the apparent degradation is attributable in part to label incompleteness, and the true hardness curve is likely flatter than we measured" |
| Both high (>30%) | The benchmark cannot support a clean hardness curve at all | Report it as the headline. This is a legitimate, publishable-style finding about the benchmark |

None of these is a failure. All three are more interesting than a bare F1 table.

---

## 4. What you do about it

| # | Mitigation | Cost | Where |
|---|---|---|---|
| 1 | **Exclude labelled clones** before using a fragment as a negative, with a unit test on an adversarial case | 1 hour | §7.2 of the spec |
| 2 | **Measure the false-negative rate** for C2 and C3 by manual judgement (§3.1) | 2 hours | §7.2 of the spec |
| 3 | **Report differences, not levels**, in every table and every claim | free | §2.1 here |
| 4 | **State the defects explicitly** in the limitations section, with the 93% and 15% figures cited | 30 min | §12 of the spec |
| 5 | **Report the functionality imbalance** of your training subset, and its concentration | 30 min | New; see below |
| 6 | **Restrict the generalisation test to well-represented functionalities**, since 22 of 43 have under 1% of the pairs | free | §9.3 of the spec |

### Mitigation 5, concretely

You probably cannot recover functionality IDs for the CodeXGLUE fragments cheaply (spec §9.3).
But you can still quantify the concentration of your training subset, which is what actually
biases your model:

```python
from collections import Counter
c = Counter()  # count how often each fragment appears across training pairs
for a, p in positive_pairs:
    c[a] += 1; c[p] += 1
counts = sorted(c.values(), reverse=True)
print("fragments:", len(c))
print("top-10 fragments cover", sum(counts[:10]) / sum(counts), "of positive-pair slots")
```

If ten fragments account for a large share of your training pairs, your model is being trained to
recognise a handful of very common Java methods, and your F1 is partly a measure of that.
**Report the number.** It is a one-line computation and it pre-empts the sharpest question you
will get.

---

## 5. What you cannot fix

Be explicit about these rather than quietly hoping nobody asks.

* **You cannot de-noise BigCloneBench.** Repair attempts exist (Li et al.) and the 2025 paper
  argues they are not demonstrably effective `[verified]`. Do not build one; you do not have the
  time and it would become a second project.
* **You cannot recover the missing functionality labels from the CodeXGLUE file alone.** Use
  Kitsios et al.'s released BCB s′ for the generalisation test instead.
* **You cannot make your absolute F1 mean "real-world clone detection accuracy".** It means
  "agreement with BigCloneBench's labels", and those labels are ~15% contested.

---

## 6. Why not switch benchmarks?

A fair question, since you now know all this. Answer it in the report.

| Alternative | Why not, for this project |
|---|---|
| **POJ-104** (CodeXGLUE's other clone dataset, C/C++, 32K/8K/12K) | Smaller and program-level; a retrieval task, not pair classification. Would need a different evaluation harness |
| **OJClone / CodeNet** | Same functionality-labelled structure, but you lose comparability with every published number, and the data engineering is a week |
| **GPTCloneBench / LLM-generated clones** | Very recent; construction provenance matters enormously, and you would spend the project validating the benchmark instead of the model |
| **Build your own** | Out of scope. Months, and no one can compare to your numbers |

**The honest position:** BigCloneBench is flawed *and* universal. Its flaws are documented,
quantified, and — crucially — **the one that interacts with your variable is measurable**. Use it,
cite the criticism, measure the interaction, and report differences rather than levels. That is a
defensible project. Switching to a cleaner-but-unknown benchmark four weeks before the deadline
is not.

---

## 7. Sentences you can put straight into the report

Adapt, do not paste.

> We use BigCloneBench via CodeXGLUE (9,134 Java fragments; 901,028 / 415,416 / 415,416 pairs)
> because it is the standard benchmark for this task and provides fixed splits. We acknowledge
> that its ground truth is contested: Krinke (IWSC 2022) reports that at least 15% of labelled
> snippets are subjective or contain validation errors, that an audit of 406 Weak Type-3/Type-4
> clone pairs found 93% to be mislabelled, and that over 90% of true clone pairs belong to just
> eight of the 43 functionalities.

> Because label errors are approximately independent of our manipulation, we report **differences
> between conditions** rather than absolute F1 levels, and we treat our absolute F1 as agreement
> with BigCloneBench's labels rather than as real-world clone-detection accuracy.

> A specific concern is that BigCloneBench's clone labels are incomplete both within and across
> functionalities, and that harder negative mining selects for exactly the unlabelled clones this
> incompleteness produces. We therefore measured the false-negative rate of our mined sets
> directly: 50 sampled negatives per strategy, manually judged for functional equivalence. The
> measured rate was X% for BM25 and Y% for semantic mining.

---

## 8. Sources for this document

| Source | Used for |
|---|---|
| Krinke, *BigCloneBench Considered Harmful for Machine Learning*, IWSC 2022 ([pdf](http://www0.cs.ucl.ac.uk/staff/j.krinke/publications/iwsc22.pdf)) | Findings 1–6; the imbalance figures |
| *How the Misuse of a Dataset Harmed Semantic Clone Detection*, 2025 ([arXiv:2505.04311](https://arxiv.org/html/2505.04311v1)) | The 406-pair audit; the 93% figure; the CodeXGLUE-specific analysis |
| Kitsios et al., *Detecting Semantic Clones of Unseen Functionality*, ASE 2025 ([arXiv:2510.04143](https://arxiv.org/pdf/2510.04143)) | The unseen-functionality result; the released BCB s′ dataset |
| CodeXGLUE ([GitHub](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench)) | Data format, split sizes, official F1 metric, the 10%-subsample precedent |
