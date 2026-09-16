# 07 — Four-week plan

Assumes ~25–30 hours/week. Each week has **exit criteria** — do not start the next week until they are met. If you are behind, cut from the SHOULD/NICE list in [`05`](05-clean-experiment-and-scope.md), never from the MUST list.

---

## Week 1 — Literature, dataset, and a validated harness

**Read (about 8 hours total):**

| Paper | Sections | Why now |
|---|---|---|
| [SBERT (Reimers & Gurevych, 2019)](https://aclanthology.org/D19-1410/) | §3, §3.1, §7 | The architecture you are copying |
| [SimCSE (Gao et al., 2021)](https://aclanthology.org/2021.emnlp-main.552/) | §2, §3 | The objective and the hard-negative trick |
| [Wang & Isola, ICML 2020](https://mlanthology.org/icml/2020/wang2020icml-understanding/) | all (11 pp) | Alignment/uniformity — your diagnostic |
| [DPR (Karpukhin et al., 2020)](https://arxiv.org/abs/2004.04906) | §3.2, §5.2 | In-batch negatives; the negative-type ablation |
| [ANCE (Xiong et al., 2021)](https://openreview.net/forum?id=zeFrfgyZln) | §1–§3 | Why in-batch negatives can be uninformative |

**Do:**

* **Day 1–2.** Install the stack. `pip install sentence-transformers ir_datasets rank_bm25 ranx faiss-cpu pandas scikit-learn matplotlib`. Encode three sentences with `all-MiniLM-L6-v2` and print a cosine similarity matrix. Confirm you understand what the numbers mean.
* **Day 2.** Load **three** subforums (`android`, `programmers`, `unix`) via `ir_datasets`. Write `corpus.jsonl`, `queries.jsonl`, `qrels.tsv`.
* **Day 2–3.** Cleaning: strip HTML, fenced code blocks, inline code, URLs. Build `text = title + ". " + body`.
* **Day 3.** **EDA and write it down:** queries/docs per subforum; group-size distribution; **gold-per-query distribution** (your ceiling argument); token-length percentiles (justifies 256 tokens).
* **Day 3.** `build_groups.py` — duplicate graph → connected components.
* **Day 4.** **♣ The critical milestone: reproduce a published number.** Implement BM25 + `ranx` nDCG@10 and confirm your BM25 score on CQADupStack lands in the published range (published CQADupStack nDCG@10 figures cluster around the high-20s to mid-30s; [GPL](https://aclanthology.org/2022.naacl-main.168/) reports 29.6 for a zero-shot dense model and 34.5 for GPL, and BEIR reports BM25 separately). **If your number is wildly off, stop and debug — your harness is wrong and everything downstream is worthless.**
* **Day 4–5.** `split.py` with group-constrained assignment + the leakage assertion. `positives.py`.

**Exit criteria for week 1:**
- [ ] BM25 nDCG@10 computed and in a sane range; number recorded.
- [ ] `groups.json` exists; every query has a `group_id`.
- [ ] Train/val/test split exists **and** the leakage assertion passes and is printed.
- [ ] EDA numbers written into a note you will paste into the paper.
- [ ] You can explain, out loud, what MNRL is minimising.

---

## Week 2 — Baselines and pair generation. **No training yet.**

**Read (about 4 hours):**

| Paper | Sections | Why now |
|---|---|---|
| [Robinson et al., ICLR 2021](https://openreview.net/forum?id=CR1XOQ0UTh-) | all | The two principles; why "hardest" is wrong |
| [RocketQA (Qu et al., 2021)](https://aclanthology.org/2021.naacl-main.466/) | §3–§4 | Denoised hard negatives — your N5 |
| [AugSBERT (Thakur et al., 2021)](https://aclanthology.org/2021.naacl-main.28/) | §3–§4 | Pair-sampling strategies: Random / BM25 / Semantic Search / KDE |

**Do:**

* **Day 1.** Finish `run_retrieval.py` so that **any** system (any `encode_fn`) produces a run file and metrics through the identical path.
* **Day 1–2.** Baselines:
  * **B0** BM25 (`rank_bm25`)
  * **B1** TF-IDF + cosine (`scikit-learn`)
  * **B2** `all-MiniLM-L6-v2` zero-shot
  * **B3** `BAAI/bge-small-en-v1.5` zero-shot — **with the query prefix** `"Represent this sentence for searching relevant passages: "`. Verify the prefix changes the score; if it does not, you have a bug.
  * **B4** BM25 ∪ dense hybrid (normalise both score distributions, then sum)
* **Day 2.** **Write the complete baseline table.** This is the moment your project becomes safe: whatever happens in weeks 3–4, you already have a valid empirical result.
* **Day 3.** **Pre-register.** Create `PRE_REGISTRATION.md`: the research question, H1 and H0 verbatim from [`04`](04-project-specification.md), the primary metric (nDCG@10), the five negative strategies, the seed list, and the ablation plan. Commit it with a timestamp.
* **Day 3–4.** `negatives.py`: implement N1 (random), N3 (BM25 lexical), N4 (FAISS model-mined), N5 (N4 + margin filter). Write `test_pairs.py`: **no positive may appear in its own negative set**.
* **Day 4–5.** `diagnostics.py`: the false-negative estimator. Sanity-check it by hand on 20 examples — does the cross-encoder agree with your eye?

**Exit criteria for week 2:**
- [ ] Baseline table complete: 5 systems × 3 subforums × {nDCG@10, R@1, R@10, R@100, MRR@10, MAP}.
- [ ] `PRE_REGISTRATION.md` committed.
- [ ] `negatives.py` produces N1/N3/N4/N5 with unit tests passing.
- [ ] False-negative estimator runs and you have hand-checked ≥20 of its judgements.

---

## Week 3 — Contrastive training

**Do:**

* **Day 1.** Get **one** training run working end to end: N2 (in-batch only), seed 42, batch 64, lr 2e-5, 3 epochs, 256 tokens. Evaluate on **validation** every 500 steps; save the best checkpoint.
* **Day 1.** Sanity check: does validation nDCG@10 actually move? If it collapses, check LR first ([Rosane et al.](https://doi.org/10.5753/sbes.2025.9809) found LR 2e-7 catastrophically bad in a closely related setting).
* **Day 2.** Sweep LR {1e-5, 2e-5, 5e-5} on N2. Pick one. **From here on, LR is frozen.**
* **Day 2–4.** Launch **A1**: N1, N2, N3, N4, N5 × 3 seeds = 15 runs. Run them overnight; babysitting a progress bar is wasted time.
* **Day 3 (parallel).** While training runs, compute the **false-negative rate** for each strategy's mined negatives. This does not need the trained model — it needs the mined sets and a cross-encoder.
* **Day 4.** Evaluate all 15 checkpoints on **validation**. Assemble the A1 table.
* **Day 5.** **A2: number of hard negatives {0, 1, 3, 5}** on the best strategy, 3 seeds. Four runs; cheap.

**Exit criteria for week 3:**
- [ ] 15 A1 runs completed, all metrics logged.
- [ ] A2 completed.
- [ ] False-negative rates computed for N3/N4/N5.
- [ ] You know, roughly, which pattern from [`04`](04-project-specification.md) §"Expected results" you are looking at. **Resist writing the conclusion yet.**

---

## Week 4 — Evaluation, analysis, writing

**Do:**

* **Day 1.** **Touch the test set for the first and only time.** Evaluate every system (all baselines + best-of-each-strategy by validation score) on test. Primary: nDCG@10. Secondary: Recall@1/10/100, MRR@10, MAP.
* **Day 1.** Statistics: mean ± std over 3 seeds; **paired bootstrap over queries** (10k resamples) for the headline comparisons.
* **Day 2.** **A4 (SHOULD): held-out subforum.** Train on 11 subforums, test on the 12th. This takes one training run and gives you the domain-generalisation result.
* **Day 2.** Diagnostics: alignment and uniformity for the best and worst configuration.
* **Day 3.** **Error analysis** — see [`08`](08-error-analysis.md). FP/FN inspection, lexical-overlap buckets, the 100-query ceiling judgement.
* **Day 3–4.** Figures: (1) bar chart nDCG@10 by system; (2) nDCG@10 vs. number of hard negatives; (3) nDCG@10 vs. measured false-negative rate, one point per strategy — **this is your money figure**; (4) per-subforum heatmap.
* **Day 4–5.** Write. Structure: Intro → Related work → Method → Experimental setup → Results → Threats to validity → Discussion → Reproducibility.
* **Day 5.** `README.md` with exact reproduction steps; pin versions; push the repo; archive `results/`.

**Exit criteria for week 4:**
- [ ] Main table: all systems × primary + secondary metrics, mean ± std.
- [ ] At least one ablation table.
- [ ] Error analysis section with concrete examples.
- [ ] Threats-to-validity section that names: sparse labels, false negatives, single-domain-family risk, 3 seeds only, one model family.
- [ ] `README.md` + `run_all.sh` reproduce the numbers.

---

## Contingency: what to cut, in this order

1. NICE list (N1–N7 in [`05`](05-clean-experiment-and-scope.md)).
2. A2 (number of hard negatives).
3. The 12-subforum run → fall back to 3 subforums.
4. A4 (held-out subforum).
5. B3 and B4 baselines (**never** B0 or B2).
6. The second seed. **Never cut to one seed** — if you must, report the single run as a pilot and say so explicitly.

## Contingency: what to do if you are *ahead*

1. A3 data efficiency (four short runs, high interpretive value).
2. All 12 subforums → a proper cross-domain matrix.
3. Triplet-loss comparison.
4. The ceiling analysis with a second human annotator (ask a friend to independently label 100 pairs; report inter-rater agreement — this makes the ceiling claim credible rather than anecdotal).

## Daily hygiene (30 minutes)

* Push to git at the end of every day.
* Any result you compute, write to `results/metrics/*.json` immediately. Do not keep numbers in your head or in a scratch terminal.
* One `NOTES.md` where you log what you tried, what happened, and what you concluded. This file becomes your paper's first draft.
