# 03 — Feasibility matrix and five candidate paths

No subjective "best". Every cell below is a fact about the dataset/paper with a source. You decide, using the evidence, which paths fit one month.

---

## Part A — Candidate paths at a glance

| ID | Path | Domain notion of similarity | Primary dataset |
|---|---|---|---|
| **A** | Duplicate bug reports | "These two reports describe the same defect" | [GitBugs](https://arxiv.org/abs/2504.09651) (150k+ reports, 9 projects) |
| **B** | Code similarity | "These two programs solve the same problem" | [POJ-104](https://arxiv.org/abs/2102.04664) via CodeXGLUE (32K/8K/12K) |
| **C** | Duplicate questions ⭐ | "This question has been asked before" | [CQADupStack](https://eltimster.github.io/www/pubs/adcs2015.pdf) in [BEIR](https://arxiv.org/abs/2104.08663) (13,145 queries / 457K docs / 12 subforums) |
| **D** | Scientific/technical documents | "These papers are about the same thing" | [SciDocs](https://aclanthology.org/2020.acl-main.207/) / [SciFact](https://arxiv.org/abs/2104.08663) |
| **E** | Cybersecurity | "This CVE is the same weakness as that CWE" | NVD CVE/CWE ([example resource](https://huggingface.co/datasets/xamxte/cve-to-cwe)) |

---

## Part B — The feasibility matrix

Legend: ✅ good · ⚠️ mixed / needs care · ❌ disqualifying or severe

| Criterion | A. Bug reports | B. Code (POJ-104) | C. Dup. questions ⭐ | D. Scientific docs | E. Cybersecurity |
|---|---|---|---|---|---|
| **Dataset availability** | ✅ [GitHub](https://github.com/av9ash/gitbugs/) + [Kaggle](https://www.kaggle.com/datasets/av9ash/gitbugs), CC BY 4.0 | ✅ [CodeXGLUE](https://github.com/microsoft/CodeXGLUE) + [HF](https://huggingface.co/datasets/code_search_net) | ✅ One line via [ir_datasets](https://ir-datasets.com/beir.html) / [BEIR](https://github.com/UKPLab/beir) | ✅ SciDocs via [SPECTER](https://github.com/allenai/specter); SciFact in BEIR | ✅ NVD is public; [HF dataset](https://huggingface.co/datasets/xamxte/cve-to-cwe) exists |
| **Dataset size** | ⚠️ 150k reports but duplicates are only **2.0–28.2%** by project | ✅ 32K train / 8K dev / 12K test — ideal | ✅ 13,145 queries, 457K docs; per-subforum 10K–90K docs | ⚠️ SciDocs ~25.7K docs; large-scale training needs S2ORC | ⚠️ 290K CVEs, but the task is **classification**, not similarity |
| **Label quality** | ⚠️ Dataset paper is arXiv-only; **duplicate-label extraction is not quantitatively validated** (a public review flags this). Must spot-check. | ✅ Program→problem assignment is objective and machine-generated | ⚠️ Community duplicate marks; **~1.4 labelled duplicates per query** ⇒ many true duplicates unlabelled | ✅ Citation links are explicit and reliable | ❌ Known inconsistent/incomplete CVE→CWE mappings ([FixV2W](https://arxiv.org/abs/2604.22176)); AI-refined labels in some releases |
| **Positive-pair quality** | ✅ Natural: human "duplicate of" links | ✅ **Best of all paths** — same problem = genuinely equivalent | ✅ Natural: human duplicate marks | ✅ Citations are a natural relatedness signal | ❌ Not a pair task. "Same CWE" is a coarse class, not a similarity judgement |
| **Negative-pair quality** | ⚠️ "Not marked duplicate" is weak; many unlabelled duplicates | ✅ Different problem = genuinely different | ⚠️ Same weakness as A — but **this is the thing you measure** | ✅ Random papers are mostly genuinely unrelated | ❌ Two CVEs of different CWEs can still describe the same weakness |
| **Compute requirements** | ✅ ≤110M encoder, minutes per epoch | ✅ Same | ✅ Same; smallest corpora of the five | ⚠️ SPECTER-style pretraining is heavier; fine-tuning is not | ✅ Small |
| **Implementation complexity** | ⚠️ HTML/code-block stripping, long descriptions, duplicate-graph construction, temporal split | ✅ CodeXGLUE ships processed data; a plain Transformer works (no AST needed) | ✅ BEIR/ir_datasets handles loading and qrels | ⚠️ Metadata/scales vary; SciDocs needs the SPECTER eval code | ⚠️ Easy to load, hard to define a *similarity* task |
| **Literature support for the exact setup** | ✅ Strong ([Zhang TOSEM](https://doi.org/10.1145/3576042), [Jiang JSS](https://doi.org/10.1016/j.jss.2023.111607), [Rosane SBES](https://doi.org/10.5753/sbes.2025.9809)) | ✅ Strong, but **crowded** ([ContraCode](https://aclanthology.org/2021.emnlp-main.482/), [Corder](https://doi.org/10.1145/3404835.3462840), [CoCoSoDa](https://arxiv.org/abs/2204.03293), [UniXcoder](https://aclanthology.org/2022.acl-long.499/)) | ✅ Strong, and **published reference numbers exist** (see below) | ✅ Strong ([SPECTER](https://aclanthology.org/2020.acl-main.207/), [Ostendorff et al.](https://doi.org/10.1145/3529372.3530912), [TSDAE](https://aclanthology.org/2021.findings-emnlp.59/)) | ⚠️ Mostly classification papers, not embedding/contrastive work |
| **Can you validate your harness against a published number?** | ❌ No — [Zhang et al.](https://doi.org/10.1145/3576042) use their own benchmark; your split will differ | ⚠️ Partially — CodeBERT MAP 84.29 on POJ-104 is reported, so you can sanity-check roughly | ✅ **Yes.** [GPL's table](https://aclanthology.org/2022.naacl-main.168/) gives CQADupStack nDCG@10: zero-shot **29.6**, TSDAE **31.8**, GPL **34.5**, TSDAE+GPL **35.1**. BEIR also reports per-model numbers. | ⚠️ Partially — SciDocs numbers published but eval code is custom | ❌ No |
| **Evaluation difficulty** | ⚠️ You must build the retrieval harness and the temporal/graph split yourself | ⚠️ You must build the retrieval harness (CodeXGLUE reports MAP) | ✅ BEIR + ir_datasets give you queries, docs, qrels and standard metrics | ⚠️ Mixed task formats | ❌ No natural retrieval evaluation |
| **Reproducibility by a 2nd-year student** | ⚠️ Moderate — data cleaning is the risk | ✅ Easy | ✅ **Easiest** — the benchmark is designed for this | ⚠️ Moderate | ❌ Task definition is the risk |
| **Expected experimentation time** | ⚠️ 1 week of data engineering before any experiment | ✅ ~2 days to first result | ✅ **~2 days to first result** | ⚠️ ~1 week | ⚠️ open-ended |
| **Main technical risk** | Data cleaning + label noise + no external validation | Literature is crowded ⇒ hard to say anything new | **Sparse labels cap achievable Recall**; must report this honestly | Heavier preprocessing; less crisp "duplicate" notion | No clean pairwise similarity task; known leakage in prior work |

---

## Part C — Reading the matrix

**Which paths satisfy the one-month constraint?**

* **C (duplicate questions)** — satisfies every criterion except "negative-pair quality", and that weakness is exactly the research question. ✅
* **B (code)** — satisfies every criterion, with the best label quality of all five. ✅ Its only real drawback is that the literature is crowded.
* **A (bug reports)** — feasible but consumes ~1 week on data engineering and **cannot be externally validated**, which is the single biggest threat to a 4-week project. ⚠️
* **D (scientific docs)** — feasible if you only *fine-tune* (rather than pretrain), but positives are coarser and the evaluation code is custom. ⚠️
* **E (cybersecurity)** — **eliminate.** CVE→CWE is a classification task with a label hierarchy, not a pairwise similarity task with natural positives. Worse, a documented leakage issue exists in this literature: random train/test splits on historical NVD data put near-identical CVEs on both sides ([noted in arXiv:2603.14911](https://arxiv.org/html/2603.14911v1) about V2W-BERT).

**Why C over B, given B has better labels?**

B is the safer *engineering* choice; C is the better *research* choice for one month, for three evidence-backed reasons:

1. **C lets you verify your evaluation harness.** In week 1 you can reproduce a published CQADupStack nDCG@10 ([GPL](https://aclanthology.org/2022.naacl-main.168/): 29.6 zero-shot / 34.5 GPL). If your BM25 number lands near the published BEIR figure, your harness is probably correct. With B you have a rough sanity check (CodeBERT MAP 84.29) but not an exact one.
2. **C has 12 built-in domains.** Its twelve subforums (android, english, gaming, gis, mathematica, physics, programmers, stats, tex, unix, webmasters, wordpress) give you a cross-domain generalisation experiment *for free* — you hold out one subforum and train on the rest. That directly operationalises "domain-specific" and answers Gap 3.
3. **C's sparse labels make the false-negative question sharp and measurable** (Gap 2). With ~1.4 labelled duplicates per query, "is this mined hard negative really a negative?" is a live, quantifiable question. In POJ-104 the labels are complete, so the false-negative problem is much weaker — which means the most interesting variable in your project would be nearly inert.

**Counter-argument you should weigh honestly:** POJ-104 gives you cleaner supervision, so your *model-quality* numbers would be more trustworthy. If your priority is "learn the machinery and get a clean, strong result", B is defensible. If your priority is "produce a defensible empirical finding about a contested variable", C is better. **This document recommends C, with B as the documented fallback.**

---

## Part D — The five paths in full

---

### Path A — Duplicate Bug Reports

```text
Bug reports (GitBugs: 9 projects, 150k+)
    ↓
"duplicate of" links  →  duplicate-group graph
    ↓
Positive pairs:  (a, p) in same group, p reported EARLIER than a
    ↓
Negative sampling:  random / BM25-lexical / model-mined / denoised
    ↓
Base: all-MiniLM-L6-v2  (zero-shot = baseline)
    ↓
Contrastive fine-tuning (MultipleNegativesRankingLoss)
    ↓
FAISS retrieval over all EARLIER reports in the project
    ↓
Recall@k / MRR@10 / MAP / nDCG@10
```

* **Research question:** Does contrastive fine-tuning on duplicate links improve duplicate-report retrieval over BM25 and over the zero-shot encoder, and does the negative-sampling strategy determine the effect?
* **Dataset:** [GitBugs](https://github.com/av9ash/gitbugs/) (CC BY 4.0). Use high-duplicate-rate projects: VS Code (28.2%), Thunderbird (27.6%), Mozilla Core (20.9%), Firefox (21.7%).
* **Model / loss / training:** `all-MiniLM-L6-v2`, MNRL, batch 64, lr 2e-5, 3 epochs, max 256 tokens, fp16.
* **Baselines:** BM25, TF-IDF, `all-MiniLM-L6-v2` zero-shot, `BAAI/bge-small-en-v1.5` zero-shot, BM25∪dense hybrid.
* **Split:** **temporal and group-constrained** — sort by timestamp, cut at quantiles, and assign an entire duplicate group to the split of its earliest member.
* **Evaluation:** Recall@1/5/10/20, MRR@10, MAP, nDCG@10. Report per project + 3 seeds.
* **Expected difficulty:** High (data engineering dominates).
* **Biggest risk:** No external validation of your harness. Mitigate by also running Path C's validation in week 1 if you choose this path.
* **Literature you must beat / cite:** [Zhang et al. TOSEM 2023](https://doi.org/10.1145/3576042) (RR@5 ≈ 0.4–0.6; simple retrieval wins), [Jiang et al. JSS 2023](https://doi.org/10.1016/j.jss.2023.111607) (lexical > semantic), [Rosane et al. SBES 2025](https://doi.org/10.5753/sbes.2025.9809) (fine-tuning marginal).
* **A meaningful result:** either (i) fine-tuning with denoised hard negatives beats BM25 and zero-shot by a margin larger than the seed-to-seed std, with a measured false-negative rate explaining why; or (ii) it does not, and the false-negative rate plus the lexical-overlap analysis explains why.

---

### Path B — Code Similarity (POJ-104)

```text
C/C++ programs (POJ-104, 104 problems)
    ↓
Semantic equivalence = "solves the same problem"
    ↓
Positives: two programs for the same problem
    ↓
Negatives: random / same-language-hard / model-mined / denoised
    ↓
Base: all-MiniLM-L6-v2 or microsoft/codebert-base
    ↓
Contrastive fine-tuning (MNRL)
    ↓
Retrieval: for a query program, rank all test programs
    ↓
MAP / Recall@10 / MRR
```

* **Research question:** Does contrastive fine-tuning with mined hard negatives improve *semantic* (not lexical) code retrieval over a zero-shot encoder and over token-based clone detection?
* **Dataset:** [POJ-104 via CodeXGLUE](https://github.com/microsoft/CodeXGLUE) — 32K/8K/12K train/dev/test. Published baseline: **CodeBERT MAP 84.29**, RoBERTa 79.96, code2vec 1.98, NCC 54.19, Aroma 55.12, MISIM-GNN 82.45 ([CodeXGLUE](https://arxiv.org/abs/2102.04664) Table 8).
* **Model / loss / training:** `microsoft/codebert-base` (125M) **or** `all-MiniLM-L6-v2`. MNRL. Longer sequences needed — budget 512 tokens.
* **Baselines:** token/lexical similarity, BM25 over code tokens, CodeBERT zero-shot `[CLS]`, your base encoder zero-shot.
* **Split:** use the CodeXGLUE split. **Critical:** group by *problem*, so no problem appears in both train and test.
* **Evaluation:** MAP (the POJ-104 convention), Recall@10, MRR.
* **Expected difficulty:** Low–Medium.
* **Biggest risk:** **Crowded literature** — [ContraCode](https://aclanthology.org/2021.emnlp-main.482/), [Corder](https://doi.org/10.1145/3404835.3462840), [CoCoSoDa](https://arxiv.org/abs/2204.03293) have all done contrastive learning for code. Your result must be framed as a controlled negative-sampling study, not as "contrastive learning for code".
* **A meaningful result:** a clean ablation showing *which* negative strategy matters for semantic-vs-lexical code similarity, with the false-negative rate reported. Also: an analysis of whether hard negatives specifically help on **Type-4 (semantic) clones** while hurting on lexical ones.

---

### Path C — Duplicate Question Retrieval (CQADupStack) ⭐ RECOMMENDED

```text
StackExchange posts (12 subforums, 13,145 queries / 457K docs)
    ↓
Community "duplicate" marks  →  duplicate-group graph
    ↓
Positives: (q, d+) where d+ is a marked duplicate
    ↓
Negatives: N1 random / N2 in-batch / N3 BM25-lexical
           / N4 model-mined / N5 mined+denoised
    ↓
Base: all-MiniLM-L6-v2  (zero-shot = baseline)
    ↓
Contrastive fine-tuning (MNRL), 3 seeds
    ↓
FAISS retrieval over the subforum's document pool (FIXED)
    ↓
nDCG@10 (primary) · Recall@1/10/100 · MRR@10 · MAP
    ↓
+ held-out subforum = domain-generalisation experiment
```

* **Research question:** see [`04-project-specification.md`](04-project-specification.md).
* **Dataset:** [CQADupStack](https://eltimster.github.io/www/pubs/adcs2015.pdf), accessed via BEIR / `ir_datasets` (`beir/cqadupstack/<subforum>`). 12 subforums; per-subforum query counts range from ~652 (stats) to ~2,900 (tex).
* **Model / loss / training:** `all-MiniLM-L6-v2`, MNRL, batch 64, lr 2e-5, 2–4 epochs, 256 tokens, fp16.
* **Baselines:** BM25, TF-IDF+cosine, `all-MiniLM-L6-v2` zero-shot, `BAAI/bge-small-en-v1.5` zero-shot, BM25∪dense hybrid. Published reference: CQADupStack nDCG@10 = **29.6** zero-shot / **31.8** TSDAE / **34.5** GPL / **35.1** TSDAE+GPL ([GPL, Table 7](https://aclanthology.org/2022.naacl-main.168/)).
* **Split:** duplicate-group-constrained within subforum; **whole-subforum hold-out** for the domain-generalisation run.
* **Evaluation:** nDCG@10 primary; Recall@1/10/100, MRR@10, MAP secondary; false-negative rate diagnostic.
* **Expected difficulty:** Low.
* **Biggest risk:** sparse labels cap absolute Recall. Mitigation: this is expected and reported; you compare *systems*, not absolute quality, and you report the label-sparsity ceiling explicitly.
* **A meaningful result:** (i) a validated harness reproducing published BEIR-scale numbers; (ii) a 5-way negative-strategy ablation with 3 seeds; (iii) the first reported **false-negative rate** for mined hard negatives in duplicate-question retrieval; (iv) a cross-subforum transfer result.

---

### Path D — Scientific / Technical Documents

```text
Papers (title + abstract)
    ↓
Citation / co-citation / shared-aspect relations
    ↓
Positives: papers linked by a citation or a shared aspect label
    ↓
Negatives: random / "citations of citations" (SPECTER's hard negatives)
    ↓
Base: allenai/scibert_scivocab_uncased
    ↓
Triplet-margin or MNRL fine-tuning
    ↓
Retrieval / k-NN recommendation
    ↓
nDCG@10 / MAP / Precision@k (SciDocs)
```

* **Research question:** Can a small domain encoder be specialised to a *specific aspect* of similarity (task / method / dataset), and does that specialisation transfer?
* **Dataset:** [SciDocs](https://aclanthology.org/2020.acl-main.207/) (7 document-level tasks, 25.7K docs in BEIR) or [SciFact](https://arxiv.org/abs/2104.08663) (300 queries / 5.18K docs). Aspect labels: Papers-with-Code, 157,606 papers ([Ostendorff et al.](https://doi.org/10.1145/3529372.3530912)).
* **Model / loss:** SciBERT (110M); **MultipleNegativesRankingLoss** (this is exactly what [Ostendorff et al.](https://doi.org/10.1145/3529372.3530912) used, with code released under MIT).
* **Baselines:** BM25, SciBERT zero-shot, SPECTER, FastText.
* **Expected difficulty:** Medium.
* **Biggest risk:** Heavier preprocessing (metadata, abstracts, citation graph) and a less crisp notion of "the same thing". Also, if you want to *pretrain* (SPECTER-style), you need far more compute than one month allows — restrict yourself to fine-tuning.
* **A meaningful result:** reproduction of the aspect-specialisation finding at small scale, plus a negative-sampling ablation on top.

---

### Path E — Cybersecurity / Vulnerability Descriptions — ❌ ELIMINATED

```text
CVE descriptions → CWE categories
    ↓  (this is CLASSIFICATION, not pairwise similarity)
```

* **Why eliminated:**
  * The natural task here is **CVE→CWE classification**, not similarity retrieval. There is no naturally occurring *pair* of "these two vulnerability descriptions mean the same thing".
  * Labels are unreliable: NVD's CVE→CWE mappings are documented as inconsistent and incomplete ([FixV2W](https://arxiv.org/abs/2604.22176)).
  * Known leakage: random train/test splits on historical NVD data place near-identical CVEs on both sides, inflating reported accuracy ([arXiv:2603.14911](https://arxiv.org/html/2603.14911v1), discussing V2W-BERT's 94–97%).
  * "Two CVEs with different CWEs" is not a reliable negative — different weakness labels can still describe the same underlying flaw.
* **If you insist on security text anyway:** reframe as *retrieval of similar vulnerability reports* using the MITRE ATT&CK → CAPEC → CWE → CVE link chain as positives (this is done in [arXiv:2509.02077](https://pith.science/paper/2509.02077), which fine-tunes a sentence transformer with CosineSimilarityLoss). But be aware the positives are *derived from chained metadata*, not from human similarity judgements, and the paper's own reviewers flag oracle incompleteness and two-author subjective validation. Higher risk, lower payoff than Path C.

---

## Part E — Decision

| Path | Verdict |
|---|---|
| **C — Duplicate questions (CQADupStack)** | ✅ **Recommended.** Best ratio of research value to execution risk. Only path where you can externally validate your harness in week 1. |
| **B — Code (POJ-104)** | ✅ **Fallback / alternative.** Cleanest labels, easiest execution, but crowded literature and the false-negative variable is weaker here. |
| **A — Bug reports (GitBugs)** | ⚠️ **Only if you have a supervisor's existing cleaned data or are willing to spend week 1 entirely on data engineering, and you accept having no external validation number.** |
| **D — Scientific docs** | ⚠️ Viable but heavier; choose it only if you have a specific interest in scholarly document processing. |
| **E — Cybersecurity** | ❌ Not suitable for a contrastive-similarity project in one month. |
