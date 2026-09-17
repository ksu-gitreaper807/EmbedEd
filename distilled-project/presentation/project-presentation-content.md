# Domain-Specific Contrastive Embeddings for Code Similarity Detection

*Slide content — strictly following the required structure. Core framing: the topic is whether domain-specific contrastive fine-tuning (specifically, negative-sampling strategy) improves embeddings over generic ones. Code similarity detection is used as the test case, not the topic itself.*

---

## 1) Introduction

### Overview of the Topic

- Generic pretrained embedding models are trained on broad, general-purpose objectives — they don't inherently capture what "similarity" means within a specific, specialized domain
- **Contrastive learning** is a technique that trains embeddings by showing pairs of similar and dissimilar examples, teaching the model to represent domain-specific similarity directly
- This project investigates a specific, underexplored lever within that training process: **how does the choice of negative-sampling strategy** (random vs. keyword-based vs. semantic) affect the quality and generalization of the resulting embeddings?
- **Code similarity detection** is used as our test case — a domain with strong existing benchmarks, allowing rigorous, comparable evaluation

### Importance of the Topic

- Domain-specific similarity is needed across many fields — legal clause matching, duplicate bug reports, code clone detection, medical record matching — where generic, off-the-shelf embeddings are a one-size-fits-all compromise
- In our test domain specifically: BigCloneBench documents **8+ million** validated code-clone pairs drawn from **25,000** real open-source projects, showing the practical scale of similarity-detection needs
- Negative-sampling strategy is a known lever in contrastive learning generally, but its specific effect for domain adaptation remains under-studied and rarely compared systematically
- **Justification:** as AI-assisted content generation (across text, code, and other domains) makes superficially different-but-equivalent content increasingly easy to produce, training embeddings that capture true semantic similarity — rather than surface similarity — becomes more urgent, not less

---

## 2) Literature Review

### Key Studies and Findings

| Study | Contribution |
|---|---|
| Guo et al., "GraphCodeBERT: Pre-training Code Representations with Data Flow," ICLR, 2021 | Introduced structural, data-flow-aware pretraining for code, improving on purely token-based models |
| Jain et al., "Contrastive Code Representation Learning," EMNLP, 2021 | Applied contrastive learning directly to source code representation, demonstrating its viability for this modality |
| Sonnekalb et al., "Generalizability of Code Clone Detection on CodeBERT," ASE, 2022 | Found that a ~96.5% F1 score on standard benchmarks drops significantly when evaluated on unseen functionalities |
| Liu et al., "ContraBERT: Enhancing Code Pre-trained Models via Contrastive Learning," ICSE, 2023 | Showed contrastive learning improves robustness of code pretrained models beyond standard fine-tuning |
| Khajezade, Fard & Shehata, "Evaluating Few-shot and Contrastive Learning Methods for Code Clone Detection," Empirical Software Engineering, 2024 | Directly evaluated contrastive learning under limited-data and unseen-problem conditions for clone detection |

### Trends in Literature

- Shift from **generic-purpose embeddings** toward **domain-adapted representations** trained via contrastive objectives (Jain et al., 2021; Liu et al., 2023)
- Increasing scrutiny of **generalizability claims** — multiple 2022–2024 studies question whether strong benchmark scores reflect genuine understanding or benchmark-specific exposure
- Contrastive learning increasingly favored over classification-based fine-tuning for producing reusable, general-purpose embeddings
- **Gap observed across this literature:** negative-sampling strategy is acknowledged as influential in contrastive learning generally, but is rarely isolated and compared as an explicit experimental variable

### Theoretical Framework

- **Contrastive Representation Learning** (Le-Khac, Healy & Smeaton, IEEE Access, 2020): embeddings are learned by pulling similar pairs together and pushing dissimilar pairs apart in vector space, rather than fitting fixed category labels
- **Classical Information Retrieval theory (BM25):** a keyword-relevance ranking model, used here as one method for constructing "hard" negative training pairs
- **Relevance:** this framework directly supports domain adaptation, since similarity is learned relative to explicitly chosen pairs — making the pairs (and how negatives are chosen) a first-class design decision, not an implementation detail

### Relevance to Current Research

- Adopts the same base models (GraphCodeBERT) and benchmark (BigCloneBench) used in Guo et al. (2021), Sonnekalb et al. (2022), and Khajezade et al. (2024), ensuring our results are directly comparable
- Directly extends Sonnekalb et al. (2022)'s generalizability concern by making generalization testing a **built-in, core experiment** rather than a follow-up critique
- Directly extends Khajezade et al. (2024) and Liu et al. (2023) by isolating **negative-sampling strategy specifically** as the variable under study — a systematic comparison not present in existing published work

### What Already Exists vs. What We're Doing Differently

*(Explicit slide per evaluator instruction)*

| Already Exists | Our Contribution |
|---|---|
| Contrastive learning shown to improve code representations generally (Jain et al., 2021; Liu et al., 2023) | We isolate **negative-sampling strategy** as the specific variable, comparing random / BM25 / semantic approaches directly |
| Generalization gaps documented for fine-tuned code models (Sonnekalb et al., 2022; Khajezade et al., 2024) | We build generalization testing into the **core experimental design**, not a separate critique |
| Domain-specific fine-tuning studied per-domain, in isolation | We frame this as a **method-level question** (does negative strategy matter?) using code as one concrete, well-supported test case |

---

## 3) Problem Definition

### Clear Statement of the Problem

> "It is unclear whether the method used to select negative training examples during contrastive fine-tuning meaningfully affects the quality and generalization of domain-specific embeddings, or whether such improvements — if any — hold up beyond the training distribution."

### Context of the Problem

- Generic pretrained models are trained on broad objectives (e.g., masked language modeling), not explicitly for domain-specific similarity judgment
- Contrastive fine-tuning is a known method for adapting embeddings to a domain, but the specific choice of *negative* pairs used during this training is rarely treated as a controlled variable in existing studies
- Prior work (Sonnekalb et al., 2022) shows benchmark performance can significantly overstate real generalization — raising the question of whether this is affected by *how* a model was trained, including negative-sampling method

### Significance of Addressing the Problem

- **If unaddressed:** practitioners adapting embeddings for a new domain default to the simplest option (random negatives) without evidence this is actually the best choice
- **Gap:** without a systematic comparison, real, available performance improvements from better negative-sampling may be left unclaimed
- **Consequence:** deployed similarity-detection systems (in code, legal text, support tickets, or any specialized domain) may underperform silently on real-world, unseen inputs, while appearing strong on standard benchmarks

---

## 4) Objectives

### Main Objective

> "To evaluate whether the choice of negative-sampling strategy in contrastive fine-tuning meaningfully affects domain-specific embedding quality and generalization, using code similarity detection as a controlled test case."

### Specific Objectives

1. Fine-tune a pretrained model (GraphCodeBERT) using three distinct negative-sampling strategies: random, BM25-based, and semantically-mined
2. Compare the resulting embedding quality of each strategy against an off-the-shelf, non-fine-tuned baseline
3. Evaluate whether performance improvements generalize to code functionalities not represented during training
4. Visually and empirically demonstrate differences in embedding quality across strategies

### Expected Contributions to the Field

- A direct, controlled comparison of negative-sampling strategies for contrastive fine-tuning — a comparison not present in existing published literature for this domain
- Evidence-based guidance on whether "harder" negative-mining strategies justify their added implementation cost over simple random sampling
- A generalization-focused evaluation methodology directly responding to documented benchmark-overfitting concerns, applicable beyond this specific test case

---

## 5) Methodology

### Design

- Quantitative, controlled experimental design
- Four conditions compared under identical settings: baseline (no fine-tuning) + three fine-tuned variants (random / BM25 / semantic negatives)
- Only the negative-sampling method differs between the three fine-tuned conditions — all other training parameters held constant for interpretability

### Data Collection Methods

- **Source dataset:** BigCloneBench (CodeXGLUE-processed version) — real, labeled Java code-clone pairs
- **Positive pairs:** true clone pairs from the dataset, used unchanged across all four conditions
- **Negative pairs, constructed three ways:**
  - Random selection of non-clone snippets
  - BM25-based retrieval of keyword-similar but non-clone snippets
  - Embedding-based retrieval (using the base model itself) of semantically-similar but non-clone snippets

### Data Analysis Techniques

- Quantitative comparison of retrieval/similarity metrics (e.g., MAP@R, precision/recall at threshold) across all four conditions
- Generalization analysis: performance on standard test split vs. held-out, unseen functionality categories
- Visual/qualitative analysis: t-SNE/UMAP dimensionality reduction to compare embedding cluster separation across conditions

### Tools and Resources Used

- Python, PyTorch
- Hugging Face Transformers, sentence-transformers
- rank_bm25 (BM25 implementation)
- BigCloneBench / CodeXGLUE dataset
- GraphCodeBERT (base pretrained model)
- t-SNE / UMAP (scikit-learn, umap-learn)
- Compute: free-tier GPU (Google Colab)

### Research Process Flowchart

```
     Domain Dataset (BigCloneBench)
                │
                ▼
      Positive Pairs (fixed)
                │
   ┌────────────┼────────────┐
   ▼            ▼            ▼
 Random       BM25        Semantic
 Negatives   Negatives    Negatives
   │            │            │
   ▼            ▼            ▼
 Fine-tune 1  Fine-tune 2  Fine-tune 3
   │            │            │
   └────────────┼────────────┘
                ▼
     + Baseline (no fine-tuning)
                │
                ▼
    Standard Benchmark Evaluation
                │
                ▼
    Generalization Evaluation
    (unseen functionality categories)
                │
                ▼
    Results Comparison + Visualization
```

---

## 6) Proposed Outcomes

### Expected Results

- Fine-tuned embeddings (at least one strategy) expected to outperform the off-the-shelf baseline
- Harder negative-mining strategies (BM25, semantic) expected to outperform random negatives, consistent with general contrastive learning literature — though the *degree* of difference is the open question being tested
- Some performance drop expected on the generalization test relative to the standard benchmark split, consistent with Sonnekalb et al. (2022) — the *size* of this drop across strategies is a new, specific finding

### Potential Impact on the Field

- Provides evidence-based guidance for selecting negative-sampling strategy when adapting embeddings to any specialized domain — not limited to code
- Reinforces generalization testing as standard practice in domain-adaptation research, rather than relying solely on standard benchmark splits

### Future Research Directions

- Apply the same negative-sampling comparison to other domains (legal text, bug reports) to test whether findings generalize beyond code
- Explore automatic negative-strategy selection for new, unseen domains based on pilot experiments (deliberately excluded from this project to preserve a clean, interpretable comparison)
- Investigate hybrid negative-mining strategies combining random, BM25, and semantic approaches in weighted proportions

---

## 7) Conclusion

### Summary of Key Points

- Generic embeddings may not capture domain-specific similarity; contrastive fine-tuning is a known adaptation method, but *how* negative examples are chosen remains under-studied
- Code similarity detection serves as a rigorous, well-supported test case, using GraphCodeBERT and BigCloneBench
- The project directly investigates negative-sampling strategy as a controlled variable, with generalization testing built in from the start

### Final Thoughts

> "How we train a model to recognize similarity is as important as the architecture itself — this project treats the training methodology as a genuine subject of investigation, not an implementation detail."

### Call to Action / Next Steps

- Implement and validate the three negative-mining pipelines
- Run comparative training and generalization evaluation across all four conditions
- Extend the comparison to a second domain to test whether findings transfer beyond code

---

## 8) References (IEEE Style)

```
[1] D. Guo et al., "GraphCodeBERT: Pre-training Code Representations
    with Data Flow," in Proc. Int. Conf. Learning Representations
    (ICLR), 2021.

[2] P. Jain, A. Jain, T. Zhang, P. Abbeel, J. Gonzalez, and I. Stoica,
    "Contrastive Code Representation Learning," in Proc. Conf.
    Empirical Methods in Natural Language Processing (EMNLP), 2021,
    pp. 5954-5971.

[3] T. Sonnekalb, B. Gruner, C.-A. Brust, and P. Mader, "Generalizability
    of Code Clone Detection on CodeBERT," in Proc. 37th IEEE/ACM Int.
    Conf. Automated Software Engineering (ASE), Rochester, MI, USA,
    2022.

[4] S. Liu, B. Wu, X. Xie, G. Meng, and Y. Liu, "ContraBERT: Enhancing
    Code Pre-trained Models via Contrastive Learning," in Proc.
    IEEE/ACM 45th Int. Conf. Software Engineering (ICSE), 2023,
    pp. 2476-2487.

[5] M. Khajezade, F. H. Fard, and M. S. Shehata, "Evaluating Few-shot
    and Contrastive Learning Methods for Code Clone Detection,"
    Empirical Software Engineering, vol. 29, no. 6, p. 163, 2024.

[6] P. H. Le-Khac, G. Healy, and A. F. Smeaton, "Contrastive
    Representation Learning: A Framework and Review," IEEE Access,
    vol. 8, pp. 193907-193934, 2020.
```

---

## 9) Expected Outcomes (EL Deliverable)

*EL work is expected to produce one of the following tangible outputs:*

- **Patent Filing:** Not the primary target for this project — the contribution is a comparative empirical study (evaluating negative-sampling strategies), not a novel functional prototype or process, so patentability is limited. Not pursued as the main outcome.

- **Journal Publication:** Strong fit. A paper detailing the methodology (four-condition comparison), experimental design, and results (standard-benchmark and generalization findings) is well-suited for submission to a software engineering or applied machine learning journal — following the same format as several of the reviewed studies (e.g., Empirical Software Engineering).

- **Conference Publication:** Primary planned outcome. Prepare a full paper in **IEEE format** covering the comparative methodology, key findings across the three negative-sampling strategies, and the generalization evaluation, for submission to a relevant software engineering or AI conference (format to be finalized as shared later).

- **Research Proposal:** A natural extension — propose applying the same negative-sampling comparison methodology to additional domains beyond code (e.g., legal text, bug reports), and explore the "auto-selector" direction (automatically choosing a negative-sampling strategy for a new domain) as a larger follow-up project, including objectives, proposed methodology, and expected impact.

**Planned focus for this EL:** Conference publication (IEEE format), with the underlying study structured to also support a future journal submission or research proposal extension.
