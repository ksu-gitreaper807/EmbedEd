## Phase 5 — verified primary results

Primary benchmark: seeds 13–15; C0 once. Mean ± sample SD. Pinned = threshold ≥0.999.

| condition | F1 | MAP@R | thresholds | pinned | predicted-positive range |
|---|---|---|---|---|---|
| C0 | 0.2572 (single reference) | 0.1024 | 0.977894 | 0/1 | 22.3%–22.3% |
| C1 | 0.7533 ± 0.0073 | 0.6779 | 0.435805, 0.435738, 0.449245 | 0/3 | 14.2%–14.5% |
| C2 | 0.2542 ± 0.0108 | 0.0539 | 0.9999999, 0.9999982, 0.835525 | 2/3 | 48.1%–87.3% |
| C3 | 0.2728 ± 0.0136 | 0.0770 | 0.9999935, 0.9999979, 0.9999989 | 3/3 | 33.6%–42.7% |
| C4 | 0.4208 ± 0.0284 | 0.2632 | 0.496528, 0.541418, 0.497701 | 0/3 | 19.3%–23.4% |
| C5 | 0.7288 ± 0.0029 | 0.6433 | 0.394478, 0.404841, 0.431438 | 0/3 | 14.5%–14.8% |
| C6 | 0.4688 ± 0.0278 | 0.1627 | 0.825167, 0.795524, 0.834423 | 0/3 | 28.0%–34.9% |

### Generalisation (different pair regime; do not compare absolute F1 across corpora)

| condition | F1 seen | F1 unseen | Δ ± sample SD | pinned |
|---|---|---|---|---|
| C1 | 0.7570 | 0.7031 | +0.0539 ± 0.0222 | 0/3 |
| C2 | 0.6781 | 0.6674 | +0.0107 ± 0.0056 | 0/3 |
| C3 | 0.7004 | 0.7057 | -0.0053 ± 0.0218 | 3/3 |
| C4 | 0.6731 | 0.6687 | +0.0044 ± 0.0055 | 0/3 |
| C5 | 0.7606 | 0.7244 | +0.0362 ± 0.0219 | 0/3 |
| C6 | 0.7169 | 0.6865 | +0.0304 ± 0.0143 | 0/3 |
| C0 | 0.7118 | 0.6934 | — | — |

### Audit evidence (agent/AI-reviewed, not independently human-validated)

| condition | clone / n | point rate [Wilson 95%] | including unsure | method |
|---|---|---|---|---|
| C1 | — | ≈0–2% | — | historical estimate, not a new sample |
| C2 | 21/50 | 42% [29.4%, 55.8%] | 42% | historical agent review |
| C3 | 15/50 | 30% [19.1%, 43.8%] | 32% | historical agent review |
| C4 | 6/50 | 12% [5.6%, 23.8%] | 16% | P4-AI-AUDIT-2026-10-23 |
| C5 | 0/50 | 0% [0.0%, 7.1%] | 0% | P4-AI-AUDIT-2026-10-23 |
| C6 | 11/50 | 22% [12.8%, 35.2%] | 24% | P4-AI-AUDIT-2026-10-23 |

### Benchmark extension — six seeds 13–18 (not the primary table)

| condition | F1 ± sample SD | MAP@R | pinned |
|---|---|---|---|
| C1 | 0.7464 ± 0.0176 | 0.6674 | 0/6 |
| C2 | 0.2555 ± 0.0123 | 0.0474 | 4/6 |
| C3 | 0.2733 ± 0.0091 | 0.0772 | 6/6 |
| C4 | 0.4349 ± 0.0246 | 0.2760 | 0/6 |
| C5 | 0.7399 ± 0.0140 | 0.6629 | 0/6 |
| C6 | 0.4722 ± 0.0282 | 0.1826 | 0/6 |
