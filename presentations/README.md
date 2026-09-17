# Presentation package

A 36-slide **EL Phase-I project presentation** (`.pptx`) covering the code-similarity-detection
project: *Domain-Specific Contrastive Embeddings for Code Similarity Detection — Comparing
Negative-Sampling Strategies*.

---

## Structure

```text
presentations/
├── presentation.pptx                  ← the deliverable (36 slides, 16:9)
├── README.md                          ← this file
├── conflict.md                        ← discrepancies vs current project context
├── assets/
│   ├── hypotheses.png                 ← generated: H1 / H0a / H0b schematic curves
│   └── closing-embedding-space.png    ← generated: image-only closing slide
├── build/
│   └── build_deck.py                  ← the generator script
└── notes/
    └── current-context-baseline.md    ← working note: authoritative facts used for conflict diffing
```

---

## Source files used

Both are read from the repository path:

| File | Role |
|---|---|
| `distilled-project/presentation/project-presentation-content.md` | **Content source** — the authoritative material the deck communicates |
| `distilled-project/presentation/EL-phase-1.md` | **Format / style reference** — structure, slide ordering, visual hierarchy, density and presentation conventions |

The deck follows the content source for *what* it says and the format reference for *how* it is
laid out.

### Design language reproduced from `EL-phase-1.md`

The reference is a 14-slide EL Phase-I academic template. The following conventions were carried
over and the structure expanded from 14 to 36 slides to accommodate the real content:

- Numbered section narrative — Introduction → Literature Review → Problem Definition →
  Objectives → Methodology → Proposed Outcomes → Timeline → Conclusion → References
- A dedicated **outline slide** listing all nine sections
- **Bold lead-in labels** on bullets ("Design: …", "Context 1 — …"), matching the reference's
  `**Heading:** text` convention
- **Section divider slides** with a large section number
- A **dedicated flowchart slide** for the research process
- A **phase-based timeline slide** (Initiation / Planning / Execution / Closure)
- An **IEEE-style references slide**
- An **"Expected Outcome by the End of First Year"** slide (patent / journal / conference /
  research proposal)
- An **image-only closing slide**
- Title slide carrying **title + theme**, per the reference

In addition, the deck applies the explanatory progression required by the brief
(problem → why existing approaches fall short → our idea → how it works → worked example →
implementation → evaluation), with pipelines and comparisons drawn as diagrams rather than prose.

---

## Slide structure

| # | Slide | Section |
|---|---|---|
| 1 | Title — title + theme | — |
| 2 | Presentation Outline | — |
| 3–5 | Introduction — overview, importance | 01 |
| 6–11 | Literature Review — key studies, trends, framework, relevance, exists-vs-contribution | 02 |
| 12–14 | Problem Definition — statement, significance | 03 |
| 15–17 | Objectives — main + four specific, expected contributions | 04 |
| 18–25 | Methodology — design, negatives, false-negative trap, data, flowchart, analysis, tools | 05 |
| 26–29 | Proposed Outcomes — expected results, impact, future directions | 06 |
| 30–31 | Timeline of the Project | 07 |
| 32–33 | Conclusion | 08 |
| 34 | References | 09 |
| 35 | Expected Outcome by End of First Year (EL deliverable) | EL |
| 36 | Closing — image only | — |

---

## Generated assets

| Asset | How it is produced |
|---|---|
| `assets/hypotheses.png` | **Regenerable.** Matplotlib, inside `build_deck.py` (`make_hypotheses_figure`). Schematic only — the axes carry no measured values, because no results exist yet. |
| `assets/closing-embedding-space.png` | **Not script-regenerable.** AI-generated artwork for the image-only closing slide. If deleted, the builder falls back to a plain navy slide. |

---

## Regenerating / editing

The deck is generated, so edit the script rather than the `.pptx`.

```bash
python -m venv /tmp/pptxenv
/tmp/pptxenv/bin/pip install python-pptx matplotlib
/tmp/pptxenv/bin/python presentations/build/build_deck.py
```

This overwrites `presentations/presentation.pptx`. Everything is inside the script:

- **Palette, typography, canvas and margins** — constants at the top
- **Reusable primitives** — `card`, `callout`, `table`, `chip`, `stat`, `code_panel`,
  `bullet`, `content_slide`, `divider`
- **Auto-sizing** — `card()` and `callout()` grow to fit their text, so content edits do not
  spill outside a coloured panel
- **Slide content** — one block per slide, in narrative order, near the bottom

To add a slide, copy an existing block and call `content_slide(...)`. To reorder, move the block.

---

## Conflicts

**Seven conflicts** were found between the supplied material and the current project context, and
are recorded in [`conflict.md`](conflict.md). In all cases the current-context version was retained
and the presentation follows it. The main ones:

- Dataset scale — "8+ million clone pairs" describes BigCloneBench, not the CodeXGLUE subset
  (9,134 fragments) actually loaded; the deck states both
- Primary metric — F1 primary / MAP@R secondary, not MAP@R alone
- Expected direction — the inverted-U (hardness helps, then hurts) is the literature-supported
  prediction, not a straight "harder is better"
- Two novelty claims softened — the strategy comparison is established for text retrieval, and
  generalisation to unseen functionality is established by Kitsios et al. (ASE 2025), so neither
  is presented as new

---

## Assumptions and limitations

- **No results exist yet.** This is a Phase-I proposal deck. Every number on the results slide is
  either a literature figure or explicitly schematic; the hypotheses chart has no measured values
  on its axes.
- **Current context is authoritative** for anything not covered by the two source files. Material
  absent from `project-presentation-content.md` but present in the current context — the
  false-negative trap, the clone-exclusion verification step, the CodeXGLUE subset figures, the
  Kitsios generalisation baseline, the GraphCodeBERT token-only caveat — was added, because
  omitting it would misrepresent the project.
- **The timeline slide merges two sources:** the generic 13-week EL phase structure from
  `EL-phase-1.md` and the project's 4-week research execution plan from the current context.
- **Reference list extends the source.** References 1–6 are the content file's; 7–10 (Kitsios,
  AugSBERT, STAR/ADORE, Krinke) were added because the deck cites them.
- **Fonts** are Calibri and Consolas. If a viewer lacks them, Windows/PowerPoint substitutes
  metrically similar faces; layout was built with ~10% slack to absorb this.
- **Layout was verified programmatically** — no blank slides, no out-of-canvas shapes, no
  placeholder text, and no text estimated to collide with a following element. It was **not**
  visually proofed in PowerPoint, as no renderer was available in this environment; a quick
  eyeball pass before presenting is advisable.
