# IEEE conference manuscript

The paper uses the supplied June 2024 conference template and its unmodified
IEEEtran.cls, in conference mode: two columns, the class's default letter paper,
fonts and margins, IEEE author blocks, keywords, numbered citations, and table
captions above the tables. No conference-specific page limit or copyright line
was supplied; these must be checked against the target venue before submission.

Six contributor blocks are intentionally blank, with name and four affiliation
lines reserved in each. In paper.tex, find the six `Contributor` comments and
replace each complete `\phantom{...}` expression with the corresponding name,
department, institution, city/country, and email or ORCID. These blanks are not
an anonymization or submission-ready authorship declaration.

## Build

From this directory, with a standard TeX Live or MiKTeX installation:

```sh
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../output/pdf paper.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../output/pdf paper.tex
```

Keep the adjacent IEEEtran.cls and ../report/final/ table inputs. The bibliography
is inline, so BibTeX is not required. The TeX installation must provide the Times
font metrics and Type 1 fonts (TeX Live packages `times` and `courier`); do not substitute a
different document class or change margins to work around a missing font.

For this workstation, the missing font packages were installed into the isolated
tree at ../tmp/pdfs/ieee/texmf, not the system TeX tree. Set TEXMFHOME to that
tree's absolute path for local builds. The Overleaf-ready ZIP under output/latex
uses main.tex and a tables/ subdirectory, so it does not depend on this repo's
directory layout; a normal hosted TeX installation needs no local font override.

## Confusion-matrix figures

The manuscript uses regenerated primary and six-seed mean comparison boards
under ../report/figures/confusion_matrices. That folder also contains all 37
separate benchmark model/seed images, 7 primary means and 7 all-seed means.
There is also a five-column overview of all 37 runs. Matplotlib styling follows
`Image_Graphs/phase2_confusion_matrices(1).png`: condition-specific colours,
bold percentages, and smaller TP/FN/FP/TN tags and counts. The reference's
15/8/8 pt cell fonts are reduced by 50% to 7.5/4/4 pt. Every palette retains
the same 0-100% actual-row normalization; hue identifies the condition.
The all-run overview is 150 DPI, while separate images and mean boards default
to 300 DPI. Original reference files and the UMAP illustration are unchanged.
Rebuild them without loading models:

```sh
python -m scripts.export_confusion_matrices --snapshot SNAPSHOT --out-dir report/figures/confusion_matrices
```

Run that command from the repository root, using the pinned local snapshot.
Counts replay frozen NPZ scores/thresholds, not legacy JSON counts. Six legacy
count differences are documented in the figure README and numerical record;
F1 agreement alone did not prove count or precision/recall equality. Averages
are elementwise mean counts, not summed counts, threshold averages or ensembles.
Benchmark and s′ runs are not mixed. The figure pack provides full-resolution
images; all 37 individual plots need not be crowded into the conference paper.

This is a formatting revision, not a new experiment. The pinned results and
audit limitations are unchanged. The earlier Phase 5 release ZIP/receipt remains
an immutable record of the pre-IEEE version, not a certification of this revision.
