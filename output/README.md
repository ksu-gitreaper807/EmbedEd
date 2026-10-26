# Generated deliverables

The actual files were organized on 26 October 2026; this directory retains
relative compatibility symlinks for historical receipts and existing commands.
All 33 original output files retain their exact SHA-256 hashes. Nothing was
deleted or regenerated during relocation.

| Previous path | Canonical location |
|---|---|
| `output/pdf/` | [`paper/compiled/`](../paper/compiled/) |
| `output/latex/` | [`paper/latex/`](../paper/latex/) |
| `output/figures/` | [`report/figures/`](../report/figures/) |
| `output/phase5/` | [`artifacts/releases/phase5/`](../artifacts/releases/phase5/) |
| `output/phase6/` | [`artifacts/releases/phase6/`](../artifacts/releases/phase6/) |

Final tables remain in [`report/final/`](../report/final/). The editable paper
is [`paper/paper.tex`](../paper/paper.tex); source notebooks remain under
[`notebooks/`](../notebooks/). Phase 6's runnable source ZIP does not contain
model weights or credentials; use its notebook to download pinned public assets.

The Phase 5/6 receipts, source manifests and ZIPs are immutable historical
snapshots. Their publication-status statements reflect the dates on those
records; organizing or committing them is not a new scientific validation.
The owner authorized only a GitHub `main` push, not an HF upload, paper
submission or public website deployment.

`paper/compiled/paper.log` is retained as evidence referenced by the original
formatting record. Other LaTeX intermediates stay local and are ignored by Git.
The relocation manifest includes them so preservation can also be checked on
the original workstation; a fresh clone need not contain ignored intermediates.

Run `python -m scripts.verify_output_layout` to verify the committed deliverables,
compatibility links, historical receipts and ZIP payload manifests. Use
`--include-build-scratch` on the original workstation to check all 33 files.
