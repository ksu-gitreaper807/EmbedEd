# Working on Phases 0–1 in Google Colab (collaboration guide)

The division of labour is deliberate:

| Piece | Lives in | Why |
|---|---|---|
| Specs, plan, tests, `code/`, `scripts/`, `report/measurements.md` | **this repo** (source of truth, reviewed via commits/PRs) | Colab notebooks are terrible diff targets; code is where review matters |
| GPU execution | [`notebooks/Phase0_Phase1_Colab.ipynb`](notebooks/Phase0_Phase1_Colab.ipynb) — a thin **sequencer** that `git clone`s the repo and runs `python -m …` cells | notebook contains no logic to drift out of sync; re-running always uses the latest committed code |
| Big artifacts (fragment cache, `corpus_emb.npy`, triples, HF caches) | **Drive** (`/content/drive/MyDrive/embeded/`, set via `EMBEDED_ARTIFACTS`/`HF_HOME` in notebook cell 1) | Colab disks are ephemeral; Drive mount makes reconnects ~free |

## Sharing entry points (pick per audience)

1. **The one-URL opener (best for anyone with repo read access).** They click:
   `https://colab.research.google.com/github/ksu-gitreaper807/EmbedEd/blob/arena/01a0c914-embeded/notebooks/Phase0_Phase1_Colab.ipynb`
   — Colab materializes the notebook, they attach a T4, and run. No local setup at all. Every
   step then pulls the *committed* code, so there is exactly one version of the pipeline.
2. **Classic Drive share.** Upload the `.ipynb` to Drive → Share → *Viewer* (run-only demos),
   *Commenter* (review sessions — feedback lands on cells), *Editor* (co-runners). This works
   even for people without GitHub access. The repo still holds the code they execute.
3. **Real-time same-session collab.** Two editors on one open notebook share the live session:
   cursors, outputs, and *one* GPU runtime. Rules that keep it sane: one person "owns" the run
   at a time (say it in chat before hitting Run All — parallel re-runs corrupt the Drive
   artifact cache), gates (G0/G1 cells) are announced in channel, and nobody `pip install`s
   anything ad hoc (versions come from `code/settings.py::PINNED`; changes go through a commit).

## Get results back into the repo

`report/measurements.md` is written inside the git clone on the VM; the final notebook cell
copies it to Drive **and** offers a direct `measurements.zip` download. Either route, a repo
owner (or a Colab session with the [Colab GitHub app](https://github.com/marketplace/google-colab)
installed on the repo) commits it. **Never paste a GitHub token into a shared notebook** —
if in-VM commits are wanted, use the official `github` Drive mount / the app, not PATs.

## Per-person artifact isolation (optional, for parallel Phase-2 runs)

`EMBEDED_ARTIFACTS` is an env var, so two people can run concurrently without clobbering:
set it to `.../embeded/<name>/artifacts`. Mining is deterministic given the code hash, so
duplicate caches cost nothing; compare hashes, not trust.

## Free-tier constraints this plan assumes (from `IMPLEMENTATION_PLAN.md` §5)

- T4, 16 GB; ~90-min idle disconnect; hard session cap ~12 h; GPU quota ~12 h/day.
  The throughput test (Phase 0.5) exists precisely to size the training subset to this box.
- Long downloads (`s′`, model, dataset) happen once per cache, then live on Drive.
- If quota blocks you, the same notebook runs on Kaggle kernels (GPU, ~30 h/week): keep the
  clone/install cells, drop `drive.mount`, use the Kaggle "attach dataset" for artifacts or a
  kaggle API key on Drive.

## Why not "just edit the notebook" for the pipeline itself?

Everything testable here is already unit-tested offline (`pytest -q code/tests`, 12 tests,
no GPU/network needed — incl. the §7.2 adversarial BM25/overlap case and the hardness-gate
logic). The notebook adds only what *needs* a GPU or the real 9,134-fragment download: the
encoding pass, the gates on real numbers, and the smoke run. That split is what makes the
Colab side safe to share: the notebook can't silently fork the logic.
