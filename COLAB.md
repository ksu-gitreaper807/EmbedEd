# Working on Phases 0–1 in Google Colab (collaboration guide)

The division of labour is deliberate:

| Piece | Lives in | Why |
|---|---|---|
| Specs, plan, tests, `embeded/`, `scripts/`, `report/measurements.md` | **this repo** (source of truth, reviewed via commits/PRs) | notebooks are terrible diff targets; code is where review matters |
| GPU execution | [`notebooks/Phase0_Phase1_Colab.ipynb`](notebooks/Phase0_Phase1_Colab.ipynb) — a thin **sequencer** that `git clone`s the repo and runs `python -m …` cells | notebook contains no logic to drift out of sync; re-running always uses the latest committed code |
| Big generated artifacts (fragment cache, `corpus_emb.npy`, triples, measurements) | **Hugging Face Hub** dataset repo (`EMBEDED_HF_REPO_ID`, optionally `EMBEDED_HF_SUBDIR`) | Colab disks are ephemeral; a dataset repo gives resumable checkpoints without a Drive mount |
| Model + dataset download cache | local Colab disk (`HF_HOME=/content/hf-home`) | easy to recreate from HF; no need to upload transient caches |

## One-time setup for resumable runs

In Colab's **Secrets** panel, add:

- `EMBEDED_HF_REPO_ID` — e.g. `your-name/embeded-artifacts`
- `HF_TOKEN` — token with write access to that repo if you want notebook pushes
- optional `EMBEDED_HF_SUBDIR` — e.g. `alice/phase01` for per-person isolation

Cell 1 reads those secrets into env vars, and the notebook then runs:

```bash
python -m scripts.hf_artifacts pull --if-configured
```

at startup, plus

```bash
python -m scripts.hf_artifacts push --if-configured --include-report
```

in the final cell.

## Sharing entry points (pick per audience)

1. **The one-URL opener (best for anyone with repo read access).** They click:
   `https://colab.research.google.com/github/ksu-gitreaper807/EmbedEd/blob/main/notebooks/Phase0_Phase1_Colab.ipynb`
   — Colab materializes the notebook, they attach a T4, and run. No local setup at all. Every
   step then pulls the *committed* code, so there is exactly one version of the pipeline.
2. **Notebook-file sharing.** Send the `.ipynb` itself however you like (Drive, email, Slack,
   GitHub). The execution logic still lives in the repo, and artifacts still checkpoint to HF.
3. **Real-time same-session collab.** Two editors on one open notebook share the live session:
   cursors, outputs, and *one* GPU runtime. Rules that keep it sane: one person "owns" the run
   at a time (say it in chat before hitting Run All — parallel re-runs corrupt the artifact
   checkpoint), gates (G0/G1 cells) are announced in channel, and nobody `pip install`s
   anything ad hoc (versions come from `embeded/settings.py::PINNED`; changes go through a commit).

## Get results back into the repo

`report/measurements.md` is written inside the git clone on the VM; the final notebook cell
uploads it to the configured HF dataset repo **and** offers a direct `measurements.zip` download.
Either route, a repo owner commits it. If in-VM commits are wanted, use the official Colab GitHub
integration — **never paste a GitHub token into a shared notebook**.

## Per-person artifact isolation (optional, for parallel Phase-2 runs)

`EMBEDED_ARTIFACTS` is the local path and `EMBEDED_HF_SUBDIR` is the remote path, so two people
can run concurrently without clobbering each other: point the local env var at a different local
folder and/or set the remote subdir to `.../embeded/<name>`. Mining is deterministic given the
code hash, so compare hashes, not trust.

## Free-tier constraints this plan assumes (from `IMPLEMENTATION_PLAN.md` §5)

- T4, 16 GB; ~90-min idle disconnect; hard session cap ~12 h; GPU quota ~12 h/day.
  The throughput test (Phase 0.5) exists precisely to size the training subset to this box.
- Long downloads (s′, model, dataset) happen once per local cache, then are cheap to rehydrate.
- If quota blocks you, the same notebook runs on Kaggle kernels (GPU, ~30 h/week): keep the
  clone/install cells, drop the Colab-secrets lines, and set the same env vars manually.

## Why not "just edit the notebook" for the pipeline itself?

Everything testable here is already unit-tested offline (`pytest -q embeded/tests`, 12+ tests,
no GPU/network needed — incl. the §7.2 adversarial BM25/overlap case and the hardness-gate
logic). The notebook adds only what *needs* a GPU or the real 8,063-fragment corpus (9,126-line
`data.jsonl`): the encoding pass, the gates on real numbers, and the smoke run. That split is
what makes the Colab side safe to share: the notebook can't silently fork the logic.
