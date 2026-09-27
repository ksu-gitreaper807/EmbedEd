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
- optional `GITHUB_TOKEN` — a fine-grained PAT, only if you want to push commits *from* the VM
  (see § "Get results back into the repo")

Cell 1 reads those secrets into env vars, and the notebook then runs:

```bash
python -m scripts.hf_artifacts pull --if-configured
```

at startup, plus

```bash
python -m scripts.hf_artifacts push --if-configured --include-report
```

in the final cell.

## Resume protocol (what a disconnect costs, and why it is not much)

A Colab VM can vanish at any time (~90 min idle, ~12 h hard cap). The environment is made
recoverable by treating it as three layers, each restored by a different mechanism — none of
them a Drive mount:

| layer | where it lives between sessions | restored by | cost |
|---|---|---|---|
| code + frozen constants (`settings.py`) | GitHub | notebook cell 0 | seconds |
| pinned Python packages | nowhere (Colab image) | notebook cell 2 (`pip`) | ~1 min |
| model + dataset download cache (`HF_HOME`) | the Hub itself | re-downloaded by the first stage that needs it | ~1 min (GraphCodeBERT 500 MB + 1.3 GB of parquet at >100 MB/s) |
| **generated artifacts** (`fragments.jsonl`, `pairs_*.tsv`, `corpus_emb.npy`, `triples_*.jsonl`, `mining_summary.json`, `report/measurements.md`) | **your HF dataset repo** (`EMBEDED_HF_REPO_ID`) | notebook cell 2 (`hf_artifacts pull`) | seconds |

The fourth row is the one that matters: it holds the only things that are expensive to make
(the 4-min GPU encode, the 15–30-min BM25 mining, and every measured number). The GPU-heavy
notebook cells push a checkpoint the moment they finish; the final cell pushes everything.

**What is *not* pushed:** `sprime/*` and any `*.zip` — the s′ replication package is 138 MB
plus ~2,000 extracted files, it is re-downloadable by DOI (`settings.SPRIME_DOI`) and
MD5-verified by `scripts/fetch_sprime.py`, so checkpointing it turned every push into a 481 MB /
2,149-file upload that `huggingface_hub` warns about. Override with
`EMBEDED_HF_EXCLUDE="glob1,glob2"` (paths relative to the artifacts dir; `""` pushes
everything). Exclusions only stop *future* uploads — files already in the repo stay there
until you delete them in the Hub UI.

**Every stage is artifact-gated.** `prepare_data`, `semantic_index` and `negatives` look for
their own outputs first and, if a complete set for the current `settings.VERSION` (and the
same model / `MAX_LEN` / `k` / anchor count) is present, print `[cache] … skipping` and exit.
So the recovery recipe is: *run every cell again from the top*. Only unfinished stages do
work. `--force` re-runs a stage deliberately; bumping `VERSION` invalidates everything
deliberately (that is what it is for).

Model and dataset weights are **not** checkpointed on purpose: they are already on the Hub,
and re-hydrating `HF_HOME` is cheaper than moving 1.8 GB through Drive's FUSE layer (which
also mangles the symlinks the HF cache relies on).

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
Pick whichever route you can actually finish:

1. **A repo owner commits it** from the HF copy or the `measurements.zip` download. No token on
   the VM at all; costs one round trip.
2. **The official Colab GitHub integration** (`Tools ▸ Command palette ▸ GitHub`). Handles auth
   for you, but it commits to whatever branch the integration is pointed at — check that first.
3. **A personal access token, via `scripts.colab_git`** (notebook §8). Use this when you want the
   VM's own commit, with its own message, on the branch you are working on.

### Route 3 in detail: a PAT from a Colab VM

`git push` from a fresh VM exits **128** — there is no credential helper configured and no
terminal for git to prompt on. Fetching needs no token at all (the repo is public; only pushing
does), so a failing `git pull` is usually *not* an auth problem: run `diagnose` before touching
credentials, because exit 128 also comes from a dirty worktree, a missing commit identity, or an
interrupted rebase, and each has a different fix.

Create a **fine-grained** PAT: owner = the account with write access, repository =
`ksu-gitreaper807/EmbedEd` only, permission **Contents: Read and write**, expiry as short as the
run allows. Then:

| Rule | Why |
|---|---|
| The token goes in Colab's **Secrets** panel as `GITHUB_TOKEN` | notebook source and outputs are saved to Drive and get pasted around; Secrets are not |
| Never `os.environ['GITHUB_TOKEN'] = "ghp_…"` in a cell | that literal lands in the `.ipynb` |
| Never `git remote set-url origin https://<token>@github.com/…` | the token then sits in `.git/config`, where `git remote -v` and any traceback print it |
| Never pass the token on a command line | `argv` is world-readable in `/proc` |
| Revoke it when the run is done | a Colab VM is ephemeral but a token is not |

`scripts.colab_git` implements those rules, so use it rather than hand-rolling the handshake:

```bash
python -m scripts.colab_git diagnose                 # why did git fail? read-only, nothing secret
python -m scripts.colab_git push                     # store token → pull --rebase → push
python -m scripts.colab_git push --commit "report: audit" --paths report
```

It reads the Secret, writes it to a mode-`0600` `~/.git-credentials`, points git's `store` helper
at it (with a local empty `credential.helper` first, so a helper inherited from the image's
system git config cannot answer first), sets a commit identity from the token's own account if
the clone has none, commits only the paths you name — `audit_key*` is refused outright, so the
blind audit key can never be pushed — rebases onto the remote, pushes, and redacts every byte of
git output before it is printed (known token, `ghp_`/`github_pat_` shapes, `password=` lines, and
any `://user:pass@` URL).

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
