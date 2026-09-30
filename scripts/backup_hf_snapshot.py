#!/usr/bin/env python3
"""Download a full snapshot of the EmbedEd HF dataset repo and verify it.

Usage:
    pip install "huggingface_hub>=0.23"
    python hf_backup.py                          # Kusshal/Embed -> ~/EmbedEd-hf-backup
    python hf_backup.py --dest /path/to/drive    # e.g. an external drive
    python hf_backup.py --tar                    # also write one .tar file (uncompressed)

Resumable: re-run it any time; only missing/changed files are fetched.
"""
import argparse, json, time
from pathlib import Path

EXPECTED_EVAL_FILES = {"eval_metrics.json", "eval_config.json", "predictions.npz"}
EXPECTED_TRAIN_FILES = {"checkpoint.pt", "run_config.json", "metrics.json", "train_log.jsonl"}


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="Kusshal/Embed")
    ap.add_argument("--dest", default="~/EmbedEd-hf-backup")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="needed only if the repo is private (or pass token= in code)")
    ap.add_argument("--tar", action="store_true",
                    help="also write a single uncompressed .tar next to the folder")
    args = ap.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=args.token)
    info = api.repo_info(args.repo, repo_type="dataset")
    print(f"repo dataset:{args.repo} @ {info.sha[:12]}  last modified {info.last_modified}")

    dest = Path(args.dest).expanduser()
    print(f"downloading full snapshot to {dest} (resumes if interrupted)...")
    snapshot_download(repo_id=args.repo, repo_type="dataset", token=args.token,
                      local_dir=str(dest))
    print("download complete")

    # ---- verify against the repo's own file list ------------------------------
    expected = set(api.list_repo_files(args.repo, repo_type="dataset"))
    have = {str(p.relative_to(dest)) for p in dest.rglob("*")
            if p.is_file() and ".cache" not in p.parts}
    missing = sorted(expected - have)
    print(f"files: {len(have)} on disk / {len(expected)} in repo")
    if missing:
        print(f"MISSING {len(missing)} file(s):")
        for f in missing[:20]:
            print("  ", f)
        raise SystemExit("incomplete download - re-run this script to resume")

    total = sum((dest / f).stat().st_size for f in expected)
    print(f"total size: {human(total)}")

    # ---- per-run completeness -------------------------------------------------
    runs = sorted({f.split("/")[2] for f in expected
                   if f.startswith("artifacts/runs/") and f.count("/") >= 3})
    incomplete = []
    for r in runs:
        names = {Path(f).name for f in expected if f.startswith(f"artifacts/runs/{r}/")}
        need = set(EXPECTED_TRAIN_FILES)
        if names & EXPECTED_EVAL_FILES:          # an evaluated run: expect the full set
            need |= EXPECTED_EVAL_FILES
        if not need <= names:
            incomplete.append((r, sorted(need - names)))
    evaluated = [r for r in runs
                 if any(f.startswith(f"artifacts/runs/{r}/eval_metrics.json") for f in expected)]
    print(f"run directories: {len(runs)} | with evaluations: {len(evaluated)}")
    for r in incomplete:
        print(f"  INCOMPLETE {r}: missing {r[1]}")
    print("  " + ", ".join(f"{r}" for r in runs))

    # ---- manifest --------------------------------------------------------------
    manifest = {
        "repo": args.repo, "revision": info.sha,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_files": len(expected), "total_bytes": total,
        "runs": runs, "evaluated_runs": evaluated, "incomplete": incomplete,
    }
    mpath = dest / f"backup-manifest-{time.strftime('%Y%m%d-%H%M%S')}.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"manifest written: {mpath}")
    print("OK" if not incomplete and not missing else "DONE WITH WARNINGS (see above)")

    if args.tar:
        import tarfile
        tar_path = str(dest).rstrip("/") + ".tar"
        print(f"writing {tar_path} (uncompressed - model weights do not shrink)...")
        with tarfile.open(tar_path, "w") as tar:
            tar.add(dest, arcname=Path(dest).name)
        print(f"written: {human(Path(tar_path).stat().st_size)}")


import os  # noqa: E402  (used in argparse default above)
if __name__ == "__main__":
    main()
