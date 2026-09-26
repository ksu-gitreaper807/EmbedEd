"""Sync pipeline artifacts to and from a Hugging Face dataset repo.

This replaces the old Colab+Drive pattern with an optional HF Hub checkpoint:

    # download the latest checkpoint into EMBEDED_ARTIFACTS / EMBEDED_REPORT
    python -m scripts.hf_artifacts pull --if-configured

    # upload the current checkpoint (creates the dataset repo if needed)
    python -m scripts.hf_artifacts push --if-configured --include-report

Configuration comes from env vars so the rest of the pipeline stays unchanged:

- EMBEDED_HF_REPO_ID   required for push/pull unless --if-configured is used
- EMBEDED_HF_REPO_TYPE defaults to "dataset"
- EMBEDED_HF_REVISION  defaults to "main"
- EMBEDED_HF_SUBDIR    optional subdirectory inside the repo (for per-person isolation)
- EMBEDED_HF_PRIVATE   "1"/"true" => create a private repo on first push
- HF_TOKEN / HUGGINGFACE_HUB_TOKEN are passed through to huggingface_hub

The synced tree inside the HF repo is:

    <subdir>/artifacts/...            # EMBEDED_ARTIFACTS contents
    <subdir>/report/measurements.md   # optional, from EMBEDED_REPORT

`HF_HOME` is *not* synced: model/dataset cache can be re-hydrated from the Hub,
while the generated experiment artifacts are the bits that must persist.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from embeded import settings as S

TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RepoConfig:
    repo_id: str
    repo_type: str = "dataset"
    revision: str = "main"
    subdir: str = ""
    private: bool = False
    token: str | None = None


def load_repo_config(*, required: bool = True, repo_id: str | None = None,
                     repo_type: str | None = None, revision: str | None = None,
                     subdir: str | None = None) -> RepoConfig | None:
    rid = (repo_id or os.environ.get("EMBEDED_HF_REPO_ID") or "").strip()
    if not rid:
        if required:
            raise RuntimeError(
                "Hugging Face artifact repo not configured: set EMBEDED_HF_REPO_ID "
                "(for example 'your-name/embeded-artifacts')"
            )
        return None
    return RepoConfig(
        repo_id=rid,
        repo_type=(repo_type or os.environ.get("EMBEDED_HF_REPO_TYPE") or "dataset").strip() or "dataset",
        revision=(revision or os.environ.get("EMBEDED_HF_REVISION") or "main").strip() or "main",
        subdir=((subdir or os.environ.get("EMBEDED_HF_SUBDIR") or "").strip().strip("/")),
        private=(os.environ.get("EMBEDED_HF_PRIVATE", "").strip().lower() in TRUTHY),
        token=(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None),
    )


def allow_patterns(subdir: str = "") -> list[str]:
    prefix = f"{subdir.strip('/')}/" if subdir else ""
    return [
        f"{prefix}artifacts/*",
        f"{prefix}artifacts/**",
        f"{prefix}report/*",
        f"{prefix}report/**",
    ]


def stage_upload_tree(stage_root: Path, artifacts_root: Path, report_md: Path,
                      *, include_report: bool = False) -> list[str]:
    """Copy the local artifacts/report into a clean staging tree.

    Returns the staged relative file paths for logging/tests.
    """
    stage_root.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []

    if artifacts_root.exists():
        shutil.copytree(artifacts_root, stage_root / "artifacts", dirs_exist_ok=True)
        staged.extend(sorted(
            p.relative_to(stage_root).as_posix()
            for p in (stage_root / "artifacts").rglob("*")
            if p.is_file()
        ))

    if include_report and report_md.exists():
        dst = stage_root / "report" / "measurements.md"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report_md, dst)
        staged.append(dst.relative_to(stage_root).as_posix())

    meta = {
        "version": S.VERSION,
        "synced_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "staged_files": staged,
    }
    (stage_root / "sync_manifest.json").write_text(json.dumps(meta, indent=2))
    staged.append("sync_manifest.json")
    return staged


def restore_download_tree(snapshot_root: Path, artifacts_root: Path, report_md: Path) -> list[str]:
    """Copy a downloaded HF snapshot back into the local pipeline locations."""
    restored: list[str] = []

    art_src = snapshot_root / "artifacts"
    if art_src.exists():
        artifacts_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(art_src, artifacts_root, dirs_exist_ok=True)
        restored.extend(sorted(
            f"artifacts/{p.relative_to(art_src).as_posix()}"
            for p in art_src.rglob("*")
            if p.is_file()
        ))

    rep_src = snapshot_root / "report" / "measurements.md"
    if rep_src.exists():
        report_md.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rep_src, report_md)
        restored.append("report/measurements.md")

    return restored


def pull_from_hf(cfg: RepoConfig, artifacts_root: Path, report_md: Path) -> list[str]:
    from huggingface_hub import HfApi, snapshot_download
    try:  # huggingface_hub moved the exception between releases
        from huggingface_hub.errors import RepositoryNotFoundError
    except Exception:  # pragma: no cover - only hit on older hub releases
        from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi(token=cfg.token)
    try:
        files = api.list_repo_files(repo_id=cfg.repo_id, repo_type=cfg.repo_type, revision=cfg.revision)
    except RepositoryNotFoundError:
        print(f"[hf] repo {cfg.repo_type}:{cfg.repo_id}@{cfg.revision} does not exist yet — starting fresh")
        return []

    prefix = f"{cfg.subdir}/" if cfg.subdir else ""
    # quick empty-check to avoid snapshot_download noise on a freshly created but empty repo
    if not any(
        f.startswith(prefix + "artifacts/") or f.startswith(prefix + "report/")
        for f in files
    ):
        print(f"[hf] repo {cfg.repo_type}:{cfg.repo_id}@{cfg.revision} has no synced artifacts yet")
        return []

    snap = Path(snapshot_download(
        repo_id=cfg.repo_id,
        repo_type=cfg.repo_type,
        revision=cfg.revision,
        allow_patterns=allow_patterns(cfg.subdir),
        token=cfg.token,
    ))
    root = snap / cfg.subdir if cfg.subdir else snap
    restored = restore_download_tree(root, artifacts_root, report_md)
    loc = f"/{cfg.subdir}" if cfg.subdir else ""
    print(f"[hf] pulled {len(restored)} files from {cfg.repo_type}:{cfg.repo_id}@{cfg.revision}{loc}")
    return restored


def push_to_hf(cfg: RepoConfig, artifacts_root: Path, report_md: Path,
               *, include_report: bool = False) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi(token=cfg.token)
    api.create_repo(repo_id=cfg.repo_id, repo_type=cfg.repo_type,
                    private=cfg.private, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="embeded-hf-stage-") as td:
        stage_root = Path(td)
        staged = stage_upload_tree(stage_root, artifacts_root, report_md,
                                   include_report=include_report)
        data_files = [p for p in staged if p != "sync_manifest.json"]
        if not data_files:
            print("[hf] nothing to upload (no artifacts dir, and report missing/not requested)")
            return []
        api.upload_folder(
            repo_id=cfg.repo_id,
            repo_type=cfg.repo_type,
            folder_path=str(stage_root),
            path_in_repo=cfg.subdir,
            revision=cfg.revision,
            ignore_patterns=["*.part"],
            commit_message=f"sync EmbedEd artifacts ({S.VERSION})",
        )
    loc = f"/{cfg.subdir}" if cfg.subdir else ""
    print(f"[hf] pushed {len(staged)} files to {cfg.repo_type}:{cfg.repo_id}@{cfg.revision}{loc}")
    return staged


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("pull", "push"))
    ap.add_argument("--if-configured", action="store_true",
                    help="exit 0 with a note instead of failing when EMBEDED_HF_REPO_ID is unset")
    ap.add_argument("--include-report", action="store_true",
                    help="when pushing, upload EMBEDED_REPORT as report/measurements.md too")
    ap.add_argument("--repo-id")
    ap.add_argument("--repo-type")
    ap.add_argument("--revision")
    ap.add_argument("--subdir")
    a = ap.parse_args(argv)

    cfg = load_repo_config(required=not a.if_configured, repo_id=a.repo_id,
                           repo_type=a.repo_type, revision=a.revision,
                           subdir=a.subdir)
    if cfg is None:
        print("[hf] EMBEDED_HF_REPO_ID not set — skipping sync")
        return

    if a.action == "pull":
        pull_from_hf(cfg, S.ARTIFACTS, S.REPORT_MD)
    else:
        push_to_hf(cfg, S.ARTIFACTS, S.REPORT_MD, include_report=a.include_report)


if __name__ == "__main__":
    main()
