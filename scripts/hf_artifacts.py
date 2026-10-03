"""Sync pipeline artifacts to and from a Hugging Face dataset repo.

This replaces the old Colab+Drive pattern with an optional HF Hub checkpoint:

    # download the latest checkpoint into EMBEDED_ARTIFACTS / EMBEDED_REPORT
    python -m scripts.hf_artifacts pull --if-configured

    # same, but skip the per-run model checkpoints (*.pt) — a few hundred MB
    # instead of ~5 GB; enough for eval_metrics/predictions.npz/train_log, i.e.
    # the results table, the confusion-matrix board and every verification block
    python -m scripts.hf_artifacts pull --if-configured --light

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

Transient network failures (dropped connections, premature response ends,
429/5xx) are retried with backoff before the sync fails; auth and config
errors surface immediately.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from embeded import settings as S

TRUTHY = {"1", "true", "yes", "on"}

# Never checkpoint what the pipeline can re-download and verify by itself. The
# s' replication package is 138 MB of zip plus ~2,000 extracted files; it has a
# DOI (settings.SPRIME_DOI) and `scripts/fetch_sprime.py` re-fetches it with an
# MD5 check, so pushing it only made every checkpoint a 481 MB / 2,149-file
# upload that huggingface_hub warns about. Override with EMBEDED_HF_EXCLUDE
# (comma-separated globs, relative to the artifacts dir; "" = exclude nothing).
DEFAULT_EXCLUDE = ("sprime/*", "*.zip")


def exclude_patterns() -> tuple[str, ...]:
    raw = os.environ.get("EMBEDED_HF_EXCLUDE")
    if raw is None:
        return DEFAULT_EXCLUDE
    return tuple(p.strip() for p in raw.split(",") if p.strip())


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


# Pull sections: named, composable subsets of the repo. A section is a virtual
# view (allow_patterns on the download), NOT a repo restructure -- the hub keeps
# one layout (artifacts/, report/) and VMs download only the slices they need,
# so a fresh Colab VM stops paying for ~4.5 GB of checkpoints or ~100 MB of npz
# it will never read. No --section at all = the full pull (unchanged).
SECTIONS: dict[str, list[str]] = {
    "mining": ["artifacts/fragments.jsonl", "artifacts/mining_summary.json",
               "artifacts/triples_C*.jsonl", "artifacts/corpus_emb.npy",
               "artifacts/corpus_emb.meta.json", "artifacts/codexglue_data.jsonl"],
    "runs-meta": ["artifacts/runs/*/eval_metrics.json", "artifacts/runs/*/metrics.json",
                  "artifacts/runs/*/run_config.json", "artifacts/runs/*/train_log.jsonl"],
    "runs": ["artifacts/runs/*", "artifacts/runs/**"],
    "audit": ["artifacts/audit/*", "artifacts/audit/**"],
    "sprime": ["artifacts/sprime_*", "artifacts/sprime/*", "artifacts/sprime/**"],
    "report": ["report/*", "report/**"],
}


def allow_patterns(subdir: str = "", sections: list[str] | None = None,
                   include: list[str] | None = None) -> list[str]:
    """The download allow-list. No sections = the full repo view (unchanged).
    With sections, the union of the named sections (+ any raw `include` globs)
    -- everything NOT matched stays on the hub. Patterns are fnmatch-style and
    fnmatch's `*` crosses '/', so mid-path stars are fine."""
    prefix = f"{subdir.strip('/')}/" if subdir else ""
    if not sections and not include:
        return [
            f"{prefix}artifacts/*",
            f"{prefix}artifacts/**",
            f"{prefix}report/*",
            f"{prefix}report/**",
        ]
    pats: list[str] = []
    for name in (sections or []):
        if name not in SECTIONS:
            raise SystemExit(f"unknown pull section {name!r} — known: {sorted(SECTIONS)}")
        pats.extend(SECTIONS[name])
    pats.extend(include or [])
    return [f"{prefix}{pat}" if not pat.startswith(prefix) else pat for pat in pats]


def _is_excluded(rel_posix: str, patterns) -> bool:
    """fnmatch on the path relative to the artifacts dir ('sprime/x/y.pkl')."""
    return any(fnmatch.fnmatch(rel_posix, pat) for pat in patterns)


def stage_upload_tree(stage_root: Path, artifacts_root: Path, report_md: Path,
                      *, include_report: bool = False,
                      exclude: tuple[str, ...] | None = None) -> list[str]:
    """Copy the local artifacts/report into a clean staging tree, skipping
    `exclude` (default: `DEFAULT_EXCLUDE`).

    Returns the staged relative file paths for logging/tests.
    """
    pats = DEFAULT_EXCLUDE if exclude is None else exclude
    stage_root.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    skipped: list[str] = []

    if artifacts_root.exists():
        dest_root = stage_root / "artifacts"
        for src in sorted(artifacts_root.rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(artifacts_root).as_posix()
            if _is_excluded(rel, pats):
                skipped.append(rel)
                continue
            dst = dest_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            staged.append(f"artifacts/{rel}")
        if skipped:
            top = sorted({s.split("/")[0] for s in skipped})
            print(f"[hf] not checkpointing {len(skipped)} file(s) under "
                  f"{', '.join(top)} — re-downloadable ({', '.join(pats)}); "
                  f"override with EMBEDED_HF_EXCLUDE")

    if include_report and report_md.exists():
        dst = stage_root / "report" / "measurements.md"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report_md, dst)
        staged.append(dst.relative_to(stage_root).as_posix())

    meta = {
        "version": S.VERSION,
        "synced_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "staged_files": staged,
        "excluded_patterns": list(pats),
        "excluded_files": len(skipped),
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


def pull_from_hf(cfg: RepoConfig, artifacts_root: Path, report_md: Path,
                 *, light: bool = False, sections: list[str] | None = None,
                 include: list[str] | None = None) -> list[str]:
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
    if sections:
        import fnmatch as _fn
        pats = allow_patterns(cfg.subdir, sections)
        have = [name for name, spats in SECTIONS.items()
                if any(_fn.fnmatch(f, f"{prefix}{sp}") for sp in spats for f in files)]
        skipped = [s for s in have if s not in sections]
        if skipped:
            print(f"[hf] PARTIAL pull (sections: {','.join(sections)}) — repo also has "
                  f"{','.join(sorted(skipped))}, NOT downloaded; a later 'push' from this "
                  "VM cannot include them (pull-before-push still applies to what you edit)")
    snap = Path(snapshot_download(
        repo_id=cfg.repo_id,
        repo_type=cfg.repo_type,
        revision=cfg.revision,
        allow_patterns=allow_patterns(cfg.subdir, sections, include),
        ignore_patterns=["*.pt"] if light else None,
        token=cfg.token,
    ))
    root = snap / cfg.subdir if cfg.subdir else snap
    restored = restore_download_tree(root, artifacts_root, report_md)
    loc = f"/{cfg.subdir}" if cfg.subdir else ""
    note = " (light — checkpoints *.pt skipped)" if light else ""
    print(f"[hf] pulled {len(restored)} files from {cfg.repo_type}:{cfg.repo_id}@{cfg.revision}{loc}{note}")
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


# --- transient network failures ---------------------------------------------
# Observed on Colab (2026-09): a pull died mid-response with ChunkedEncodingError
# ("Response ended prematurely") inside api.list_repo_files — a dropped
# connection, not an auth/config problem. Retry those; surface permanent errors
# (401/403, bad revision, ...) immediately so the notebook shows them at once.
# Exception classes are matched by NAME across the MRO instead of isinstance so
# this module keeps importing in offline environments where
# requests/urllib3/huggingface_hub are absent (huggingface_hub pulls the first
# two in only when a network call actually runs).
TRANSIENT_EXC_NAMES = frozenset({
    "ConnectionError", "ChunkedEncodingError", "ConnectTimeout", "ReadTimeout",
    "Timeout", "TimeoutError", "ProtocolError", "IncompleteRead", "SSLError",
    "MaxRetryError", "NewConnectionError", "TransientError",
})
TRANSIENT_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
SYNC_RETRIES = 3          # extra attempts after the first failure
SYNC_BACKOFF_S = 2.0      # waits 2s, 4s, 8s (exponential)
# HTTP 429 is a DIFFERENT beast: the free Hub limit is a rolling window, and
# re-hitting it quickly EXTENDS the limit (observed 2026-10-03 on a Colab VM
# whose ~200-file sectioned pull ran without a token). So: honour Retry-After,
# wait 20s/40s otherwise, and give up fast with a wait-it-out message instead
# of hammering.
SYNC_RETRIES_429 = 1      # one patient retry, then stop and tell the user to wait
BACKOFF_429_S = 20.0
BACKOFF_429_CAP_S = 120.0


def rate_limited(exc: BaseException) -> bool:
    return getattr(getattr(exc, "response", None), "status_code", None) == 429


def retry_after_seconds(exc: BaseException) -> float | None:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) or {}
    val = headers.get("Retry-After")
    if not val:
        return None
    try:
        return min(float(val), BACKOFF_429_CAP_S)
    except ValueError:
        pass
    try:  # HTTP-date form
        from datetime import datetime, timezone
        from email.utils import parsedate_to_datetime
        delta = parsedate_to_datetime(val) - datetime.now(timezone.utc)
        return max(0.0, min(delta.total_seconds(), BACKOFF_429_CAP_S))
    except Exception:
        return None


def is_transient(exc: BaseException) -> bool:
    """True for retryable network failures, False for permanent errors."""
    if any(cls.__name__ in TRANSIENT_EXC_NAMES for cls in type(exc).__mro__):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in TRANSIENT_HTTP_STATUS


def run_with_retries(fn, *args, label: str = "sync", **kwargs):
    """Run fn(*args, **kwargs), retrying transient network failures with backoff."""
    for attempt in range(SYNC_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            rl = rate_limited(exc)
            limit = SYNC_RETRIES_429 if rl else SYNC_RETRIES
            if attempt >= limit or not is_transient(exc):
                if rl:
                    raise SystemExit(
                        f"[hf] rate limited (HTTP 429) during {label}: the free Hub limit is a "
                        "rolling window — wait ~10-15 minutes WITHOUT re-running, then retry "
                        "this cell. Repeated retries extend the limit, and pushes need HF_TOKEN "
                        "attached (anonymous bursts are the usual trigger).")
                raise
            if rl:
                wait = retry_after_seconds(exc) or min(BACKOFF_429_S * (2 ** attempt),
                                                       BACKOFF_429_CAP_S)
                print(f"[hf] rate limited during {label} — waiting {wait:.0f}s "
                      "(Retry-After honoured; the window is rolling)…")
            else:
                wait = SYNC_BACKOFF_S * (2 ** attempt)
                print(f"[hf] transient network error during {label} "
                      f"({type(exc).__name__}: {exc}) — retry {attempt + 1}/{SYNC_RETRIES} in {wait:.0f}s")
            time.sleep(wait)


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
    ap.add_argument("--light", action="store_true",
                    help="when pulling, skip per-run model checkpoints (*.pt) — "
                         "keeps metrics/logs/predictions.npz (table, board, audit blocks)")
    ap.add_argument("--section", default=None,
                    help="pull only the named sections (comma-separated): "
                         f"{', '.join(sorted(SECTIONS))}. Default: the whole repo.")
    ap.add_argument("--include", default=None,
                    help="extra raw fnmatch globs to allow on a pull (comma-separated, "
                         "repo-relative, e.g. 'artifacts/runs/C4_16/*')")
    a = ap.parse_args(argv)

    cfg = load_repo_config(required=not a.if_configured, repo_id=a.repo_id,
                           repo_type=a.repo_type, revision=a.revision,
                           subdir=a.subdir)
    if cfg is None:
        print("[hf] EMBEDED_HF_REPO_ID not set — skipping sync")
        return

    if a.action == "pull":
        sections = [s.strip() for s in a.section.split(",") if s.strip()] if a.section else None
        include = [s.strip() for s in a.include.split(",") if s.strip()] if a.include else None
        run_with_retries(pull_from_hf, cfg, S.ARTIFACTS, S.REPORT_MD, label="pull",
                         light=a.light, sections=sections, include=include)
    else:
        run_with_retries(push_to_hf, cfg, S.ARTIFACTS, S.REPORT_MD, label="push",
                         include_report=a.include_report)


if __name__ == "__main__":
    main()
