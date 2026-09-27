"""One-shot VM report: everything needed to debug a Colab run, in one file.

    python -m scripts.colab_report                  # print it and write report/colab_report.md
    python -m scripts.colab_report --out /tmp/r.md  # somewhere else (e.g. files.download)

Why this exists: the agent sandbox has no Colab/MCP access and its egress is an
allowlist (github.com + api.github.com + pypi.org only; colab.research.google.com
resolves to nothing), so the only shared channel is a file. Run this on the VM and
either paste the output or push/upload the file — it carries the whole environment
in one shot instead of one traceback per round trip.

It never prints audit_key.csv contents (the audit stays blind) and masks every
environment variable whose name suggests a credential.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from embeded import settings as S

SECRET_HINTS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "PAT", "CREDENTIAL")
JSONL = (".jsonl", ".tsv", ".csv")


def _sh(*args: str) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:                                    # noqa: BLE001 - a report must not fail
        return f"<{exc.__class__.__name__}: {exc}>"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _rows(path: Path) -> str:
    if path.suffix not in JSONL or path.stat().st_size > 64_000_000:
        return ""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return str(sum(1 for _ in fh))


def collect() -> dict:
    out: dict = {
        "version": S.VERSION,
        "artifacts_dir": str(S.ARTIFACTS),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git": {"head": _sh("git", "rev-parse", "--short", "HEAD"),
                "branch": _sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": _sh("git", "status", "--porcelain").splitlines()[:10]},
        "env": {k: ("***" if any(h in k.upper() for h in SECRET_HINTS) else v)
                for k, v in sorted(os.environ.items())
                if k.startswith(("EMBEDED_", "HF_", "CUDA"))},
        "packages": {},
        "gpu": {},
        "artifacts": [],
        "audit": {},
        "runs": [],
    }
    for mod in ("torch", "transformers", "datasets", "sklearn", "numpy", "rank_bm25"):
        try:
            out["packages"][mod] = getattr(__import__(mod), "__version__", "installed (no __version__)")
        except Exception as exc:                                # noqa: BLE001
            out["packages"][mod] = f"<not importable: {exc.__class__.__name__}>"
    try:
        import torch
        out["gpu"] = {"cuda": torch.cuda.is_available(),
                      "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      "torch_cuda": torch.version.cuda,
                      "mem_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                      if torch.cuda.is_available() else None}
    except Exception as exc:                                    # noqa: BLE001
        out["gpu"] = {"error": f"{exc.__class__.__name__}: {exc}"}

    if S.ARTIFACTS.exists():
        for p in sorted(S.ARTIFACTS.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(S.ARTIFACTS).as_posix()
            if rel.endswith("audit_key.csv"):                   # blindness: existence only
                out["audit"]["audit_key.csv"] = f"present, {p.stat().st_size} bytes (contents withheld)"
                continue
            rec = {"path": rel, "bytes": p.stat().st_size, "sha256_12": _sha(p)}
            n = _rows(p)
            if n:
                rec["lines"] = n
            out["artifacts"].append(rec)
            if rel.startswith("audit/"):
                out["audit"][rel] = (f"{rec['bytes']:,} bytes, sha {rec['sha256_12']}"
                                     + (f", {rec['lines']} lines" if rec.get("lines") else ""))
            if rel == "audit/audit_labels.csv":
                with open(p, newline="", encoding="utf-8", errors="replace") as fh:
                    rows = list(csv.DictReader(fh))
                out["audit"]["filled_labels"] = (
                    f"{sum(1 for r in rows if (r.get('label') or '').strip())}/{len(rows)}")
    for run in sorted((S.ARTIFACTS / S.RUNS_SUBDIR).glob("*")) if (S.ARTIFACTS / S.RUNS_SUBDIR).exists() else []:
        entry = {"run": run.name, "files": sorted(p.name for p in run.iterdir())}
        for name in ("metrics.json", "eval_metrics.json"):
            if (run / name).exists():
                entry[name] = json.loads((run / name).read_text())
        out["runs"].append(entry)
    for name in ("mining_summary.json", "corpus_emb.meta.json", "gate_runs.json"):
        p = S.ARTIFACTS / name
        if p.exists():
            out[name] = json.loads(p.read_text())
    return out


def render(rep: dict) -> str:
    md = [f"# Colab VM report — {rep['version']}", "",
          f"python {rep['python']} | {rep['platform']}",
          f"git {rep['git']['head']} on `{rep['git']['branch']}`"
          + (f" | dirty: {rep['git']['dirty']}" if rep["git"]["dirty"] else " | clean"),
          f"ARTIFACTS = `{rep['artifacts_dir']}`", "",
          "## packages", "",
          *[f"- {k}: {v}" for k, v in rep["packages"].items()], "",
          "## gpu", "", "```json", json.dumps(rep["gpu"], indent=2), "```", "",
          "## environment (credentials masked)", "",
          *[f"- `{k}={v}`" for k, v in rep["env"].items()], "",
          "## artifacts", "",
          "| path | bytes | sha256/12 | lines |", "|---|---:|---|---:|"]
    for a in rep["artifacts"]:
        md.append(f"| {a['path']} | {a['bytes']:,} | `{a['sha256_12']}` | {a.get('lines', '')} |")
    if rep.get("mining_summary.json"):
        md += ["", "## mining_summary `_run`", "", "```json",
               json.dumps(rep["mining_summary.json"].get("_run"), indent=2), "```"]
    if rep["audit"]:
        md += ["", "## audit dir", "", *[f"- {k}: {v}" for k, v in rep["audit"].items()]]
    if rep["runs"]:
        md += ["", "## runs", "", "```json", json.dumps(rep["runs"], indent=2), "```"]
    return "\n".join(md) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(S.ARTIFACTS / "colab_report.md"),
                    help="where to write the report (default: <ARTIFACTS>/colab_report.md, "
                         "outside the repo so it is never committed by accident)")
    ap.add_argument("--json", action="store_true", help="also dump the raw dict as JSON")
    a = ap.parse_args(argv)
    rep = collect()
    text = render(rep)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    if a.json:
        Path(a.out).with_suffix(".json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(text)
    print(f"[written] {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
