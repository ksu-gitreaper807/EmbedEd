"""Phase 0.7 — acquire and probe Kitsios et al.'s BCB s′ (FINAL_SPEC §0.2).

    python -m scripts.fetch_sprime            # download + probe (needs zenodo.org)
    python -m scripts.fetch_sprime --dry-run  # just list what would download

Downloads stream with a progress bar, resume via HTTP Range if a previous run
left a *.part file, skip size-verified complete files, and check the MD5 from
the Zenodo record.

What the probe must confirm (RQ3 gate, FINAL_SPEC §0.2):
  The zip is a snapshot of github.com/kitsiosk/unseen-clones. The s′ dataset
  is `datasets/bcb_v2_sampled_bf/data_bcb_v2_sampled_bf.pickle` — a pandas
  DataFrame with columns code1, code2, label, functionality_id:
    * 4,600 pairs = 2,300 clone + 2,300 non-clone (functionality-balanced)
    * 23 unique functionality ids
  The package also carries the SCB corpora (datasets/scb/{Java,C}) used for
  the paper's cross-dataset protocols.

Verdict printed + written to report/measurements.md:
  s' GATE: PASS          — pair DataFrame with functionality labels found
  s' GATE: INCONCLUSIVE  — package listed but s′ not identifiable (exit 2,
                           manual check of the file list; the download stays)

zenodo.org is blocked in the Arena sandbox — run the download from a
machine/Colab with access. Reruns are cheap: complete files are skipped and
only the probe repeats. RQ3 is blocked on this file — discover breakage on
day 2, not week 3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

RECORD = "17238379"          # doi:10.5281/zenodo.17238379
API = f"https://zenodo.org/api/records/{RECORD}"
DATA_EXT = {".jsonl", ".json", ".csv", ".tsv", ".txt", ".parquet", ".zip"}
SKIP_MEMBERS = (".git", ".DS_Store")


# --------------------------------------------------------------------------- download

def _download(url: str, dest: Path, expected_size: int | None = None) -> None:
    """Stream to disk with a progress bar; resume a partial *.part via Range;
    skip files already complete (size-verified against the Zenodo record)."""
    if dest.exists():
        have = dest.stat().st_size
        if expected_size is None or have == expected_size:
            print(f"  {dest.name}: already complete ({have / 1e6:.1f} MB) — skipping")
            return
        print(f"  {dest.name}: existing file truncated ({have} != {expected_size}) — re-downloading")
        dest.unlink()
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    if start and expected_size is not None and start >= expected_size:
        part.unlink()
        start = 0
    req = urllib.request.Request(url, headers={"User-Agent": "embeded/phase0"})
    if start:
        req.add_header("Range", f"bytes={start}-")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        whole = r.status == 200          # server ignored Range: start over
        if whole and start:
            start = 0
        total = expected_size or (int(r.headers.get("Content-Length") or 0) + start)
        with open(part, "ab" if (not whole and start) else "wb") as fh:
            done = start
            last = 0.0
            while chunk := r.read(1 << 16):
                fh.write(chunk)
                done += len(chunk)
                now = time.time()
                if total and now - last >= 0.5:
                    last = now
                    rate = (done - start) / max(now - t0, 1e-9)
                    print(f"  {dest.name}: {done / 1e6:.1f}/{total / 1e6:.1f} MB "
                          f"({done / total:.0%}) {rate / 1e6:.2f} MB/s", end="\r", flush=True)
    print()
    print(f"  {dest.name}: {done / 1e6:.1f} MB done in {time.time() - t0:.0f}s")
    if expected_size is not None and part.stat().st_size != expected_size:
        part.unlink(missing_ok=True)
        raise IOError(f"{dest.name}: size mismatch (got {part.stat().st_size}, "
                      f"want {expected_size}) — will retry next run")
    part.replace(dest)


def _check_md5(dest: Path, expected_md5: str | None) -> None:
    if not expected_md5:
        return
    h = hashlib.md5()
    with open(dest, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    if h.hexdigest() != expected_md5:
        dest.unlink()
        raise IOError(f"{dest.name}: MD5 mismatch (got {h.hexdigest()}, "
                      f"want {expected_md5}) — file deleted, rerun to re-download")
    print(f"  {dest.name}: md5 {expected_md5[:12]}… verified")


# --------------------------------------------------------------------------- extract + probe

def _extract(zf: zipfile.ZipFile, dest: Path) -> tuple[int, int]:
    """Extract, skipping .git/ internals and .DS_Store. Returns (n, n_skipped)."""
    n = skipped = 0
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = info.filename.split("/")
        if ".git" in parts or parts[-1] in SKIP_MEMBERS:
            skipped += 1
            continue
        zf.extract(info, dest)
        n += 1
    return n, skipped


def _probe_pickle(path: Path) -> dict:
    try:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
    except Exception as e:
        return {"kind": "pickle", "error": f"load failed: {e.__class__.__name__}: {str(e)[:120]}"}
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            d = {"kind": "dataframe", "rows": int(len(obj)),
                 "columns": [str(c) for c in obj.columns]}
            if "label" in obj.columns:
                try:
                    d["label_counts"] = {int(k): int(v) for k, v in
                                         obj["label"].value_counts().items()}
                except Exception:
                    pass
            for c in obj.columns:
                try:
                    s = obj[c]
                    if s.dtype.kind in "iub":
                        d[f"unique_{c}"] = int(s.nunique())
                    elif s.dtype == object and s.nunique() < 200:
                        if s.dropna().astype(str).str.len().max() < 60:
                            d[f"unique_{c}"] = int(s.nunique())
                except Exception:
                    continue
            return d
    except Exception:
        pass
    return {"kind": "pickle", "type": type(obj).__name__,
            "len": len(obj) if hasattr(obj, "__len__") else None}


def _probe_text(path: Path, cap: int = 100_000_000) -> dict:
    if path.stat().st_size > cap:
        return {"kind": "text", "note": "skipped full probe (file too large)"}
    with open(path, encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
        n = 1 + sum(1 for _ in fh)
    cols = None
    if path.suffix == ".jsonl":
        try:
            cols = sorted(json.loads(first).keys())
        except Exception:
            pass
    else:
        for delim in ("\t", ","):
            if delim in first.strip():
                cols = first.strip().split(delim)
                break
    return {"kind": "text", "rows": n, "cols": cols}


def _probe_scb(root: Path) -> dict:
    """SCB-style layout: datasets/scb/<lang>/<subdir> with files (here flat
    `CloneN` files; other releases use CloneN/ directories)."""
    out = {}
    for scb in [p for p in root.rglob("scb")
                if p.is_dir() and p.parent.name == "datasets"]:
        for lang in sorted(p.name for p in scb.iterdir() if p.is_dir()):
            d = {}
            for p in sorted((scb / lang).iterdir()):
                if not p.is_dir():
                    continue
                names = [n for _, _, fs in os.walk(p) for n in fs]
                d[p.name] = {"files": len(names),
                             "clone_named": sum(n.startswith("Clone")
                                                for n in names)}
            out[f"{scb.relative_to(root)}/{lang}"] = d
    return out


def _list_package(root: Path) -> tuple[list[tuple[str, int]], int]:
    files, git_files = [], 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if ".git" in rel.parts:
            git_files += 1
            continue
        files.append((str(rel), p.stat().st_size))
    files.sort(key=lambda t: -t[1])
    return files, git_files


def probe_package(root: Path) -> dict:
    files, git_files = _list_package(root)
    summary = {
        "n_files": len(files),
        "n_git_files_skipped": git_files,
        "files": [{"path": p, "size": s} for p, s in files[:30]],
        "probes": {},
    }
    for p, s in files:
        path = root / p
        if s > 300_000_000:
            continue
        if path.suffix in {".pickle", ".pkl"}:
            summary["probes"][p] = _probe_pickle(path)
        elif path.suffix in {".tsv", ".csv", ".jsonl"}:
            summary["probes"][p] = _probe_text(path)
    scb = _probe_scb(root)
    if scb:
        summary["scb"] = scb
    summary["verdict"] = _gate(summary)
    return summary


def _gate(summary: dict) -> tuple[str, str]:
    """RQ3 gate: find the s′ pair DataFrame — label column + a low-cardinality
    functionality column + a healthy row count. Reports the canonical
    datasets/ copy first (the release has a duplicate under llms/)."""
    matches = []
    for p, pr in summary.get("probes", {}).items():
        if pr.get("kind") != "dataframe" or "label" not in pr.get("columns", []):
            continue
        func_cols = {k[len("unique_"):]: v for k, v in pr.items()
                     if k.startswith("unique_") and k != "unique_label"}
        if pr["rows"] >= 1000 and any(5 <= n <= 100 for n in func_cols.values()):
            matches.append((p, pr, func_cols))
    if not matches:
        return ("INCONCLUSIVE",
                "no pair DataFrame with a label + low-cardinality functionality "
                "column found — inspect the file list (s′ should be a DataFrame "
                "code1/code2/label/functionality_id, 4,600 rows, 23 functionalities)")
    # canonical copy lives under datasets/ (the release also duplicates it under
    # llms/ and astnn/); the zip may or may not carry a top-level wrapper folder
    matches.sort(key=lambda t: 0 if "datasets/" in t[0] else 1)
    p, pr, func_cols = matches[0]
    why = (f"{p}: {pr['rows']} pairs, labels {pr.get('label_counts')}, "
           f"functionality cols {func_cols}")
    if len(matches) > 1:
        why += f" (+{len(matches) - 1} duplicate copy/copies: " \
               + ", ".join(m[0] for m in matches[1:]) + ")"
    return ("PASS", why)


# --------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    with urllib.request.urlopen(API, timeout=120) as r:
        rec = json.load(r)
    files = [f for f in rec.get("files", []) if Path(f["key"]).suffix.lower() in DATA_EXT]
    print(f"record: {rec['metadata']['title']!r}")
    for f in files:
        print(f"  {f['key']}  {f.get('size', '?')} bytes")
    if not files:
        print("!! no data files found in record — STOP, report blocker")
        return 2

    if a.dry_run:
        print("(--dry-run: nothing downloaded)")
        return 0

    from embeded import settings as S
    outdir = S.artifact("sprime")
    outdir.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = outdir / f["key"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        _download(f["links"]["self"], dest, f.get("size"))
        _check_md5(dest, (f.get("checksums") or {}).get("md5"))
        if dest.suffix == ".zip":
            with zipfile.ZipFile(dest) as z:
                names = z.namelist()
                n, skipped = _extract(z, dest.parent)
            print(f"  extracted {n} files ({skipped} .git members skipped, "
                  f"{len(names)} total members)")

    probe = probe_package(outdir)
    print(json.dumps({k: v for k, v in probe.items() if k != "files"}, indent=2))
    print("top files:")
    for item in probe["files"][:15]:
        print(f"  {item['size'] / 1e6:8.2f} MB  {item['path']}")
    verdict, why = probe["verdict"]
    print(f"s' GATE: {verdict} — {why}")
    block = ["", "## s′ acquisition (Phase 0.7)", "",
             f"**verdict: {verdict}** — {why}", "",
             "```json", json.dumps(probe, indent=2), "```", ""]
    _append_report(block)
    return 0 if verdict == "PASS" else 2


def _append_report(block: list[str]) -> None:
    from embeded import settings as S
    import re
    if S.REPORT_MD.exists():
        old = S.REPORT_MD.read_text()
        if "## s′ acquisition" in old:
            old = re.sub(r"\n## s′ acquisition.*?(?=\n## |\Z)", "\n" + "\n".join(block), old, flags=re.S)
        else:
            old += "\n".join(block)
        S.REPORT_MD.write_text(old)
    else:
        S.REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
        S.REPORT_MD.write_text("# Measurements\n" + "\n".join(block))


if __name__ == "__main__":
    sys.exit(main())
