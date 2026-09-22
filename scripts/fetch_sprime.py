"""Phase 0.7 — acquire and probe Kitsios et al.'s BCB s′ (FINAL_SPEC §0.2/§9.3).

    python -m scripts.fetch_sprime            # download + probe (needs zenodo.org)
    python -m scripts.fetch_sprime --dry-run  # just list what would download

Downloads stream with a progress bar, resume via HTTP Range if a previous run
left a *.part file, and skip files that are already complete (verified by
size). This Arena sandbox blocks zenodo.org — run the download from a
machine/Colab with access. The probe must confirm: loads, has functionality
labels, ~23 functionalities, ~2,300 + ~2,300 pairs. Writes a "s′ acquisition"
block into report/measurements.md. RQ3 is blocked on this file — discover
breakage on day 2, not week 3.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

RECORD = "17238379"          # doi:10.5281/zenodo.17238379
API = f"https://zenodo.org/api/records/{RECORD}"
DATA_EXT = {".jsonl", ".json", ".csv", ".tsv", ".txt", ".parquet", ".zip"}


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
    if total:
        print(f"  {dest.name}: {done / 1e6:.1f} MB done in {time.time() - t0:.0f}s      ")
    if expected_size is not None and part.stat().st_size != expected_size:
        part.unlink(missing_ok=True)
        raise IOError(f"{dest.name}: size mismatch (got {part.stat().st_size}, "
                      f"want {expected_size}) — will retry next run")
    part.replace(dest)


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
    summary = {}
    for f in files:
        dest = outdir / f["key"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        _download(f["links"]["self"], dest, f.get("size"))
        if dest.suffix == ".zip":
            with zipfile.ZipFile(dest) as z:
                names = z.namelist()
                z.extractall(dest.parent)
            summary[f["key"]] = {"kind": "zip", "members": names[:10]}
            dest = next((p for p in dest.parent.rglob("*")
                         if p.suffix.lower() in DATA_EXT - {".zip"}), dest)
        if dest.suffix.lower() in {".jsonl", ".tsv", ".csv", ".txt"}:
            with open(dest, encoding="utf-8", errors="replace") as fh:
                first = fh.readline()
                n = 1 + sum(1 for _ in fh)
            cols = re_cols(first, dest)
            summary[f["key"]] = {"rows": n, "cols": cols, "size": dest.stat().st_size}
    print(json.dumps(summary, indent=2))

    block = ["", "## s′ acquisition (Phase 0.7)", "",
             "```json", json.dumps(summary, indent=2), "```", ""]
    if S.REPORT_MD.exists():
        old = S.REPORT_MD.read_text()
        import re as _re
        if "## s′ acquisition" in old:
            old = _re.sub(r"\n## s′ acquisition.*?(?=\n## |\Z)", "\n" + "\n".join(block), old, flags=_re.S)
        else:
            old += "\n".join(block)
        S.REPORT_MD.write_text(old)
    else:
        S.REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
        S.REPORT_MD.write_text("# Measurements\n" + "\n".join(block))
    return 0


def re_cols(first_line: str, path: Path):
    first_line = first_line.strip()
    if path.suffix == ".jsonl":
        try:
            return sorted(json.loads(first_line).keys())
        except Exception:
            pass
    for delim in ("\t", ","):
        if delim in first_line:
            return first_line.split(delim)
    return None


if __name__ == "__main__":
    sys.exit(main())
