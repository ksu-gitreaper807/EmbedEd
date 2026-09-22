"""Phase 0.7 — acquire and probe Kitsios et al.'s BCB s′ (FINAL_SPEC §0.2/§9.3).

    python -m scripts.fetch_sprime            # download + probe (needs zenodo.org)
    python -m scripts.fetch_sprime --dry-run  # just list what would download

This sandbox blocks zenodo.org; run the download in the Colab notebook. The
probe must confirm: loads, has functionality labels, ~23 functionalities,
~2,300 + ~2,300 pairs. Writes a "s′ acquisition" block into
report/measurements.md. RQ3 is blocked on this file — discover breakage on
day 2, not week 3.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

RECORD = "17238379"          # doi:10.5281/zenodo.17238379
API = f"https://zenodo.org/api/records/{RECORD}"
DATA_EXT = {".jsonl", ".json", ".csv", ".tsv", ".txt", ".parquet", ".zip"}


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

    from code import settings as S
    outdir = S.artifact("sprime")
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for f in files:
        dest = outdir / f["key"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f["links"]["self"], dest)
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
