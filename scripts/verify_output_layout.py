"""Read-only verification of the byte-preserving generated-output relocation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRATCH = {"paper.aux", "paper.out", "missfont.log"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def verify(root: Path = ROOT, *, include_build_scratch: bool = False) -> dict:
    manifest = json.loads((root / "report/output_relocation.json").read_text())
    for old, target in manifest["compatibility_links"].items():
        link = root / old
        require(link.is_symlink(), f"Missing compatibility symlink: {old}")
        require(link.readlink() == Path(target), f"Unexpected symlink target: {old}")
        require(link.is_dir(), f"Broken compatibility symlink: {old}")

    checked = 0
    archives = []
    for entry in manifest["files"]:
        old, new = root / entry["from"], root / entry["to"]
        if new.name in BUILD_SCRATCH and not include_build_scratch:
            continue
        require(new.is_file(), f"Missing relocated file: {entry['to']}")
        require(old.resolve() == new.resolve(), f"Path mapping differs: {entry['from']}")
        require(digest(new) == entry["sha256"], f"Content changed: {entry['to']}")
        checked += 1
        if new.suffix == ".zip":
            archives.append(new)

    payload_hashes = 0
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)), f"Duplicate ZIP members: {path}")
            require(archive.testzip() is None, f"ZIP CRC failure: {path}")
            for name in ("source_manifest.json", "release_manifest.json",
                         "phase6_source_manifest.json"):
                if name not in names:
                    continue
                entries = json.loads(archive.read(name))
                require(set(names) == set(entries) | {name}, f"ZIP inventory changed: {path}")
                for member, expected in entries.items():
                    actual = hashlib.sha256(archive.read(member)).hexdigest()
                    require(actual == expected, f"ZIP payload changed: {path.name}:{member}")
                    payload_hashes += 1

    phase5 = json.loads((root / "output/phase5/closure.json").read_text())
    package = phase5["package"]
    require(digest(root / package["path"]) == package["sha256"], "Phase 5 ZIP changed")
    require(digest(root / "output/phase5/review.json") == phase5["review_sha256"],
            "Phase 5 review changed")
    phase6 = json.loads((root / "output/phase6/closure.json").read_text())
    for path, meta in phase6["artifacts"].items():
        require(digest(root / path) == meta["sha256"], f"Phase 6 ZIP changed: {path}")
    for path, expected in phase6["evidence_sha256"].items():
        require(digest(root / path) == expected, f"Phase 6 evidence changed: {path}")
    # The IEEE record binds the paper, portable source and figure ZIP, plus
    # other unchanged scientific inputs; source revisions are checked too.
    ieee = json.loads((root / "report/ieee_paper_format.json").read_text())
    for path, expected in ieee["sha256"].items():
        require(digest(root / path) == expected, f"IEEE evidence changed: {path}")

    return dict(status="passed", relocated_files_checked=checked,
                build_scratch_included=include_build_scratch,
                compatibility_links_checked=len(manifest["compatibility_links"]),
                zip_crc_checks=len(archives), zip_payload_hashes_checked=payload_hashes,
                historical_receipts_verified=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-build-scratch", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(include_build_scratch=args.include_build_scratch), indent=2))


if __name__ == "__main__":
    main()
