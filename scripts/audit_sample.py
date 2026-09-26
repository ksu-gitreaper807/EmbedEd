"""False-negative audit, 50 + 50 (SCOPE P2-9, IMPLEMENTATION_PLAN Phase 2.5) — BLIND.

    python -m scripts.audit_sample make  [--n 50] [--seed 13]
    python -m scripts.audit_sample score

`make` samples n (anchor, mined-negative) pairs from C2 and n from C3 — distinct
anchors, one random negative each — shuffles them, and writes under
ARTIFACTS/audit/:

    audit_pairs.md    the pairs to read (id + both code fragments), NO condition shown
    audit_labels.csv  id,label,note — fill `label` with clone | not_clone | unsure
    audit_key.csv     id -> condition/anchor/negative — do not open until scoring

`score` joins labels with the key and reports, per condition, the share of
mined "negatives" that are in fact clones (Wilson 95% CI), with `unsure`
counted separately (headline rate counts unsure as not_clone; an upper bound
counts it as clone). Appends a '## False-negative audit' section to
report/measurements.md and writes audit_result.json.

Why blind: the labeller must not know whether a pair came from BM25 or from
the semantic miner — the whole point is to compare the two rates.

RUBRIC (BigCloneBench's own notion of a clone = functional equivalence):
  clone      both fragments implement the same functionality (copy a file,
             MD5 a string, ...), whatever the syntax — Type-1..4 incl. weak T3/T4
  not_clone  different functionality, or one is only a small part of the other
  unsure     cannot tell in about a minute — do not agonise, mark it and move on
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re

from embeded import settings as S

CONDS = ("C2", "C3")
LABELS = ("clone", "not_clone", "unsure")


def audit_dir():
    d = S.ARTIFACTS / "audit"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _triples(cond):
    rows = [json.loads(l) for l in open(S.artifact(f"triples_{cond}.jsonl"), encoding="utf-8")]
    by_anchor: dict[int, list[int]] = {}
    for r in rows:
        by_anchor.setdefault(r["anchor"], []).append(r["negative"])
    return by_anchor


def sample_pairs(n: int, seed: int) -> list[dict]:
    """n pairs per condition, distinct anchors within a condition, one random
    negative per anchor, then a seeded shuffle across conditions."""
    rng = random.Random(seed)
    out = []
    for cond in CONDS:
        by_anchor = _triples(cond)
        anchors = sorted(by_anchor)
        if len(anchors) < n:
            raise SystemExit(f"{cond}: only {len(anchors)} anchors, cannot sample {n}")
        for a in rng.sample(anchors, n):
            out.append({"condition": cond, "anchor": a, "negative": rng.choice(by_anchor[a])})
    rng.shuffle(out)
    for i, p in enumerate(out, 1):
        p["id"] = f"P{i:03d}"
    return out


def make(n: int, seed: int) -> dict:
    from embeded.data.prepare_data import load_fragments
    frags = load_fragments()
    pairs = sample_pairs(n, seed)
    d = audit_dir()

    with open(d / "audit_key.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "condition", "anchor", "negative"])
        w.writeheader()
        for p in pairs:
            w.writerow({k: p[k] for k in w.fieldnames})

    with open(d / "audit_labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "label", "note"])
        w.writeheader()
        for p in pairs:
            w.writerow({"id": p["id"], "label": "", "note": ""})

    md = [f"# False-negative audit — {len(pairs)} pairs ({n} per condition, blind)", "",
          f"seed {seed}, version `{S.VERSION}`. For each pair decide whether the two fragments implement the "
          "**same functionality** (BigCloneBench's clone definition). Write `clone`, `not_clone` or `unsure` in "
          "`audit_labels.csv`; do not open `audit_key.csv` until you run `score`.", "",
          "Rubric: `clone` = same functionality whatever the syntax (Type-1..4, incl. weak T3/T4); `not_clone` = "
          "different functionality or one is only a small part of the other; `unsure` = cannot tell in about a minute.", ""]
    for p in pairs:
        md += [f"## {p['id']}", "", "**A**", "", "```java", frags[p["anchor"]].strip(), "```", "",
               "**B**", "", "```java", frags[p["negative"]].strip(), "```", ""]
    (d / "audit_pairs.md").write_text("\n".join(md))
    return {"n_pairs": len(pairs), "per_condition": n, "seed": seed, "dir": str(d)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def score_rows(labels: dict[str, str], key: list[dict]) -> dict:
    out = {}
    for cond in CONDS:
        ids = [r["id"] for r in key if r["condition"] == cond]
        got = {i: labels.get(i, "").strip().lower() for i in ids}
        bad = sorted(i for i, v in got.items() if v not in LABELS)
        if bad:
            raise SystemExit(f"{cond}: {len(bad)} pair(s) unlabelled or mislabelled "
                             f"(allowed: {', '.join(LABELS)}): {', '.join(bad[:8])}{' ...' if len(bad) > 8 else ''}")
        n = len(ids)
        c = sum(v == "clone" for v in got.values())
        u = sum(v == "unsure" for v in got.values())
        lo, hi = wilson(c, n)
        lo_u, hi_u = wilson(c + u, n)
        out[cond] = {"n": n, "clone": c, "not_clone": n - c - u, "unsure": u,
                     "fn_rate": round(c / n, 3), "fn_rate_ci95": [round(lo, 3), round(hi, 3)],
                     "fn_rate_upper_incl_unsure": round((c + u) / n, 3),
                     "fn_rate_upper_ci95": [round(lo_u, 3), round(hi_u, 3)]}
    return out


def write_section(res: dict) -> None:
    lines = ["## False-negative audit (50 + 50, blind; SCOPE P2-9)", "",
             f"`python -m scripts.audit_sample score` — version `{S.VERSION}`. Share of mined negatives judged "
             "to be clones (functional equivalence). Headline counts `unsure` as not_clone; the upper bound counts it as clone.",
             "", "| condition | n | clone | not_clone | unsure | FN rate [95% CI] | upper bound incl. unsure |",
             "|---|---|---|---|---|---|---|"]
    for cond in CONDS:
        m = res[cond]
        lines.append(f"| {cond} | {m['n']} | {m['clone']} | {m['not_clone']} | {m['unsure']} | "
                     f"**{m['fn_rate']:.0%}** [{m['fn_rate_ci95'][0]:.0%}, {m['fn_rate_ci95'][1]:.0%}] | "
                     f"{m['fn_rate_upper_incl_unsure']:.0%} [{m['fn_rate_upper_ci95'][0]:.0%}, {m['fn_rate_upper_ci95'][1]:.0%}] |")
    lines.append("")
    block = "\n".join(lines)
    S.REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    if S.REPORT_MD.exists():
        old = S.REPORT_MD.read_text()
        if "## False-negative audit" in old:
            old = re.sub(r"## False-negative audit.*?(?=\n## |\Z)", lambda _m: block + "\n", old, flags=re.S)
        else:
            old = old.rstrip("\n") + "\n\n" + block + "\n"
        S.REPORT_MD.write_text(old)
    else:
        S.REPORT_MD.write_text("# Measurements\n\n" + block + "\n")
    print(f"wrote {S.REPORT_MD}")


def score() -> dict:
    d = audit_dir()
    with open(d / "audit_key.csv", encoding="utf-8") as fh:
        key = list(csv.DictReader(fh))
    with open(d / "audit_labels.csv", encoding="utf-8") as fh:
        labels = {r["id"]: r["label"] for r in csv.DictReader(fh)}
    res = score_rows(labels, key)
    (d / "audit_result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    write_section(res)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("make", "score"))
    ap.add_argument("--n", type=int, default=50, help="pairs per condition")
    ap.add_argument("--seed", type=int, default=S.SEED)
    a = ap.parse_args(argv)
    if a.action == "make":
        info = make(a.n, a.seed)
        print(json.dumps(info, indent=2))
        print(f"read  {info['dir']}/audit_pairs.md\nfill  {info['dir']}/audit_labels.csv\nthen  python -m scripts.audit_sample score")
    else:
        score()


if __name__ == "__main__":
    main()
