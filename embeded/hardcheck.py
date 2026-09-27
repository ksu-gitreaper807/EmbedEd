"""Gate G1b — the hardness check (FINAL_SPEC §6.1, correction 4). Run BEFORE
training; if the manipulation did not take effect, the experiment produces no
evidence.

    python -m embeded.hardcheck          # real data (needs embeded/artifacts/corpus_emb.npy)
    python -m embeded.hardcheck --recorded report/gate_g1/run_2026-09-26_phase01-v5.json
                                         # re-judge an already-recorded run, no artifacts needed

Rule in force (decision D1, recorded 2026-09-27 — see the comment above
`settings.HARDNESS_D_MIN`):

    C1 < C2 <= C3   AND   d(C1->C2) >= settings.HARDNESS_D_MIN

where d is the standardised gap (Cohen's d, pooled sd) — the same definition
`scripts/hardness_diagnostics.py` reports, so gate and diagnostics cannot drift
apart. The legacy absolute-margin reading (`evaluate`, margin 0.02) is still
computed and written into every recorded run: the 2026-09-26 FAIL that prompted
the change must stay legible next to the PASS (SCOPE P6-3).

evaluate()/evaluate_gap() are pure so the gate logic itself is unit-tested.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict

import numpy as np

from . import settings as S

CONDS = ("C1", "C2", "C3")
SECTION = "## Gate G1 — hardness check"


def standardized_gap(mean_a: float, sd_a: float, mean_b: float, sd_b: float) -> float:
    """Cohen's d between two conditions, pooled sd — the ONE definition, shared
    with `scripts.hardness_diagnostics.py` (positive = b is harder than a)."""
    pooled = math.sqrt((sd_a ** 2 + sd_b ** 2) / 2)
    if pooled <= 0:
        return 0.0
    return (mean_b - mean_a) / pooled


def evaluate(means: dict[str, float], margin: float) -> tuple[bool, list[str]]:
    """LEGACY v1–v5 rule: absolute cosine margin. Kept (and still recorded on
    every run) so the 2026-09-26 FAIL stays reproducible; it is no longer the
    verdict — see `evaluate_gap`."""
    msgs = []
    ok = True
    if not all(k in means for k in CONDS):
        return False, [f"missing conditions: have {sorted(means)}, need C1/C2/C3"]
    if means["C2"] - means["C1"] < margin:
        ok = False
        msgs.append(f"C1→C2 gap {means['C2'] - means['C1']:+.4f} < margin {margin} "
                    "(mining produced equally easy negatives)")
    if means["C3"] < means["C2"] - 1e-9:
        ok = False
        msgs.append(f"C3 ({means['C3']:.4f}) harder than C2 ({means['C2']:.4f}) "
                    "violates C2 <= C3 — check semantic exclusion order")
    if not msgs:
        msgs.append(f"order C1 < C2 <= C3 holds with visible first gap "
                    f"({means['C1']:.4f} < {means['C2']:.4f} <= {means['C3']:.4f})")
    return ok, msgs


def evaluate_gap(means: dict[str, float], sds: dict[str, float], *,
                 d_min: float = S.HARDNESS_D_MIN, margin: float = S.HARDNESS_MARGIN,
                 ci: tuple[float, float] | None = None) -> dict:
    """The D1 verdict, plus the legacy reading for the record.

    Returns {"ok", "msgs", "d", "gap", "ci", "legacy_ok", "legacy_msgs"}.
    `ok` is the verdict; `legacy_ok` is reported, never decisive.
    """
    out: dict = {"ok": False, "msgs": [], "d": None, "gap": None, "ci": ci,
                 "legacy_ok": False, "legacy_msgs": []}
    if not all(k in means for k in CONDS):
        out["msgs"] = [f"missing conditions: have {sorted(means)}, need C1/C2/C3"]
        return out
    if not all(k in sds and sds[k] is not None for k in CONDS):
        out["msgs"] = ["no per-condition dispersion recorded — the standardised "
                       "gap cannot be computed (re-measure with `hardcheck`)"]
        return out

    msgs = []
    failed = []
    gap = means["C2"] - means["C1"]
    d = standardized_gap(means["C1"], sds["C1"], means["C2"], sds["C2"])
    out["gap"], out["d"] = round(gap, 5), round(d, 3)

    if means["C2"] <= means["C1"]:
        failed.append(f"C2 ({means['C2']:.4f}) is not harder than C1 ({means['C1']:.4f}) "
                      "— the manipulation did not take effect")
    if means["C3"] < means["C2"] - 1e-9:
        failed.append(f"C3 ({means['C3']:.4f}) below C2 ({means['C2']:.4f}) "
                      "violates C2 <= C3 — check semantic exclusion order")
    if d < d_min:
        failed.append(f"standardised gap d(C1→C2) = {d:.3f} < {d_min} "
                      f"(absolute gap {gap:+.4f}) — C2 is not visibly harder than C1 "
                      "on the scale of this embedding space")
    if not failed:
        msgs.append(f"order C1 < C2 <= C3 holds ({means['C1']:.4f} < {means['C2']:.4f} "
                    f"<= {means['C3']:.4f}) and d(C1→C2) = {d:.3f} >= {d_min} "
                    f"(absolute gap {gap:+.4f})")
    if ci is not None and ci[0] < d_min:
        msgs.append(f"note: the anchor-resampled {S.HARDNESS_CI:.0%} CI for d reaches "
                    f"{ci[0]:.3f}, below {d_min} — the point estimate passes but the gap "
                    "is not separated from the threshold at this sample size")

    out["ok"] = not failed
    out["msgs"] = failed + msgs
    out["legacy_ok"], out["legacy_msgs"] = evaluate(means, margin)
    return out


def bootstrap_d(per_anchor: dict[str, dict[int, float]], *,
                n_boot: int = S.HARDNESS_BOOTSTRAP, seed: int = S.SEED,
                ci: float = S.HARDNESS_CI) -> tuple[float, float] | None:
    """Percentile CI for d(C1→C2), resampling ANCHORS with replacement.

    Never resample the individual negatives: one anchor contributes k = 20
    correlated negatives, so treating 32,000 of them as independent understates
    the interval by ~sqrt(k) (PHASE2_PLAN §2 item 3). Returns None when there
    are too few shared anchors to resample.
    """
    if "C1" not in per_anchor or "C2" not in per_anchor:
        return None
    shared = sorted(set(per_anchor["C1"]) & set(per_anchor["C2"]))
    if len(shared) < 8 or n_boot <= 0:
        return None
    x1 = np.array([per_anchor["C1"][a] for a in shared], dtype=float)
    x2 = np.array([per_anchor["C2"][a] for a in shared], dtype=float)
    rng = np.random.default_rng(seed)
    n = len(shared)
    ds = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        a, c = x1[idx], x2[idx]
        ds[b] = standardized_gap(a.mean(), a.std(), c.mean(), c.std())
    lo, hi = np.quantile(ds, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return round(float(lo), 3), round(float(hi), 3)


def measure(triples_files: dict[str, str], emb: np.ndarray, row_of: dict[int, int],
            n_anchors: int, seed: int) -> dict:
    """Mean/sd cos(anchor, negative) per condition over a seeded anchor sample,
    plus the per-anchor means the bootstrap needs."""
    rng = np.random.default_rng(seed)
    out = {}
    for cond, path in triples_files.items():
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        anchors = sorted({r["anchor"] for r in rows})
        take = anchors if len(anchors) <= n_anchors else list(
            rng.choice(anchors, size=n_anchors, replace=False))
        take = set(int(t) for t in take)
        by_anchor: dict[int, list[float]] = defaultdict(list)
        sims = []
        for r in rows:
            if r["anchor"] in take and r["negative"] in row_of:
                s = float(emb[row_of[r["anchor"]]] @ emb[row_of[r["negative"]]])
                by_anchor[int(r["anchor"])].append(s)
                sims.append(s)
        if not sims:
            raise SystemExit(f"{cond}: no (anchor,negative) pair found in corpus index — stale artifacts?")
        out[cond] = {"mean_cos": round(float(np.mean(sims)), 5),
                     "std": round(float(np.std(sims)), 5),
                     "n_samples": len(sims),
                     "n_anchors": len(by_anchor),
                     "per_anchor": {a: round(float(np.mean(v)), 6) for a, v in by_anchor.items()}}
    return out


def percentile_note(diagnostics_path=None) -> tuple[list[str], dict | None]:
    """Corpus-percentile diagnostics reported alongside the verdict (D1 asks for
    them so a PASS is readable without re-deriving the scale of the space)."""
    p = diagnostics_path or S.artifact("hardness_diagnostics.json")
    try:
        res = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return ([f"corpus percentiles: not available — run "
                 f"`python -m scripts.hardness_diagnostics` (no {p.name})"], None)
    cd = res.get("conditions", {})
    if not all(c in cd for c in CONDS):
        return ([f"corpus percentiles: {p.name} has no per-condition block — rerun "
                 f"`python -m scripts.hardness_diagnostics`"], None)
    med = {c: cd[c].get("percentile_median") for c in CONDS}
    top20 = {c: cd[c].get("in_anchor_top20_frac") for c in CONDS}
    line = ("corpus percentile of the negative in its anchor's similarity ranking, median: "
            + " / ".join(f"{c} {med[c]:.1f}" for c in CONDS)
            + "  (50 = random, 100 = nearest); inside the anchor's top-20: "
            + " / ".join(f"{c} {top20[c]:.1%}" for c in CONDS))
    return [line], {"percentile_median": med, "in_anchor_top20_frac": top20,
                    "source": str(p)}


def append_measurements(per_cond, ok, msgs, *, margin=None, stamp=None, rule=None,
                        extra=()) -> None:
    """Record this run under the G1 section — as HISTORY, never replacing
    earlier runs: a PASS obtained after changing the mining or the rule must
    sit next to the FAIL that prompted the change (SCOPE P6-3 transparency)."""
    import time
    stamp = stamp or time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    margin = S.HARDNESS_MARGIN if margin is None else margin
    rule = rule or f"legacy absolute margin {margin}"
    run = ["", f"### run {stamp} — version `{S.VERSION}`, rule: {rule}", ""]
    for c in CONDS:
        m = per_cond[c]
        na = f", anchors = {m['n_anchors']}" if "n_anchors" in m else ""
        run.append(f"- {c}: mean cos(anchor, negative) = {m['mean_cos']:.5f} "
                   f"(sd {m['std']:.3f}, n = {m['n_samples']}{na})")
    run.extend(f"- {line}" for line in extra)
    run.append(f"- **verdict: {'PASS' if ok else 'FAIL'}** — " + "; ".join(msgs))
    run.append("")
    run_text = "\n".join(run)
    S.REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    if S.REPORT_MD.exists():
        t = S.REPORT_MD.read_text()
        m = re.search(r"\n" + re.escape(SECTION) + r".*?(?=\n## |\Z)", t, flags=re.S)
        if m:   # append this run at the end of the existing section
            t = t[:m.end()].rstrip("\n") + "\n" + run_text + t[m.end():]
        else:
            t = t.rstrip("\n") + "\n\n" + SECTION + "\n" + run_text
        S.REPORT_MD.write_text(t)
    else:
        S.REPORT_MD.write_text("# Measurements\n\n" + SECTION + "\n" + run_text)


def load_recorded(path) -> dict:
    """A previously recorded gate run, machine-readable, so a rule change can be
    re-judged offline instead of re-spending GPU time on the same artifacts.
    Summary-only records have no per-anchor data, so no anchor-resampled CI."""
    rec = json.loads(open(path, encoding="utf-8").read())
    per = rec.get("per_condition") or rec
    if not all(c in per for c in CONDS):
        raise SystemExit(f"{path}: no C1/C2/C3 block — expected "
                         f'{{"per_condition": {{"C1": {{"mean_cos": ..., "std": ...}}}}}}')
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--margin", type=float, default=S.HARDNESS_MARGIN,
                    help="legacy absolute margin, reported alongside the verdict")
    ap.add_argument("--d-min", type=float, default=S.HARDNESS_D_MIN,
                    help="required standardised C1→C2 gap (decision D1)")
    ap.add_argument("--anchors", type=int, default=S.HARDNESS_CHECK_ANCHORS)
    ap.add_argument("--rule", choices=("scale-aware", "legacy"), default="scale-aware",
                    help="scale-aware = decision D1 (default); legacy = the v1–v5 absolute margin")
    ap.add_argument("--recorded", type=str, default=None,
                    help="re-judge a recorded run (JSON) instead of measuring artifacts")
    ap.add_argument("--no-bootstrap", action="store_true", help="skip the anchor-resampled CI")
    ap.add_argument("--stamp", default=None, help="override the run stamp (tests/replays)")
    a = ap.parse_args(argv)

    extra: list[str] = []
    if a.recorded:
        rec = load_recorded(a.recorded)
        per = {c: {k: v for k, v in rec["per_condition"][c].items() if k != "per_anchor"}
               for c in CONDS}
        stamp = a.stamp or (f"{rec.get('measured', 'recorded')} measurement, "
                            f"re-evaluated under the rule now in force")
        extra.append(f"**no new measurement**: this re-judges the recorded run "
                     f"`{rec.get('measured', '?')}` (version `{rec.get('version', '?')}`, "
                     f"{rec.get('n_anchors', '?')} anchors × k) under the rule now in force. "
                     "Anchor-level resampling is impossible from a summary record, so no CI.")
        ci = None
    else:
        from .data.prepare_data import load_fragments
        from .mining.semantic_index import load_corpus_emb

        stamp = a.stamp
        frag_path = S.ARTIFACTS / "fragments.jsonl"
        if not frag_path.exists():
            raise SystemExit(missing_artifact_message(
                frag_path, hint="regenerate the Phase 0 artifacts first: "
                                "python -m embeded.data.prepare_data --hf --verify-spec"))
        files = {c: str(S.artifact(f"triples_{c}.jsonl")) for c in CONDS}
        for c, p in files.items():
            try:
                open(p).close()
            except FileNotFoundError:
                raise SystemExit(missing_artifact_message(p))
        try:
            emb, meta = load_corpus_emb()
        except FileNotFoundError as e:
            raise SystemExit(missing_artifact_message(e.filename or S.artifact("corpus_emb.npy")))
        row_of = {c: i for i, c in enumerate(sorted(load_fragments()))}
        per = measure(files, emb, row_of, a.anchors, S.SEED)
        ci = None if a.no_bootstrap else bootstrap_d(
            {c: per[c].get("per_anchor", {}) for c in CONDS})
        lines, _diag = percentile_note()
        extra.extend(lines)
        extra.append(f"embeddings: {meta.get('model')} max_len={meta.get('max_len')} "
                     f"n={meta.get('n')} version={meta.get('version')}")

    means = {c: per[c]["mean_cos"] for c in CONDS}
    sds = {c: per[c]["std"] for c in CONDS}
    d = standardized_gap(means["C1"], sds["C1"], means["C2"], sds["C2"])
    legacy_ok, legacy_msgs = evaluate(means, a.margin)

    if a.rule == "legacy":
        ok, msgs, rule = legacy_ok, legacy_msgs, f"legacy absolute margin {a.margin}"
    else:
        v = evaluate_gap(means, sds, d_min=a.d_min, margin=a.margin, ci=ci)
        ok, msgs = v["ok"], v["msgs"]
        rule = f"D1 scale-aware: C1 < C2 <= C3 and d(C1->C2) >= {a.d_min}"
        if ci is not None:
            extra.append(f"d(C1→C2) = {d:.3f}, 95% CI {ci[0]:.3f}–{ci[1]:.3f} "
                         f"({S.HARDNESS_BOOTSTRAP} resamples over anchors, not negatives)")
        extra.append(f"legacy absolute-margin rule ({a.margin}): "
                     f"{'PASS' if legacy_ok else 'FAIL'} — {legacy_msgs[0]} "
                     "(reported for the record; not the verdict since v6)")

    payload = {c: {k: v for k, v in per[c].items() if k != "per_anchor"} for c in CONDS}
    print(json.dumps(payload, indent=2))
    print(f"d(C1→C2) = {d:.3f}   gap = {means['C2'] - means['C1']:+.5f}")
    print("HARDNESS GATE:", "PASS" if ok else "FAIL", f"[{rule}]")
    for m in msgs:
        print(" -", m)
    for line in extra:
        print(" *", line)
    append_measurements(per, ok, msgs, margin=a.margin, stamp=stamp, rule=rule, extra=extra)
    _append_run_log(payload, ok, rule, stamp, d, ci, recorded=a.recorded)
    raise SystemExit(0 if ok else 1)


def missing_artifact_message(path, *, hint: str = "") -> str:
    """One actionable message for every 'the Phase 1 artifacts are not here'
    failure, so a fresh Colab VM says what to run instead of raising a bare
    FileNotFoundError out of numpy."""
    steps = hint or (
        "restore a checkpoint:  python -m scripts.hf_artifacts pull --if-configured\n"
        "  or regenerate (all three conditions together, PHASE2_PLAN §2 item 2):\n"
        "    python -m embeded.data.prepare_data --hf --verify-spec\n"
        "    python -m embeded.mining.semantic_index   # GPU encode, ~4 min on a T4\n"
        "    python -m embeded.negatives               # ~15-30 min CPU for BM25")
    return (f"missing {path}\n"
            f"  this stage needs the Phase 1 artifacts and they are not in {S.ARTIFACTS}\n"
            f"  {steps}")


def _append_run_log(per_cond, ok, rule, stamp, d, ci, *, recorded=None) -> None:
    """Machine-readable twin of the report history, so a future rule change can
    be replayed without parsing markdown."""
    p = S.artifact("gate_runs.json")
    runs = []
    if p.exists():
        try:
            runs = json.loads(p.read_text()).get("runs", [])
        except json.JSONDecodeError:
            runs = []
    runs.append({"stamp": stamp, "version": S.VERSION, "rule": rule, "ok": ok,
                 "d_c1_c2": round(d, 4), "ci": ci, "recorded_source": recorded,
                 "per_condition": per_cond})
    p.write_text(json.dumps({"runs": runs}, indent=2))


if __name__ == "__main__":
    main()
