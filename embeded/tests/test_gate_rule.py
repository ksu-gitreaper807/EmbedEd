"""Gate G1 under decision D1 (scale-aware rule), recorded 2026-09-27.

Covers: the one shared definition of d, the verdict logic, the anchor-resampled
CI (and why it is not the naive one), the recorded 2026-09-26 replay, and the
rule that a later PASS never erases the FAIL that prompted the change.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from embeded import settings as S
from embeded.hardcheck import (append_measurements, bootstrap_d, evaluate,
                               evaluate_gap, load_recorded, measure,
                               standardized_gap)
from embeded.tests.test_semantic_hardcheck import _fake_emb
from scripts import hardness_diagnostics as D

ROOT = Path(__file__).resolve().parents[2]
RECORDED = ROOT / "report" / "gate_g1" / "run_2026-09-26_phase01-v5.json"

# the measured Phase 1 run (report/measurements.md, PROGRESS §2.5)
PHASE1 = {"C1": 0.96068, "C2": 0.97526, "C3": 0.98845}
PHASE1_SD = {"C1": 0.0257, "C2": 0.0216, "C3": 0.0072}


def test_recorded_run_file_is_replayable():
    """The shipped record must contain exactly what hardcheck --recorded needs,
    and its numbers must match the report they were copied from."""
    rec = load_recorded(RECORDED)
    per = rec["per_condition"]
    assert set(per) == {"C1", "C2", "C3"}
    for c in per:
        assert per[c]["mean_cos"] == pytest.approx(PHASE1[c])
        assert per[c]["std"] == pytest.approx(PHASE1_SD[c])
        assert per[c]["n_samples"] == 20_000
    text = (ROOT / "report" / "measurements.md").read_text()
    assert "verdict: FAIL" in text and "0.0146" in text   # the original FAIL stays


def test_standardized_gap_matches_diagnostics_definition():
    """Gate and diagnostics must quote the same d for the same run — the
    diagnostics script imports this function, so assert the arithmetic too."""
    d = standardized_gap(0.96068, 0.0257, 0.97526, 0.0216)
    pooled = math.sqrt((0.0257 ** 2 + 0.0216 ** 2) / 2)
    assert d == pytest.approx((0.97526 - 0.96068) / pooled)
    assert d == pytest.approx(0.614, abs=5e-3)   # PROGRESS §2.5 quotes d ≈ 0.63 from the
    # all-32,000-negative diagnostics run; the gate samples 1,000 anchors instead
    assert standardized_gap(0.5, 0.0, 0.7, 0.0) == 0.0     # zero dispersion => no claim


def test_d1_passes_the_measured_phase1_run_and_legacy_does_not():
    v = evaluate_gap(PHASE1, PHASE1_SD)
    assert v["ok"] and v["d"] == pytest.approx(0.614, abs=5e-3)
    assert v["gap"] == pytest.approx(0.01458, abs=1e-4)
    assert not v["legacy_ok"]                             # 0.02 margin still fails
    assert "margin" in v["legacy_msgs"][0]
    # the legacy rule, asked directly, gives the historical verdict
    ok, msgs = evaluate(PHASE1, margin=S.HARDNESS_MARGIN)
    assert not ok and "gap" in msgs[0]


def test_d1_failure_modes():
    # gap too small on the scale of the space, even though the ordering holds
    v = evaluate_gap({"C1": 0.960, "C2": 0.963, "C3": 0.988},
                     {"C1": 0.026, "C2": 0.022, "C3": 0.007})
    assert not v["ok"] and any("standardised gap" in m for m in v["msgs"])
    # C2 not harder than C1 at all
    v = evaluate_gap({"C1": 0.970, "C2": 0.965, "C3": 0.988},
                     {"C1": 0.02, "C2": 0.02, "C3": 0.01})
    assert not v["ok"] and any("did not take effect" in m for m in v["msgs"])
    # ordering C2 <= C3 violated (the C2-looks-semantic failure)
    v = evaluate_gap({"C1": 0.90, "C2": 0.99, "C3": 0.95},
                     {"C1": 0.02, "C2": 0.01, "C3": 0.01})
    assert not v["ok"] and any("C2 <= C3" in m for m in v["msgs"])
    # a threshold above what was measured must fail, not pass
    v = evaluate_gap(PHASE1, PHASE1_SD, d_min=0.8)
    assert not v["ok"]
    # missing dispersion => no verdict at all (never a silent pass)
    v = evaluate_gap(PHASE1, {"C1": 0.02, "C2": None, "C3": 0.01})
    assert not v["ok"] and "dispersion" in v["msgs"][0]
    v = evaluate_gap({"C1": 0.96}, {"C1": 0.02})
    assert not v["ok"] and "missing conditions" in v["msgs"][0]


def test_bootstrap_ci_resamples_anchors_not_negatives():
    """k = 20 correlated negatives per anchor: resampling them individually
    would claim ~sqrt(k) times more precision than the data supports."""
    rng = np.random.default_rng(7)
    n_anchors, k = 60, 20
    a_mean = rng.normal(0.960, 0.03, n_anchors)
    per_anchor = {"C1": {}, "C2": {}}
    for i, mu in enumerate(a_mean):
        n1 = rng.normal(mu, 0.01, k)
        n2 = rng.normal(mu + 0.015, 0.01, k)
        per_anchor["C1"][i] = float(n1.mean())
        per_anchor["C2"][i] = float(n2.mean())
    ci = bootstrap_d(per_anchor, n_boot=400, seed=S.SEED)
    assert ci is not None and ci[0] < ci[1]
    assert bootstrap_d({"C1": {i: 0.5 for i in range(3)}, "C2": {i: 0.6 for i in range(3)}}) is None
    assert bootstrap_d({"C2": {}}) is None
    # a CI built by resampling individual negatives would be far narrower
    all1 = np.concatenate([rng.normal(per_anchor["C1"][i], 0.01, k) for i in range(n_anchors)])
    all2 = np.concatenate([rng.normal(per_anchor["C2"][i], 0.01, k) for i in range(n_anchors)])
    naive = []
    for _ in range(400):
        i1 = rng.integers(0, len(all1), len(all1))
        i2 = rng.integers(0, len(all2), len(all2))
        naive.append(standardized_gap(all1[i1].mean(), all1[i1].std(),
                                      all2[i2].mean(), all2[i2].std()))
    assert (ci[1] - ci[0]) > (np.quantile(naive, 0.975) - np.quantile(naive, 0.025))


def test_recorded_replay_end_to_end(fx, capsys):
    """`hardcheck --recorded` re-judges the 2026-09-26 run under the rule now in
    force, without artifacts, and appends to the report history."""
    from embeded import hardcheck as H

    # the historical FAIL, as recorded on 2026-09-26, is already in the report
    per = {c: {"mean_cos": PHASE1[c], "std": PHASE1_SD[c], "n_samples": 20_000}
           for c in ("C1", "C2", "C3")}
    append_measurements(per, False, ["C1→C2 gap +0.0146 < margin 0.02"],
                        margin=0.02, stamp="2026-09-26 (original, absolute margin)")

    with pytest.raises(SystemExit) as e:
        H.main(["--recorded", str(RECORDED), "--stamp", "2026-09-27 re-evaluation"])
    assert e.value.code == 0                              # PASS under D1
    out = capsys.readouterr().out
    assert "HARDNESS GATE: PASS" in out and "d(C1→C2) = 0.614" in out
    assert "legacy absolute-margin rule (0.02): FAIL" in out
    assert "no new measurement" in out

    t = S.REPORT_MD.read_text()
    assert t.count("## Gate G1 — hardness check") == 1
    assert t.index("2026-09-26 (original") < t.index("2026-09-27 re-evaluation")
    assert "verdict: FAIL" in t and "verdict: PASS" in t
    assert "rule: D1 scale-aware" in t
    # the machine-readable twin exists and carries both verdicts' inputs
    runs = json.loads((S.ARTIFACTS / "gate_runs.json").read_text())["runs"]
    assert runs[-1]["ok"] is True and runs[-1]["d_c1_c2"] == pytest.approx(0.614, abs=5e-3)
    assert runs[-1]["recorded_source"] == str(RECORDED)


def test_legacy_rule_still_selectable(fx, capsys):
    from embeded import hardcheck as H
    with pytest.raises(SystemExit) as e:
        H.main(["--recorded", str(RECORDED), "--rule", "legacy",
                "--stamp", "legacy replay"])
    assert e.value.code == 1                              # the old verdict, unchanged
    assert "HARDNESS GATE: FAIL" in capsys.readouterr().out


def test_measure_reports_per_anchor_for_the_bootstrap(fx):
    from embeded import negatives as NG
    NG.main(["--strategies", "random,bm25", "--k", "3"])
    emb = _fake_emb()
    row_of = {c: i for i, c in enumerate(sorted(fx.frags))}
    files = {c: str(S.ARTIFACTS / f"triples_{c}.jsonl") for c in ("C1", "C2")}
    out = measure(files, emb, row_of, n_anchors=1000, seed=S.SEED)
    for c in ("C1", "C2"):
        assert {"mean_cos", "std", "n_samples", "n_anchors", "per_anchor"} <= set(out[c])
        pa = out[c]["per_anchor"]
        assert len(pa) == out[c]["n_anchors"]
        # per-anchor means must average back to the reported mean (3 negatives each)
        assert float(np.mean(list(pa.values()))) == pytest.approx(out[c]["mean_cos"], abs=1e-3)


def test_diagnostics_and_gate_agree_on_d(fx):
    """The diagnostics table and the gate must not drift: same function."""
    from embeded import negatives as NG
    emb = _fake_emb()
    all_ids = sorted(fx.frags)
    np.save(S.ARTIFACTS / "corpus_emb.npy", emb)
    (S.ARTIFACTS / "corpus_emb.meta.json").write_text(json.dumps(
        {"model": "fake", "pooling": "mean", "max_len": 8, "n": len(emb),
         "corpus_first": all_ids[:5], "version": S.VERSION}))
    NG.main(["--strategies", "random,bm25,semantic", "--k", "3"])
    D.main(["--examples", "0"])
    res = json.loads((S.ARTIFACTS / "hardness_diagnostics.json").read_text())
    cd = res["conditions"]
    for c in ("C2", "C3"):
        expect = standardized_gap(cd["C1"]["mean_cos"], cd["C1"]["sd"],
                                  cd[c]["mean_cos"], cd[c]["sd"])
        assert cd[c]["d_vs_C1"] == pytest.approx(round(expect, 3))
    text = S.REPORT_MD.read_text()
    assert "Rule in force" in text and str(S.HARDNESS_D_MIN) in text


def test_missing_artifacts_say_what_to_run(fx, capsys):
    from embeded import hardcheck as H
    (S.ARTIFACTS / "triples_C1.jsonl").unlink(missing_ok=True)
    with pytest.raises(SystemExit) as e:
        H.main([])
    msg = str(e.value)
    assert "triples_C1.jsonl" in msg and "hf_artifacts pull" in msg
    assert "embeded.negatives" in msg
    with pytest.raises(SystemExit) as e:
        D.main(["--examples", "0"])
    assert "corpus_emb.npy" in str(e.value)
