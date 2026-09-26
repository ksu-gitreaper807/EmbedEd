"""Phase 0.5 selection rule (scripts/phase0_throughput.suggest) — no GPU needed.

The rule: fastest candidate that (a) did not OOM and (b) respects the MAX_LEN
floor fixed by the Phase 0.4 token-length measurement. Regression for the
2026-09 sweep where plain-fp32 256:32 OOMed the T4 and the old script both died
on the first OOM and would have suggested MAX_LEN=256 (fastest) over 512.
"""
from scripts import phase0_throughput as T


def _r(max_len, batch, rate=None, oom=False, cap=None, amp="float16", ckpt=False):
    rec = {"max_len": max_len, "batch": batch, "amp": amp, "grad_checkpoint": ckpt, "gpu": "T4"}
    if oom:
        return {**rec, "oom": True, "steps_before_oom": 0}
    return {**rec, "steps": 10, "triples_per_sec": rate,
            "cap_for_budget": cap if cap is not None else int(rate * 40 * 60), "peak_mem_gb": 9.5}


def test_suggest_skips_oom_and_short_max_len():
    results = [
        _r(256, 32, rate=40.0),          # fastest overall but below the MAX_LEN floor
        _r(512, 32, oom=True),           # does not fit
        _r(512, 8, rate=9.0),
        _r(512, 16, rate=11.0),          # fastest that fits at 512
    ]
    best = T.suggest(results, min_max_len=512)
    assert best["MAX_LEN"] == 512 and best["BATCH"] == 16
    assert best["TRAIN_PAIRS_CAP"] == int(11.0 * 40 * 60)
    assert best["EPOCHS"] == 1
    assert best["AMP_DTYPE"] == "float16" and best["GRAD_CHECKPOINT"] is False


def test_suggest_caps_at_codexglue_ten_percent():
    best = T.suggest([_r(512, 16, rate=100.0)], min_max_len=512)
    assert best["TRAIN_PAIRS_CAP"] == 90_000


def test_suggest_none_when_nothing_fits():
    assert T.suggest([_r(512, 16, oom=True), _r(256, 8, rate=5.0)], min_max_len=512) is None
    assert T.suggest([], min_max_len=512) is None
