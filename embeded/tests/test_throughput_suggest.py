"""Phase 0.5 selection rule (scripts/phase0_throughput.suggest) — no GPU needed.

The rule: drop OOM / below-MAX_LEN candidates, keep those within 10% of the
best throughput, take the lowest peak memory. Regressions for the 2026-09-26
T4 sweep: plain-fp32 256:32 OOMed and killed the old script, which would also
have suggested MAX_LEN=256 (fastest) over 512; and 512:16 beat 512:8 by 4% at
almost twice the memory.
"""
from scripts import phase0_throughput as T


def _r(max_len, batch, rate=None, oom=False, peak=9.5, cap=None, amp="float16", ckpt=False):
    rec = {"max_len": max_len, "batch": batch, "amp": amp, "grad_checkpoint": ckpt, "gpu": "T4"}
    if oom:
        return {**rec, "oom": True, "steps_before_oom": 0}
    return {**rec, "steps": 10, "triples_per_sec": rate, "peak_mem_gb": peak,
            "cap_for_budget": cap if cap is not None else int(rate * 40 * 60)}


def test_suggest_skips_oom_and_short_max_len():
    results = [
        _r(256, 32, rate=40.0, peak=3.0),   # fastest overall but below the MAX_LEN floor
        _r(512, 32, oom=True),              # does not fit
        _r(512, 8, rate=9.0, peak=6.7),     # >10% slower than the best -> not "close"
        _r(512, 16, rate=11.0, peak=11.4),
    ]
    best = T.suggest(results, min_max_len=512)
    assert best["MAX_LEN"] == 512 and best["BATCH"] == 16
    assert best["TRAIN_PAIRS_CAP"] == int(11.0 * 40 * 60)
    assert best["EPOCHS"] == 1
    assert best["AMP_DTYPE"] == "float16" and best["GRAD_CHECKPOINT"] is False


def test_suggest_prefers_headroom_within_ten_percent():
    # the real 2026-09-26 T4 numbers
    results = [
        _r(512, 4, rate=11.9, peak=4.37, cap=28_648),
        _r(512, 8, rate=13.5, peak=6.71, cap=32_367),
        _r(512, 16, rate=14.1, peak=11.39, cap=33_951),
        _r(512, 32, oom=True),
    ]
    best = T.suggest(results, min_max_len=512)
    assert best["BATCH"] == 8                     # 13.5 >= 0.9 * 14.1, and 6.7 GB < 11.4 GB
    assert best["TRAIN_PAIRS_CAP"] == 32_367
    assert best["top_triples_per_sec"] == 14.1
    # widen the tolerance and the tiny batch wins on memory; tighten it and speed wins
    assert T.suggest(results, min_max_len=512, within=0.20)["BATCH"] == 4
    assert T.suggest(results, min_max_len=512, within=0.01)["BATCH"] == 16


def test_suggest_caps_at_codexglue_ten_percent():
    best = T.suggest([_r(512, 16, rate=100.0)], min_max_len=512)
    assert best["TRAIN_PAIRS_CAP"] == 90_000


def test_suggest_none_when_nothing_fits():
    assert T.suggest([_r(512, 16, oom=True), _r(256, 8, rate=5.0)], min_max_len=512) is None
    assert T.suggest([], min_max_len=512) is None


def test_record_measurements_upserts_section(tmp_path, monkeypatch):
    from embeded import settings as S
    monkeypatch.setattr(S, "REPORT_MD", tmp_path / "measurements.md")
    results = [_r(512, 8, rate=13.5, peak=6.71, cap=32_367), _r(512, 32, oom=True)]
    best = T.suggest(results, min_max_len=512)
    T.record_measurements(results, best, "cmd-1")
    T.record_measurements(results, best, "cmd-2")          # rerun replaces, never duplicates
    text = S.REPORT_MD.read_text()
    assert text.count("## Phase 0.5") == 1 and "cmd-2" in text and "cmd-1" not in text
    assert "| 512:8 | 13.5 | 6.71 | 32,367 |" in text
    assert "| 512:32 | OOM on T4 | — | — |" in text
