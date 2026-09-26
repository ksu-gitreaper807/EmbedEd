"""Resume protocol: every stage skips itself when its artifacts for THIS
settings.VERSION are already present (e.g. restored by `hf_artifacts pull`),
and `--force` overrides. Nothing here needs torch or the network."""
import json

import numpy as np

from embeded import negatives as NG
from embeded import settings as S
from embeded.mining import semantic_index as SI
from embeded.tests.test_semantic_hardcheck import _fake_emb


def test_prepare_data_skips_when_artifacts_present(fx, capsys):
    before = (S.ARTIFACTS / "manifest.json").stat().st_mtime_ns
    fx.P.main(["--dir", str(fx.fdir), "--verify-spec"])          # second run
    out = capsys.readouterr().out
    assert "[cache] artifacts for version" in out and "SPEC CHECK" in out
    assert (S.ARTIFACTS / "manifest.json").stat().st_mtime_ns == before
    # a VERSION bump (or --force) invalidates the cache
    (S.ARTIFACTS / "manifest.json").write_text(json.dumps({"version": "stale"}))
    assert fx.P.cached_manifest() is None
    fx.P.main(["--dir", str(fx.fdir)])
    assert json.loads((S.ARTIFACTS / "manifest.json").read_text())["version"] == S.VERSION
    fx.P.main(["--dir", str(fx.fdir), "--force"])
    assert "[cache]" not in capsys.readouterr().out.split("SPEC CHECK")[-1]


def test_semantic_index_skips_on_matching_meta(fx, capsys):
    n = len(fx.frags)
    np.save(S.ARTIFACTS / "corpus_emb.npy", _fake_emb(n=n))
    meta = {"model": S.MODEL_ID, "pooling": "mean", "max_len": S.MAX_LEN, "n": n,
            "corpus_first": sorted(fx.frags)[:5], "version": S.VERSION}
    (S.ARTIFACTS / "corpus_emb.meta.json").write_text(json.dumps(meta))
    SI.main([])                                   # must return before importing torch
    assert "[cache] corpus_emb.npy already encoded" in capsys.readouterr().out
    assert SI.cached_meta(model_id=S.MODEL_ID, max_len=S.MAX_LEN, n=n) is not None
    assert SI.cached_meta(model_id=S.MODEL_ID, max_len=S.MAX_LEN + 1, n=n) is None   # different max_len
    assert SI.cached_meta(model_id="other/model", max_len=S.MAX_LEN, n=n) is None
    assert SI.cached_meta(model_id=S.MODEL_ID, max_len=S.MAX_LEN, n=n + 1) is None  # corpus changed


def test_negatives_skips_on_matching_summary(fx, capsys):
    NG.main(["--strategies", "random,bm25", "--k", "3"])
    first = (S.ARTIFACTS / "triples_C1.jsonl").stat().st_mtime_ns
    run = json.loads((S.ARTIFACTS / "mining_summary.json").read_text())["_run"]
    assert run["version"] == S.VERSION and run["k"] == 3 and run["strategies"] == ["random", "bm25"]
    capsys.readouterr()
    NG.main(["--strategies", "random,bm25", "--k", "3"])
    assert "[cache] triples for version" in capsys.readouterr().out
    assert (S.ARTIFACTS / "triples_C1.jsonl").stat().st_mtime_ns == first
    # a strategy that was not mined, a different k, or --force all re-mine
    assert NG.cached_summary(["random", "bm25", "semantic"], k=3, n_anchors=run["n_anchors"]) is None
    assert NG.cached_summary(["random"], k=4, n_anchors=run["n_anchors"]) is None
    NG.main(["--strategies", "random", "--k", "3", "--force"])
    assert "[cache]" not in capsys.readouterr().out
