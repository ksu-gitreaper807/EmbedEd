"""demo/app.py headless contract: real pairs only, D6 four-cell coverage, no
training and no encoder work at import, the illustrative label verbatim."""
import importlib
import sys

import pytest


def _make_run(root, condition, seed, *, thr, n, seed_scores):
    import numpy as np
    d = root / f"{condition}_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    scores = np.array([seed_scores(k) for k in range(n)])
    labels = np.array([k % 2 for k in range(n)])
    np.savez(d / "predictions.npz", threshold=np.array(thr), valid_scores=rng.random(4),
             valid_labels=np.array([0, 1, 0, 1]), test_scores=scores, test_labels=labels)


def _world(tmp_path, *, thr0=0.5, thrb=0.5, n=8):
    """pairs_test.tsv + two runs whose scores put pairs into all four D6 cells:
    k even  -> C0 says clone  (score 0.9 >= thr0)
    k < n/2 -> C_best says clone (0.9 >= thrb) else non-clone (0.1 < thrb)"""
    root = tmp_path / "runs"
    _make_run(root, "C0", 13, thr=thr0, n=n, seed_scores=lambda k: 0.9 if k % 2 == 0 else 0.1)
    _make_run(root, "C1", 13, thr=thrb, n=n, seed_scores=lambda k: 0.9 if k < n // 2 else 0.1)
    pairs = [(k, 100 + k, k % 2) for k in range(n)]
    frags = {i: f"frag text {i}" for i in range(200)}
    return root, pairs, frags


def test_check_passes_with_all_four_cells(tmp_path):
    import demo.app as app
    root, pairs, frags = _world(tmp_path)
    assert app.check(root, seed0=13, seedb=13, pairs=pairs, frags=frags) == 0


def test_check_refuses_empty_cell(tmp_path):
    import demo.app as app
    root, pairs, frags = _world(tmp_path)
    # C_best predicts clone for EVERYTHING -> the two C_best-non-clone cells empty
    _make_run(root, "C1", 13, thr=0.05, n=8, seed_scores=lambda k: 0.9)
    with pytest.raises(SystemExit, match="no real pair"):
        app.check(root, seed0=13, seedb=13, pairs=pairs, frags=frags)


def test_rows_align_with_pairs_test_order(tmp_path):
    import demo.app as app
    root, pairs, frags = _world(tmp_path)
    rows, meta = app.collect(root, seed0=13, seedb=13, pairs=pairs, frags=frags)
    assert meta["n_test_pairs"] == len(pairs)
    assert rows[1]["text_a"] == "frag text 1" and rows[1]["gold"] == 1
    assert rows[0]["cell"] == (True, True) and rows[5]["cell"] == (False, False)
    assert rows[6]["cell"] == (True, False) and rows[3]["cell"] == (False, True)
    assert meta["worst_cell"] >= 1 and len(meta["empty"]) == 0


def test_seed_policy_maximises_worst_cell(tmp_path):
    import demo.app as app
    root, pairs, frags = _world(tmp_path)
    # seed 15 for C_best: everything clone -> worse coverage; policy must keep (13, 13)
    _make_run(root, "C1", 15, thr=0.05, n=8, seed_scores=lambda k: 0.9)
    _, meta = app.collect(root, pairs=pairs, frags=frags)      # no seeds pinned
    assert (meta["seed0"], meta["seedb"]) == (13, 13)


def test_missing_run_names_the_pull(tmp_path):
    import demo.app as app
    root, pairs, frags = _world(tmp_path)
    (root / "C1_13" / "predictions.npz").unlink()
    with pytest.raises(SystemExit, match="pull --if-configured"):
        app.check(root, seed0=13, seedb=13, pairs=pairs, frags=frags)


def test_import_does_not_touch_encoder(monkeypatch):
    """Import-time discipline: importing the module must not load any encoder —
    the demo loads models only on user action or inside --check."""
    import embeded.encoder as ENC
    monkeypatch.setattr(ENC, "load_encoder",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("encoder loaded at import")))
    sys.modules.pop("demo.app", None)
    mod = importlib.import_module("demo.app")
    assert mod.ILLUSTRATIVE_LABEL == "illustrative only — not part of the evaluation"
