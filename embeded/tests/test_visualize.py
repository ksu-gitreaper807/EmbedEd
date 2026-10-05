"""The §4 figure module (embeded.visualize): sample determinism, params contract,
injected-render path, and the actionable missing-checkpoint gate."""
import json
import zlib
from pathlib import Path

import numpy as np
import pytest

from embeded import settings as S


def _fake_embed_for(dim=8):
    """Deterministic hash vectors — no encoder, no GPU, no downloads."""
    def make(condition, model_seed=None):   # contract: (condition, model_seed)
        salt = f"panel#{condition}#".encode()

        def fn(texts):
            X = np.zeros((len(texts), dim), dtype=np.float32)
            for i, t in enumerate(texts):
                h = zlib.crc32(salt + t.encode())
                for j in range(dim):
                    X[i, j] = ((h >> (j % 24)) & 0xFF) / 255.0
            return X
        return fn, "deadbeefdeadbeef" if condition != "C0" else "base"
    return make


class _StubReducer:
    """Stands in for umap.UMAP in tests: deterministic linear projection."""
    def fit_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X[:, :2] * 10.0


def _data():
    texts = {i: f"public void m{i}() {{ }}" for i in range(12)}
    frag_func = {0: "1", 1: "1", 2: "1", 3: "1", 4: "2", 5: "2",
                 6: "2", 7: "2", 8: "3", 9: "3", 10: "3", 11: "3"}
    return texts, frag_func


def test_p2_18_constants_exist():
    """SCOPE P2-18: seed + n_neighbors + min_dist recorded in settings.py BEFORE
    the figure exists."""
    assert isinstance(S.UMAP_SEED, int) and S.UMAP_SEED >= 0
    assert S.UMAP_N_NEIGHBORS > 1 and 0.0 <= S.UMAP_MIN_DIST < 1.0
    assert S.UMAP_MODEL_SEED in S.SPRIME_SEEDS


def test_sample_fragment_ids_deterministic_and_shared():
    from embeded.visualize import sample_fragment_ids
    _, frag_func = _data()
    a = sample_fragment_ids(frag_func, 2, version="v-test")
    b = sample_fragment_ids(frag_func, 2, version="v-test")
    c = sample_fragment_ids(frag_func, 2, version="v-other")
    assert a == b == sorted(a)                 # same call, same sample, sorted
    assert a != c                              # version participates (fingerprinted runs)
    assert len(a) == 6                         # 2 per functionality x 3 functionalities


def test_fragment_functions_from_frame_mapping():
    import pandas as pd
    from embeded.visualize import fragment_functions_from_frame
    rows = []
    for fid in (7, 9):
        for i in range(2):
            rows.append({"code1": f"src{fid}_{i}", "code2": f"dst{fid}_{i}",
                         "label": 1, "functionality_id": fid})
    frame = pd.DataFrame(rows)
    frag_func = fragment_functions_from_frame(frame)
    # build_fragments keys texts 0..N-1 by first appearance over code1/code2
    assert len(frag_func) == 8
    assert all(frag_func[i] in ("7", "9") for i in range(8))


def test_panels_writes_figure_and_full_params_contract(fx):
    from embeded.visualize import panels
    texts, frag_func = _data()
    out_dir = fx.tmp / "figures"
    params = panels(["C0", "C1"], model_seed=13, n_per_func=2,
                    embed_fn_for=_fake_embed_for(), reducer_factory=_StubReducer,
                    texts=texts, frag_func=frag_func, holdout=["3"],
                    out_dir=out_dir)
    png, ppath = out_dir / "umap_panels.png", out_dir / "umap_params.json"
    assert png.exists() and png.stat().st_size > 1000
    for key in ("seed", "n_neighbors", "min_dist", "fragment_ids"):   # notebook's contract
        assert key in params, key
    assert params["version"] == S.VERSION and params["model_seed"] == 13
    assert params["conditions"] == ["C0", "C1"]
    assert params["fragment_ids"] == sorted(params["fragment_ids"])
    assert params["holdout"] == ["3"]
    assert params["checkpoints"] == {"C0": "base", "C1": "deadbeefdeadbeef"}
    # the sample is recorded id-exact: this IS the "same sample in every panel" proof
    assert params["fragment_ids"] == sorted(
        json.loads(ppath.read_text())["fragment_ids"])


def test_panels_refuses_unknown_condition_and_missing_checkpoint(fx):
    from embeded.visualize import panels
    texts, frag_func = _data()
    with pytest.raises(SystemExit, match="unknown condition"):
        panels(["C9"], embed_fn_for=_fake_embed_for(), reducer_factory=_StubReducer,
               texts=texts, frag_func=frag_func, holdout=["3"], out_dir=fx.tmp / "f2")
    # real default loading path, C1 resolved FIRST (models resolve before any
    # download/render): no s-prime run dir in the tmp artifacts -> actionable gate
    with pytest.raises(SystemExit, match="sprime_C1_13"):
        panels(["C1"], reducer_factory=_StubReducer,
               texts=texts, frag_func=frag_func, holdout=["3"], out_dir=fx.tmp / "f3")
