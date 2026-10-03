"""Phase 3 generalisation runner (`embeded.generalize`) on a synthetic s′.

The s′ pickle is rebuilt tiny (6 functionalities, 38 pairs) under
`EMBEDED_ARTIFACTS/sprime/`, so census → mine → gate G1s → run → transfer → table
runs fully offline. Production encoders are injected exactly where the module
exposes the seam (embed_fn / encoder / `embeded.encoder.load_encoder`), so nothing
here re-implements the code under test and no GPU or pretrained checkpoint is
needed. Counts asserted are FIXTURE counts, not the real 4,600 / 23.
"""
import json
import pickle

import numpy as np
import pytest

from embeded import settings as S
from embeded import generalize as GZ

torch = pytest.importorskip("torch", reason="the run path trains a real loop")

from embeded.tests.test_train_evaluate import (TinyTokenizer, fake_embed_fn,  # noqa: E402
                                               tiny_model)

# fixture shape: functionalities 101..106, per fid: (clone, non-clone) pairs
FID_COUNTS = {101: 3, 102: 3, 103: 3, 104: 4, 105: 3, 106: 3}
HOLDOUT_BY_RANK = ["104", "101", "102"]     # 8 pairs first; the 6s tie -> lowest ids
HOLDOUT_SORTED = ["101", "102", "104"]
HELD_IN = ["103", "105", "106"]


@pytest.fixture()
def sprime(fx, monkeypatch):
    """fx (mini CodeXGLUE artifacts + tmp ARTIFACTS/REPORT) + a synthetic s′ pickle."""
    base = S.ARTIFACTS / "sprime" / "datasets" / "bcb_v2_sampled_bf"
    base.mkdir(parents=True, exist_ok=True)
    rows = []
    for fid, n in FID_COUNTS.items():
        for i in range(n):
            rows.append({"code1": f"txtA{fid}_{i}", "code2": f"txtB{fid}_{i}",
                         "label": 1, "functionality_id": fid})
            rows.append({"code1": f"txtA{fid}_{i}", "code2": f"txtC{fid}_{i}",
                         "label": 0, "functionality_id": fid})
    frame = pdDataFrame(rows)
    path = base / "data_bcb_v2_sampled_bf.pickle"
    path.write_bytes(pickle.dumps(frame))
    fx.frame, fx.sprime_path = frame, path
    return fx


def pdDataFrame(rows):
    import pandas as pd
    return pd.DataFrame(rows)


def run_census():
    assert GZ.main(["--census", "--holdout-k", str(S.SPRIME_HELDOUT_K)]) == 0


def run_mine(cap=9, k=3):
    return GZ.mine(S.SPRIME_HELDOUT_K, k=k, cap=cap, embed_fn=fake_embed_fn())


def triples(cond):
    return [json.loads(l) for l in open(S.artifact(GZ.TRIPLES_NAME.format(cond=cond)),
                                        encoding="utf-8")]


def clone_sets_from(frame):
    sets = {}
    for a, b, lab in zip(frame["code1"], frame["code2"], frame["label"]):
        if int(lab) == 1:
            sets.setdefault(a, set()).add(b)
            sets.setdefault(b, set()).add(a)
    return sets


def fid_of(text):          # fixture texts are txtA<fid>_...
    return text[4:].split("_")[0]


# ------------------------------------------------------------------ census (D4)

def test_rule_name_is_the_recorded_d4_string():
    assert GZ.RULE_NAME == "max_pair_count_then_lowest_id"


def test_holdout_k_flag_is_guarded(fx):
    with pytest.raises(SystemExit, match="SPRIME_HELDOUT_K"):
        GZ.main(["--census", "--holdout-k", str(S.SPRIME_HELDOUT_K + 2)])


def test_census_counts_holdout_and_report(sprime):
    run_census()
    c = json.loads(S.artifact(GZ.CENSUS_NAME).read_text())
    assert c["rule"] == GZ.RULE_NAME and c["k"] == S.SPRIME_HELDOUT_K
    assert c["counts"] == {str(f): 2 * n for f, n in FID_COUNTS.items()}
    assert c["holdout"] == HOLDOUT_BY_RANK
    assert c["n_pairs"] == sum(2 * n for n in FID_COUNTS.values())
    assert c["label_counts"] == {"0": 19, "1": 19}
    text = S.REPORT_MD.read_text()
    assert GZ.CENSUS_SECTION in text and "104" in text
    # rebuild is deterministic and replaces, never duplicates, the section
    run_census()
    assert S.REPORT_MD.read_text().count(GZ.CENSUS_SECTION) == 1


# ------------------------------------------------------------------ mining (D3=B)

def test_mine_requires_census_first(fx):
    with pytest.raises(SystemExit, match="census"):
        run_mine()


def test_mine_cap_must_be_below_the_main_cap(sprime):
    run_census()
    with pytest.raises(SystemExit, match="TRAIN_PAIRS_CAP"):
        run_mine(cap=S.TRAIN_PAIRS_CAP)


def test_mine_shared_stream_exclusions_and_cap(sprime):
    run_census()
    summary = run_mine(cap=9, k=3)
    frame = sprime.frame
    clones = clone_sets_from(frame)

    t = {c: triples(c) for c in ("C1", "C2", "C3")}
    for c in ("C1", "C2", "C3"):
        assert len(t[c]) == 9, "cap 9 with k=3 = 3 anchors x 3 negatives"
        assert summary[c]["n_triples"] == 9
        assert summary[c]["k"] == 3 and summary[c]["n_anchors"] == 3
        for r in t[c]:
            a, p, n = r["anchor"], r["positive"], r["negative"]
            assert n != a and n not in clones.get(a, set()), "labelled clone leaked"
    # the ONLY difference between conditions is the ranking of negatives:
    # same anchors, same positives, same order
    for c in ("C2", "C3"):
        assert [(r["anchor"], r["positive"]) for r in t["C1"]] == \
               [(r["anchor"], r["positive"]) for r in t[c]]

    frags = {int(json.loads(l)["idx"]): json.loads(l)["text"]
             for l in open(S.artifact(GZ.FRAGS_NAME), encoding="utf-8")}
    split = json.loads(S.artifact(GZ.SPLIT_NAME).read_text())
    train_pairs = split["rows"]["train"]
    train_frag_ids = set()
    rows = GZ.pair_frag_rows_of(frame)
    for i in train_pairs:
        train_frag_ids.update(rows[i][:2])
    for c in ("C1", "C2", "C3"):
        for r in t[c]:
            for x in (r["anchor"], r["positive"], r["negative"]):
                assert x in train_frag_ids, "training triples must stay inside the " \
                                            "held-in train slice (no unseen code, no valid pair)"
    assert split["counts"] == {"train": 16, "valid": 2, "unseen": 20}
    assert split["holdout"] == HOLDOUT_SORTED
    assert split["per_class"]["valid"] == {"0": 1, "1": 1}

    # same mining inputs are cached, not recomputed
    again = run_mine(cap=9, k=3)
    assert again == summary


def test_fragments_are_unique_and_canonical(sprime):
    run_census()
    run_mine()
    frags = [json.loads(l) for l in open(S.artifact(GZ.FRAGS_NAME), encoding="utf-8")]
    texts = [f["text"] for f in frags]
    assert len(texts) == len(set(texts))
    assert sorted(f["idx"] for f in frags) == list(range(len(frags)))
    assert set(texts) == {t for col in ("code1", "code2") for t in sprime.frame[col]}


# ------------------------------------------------------------------ gate G1s

def _craft_gate_artifacts(sprime, c2_hard=True):
    """Fragments + triples + a matching embedding cache, so the G1s verdict is fully
    controlled: C1 negatives ~orthogonal (easy), C2 cos ~0.90, C3 cos ~0.95."""
    n_frags = 400
    with open(S.artifact(GZ.FRAGS_NAME), "w", encoding="utf-8") as fh:
        for i in range(n_frags):
            fh.write(json.dumps({"idx": i, "sha1": f"{i:040d}", "text": f"t{i}"}) + "\n")
    rng = np.random.default_rng(5)

    def neg_vec(anchor, cos):
        perp = np.zeros(8, dtype=np.float32)
        perp[(anchor + 3) % 8] = 1.0
        v = cos * np.eye(8, dtype=np.float32)[anchor] + np.sqrt(max(1 - cos * cos, 0)) * perp
        return v / np.linalg.norm(v)

    emb = rng.normal(size=(n_frags, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    for i in range(8):
        emb[i] = np.eye(8, dtype=np.float32)[i]
        for base, cos in ((100, 0.01), (200, 0.90), (300, 0.95)):
            for j in range(2):
                emb[base + 2 * i + j] = neg_vec(i, cos + rng.uniform(-0.005, 0.005))
    np.save(S.artifact(GZ.EMB_NAME), emb)
    S.artifact(GZ.EMB_META_NAME).write_text(json.dumps(
        {"model": "crafted (test)", "max_len": 8, "n": n_frags, "version": S.VERSION}))

    for cond, base in (("C1", 100), ("C2", 200), ("C3", 300)):
        if cond == "C2" and not c2_hard:
            base = 100                      # C2 as easy as C1 -> the manipulation is gone
        with open(S.artifact(GZ.TRIPLES_NAME.format(cond=cond)), "w", encoding="utf-8") as fh:
            for i in range(8):              # 8 anchors -> bootstrap resampling works
                for j in range(2):
                    fh.write(json.dumps({"anchor": i, "positive": 50 + i,
                                         "negative": base + 2 * i + j}) + "\n")


def test_gate_g1s_pass_and_fail_paths(sprime):
    run_census()
    _craft_gate_artifacts(sprime, c2_hard=True)
    assert GZ.main(["--hardness", "--holdout-k", str(S.SPRIME_HELDOUT_K)]) == 0
    text = S.REPORT_MD.read_text()
    assert GZ.GATE_SECTION in text and "verdict: PASS" in text
    log = json.loads(S.artifact(GZ.GATE_LOG_NAME).read_text())
    assert log["runs"][-1]["ok"] is True and log["runs"][-1]["ci"] is not None

    _craft_gate_artifacts(sprime, c2_hard=False)
    rc = GZ.main(["--hardness", "--holdout-k", str(S.SPRIME_HELDOUT_K)])
    assert rc == 1, "a manipulation that did not take effect must fail the gate"
    text = S.REPORT_MD.read_text()
    assert "verdict: FAIL" in text, "the FAIL is appended as history, never hidden"
    assert text.count(GZ.GATE_SECTION) == 1


def test_gate_g1s_requires_triples(fx):
    with pytest.raises(SystemExit, match="mine"):
        GZ.main(["--hardness", "--holdout-k", str(S.SPRIME_HELDOUT_K)])


# ------------------------------------------------------------------ runs

@pytest.fixture()
def mined(sprime):
    run_census()
    run_mine(cap=9, k=3)
    return sprime


@pytest.fixture()
def tiny_encoder(monkeypatch):
    """Patch the ONE encoder factory so the CLI path (train + eval + transfer C0)
    runs on the tiny model without downloading GraphCodeBERT."""
    torch.manual_seed(0)
    enc = (TinyTokenizer(), tiny_model(), "cpu")
    monkeypatch.setattr("embeded.encoder.load_encoder",
                        lambda model_id=None, device=None: enc)
    return enc


def test_smoke_run_writes_the_full_contract(mined, tiny_encoder):
    rc = GZ.main(["--condition", "C1", "--seed", "13",
                  "--holdout-k", str(S.SPRIME_HELDOUT_K), "--smoke"])
    assert rc == 0
    out = GZ.sprime_run_dir("C1", 13, smoke=True)
    assert out.name == "sprime_smoke_C1_13", "smoke must live in its own directory"
    for name in (GZ.CONFIG_NAME, GZ.METRICS_NAME, GZ.CHECKPOINT_NAME,
                 GZ.EVAL_METRICS_NAME, GZ.PREDICTIONS_NAME):
        assert (out / name).exists(), f"missing {name}"
    cfg = json.loads((out / GZ.CONFIG_NAME).read_text())
    assert cfg["smoke"] is True and cfg["condition"] == "C1"
    assert cfg["holdout_functionalities"] == HOLDOUT_SORTED, \
        "the notebook re-checks the holdout from run_config.json after every run"
    assert cfg["corpus"].startswith("sprime") and cfg["doi"] == S.SPRIME_DOI
    m = json.loads((out / GZ.EVAL_METRICS_NAME).read_text())
    assert m["smoke"] is True and m["kind"] == "generalisation"
    for key in ("f1_seen", "f1_unseen", "delta", "threshold", "holdout_functionalities"):
        assert key in m
    assert m["holdout_functionalities"] == HOLDOUT_SORTED
    assert m["delta"] == pytest.approx(m["f1_seen"] - m["f1_unseen"], abs=1e-4)
    npz = np.load(out / GZ.PREDICTIONS_NAME)
    for key in ("threshold", "valid_scores", "valid_labels", "test_scores", "test_labels"):
        assert key in npz, "same npz keys as Phase 2 (valid = seen, test = unseen)"
    assert len(npz["valid_labels"]) == m["valid"]["n"]
    assert len(npz["test_labels"]) == m["test"]["n"]


def test_run_resume_and_cache_gating(mined, tiny_encoder):
    argv = ["--condition", "C1", "--seed", "13", "--holdout-k", str(S.SPRIME_HELDOUT_K),
            "--smoke"]
    assert GZ.main(argv) == 0
    out = GZ.sprime_run_dir("C1", 13, smoke=True)
    first = (out / GZ.METRICS_NAME).read_text()
    assert GZ.main(argv) == 0                       # cached: fingerprint match
    assert (out / GZ.METRICS_NAME).read_text() == first
    assert GZ.main(argv + ["--force"]) == 0         # forced: retrained
    assert json.loads((out / GZ.METRICS_NAME).read_text())["fingerprint"] == \
        json.loads(first)["fingerprint"]


def test_c0_is_never_trained_on_sprime(mined):
    with pytest.raises(SystemExit, match="transfer"):
        GZ.main(["--condition", "C0", "--seed", "13",
                 "--holdout-k", str(S.SPRIME_HELDOUT_K)])


def test_transfer_c0_and_phase2_checkpoint_guard(mined, tiny_encoder):
    # C0 = the base model, no Phase 2 checkpoint needed
    assert GZ.main(["--condition", "C0", "--seed", "13",
                    "--holdout-k", str(S.SPRIME_HELDOUT_K), "--transfer"]) == 0
    out = GZ.sprime_transfer_dir("C0", 13)
    m = json.loads((out / GZ.EVAL_METRICS_NAME).read_text())
    assert m["kind"] == "transfer" and m["condition"] == "C0"
    assert m["holdout_functionalities"] == HOLDOUT_SORTED
    assert "f1_seen" in m and "f1_unseen" in m and "delta" in m
    # C1..C3 transfer reads a Phase 2 checkpoint that does not exist on this fixture
    with pytest.raises(SystemExit, match="embeded.train"):
        GZ.main(["--condition", "C1", "--seed", "13",
                 "--holdout-k", str(S.SPRIME_HELDOUT_K), "--transfer"])


def test_transfer_transfer_dirs_never_enter_the_delta_table(mined, tiny_encoder):
    GZ.main(["--condition", "C0", "--seed", "13",
             "--holdout-k", str(S.SPRIME_HELDOUT_K), "--transfer"])
    GZ.write_results_table()
    text = S.REPORT_MD.read_text()
    block = text.split(GZ.TABLE_SECTION, 1)[1].split("\n## ", 1)[0]
    per_seed = block.split("### Transfer", 1)[0]
    assert "| C0 |" not in per_seed, "transfer-kind rows stay out of the Δ table"
    assert "### Transfer readings" in block and "| C0 | 13 |" in block


def test_smoke_never_clobbers_a_real_run(mined, tiny_encoder):
    """Incident 2026-09-28: a resumed VM pulls finished runs, re-runs the §7 smoke
    cell, and the smoke files landing in sprime_C1_13 clobbered the real run's
    metrics -> fingerprint mismatch -> spurious retrain. The smoke run must write
    ONLY into its own directory and leave a real run byte-identical."""
    real = GZ.sprime_run_dir("C1", 13)
    real.mkdir(parents=True, exist_ok=True)
    (real / GZ.METRICS_NAME).write_text(json.dumps({"fingerprint": "real", "steps": 2000}))
    (real / "run_config.json").write_text(json.dumps({"fingerprint": "real"}))
    before = {p.name: p.read_bytes() for p in real.iterdir()}

    assert GZ.main(["--condition", "C1", "--seed", "13",
                    "--holdout-k", str(S.SPRIME_HELDOUT_K), "--smoke"]) == 0

    after = {p.name: p.read_bytes() for p in real.iterdir()}
    assert after == before, "the smoke run touched the real run's directory"
    smoke = GZ.sprime_run_dir("C1", 13, smoke=True)
    assert (smoke / GZ.METRICS_NAME).exists() and (smoke / GZ.EVAL_METRICS_NAME).exists()


# ------------------------------------------------------------------ results table

def _fake_run(condition, seed, holdout, *, f1_seen, f1_unseen, smoke=False,
              transfer=False):
    out = (GZ.sprime_transfer_dir(condition, seed) if transfer
           else GZ.sprime_run_dir(condition, seed))
    out.mkdir(parents=True, exist_ok=True)
    m = {"condition": condition, "seed": seed, "version": S.VERSION,
         "smoke": smoke, "kind": "transfer" if transfer else "generalisation",
         "threshold": 0.5, "f1_seen": f1_seen, "f1_unseen": f1_unseen,
         "delta": round(f1_seen - f1_unseen, 4),
         "holdout_functionalities": holdout,
         "valid": {"precision": 0.5, "recall": 0.5, "f1": f1_seen, "n": 10},
         "test": {"precision": 0.4, "recall": 0.4, "f1": f1_unseen, "n": 20},
         "test_f1": f1_unseen, "test_map_at_r": 0.1, "valid_map_at_r": 0.1}
    (out / GZ.EVAL_METRICS_NAME).write_text(json.dumps(m))


def test_results_table_rows_means_and_smoke_exclusion(mined):
    h = HOLDOUT_SORTED
    _fake_run("C1", 13, h, f1_seen=0.50, f1_unseen=0.30)
    _fake_run("C1", 14, h, f1_seen=0.54, f1_unseen=0.34)
    _fake_run("C2", 13, h, f1_seen=0.60, f1_unseen=0.45)
    _fake_run("C3", 13, h, f1_seen=0.70, f1_unseen=0.50, smoke=True)   # excluded
    _fake_run("C1", 13, h, f1_seen=0.40, f1_unseen=0.25, transfer=True)
    res = GZ.write_results_table()
    assert res["runs"] == 3 and res["aggregate"]["C1"]["n"] == 2
    a = res["aggregate"]["C1"]
    assert a["mean_delta"] == pytest.approx(0.20, abs=1e-6)
    assert a["per_seed_delta"] == [0.2, 0.2]
    text = S.REPORT_MD.read_text()
    block = text.split(GZ.TABLE_SECTION, 1)[1].split("\n## ", 1)[0]
    for token in ("F1_seen", "F1_unseen", "Δ", "| C1 | 13 | 0.5000 | 0.3000 | +0.2000 |"):
        assert token in block, token
    assert "| C3 |" not in block.split("### Transfer", 1)[0]
    assert "smoke-run" in block and "NOT comparable" in block
    assert text.count(GZ.TABLE_SECTION) == 1     # rebuild replaces, never duplicates


def test_results_table_refuses_mixed_holdouts(mined):
    h = HOLDOUT_SORTED
    _fake_run("C1", 13, h, f1_seen=0.5, f1_unseen=0.3)
    _fake_run("C2", 13, ["105", "106", "103"], f1_seen=0.5, f1_unseen=0.3)
    with pytest.raises(SystemExit, match="holdout"):
        GZ.write_results_table()


def test_results_table_placeholder_when_nothing_ran(sprime):
    GZ.write_results_table()
    block = S.REPORT_MD.read_text().split(GZ.TABLE_SECTION, 1)[1].split("\n## ", 1)[0]
    assert "placeholder" in block
