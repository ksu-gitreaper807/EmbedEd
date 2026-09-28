"""Phase 2 runner: metrics, threshold policy, training loop, checkpointing,
resume gating and the results table.

The training loop runs for real — forward, backward, optimizer step, checkpoint
save/reload — against a tiny randomly-initialised transformer and a stub
tokenizer, so the offline suite exercises `embeded.train` / `embeded.evaluate`
without the 500 MB pretrained checkpoint or a GPU. `embed_fn` / `encoder`
injection is the same seam the production path uses; nothing here re-implements
the code under test.
"""
import json
import zlib

import numpy as np
import pytest

from embeded import evaluate as EV
from embeded import settings as S
from embeded import train as TR

torch = pytest.importorskip("torch", reason="the training loop needs torch")


# ------------------------------------------------------------------ tiny encoder

class TinyTokenizer:
    """Whitespace tokenizer with a stable hash: same API surface as the HF
    tokenizer for the fields `embeded.encoder.forward_embed` uses."""

    def __init__(self, vocab_size: int = 32):
        self.vocab_size = vocab_size

    def __call__(self, texts, padding=True, truncation=True, max_length=None,
                 return_tensors="pt"):
        rows = []
        for t in texts:
            ids = [zlib.crc32(w.encode()) % (self.vocab_size - 2) + 1 for w in t.split()]
            rows.append(ids[:max_length] if max_length else ids)
        width = max(len(r) for r in rows)
        ids = [r + [0] * (width - len(r)) for r in rows]
        mask = [[1] * len(r) + [0] * (width - len(r)) for r in rows]
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long)}


def tiny_model(hidden: int = 16):
    from transformers import BertConfig, BertModel
    cfg = BertConfig(vocab_size=32, hidden_size=hidden, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=32,
                     max_position_embeddings=128, max_length=128)
    model = BertModel(cfg)
    model.config.hidden_size = hidden
    return model


@pytest.fixture()
def encoder():
    torch.manual_seed(0)
    return TinyTokenizer(), tiny_model(), "cpu"


@pytest.fixture()
def mined(fx):
    """Phase 1 artifacts for all three conditions on the mini fixture."""
    from embeded import negatives as NG
    emb = np.random.default_rng(3).normal(size=(len(fx.frags), 8)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(S.ARTIFACTS / "corpus_emb.npy", emb)
    (S.ARTIFACTS / "corpus_emb.meta.json").write_text(json.dumps(
        {"model": "tiny-fake", "pooling": "mean", "max_len": 8, "n": len(emb),
         "corpus_first": sorted(fx.frags)[:5], "version": S.VERSION}))
    NG.main(["--strategies", "random,bm25,semantic", "--k", "3"])
    return fx


def fake_embed_fn(dim: int = 8, seed: int = 0):
    """Deterministic, content-sensitive, L2-normalised stand-in encoder."""
    def fn(texts):
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for k, t in enumerate(texts):
            rng = np.random.default_rng(zlib.crc32(t.encode()) % (2 ** 31))
            out[k] = rng.normal(size=dim)
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out
    return fn


# ------------------------------------------------------------------ metrics

def test_precision_recall_f1_and_threshold():
    scores = np.array([0.9, 0.8, 0.4, 0.2])
    labels = np.array([1, 1, 0, 0])
    m = EV.precision_recall_f1(scores, labels, 0.5)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 0, 0)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0

    mixed = EV.precision_recall_f1(scores, np.array([1, 0, 1, 0]), 0.5)
    assert (mixed["tp"], mixed["fp"], mixed["fn"]) == (1, 1, 1)
    assert mixed["precision"] == 0.5 and mixed["recall"] == 0.5 and mixed["f1"] == 0.5
    assert EV.precision_recall_f1(scores, labels, 0.95)["n_predicted_positive"] == 0
    assert EV.precision_recall_f1(scores, labels, 0.95)["f1"] == 0.0


def test_best_f1_threshold_is_exact_and_deterministic():
    rng = np.random.default_rng(11)
    scores = rng.uniform(0.3, 0.99, 500)
    labels = (rng.uniform(size=500) < 0.4).astype(int)
    sel = EV.best_f1_threshold(scores, labels)
    # brute force over every candidate threshold, including the observed scores
    best = max(((EV.precision_recall_f1(scores, labels, t)["f1"], t)
                for t in np.unique(scores)))
    # precision_recall_f1 rounds to 4 dp; the selection itself is full precision
    assert sel["f1"] == pytest.approx(best[0], abs=1e-4)
    assert EV.precision_recall_f1(scores, labels, sel["threshold"])["f1"] == \
        pytest.approx(sel["f1"], abs=1e-4)
    assert EV.best_f1_threshold(scores, labels) == sel          # stable choice
    assert EV.best_f1_threshold(np.array([]), np.array([]))["f1"] == 0.0
    assert EV.best_f1_threshold(scores, np.zeros(500, dtype=int))["f1"] == 0.0


def test_map_at_r_known_cases():
    # perfect ranking: every positive above every negative -> AP@R = 1
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    assert EV.map_at_r(scores, labels) == pytest.approx(1.0)
    # positives at ranks 1 and 3 of 4 (R = 2): k=1 P=1 rel=1, k=2 P=1/2 rel=0
    scores = np.array([0.9, 0.7, 0.5, 0.3])
    labels = np.array([1, 0, 1, 0])
    assert EV.map_at_r(scores, labels) == pytest.approx(0.5)
    # positives at ranks 2 and 4 (R = 2): only k <= R counts, so the rank-4 hit
    # is outside the cutoff -> (1/2)*(1/2) / 2 = 0.25. This is MAP@R, not MAP.
    labels = np.array([0, 1, 0, 1])
    assert EV.map_at_r(scores, labels) == pytest.approx(0.25)
    assert EV.map_at_r(scores, np.zeros(4, dtype=int)) == 0.0


def test_subsample_is_deterministic_and_stratified():
    pairs = [(i, i + 1, i % 2) for i in range(40)]
    a = EV.subsample(pairs, 10, seed=13)
    b = EV.subsample(pairs, 10, seed=13)
    assert a == b and len(a) == 10
    assert {p[2] for p in a} == {0, 1}
    assert EV.subsample(pairs, 100, seed=13) == pairs


# ------------------------------------------------------------------ training

def test_train_smoke_runs_and_checkpoints(mined, encoder):
    m = TR.train("C1", 13, smoke=True, encoder=encoder, verbose=False)
    assert m["steps"] > 0 and m["triples"] > 0
    assert m["loss_final_mean"] is not None and np.isfinite(m["loss_final_mean"])
    assert m["smoke"] is True and m["version"] == S.VERSION
    # checkpoint save/reload round-trip is part of the harness
    assert m["checkpoint_reload"]["tensors"] > 0 and m["checkpoint_reload"]["step"] == m["steps"]

    rd = TR.run_dir("C1", 13, smoke=True)
    assert rd.name == "smoke_C1_13", "smoke must live in its own directory"
    assert (rd / TR.CHECKPOINT_NAME).exists() and (rd / TR.METRICS_NAME).exists()
    cfg = json.loads((rd / TR.CONFIG_NAME).read_text())
    assert cfg["loss"] == S.LOSS and cfg["triplet_margin"] == S.TRIPLET_MARGIN
    assert cfg["max_len"] == S.SMOKE_MAX_LEN and cfg["smoke"] is True
    assert cfg["artifacts"]["triples_C1.jsonl"]                   # identity recorded
    steps = [json.loads(l) for l in (rd / TR.LOG_NAME).read_text().splitlines()]
    assert len(steps) == m["steps"] and all("loss" in s for s in steps)


def test_training_actually_moves_the_weights(mined, encoder):
    """A forward/backward pass that changes nothing is not a training run."""
    before = {k: v.clone() for k, v in encoder[1].state_dict().items()}
    m = TR.train("C2", 13, smoke=True, encoder=encoder, verbose=False)
    after = encoder[1].state_dict()
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert m["steps"] > 0 and changed, "no parameter changed during training"
    ck = torch.load(TR.run_dir("C2", 13, smoke=True) / TR.CHECKPOINT_NAME, weights_only=False)
    assert torch.equal(ck["model"][changed[0]].cpu(), after[changed[0]].cpu())


def test_completed_run_is_skipped_and_stale_config_is_not(mined, encoder, capsys):
    first = TR.train("C3", 13, smoke=True, encoder=encoder, verbose=False)
    again = TR.train("C3", 13, smoke=True, encoder=encoder, verbose=False)
    assert again == first                                  # artifact-gated resume
    assert "[cache]" in capsys.readouterr().out

    # same run, different frozen setting => different fingerprint => retrain
    import unittest.mock as mock
    with mock.patch.object(S, "TRIPLET_MARGIN", S.TRIPLET_MARGIN + 0.05):
        other = TR.train("C3", 13, smoke=True, encoder=encoder, verbose=False)
    assert other["fingerprint"] != first["fingerprint"]
    assert "retraining" in capsys.readouterr().out


def test_resume_only_from_a_matching_checkpoint(mined, encoder, capsys):
    """A checkpoint is only a resume point if it belongs to this exact config."""
    TR.train("C1", 13, smoke=True, encoder=encoder, verbose=False)
    rd = TR.run_dir("C1", 13, smoke=True)
    cfg = json.loads((rd / TR.CONFIG_NAME).read_text())
    ck = torch.load(rd / TR.CHECKPOINT_NAME, weights_only=False)
    fresh = tiny_model()
    got = TR.load_checkpoint(rd / TR.CHECKPOINT_NAME, fresh, cfg)
    assert got and got["epoch"] == 1 and got["step"] > 0
    assert all(torch.equal(ck["model"][k], fresh.state_dict()[k]) for k in ck["model"])

    cfg["triplet_margin"] = cfg["triplet_margin"] + 0.05     # a different experiment
    fresh2 = tiny_model()
    assert TR.load_checkpoint(rd / TR.CHECKPOINT_NAME, fresh2, cfg) is None
    assert "different config" in capsys.readouterr().out
    assert TR.load_checkpoint(rd / "absent.pt", fresh2, cfg) is None


def test_interrupted_run_resumes_instead_of_restarting(monkeypatch, mined, encoder, capsys):
    """Colab disconnects mid-run: continue from the checkpoint, do not silently
    restart (PHASE2_PLAN §5) and do not double the work."""
    monkeypatch.setattr(S, "SMOKE_EPOCHS", 2)
    full = TR.train("C1", 13, smoke=True, encoder=encoder, verbose=False)
    assert full["epochs"] == 2
    rd = TR.run_dir("C1", 13, smoke=True)
    ck = torch.load(rd / TR.CHECKPOINT_NAME, weights_only=False)
    ck["epoch"], ck["step"] = 1, full["steps"] // 2          # as after epoch 1
    torch.save(ck, rd / TR.CHECKPOINT_NAME)
    (rd / TR.METRICS_NAME).unlink()                          # the run never finished

    resumed = TR.train("C1", 13, smoke=True, encoder=encoder, verbose=False)
    assert "[resume]" in capsys.readouterr().out
    assert resumed["steps"] == full["steps"]                 # same total, not 2x
    assert resumed["epochs"] == 2 and resumed["checkpoint_reload"]["step"] == full["steps"]


def test_stale_mining_summary_blocks_training(mined, encoder):
    summary = json.loads((S.ARTIFACTS / "mining_summary.json").read_text())
    summary["_run"]["version"] = "phase01-v0-stale"
    (S.ARTIFACTS / "mining_summary.json").write_text(json.dumps(summary))
    with pytest.raises(SystemExit) as e:
        TR.train("C1", 13, smoke=True, encoder=encoder, verbose=False)
    assert "stale artifacts" in str(e.value) and "--force" in str(e.value)


def test_c0_is_not_trainable(mined, encoder):
    with pytest.raises(SystemExit) as e:
        TR.train("C0", 13, smoke=True, encoder=encoder, verbose=False)
    assert "untuned baseline" in str(e.value)


def test_batches_are_deterministic_and_cover_every_triple():
    triples = [(i, i + 1, i + 2) for i in range(10)]
    a = TR.batches(triples, 4, seed=13, epoch=0)
    b = TR.batches(triples, 4, seed=13, epoch=0)
    c = TR.batches(triples, 4, seed=14, epoch=0)
    assert a == b and a != c                               # seed orders, never splits
    assert sorted(x for batch in a for x in batch) == triples
    assert TR.batches(triples, 4, seed=13, epoch=0) != TR.batches(triples, 4, seed=13, epoch=1)


def test_triplet_loss_hinge():
    a = torch.tensor([[1.0, 0.0]])
    p = torch.tensor([[1.0, 0.0]])            # cos(a,p) = 1
    n = torch.tensor([[0.0, 1.0]])            # cos(a,n) = 0
    assert float(TR.triplet_loss(a, p, n, 0.1)) == 0.0     # satisfied: no loss
    hard = torch.tensor([[1.0, 0.0]])         # cos(a,n) = 1 => loss = margin
    assert float(TR.triplet_loss(a, p, hard, 0.1)) == pytest.approx(0.1)


# ------------------------------------------------------------------ evaluation

def test_evaluate_uses_the_validation_threshold_only(mined):
    """The threshold must come from validation; test may not influence it."""
    emb = fake_embed_fn()
    m = EV.evaluate("C0", 13, smoke=True, embed_fn=emb, verbose=False)
    assert m["threshold_policy"] == S.THRESHOLD_POLICY
    assert m["test"]["n"] > 0 and m["test_f1"] == m["test"]["f1"]
    assert 0.0 <= m["threshold"] <= 1.0 and m["test_map_at_r"] >= 0.0

    # recompute the threshold from the saved validation predictions: identical
    z = np.load(TR.run_dir("C0", 13, smoke=True) / TR.PREDICTIONS_NAME)
    sel = EV.best_f1_threshold(z["valid_scores"], z["valid_labels"])
    assert sel["threshold"] == pytest.approx(m["threshold"])
    test = EV.precision_recall_f1(z["test_scores"], z["test_labels"], m["threshold"])
    assert test["f1"] == pytest.approx(m["test_f1"])
    assert EV.map_at_r(z["test_scores"], z["test_labels"]) == pytest.approx(m["test_map_at_r"])


def test_missing_checkpoint_is_reported_before_the_model_loads(monkeypatch, mined):
    """Order matters: a missing run is a one-line message, not a 500 MB model
    download that ends in 'checkpoint not found'."""
    import embeded.encoder as ENC

    def boom(*a, **k):
        raise AssertionError("load_encoder must not run when the checkpoint is missing")

    monkeypatch.setattr(ENC, "load_encoder", boom)
    with pytest.raises(SystemExit) as e:
        EV.default_embed_fn("C2", 13, smoke=True)
    assert "train it first" in str(e.value)


def test_default_embed_fn_refuses_an_untrained_condition(monkeypatch, mined, encoder):
    """Evaluating C1 without a checkpoint must fail loudly, not silently score
    the untuned model and call it a trained condition."""
    import embeded.encoder as ENC
    monkeypatch.setattr(ENC, "load_encoder", lambda *a, **k: encoder)
    with pytest.raises(SystemExit) as e:
        EV.default_embed_fn("C1", 13, smoke=True)
    assert "embeded.train --condition C1" in str(e.value)


def test_default_embed_fn_loads_the_checkpoint(monkeypatch, mined, encoder):
    """The production path: base model + this run's checkpoint, and a refusal
    when the checkpoint belongs to a different config."""
    import embeded.encoder as ENC
    monkeypatch.setattr(ENC, "load_encoder", lambda *a, **k: encoder)
    TR.train("C1", 13, smoke=True, encoder=encoder, verbose=False)
    trained = {k: v.clone() for k, v in encoder[1].state_dict().items()}

    fresh = tiny_model()
    assert any(not torch.equal(trained[k], v) for k, v in fresh.state_dict().items())
    monkeypatch.setattr(ENC, "load_encoder",
                        lambda *a, **k: (TinyTokenizer(), fresh, "cpu"))
    fn, dev, note = EV.default_embed_fn("C1", 13, smoke=True)
    assert "C1" in note and "step" in note
    loaded = fresh.state_dict()
    # the checkpoint restored the trained weights exactly
    assert all(torch.equal(trained[k], loaded[k]) for k in loaded)
    vecs = fn(["public void a() { }", "public void b() { }"])
    assert vecs.shape == (2, 16)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)   # L2-normalised

    # a checkpoint from another config must not be evaluated as this run
    ck_path = TR.run_dir("C1", 13, smoke=True) / TR.CHECKPOINT_NAME
    ck = torch.load(ck_path, weights_only=False)
    ck["fingerprint"] = "deadbeefdeadbeef"
    torch.save(ck, ck_path)
    with pytest.raises(SystemExit) as e:
        EV.default_embed_fn("C1", 13, smoke=True)
    assert "different config" in str(e.value)


def test_results_table_aggregates_and_excludes_smoke(mined):
    for cond, seed, smoke in (("C0", 13, False), ("C1", 13, False), ("C1", 14, False),
                              ("C2", 13, False), ("C3", 13, True)):
        EV.evaluate(cond, seed, smoke=smoke, embed_fn=fake_embed_fn(seed=seed),
                    verbose=False)
    res = EV.write_results_table()
    assert res["runs"] == 4
    text = S.REPORT_MD.read_text()
    assert text.count(EV.SECTION) == 1
    assert "| C1 | 13 |" in text and "| C1 | 14 |" in text and "| C0 | 13 |" in text
    assert "mean F1" in text and "sd F1" in text
    assert "smoke" in text.lower()                      # the smoke run is called out
    assert "sanity reference" in text.lower() or "Sanity reference" in text
    # rerunning replaces the section instead of stacking it
    EV.write_results_table()
    assert S.REPORT_MD.read_text().count(EV.SECTION) == 1


def test_cli_entrypoints(monkeypatch, mined, encoder, capsys):
    import embeded.encoder as ENC
    monkeypatch.setattr(ENC, "load_encoder", lambda *a, **k: encoder)
    assert EV.main(["--condition", "C0", "--seed", "13", "--smoke"]) == 0
    out = capsys.readouterr().out
    assert '"condition": "C0"' in out
    assert EV.main(["--results-table"]) == 0
    assert "wrote" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        EV.main([])                                     # --condition required
