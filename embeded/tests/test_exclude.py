"""The §7.2 adversarial exclusion test, on the fixture corpus:
anchor 16's raw BM25 top hit is its labelled clone 17; rival 20 is a
train/test-overlap fragment; 18 must be promoted to first negative."""
from embeded.mining.bm25_index import BM25Index
from embeded.mining.exclude import exclude_labeled_clones
from embeded.negatives import clean_candidates


def test_direct():
    assert exclude_labeled_clones(0, [0, 5, 6], {0: {5}}) == [6]      # self + clone gone
    assert exclude_labeled_clones(4, [1, 2, 3], {}) == [1, 2, 3]      # untouched


def test_bm25_ranking_trap(fx):
    all_ids = sorted(fx.frags)
    bm25 = BM25Index([fx.frags[i] for i in all_ids])
    raw = [i for i, _ in bm25.topk(16, 3)]
    assert raw[0] == 17, f"fixture broken: top hit should be the clone, got {raw}"
    assert raw[1] == 20, f"fixture broken: rival 20 should outrank 18, got {raw}"


def test_exclusion_promotes_rank(fx):
    all_ids = sorted(fx.frags)
    bm25 = BM25Index([fx.frags[i] for i in all_ids])
    raw = [i for i, _ in bm25.topk(16, 12)]
    stats = {}
    negs = clean_candidates(16, raw, fx.clones, set(fx.corpus), fx.corpus, 3, stats)
    assert negs[0] == 18, f"rank 2 (clone 17 removed, 20 outside corpus) must be promoted, got {negs}"
    assert 17 not in negs and 16 not in negs          # clone + self excluded
    assert 20 not in negs                             # Rule 1: test-overlap never mined
    assert len(negs) == 3 and all(c in set(fx.corpus) for c in negs)
