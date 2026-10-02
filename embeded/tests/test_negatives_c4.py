"""C4 "filtered" negative strategy: the two denoisers and their failure modes.

Every property here is a pre-registered guarantee of `clean_candidates_filtered`
(embeded/negatives.py): labelled-clone exclusion FIRST (identical to C1-C3),
then the Jaccard cut, then the head skip, deterministic order, counted padding
that itself honours the Jaccard cut.
"""
from collections import defaultdict

import pytest

from embeded import settings as S
from embeded.negatives import (_jaccard, clean_candidates, clean_candidates_filtered, mine)

# token sets under embeded.mining.bm25_index.tokenize_code (camel-split, lower):
#   anchor/dup share alpha/bravo/charlie/delta/echo  -> Jaccard ~0.71 (>= 0.40)
#   the "clean*" fragments share almost nothing       -> Jaccard ~0.18 (< 0.40)
ANCHOR = "alphaBravo charlieDelta echo();"
DUP = "alphaBravo charlieDelta foxtrot();"
CLEAN = {i: f"zuluOne{i} yorkTwo{i}(); quinine{i}" for i in range(2, 40)}
STATS0 = dict(jaccard_filtered=0, head_skipped=0, padded=0)


def frags():
    d = {0: ANCHOR, 1: DUP}
    d.update(CLEAN)
    return d


class StubBM25:                  # same ranking for every anchor: 5, 1, then clean asc
    def topk(self, anchor, depth):
        return [(i, 1.0) for i in [5, 1] + list(range(2, 40))][:depth]


def test_jaccard_thresholds_match_the_audit_cut():
    assert _jaccard(ANCHOR, DUP) >= S.C4_JACCARD_MAX      # the poison band
    assert _jaccard(ANCHOR, CLEAN[2]) < S.C4_JACCARD_MAX  # the clean band


def test_c4_excludes_clones_dups_and_skips_head():
    ranked = [5, 1] + list(range(2, 40))   # clone, dup, then 38 clean candidates
    stats = dict(STATS0)
    ok = clean_candidates_filtered(
        0, ranked, clone_sets={0: {5}}, corpus_set=set(ranked),
        corpus_list=sorted(ranked), k=3, stats=stats, frags=frags())
    # 5 clone-excluded, 1 dup-excluded, 2..12 head-skipped (10; 5 was already
    # gone), then the first three survivors
    assert ok == [13, 14, 15]
    assert stats["jaccard_filtered"] == 1 and stats["head_skipped"] == 10
    assert stats["padded"] == 0


def test_c4_pads_only_from_jaccard_legal_candidates():
    ranked = [5, 1, 2]               # clone + dup + one survivor -> head eats it
    stats = dict(STATS0)
    ok = clean_candidates_filtered(
        0, ranked, clone_sets={0: {5}}, corpus_set=set(ranked) | {20, 21, 1},
        corpus_list=[2, 20, 21, 1], k=3, stats=stats, frags=frags())
    # pad top-up runs over corpus_list order: 2, 20, 21 are Jaccard-clean; the
    # filtered dup (1) must NOT re-enter through the back door
    assert ok == [2, 20, 21]
    assert stats["padded"] >= 1 and 1 not in ok
    assert 5 not in ok and 0 not in ok


def test_c4_is_deterministic():
    ranked = [5, 1] + list(range(2, 40))
    args = dict(clone_sets={0: {5}}, corpus_set=set(ranked),
                corpus_list=sorted(ranked), k=4, frags=frags())
    a = clean_candidates_filtered(0, ranked, stats=defaultdict(int), **args)
    b = clean_candidates_filtered(0, ranked, stats=defaultdict(int), **args)
    assert a == b


def test_c4_end_to_end_through_mine_with_stub_ranker():
    stream = [(0, 39)]               # one anchor (0) with labelled positive 39
    # corpus includes the dup (1): outside-corpus candidates are corpus-skipped
    # before the Jaccard cut ever sees them, so it must be IN to be counted
    triples, stats = mine("filtered", stream=stream, corpus_list=list(range(1, 39)),
                          clone_sets={0: {5}}, k=3, all_ids=list(range(40)),
                          bm25=StubBM25(), frags=frags())
    negs = [n for _, _, n in triples]
    assert 5 not in negs and 1 not in negs and 0 not in negs and 39 not in negs
    assert stats["jaccard_filtered"] >= 1 and stats.get("padded", 0) == 0
    assert stats["anchors_short"] == 0


def test_c4_rejects_missing_frags():
    # frags is how the Jaccard cut reads fragment text; without it the strategy
    # must fail loudly rather than silently degrade to plain BM25 (C2)
    with pytest.raises(TypeError):
        mine("filtered", stream=[(0, 39)], corpus_list=[2], clone_sets={},
             k=1, all_ids=[0, 1, 2], bm25=StubBM25())


# --- shared pad-path regressions (the VM crash: the C4 pad code was committed
# inside clean_candidates, where `frags` does not exist; fixtures never pad, so
# only scale exposed it) -------------------------------------------------------

def test_clean_candidates_pad_path_needs_no_frags():
    # every ranked candidate is a labelled clone -> ok == [] -> the pad fills k
    # from the corpus tail, exactly the C1-C3 shared convention (no Jaccard here)
    ranked = [5, 6, 7, 8, 9]
    clone_sets = {0: {5, 6, 7, 8, 9}}
    corpus_list = [2, 3, 4, 10, 11]
    stats = defaultdict(int)
    ok = clean_candidates(0, ranked, clone_sets, corpus_set=set(corpus_list),
                          corpus_list=corpus_list, k=3, stats=stats)
    assert ok == [2, 3, 4] and stats["padded"] == 3


def test_mine_bm25_survives_a_ranking_that_is_mostly_excluded():
    # the exact shape of the VM crash: BM25's top-k is dominated by labelled
    # clones and out-of-corpus ids, so clean_candidates must pad; mine("bm25")
    # passes no frags and none may be required
    class PoisonedBM25:
        def topk(self, anchor, depth):
            return [(i, 1.0) for i in [5, 6, 7] + [50, 51, 52]][:depth]

    triples, stats = mine("bm25", stream=[(0, 9)],
                          corpus_list=[2, 3, 4, 9], clone_sets={0: {5, 6, 7}},
                          k=3, all_ids=list(range(60)), bm25=PoisonedBM25())
    assert [n for _, _, n in triples] == [2, 3, 4]   # all three from the pad
    assert stats["padded"] == 3
