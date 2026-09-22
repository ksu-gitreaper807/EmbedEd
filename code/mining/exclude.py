"""The true-clone exclusion (FINAL_SPEC §7.2) — standalone, adversarially tested.

SCOPE P1-5: "a leaked clone in a negative set is not a hard negative, it is a
corrupted label". Correction 5: necessary but NOT sufficient — unlabelled
clones survive; that residual rate is what the 50+50 manual audit measures.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Set


def exclude_labeled_clones(anchor_id: int,
                           candidates: Iterable[int],
                           clone_sets: Mapping[int, Set[int]]) -> list[int]:
    """Drop `anchor_id` itself and every candidate that BigCloneBench LABELLED as
    a true clone of the anchor. Preserves candidate order (= rank order)."""
    banned = clone_sets.get(anchor_id, frozenset())
    return [c for c in candidates if c != anchor_id and c not in banned]
