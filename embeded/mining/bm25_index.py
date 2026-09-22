"""C2 mining index: rank_bm25 over the train-split corpus (FINAL_SPEC §7.1).

Tokenisation for code: split identifiers (camel/snake), keep operator/keyword
runs, drop layout. rank_bm25 on ~9k short docs is seconds — no tricks needed.
"""
from __future__ import annotations

import re
from typing import Sequence

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SPLIT_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+")
_OPER = re.compile(r"[{}()\[\];,.]|[=!<>]=?|&&|\|\||\+\+|--|\+=|[-+*/%&|^~]|\.(?=[A-Za-z])")


def tokenize_code(text: str) -> list[str]:
    toks: list[str] = []
    for m in _IDENT.finditer(text):
        toks.extend(w.lower() for w in _SPLIT_CAMEL.findall(m.group(0)))
    toks.extend(m.group(0) for m in _OPER.finditer(text))
    return toks


class BM25Index:
    def __init__(self, doc_texts: Sequence[str]):
        from rank_bm25 import BM25Okapi
        self._corpus = [tokenize_code(t) or ["<empty>"] for t in doc_texts]
        self._bm = BM25Okapi(self._corpus)

    def topk(self, doc_idx: int, k: int) -> list[tuple[int, float]]:
        """(idx, score) desc, EXCLUDING the query doc itself; caller still must
        run exclude_labeled_clones — this class never touches labels."""
        scores = self._bm.get_scores(self._corpus[doc_idx])
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [(i, float(scores[i])) for i in order[: k + 5] if i != doc_idx][:k]
