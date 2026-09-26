"""C3 mining index: embed the corpus ONCE with the untouched base model
(FINAL_SPEC §7.1: "using the base model itself"). Exact top-k = one matmul;
9,134 x 768 float32 is 28 MB — FAISS is out of scope (SCOPE P5).

Option A loading (correction 3): token-only GraphCodeBERT, mean pooling,
L2-normalised. Needs torch — run this stage in the Colab notebook.

An index is just `emb @ emb.T`: store the matrix, score with `cosines`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def encode_corpus(texts: list[str], *, model_id: str, max_len: int,
                  batch_size: int = 32, device: str | None = None) -> np.ndarray:
    """(n, 768) float32, L2-normalised. Order matches `texts`."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(dev).eval()
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            enc = tok(texts[s:s + batch_size], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to(dev)
            m = model(**enc).last_hidden_state                       # (b, t, h)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            v = (m * mask).sum(1) / mask.sum(1).clamp(min=1e-6)      # mean pooling
            v = torch.nn.functional.normalize(v, dim=1)
            out[s:s + batch_size] = v.float().cpu().numpy()
            if s % (batch_size * 50) == 0:
                print(f"encoded {s + batch_size}/{len(texts)}")
    return out


def save_corpus_emb(emb: np.ndarray, corpus_ids: list[int], *,
                    model_id: str | None = None, max_len: int | None = None) -> Path:
    """Persist the matrix + a meta file recording what actually produced it
    (the values used for encoding, not whatever settings says at call time)."""
    from .. import settings as S
    p = S.artifact("corpus_emb.npy")
    np.save(p, emb)
    meta = {"model": model_id or S.MODEL_ID, "pooling": S.POOLING,
            "max_len": S.MAX_LEN if max_len is None else max_len,
            "option": "A (token-only; no DFG at inference)", "n": len(emb),
            "corpus_first": corpus_ids[:5], "version": S.VERSION}
    S.artifact("corpus_emb.meta.json").write_text(json.dumps(meta, indent=2))
    return p


def load_corpus_emb() -> tuple[np.ndarray, dict]:
    from .. import settings as S
    emb = np.load(S.artifact("corpus_emb.npy"))
    meta = json.loads(S.artifact("corpus_emb.meta.json").read_text())
    return emb, meta


class SemanticIndex:
    """Top-k by cosine over the stored matrix. `corpus_ids[i]` is the dataset
    fragment index of row i — the translation layer the rest of the pipeline needs."""

    def __init__(self, emb: np.ndarray, corpus_ids: list[int]):
        self.emb = emb
        self.ids = np.asarray(corpus_ids)
        self.pos = {int(c): i for i, c in enumerate(corpus_ids)}

    def topk_for_fragment(self, frag_idx: int, k: int) -> list[tuple[int, float]]:
        i = self.pos[frag_idx]
        sims = self.emb @ self.emb[i]          # rows are unit-norm -> cosine
        kk = min(k + 1, len(sims))
        order = np.argpartition(-sims, kk - 1)[:kk]
        order = order[np.argsort(-sims[order])]
        return [(int(self.ids[j]), float(sims[j])) for j in order if int(j) != i][:k]


def main(argv=None):
    """Encode the train-minus-test corpus and cache it. GPU here; run in Colab."""
    import argparse
    from .. import settings as S
    from ..data.prepare_data import load_fragments
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=S.MAX_LEN)
    ap.add_argument("--model", default=S.MODEL_ID)
    a = ap.parse_args(argv)
    import time
    frags = load_fragments()
    ids = sorted(frags)          # ALL fragments: BM25/semantic indexes are built
    t0 = time.time()             # over these, candidates then filtered to corpus
    emb = encode_corpus([frags[c] for c in ids], model_id=a.model,
                        max_len=a.max_len, batch_size=a.batch)
    p = save_corpus_emb(emb, ids, model_id=a.model, max_len=a.max_len)
    print(f"encoded {emb.shape} in {time.time() - t0:.1f}s -> {p}")
    print("OPTION A CAVEAT (correction 3): token-only encoder; no data-flow graphs at inference.")


if __name__ == "__main__":
    main()
