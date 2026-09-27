"""ONE encoder path for every stage (PHASE2_PLAN §3.2: "Run C0 with the same
tokenizer, pooling, truncation, and scoring path as trained models").

Phase 1's corpus embeddings, the C0 baseline, and the trained C1/C2/C3 models
must all mean the same thing by "cosine similarity of mean-pooled embeddings",
otherwise the conditions differ by more than the negative-selection strategy
and the experiment measures the wrong thing (SCOPE P1-1). So the pooling,
truncation and normalisation live here and nowhere else:

    load_encoder()   -> (tokenizer, model, device)      Option A: token-only GraphCodeBERT
    forward_embed()  -> L2-normalised mean-pooled vectors (keeps gradients)
    encode_texts()   -> the same, in batches, as float32 numpy (no gradients)

torch/transformers are imported lazily: the pure-Python logic in `train.py` and
`evaluate.py` stays testable offline, and `pytest -q embeded/tests` still runs
without torch installed.
"""
from __future__ import annotations

from contextlib import nullcontext

import numpy as np

from . import settings as S


def load_encoder(model_id: str | None = None, device: str | None = None):
    """(tokenizer, model, device). Option A (FINAL_SPEC §5, correction 3):
    token-only GraphCodeBERT — no data-flow graphs at inference."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    mid = model_id or S.MODEL_ID
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModel.from_pretrained(mid).to(dev)
    return tok, model, dev


def autocast_ctx(device: str, amp_dtype: str | None):
    """torch.autocast only where it is both supported and measured: the frozen
    recipe is fp16 on the T4 (settings.AMP_DTYPE). CPU autocast is not used —
    the Phase 1 corpus embeddings were produced in fp32 and must stay
    reproducible bit-for-bit."""
    import torch

    if not amp_dtype or amp_dtype == "float32" or not str(device).startswith("cuda"):
        return nullcontext()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[amp_dtype]
    return torch.autocast(device_type="cuda", dtype=dtype)


def mean_pool(last_hidden, attention_mask):
    """Mean over real tokens (settings.POOLING = 'mean'), mask-aware."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)


def forward_embed(model, tokenizer, texts, *, max_len: int, device: str,
                  amp_dtype: str | None = None):
    """(b, hidden) L2-normalised float32 tensor. Gradients are kept: training
    calls this directly, evaluation wraps it in `torch.no_grad()`."""
    import torch

    enc = tokenizer(list(texts), padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    with autocast_ctx(device, amp_dtype):
        hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
        v = mean_pool(hidden, mask)
        return torch.nn.functional.normalize(v.float(), dim=1)


def encode_texts(model, tokenizer, texts, *, max_len: int, device: str,
                 batch_size: int = 32, amp_dtype: str | None = None,
                 progress_every: int | None = None) -> np.ndarray:
    """(n, hidden) float32, L2-normalised, order preserved. This is the exact
    Phase 1 encode path (`mining.semantic_index.encode_corpus` delegates here)."""
    import torch

    texts = list(texts)
    hidden = int(getattr(model.config, "hidden_size", 0)) or 768
    out = np.zeros((len(texts), hidden), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            v = forward_embed(model, tokenizer, texts[s:s + batch_size],
                              max_len=max_len, device=device, amp_dtype=amp_dtype)
            v = v.float().cpu().numpy()
            out[s:s + len(v)] = v
            if progress_every and s % progress_every == 0:
                print(f"encoded {s + len(v)}/{len(texts)}", flush=True)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise cosine for already-normalised rows (a * b).sum(-1)."""
    return (a * b).sum(-1)
