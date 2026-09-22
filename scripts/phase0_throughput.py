"""Phase 0.5 — measured throughput; replaces the [illustrative] TRAIN_PAIRS_CAP
and EPOCHS in code/settings.py (SCOPE P2-13: fix in week 1 from a measured test,
then hold constant). Run on the Colab T4.

    python -m scripts.phase0_throughput --candidates 256:32 256:16 512:16

Each candidate = max_len:batch_size. Measures triples-seconds via a mini
TripletLoss run (3 encoder streams — anchor/positive/negative — like training),
prints recommended settings for a ~40-min budget per run (3 seeds x 3
conditions must fit Colab sessions with headroom).
"""
from __future__ import annotations

import argparse
import json
import time


def bench(max_len: int, batch: int, minutes: float, run_dir: str = ".",
          budget_min: float = 40.0) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    from code import settings as S
    from code.data.prepare_data import load_fragments

    frag = S.ARTIFACTS / "fragments.jsonl"
    if not frag.exists():
        raise SystemExit(
            f"{frag} not found — Phase 0.2 (data preparation) has not run.\n"
            "Run it first:  python -m code.data.prepare_data --hf --verify-spec\n"
            "(notebook: the 'Phase 0.2' cell). The throughput test sizes training on\n"
            "real fragment lengths, so it needs the prepared fragments on Drive.")
    texts = list(load_fragments().values())[:4000] or ["int x = 1;"] * 4000
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(S.MODEL_ID)
    model = AutoModel.from_pretrained(S.MODEL_ID).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=S.LR)

    def enc(ids, mask):
        h = model(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        return F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-6), dim=1)

    model.train()
    step = 0
    t0 = time.time()
    limit = minutes * 60
    while time.time() - t0 < limit:
        idx = np.random.randint(0, len(texts), size=(batch, 3))
        flat = [texts[i] for row in idx for i in row]
        e = tok(flat, padding=True, truncation=True, max_length=max_len,
                return_tensors="pt").to(dev)
        n = e["input_ids"].shape[0] // 3
        a = enc(e["input_ids"][0:n], e["attention_mask"][0:n])
        p = enc(e["input_ids"][n:2 * n], e["attention_mask"][n:2 * n])
        q = enc(e["input_ids"][2 * n:], e["attention_mask"][2 * n:])
        loss = F.triplet_margin_loss(a, p, q, margin=0.2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 20 == 0:
            print(f"  step {step} loss {loss.item():.3f} "
                  f"{step / (time.time() - t0):.2f} batch/s")
    elapsed = time.time() - t0
    pairs_per_step = batch
    rate = pairs_per_step * step / elapsed                       # pairs per second
    rec = {"max_len": max_len, "batch": batch, "pairs_per_sec": round(rate, 1),
            "cap_for_budget": int(rate * budget_min * 60),
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "note": "1 epoch at cap_for_budget triples ~= budget_min minutes/run"}
    print(json.dumps(rec, indent=2))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", default=["256:32", "256:16", "512:16"])
    ap.add_argument("--minutes", type=float, default=1.0)
    ap.add_argument("--budget", type=float, default=40.0)
    a = ap.parse_args(argv)
    results = []
    for cand in a.candidates:
        max_len, batch = (int(x) for x in cand.split(":"))
        print(f"== {cand} ==")
        results.append(bench(max_len, batch, a.minutes, budget_min=a.budget))
    best = max(results, key=lambda r: r["pairs_per_sec"])
    print("\nSUGGESTED settings.py edits (verify token-length p99 supports max_len):")
    print(f"  MAX_LEN = {best['max_len']}")
    print(f"  TRAIN_PAIRS_CAP = {min(best['cap_for_budget'], 90_000)}")
    print(f"  BATCH = {best['batch']}")
    print(f"  EPOCHS = 1")


if __name__ == "__main__":
    main()
