"""Phase 0.5 — measured throughput; replaces the [illustrative] TRAIN_PAIRS_CAP,
BATCH and EPOCHS in embeded/settings.py (SCOPE P2-13: fix in week 1 from a
measured test, then hold constant). Run on the Colab T4.

    python -m scripts.phase0_throughput --candidates 512:4 512:8 512:16 512:32

Each candidate = max_len:batch, where batch counts TRIPLES — the encoder sees 3x
that many sequences per step (anchor / positive / negative streams, like
training). Measures triples/second with a mini TripletLoss run using the SAME
memory recipe train.py must use (settings.AMP_DTYPE / GRAD_CHECKPOINT), and
prints recommended settings for a ~40-min budget per run (3 seeds x 3
conditions must fit Colab sessions with headroom).

- A candidate that runs out of GPU memory is reported as `oom` and skipped; the
  sweep continues (measured 2026-09: plain fp32 at 256x32 = 96 sequences already
  OOMs the 14.5 GB T4, so a single OOM must not kill the whole test).
- Candidates shorter than settings.MAX_LEN are benchmarked if asked for but never
  suggested: the Phase 0.4 token-length measurement fixes MAX_LEN, not this test.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

# Must be set before torch initialises CUDA: fewer fragmentation OOMs near the limit.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_AMP = {"float16": "float16", "fp16": "float16", "bfloat16": "bfloat16", "bf16": "bfloat16",
        "float32": "float32", "fp32": "float32", "none": "float32"}


def bench(max_len: int, batch: int, minutes: float, run_dir: str = ".",
          budget_min: float = 40.0, amp_dtype: str | None = None,
          grad_checkpoint: bool | None = None) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    from embeded import settings as S
    from embeded.data.prepare_data import load_fragments

    frag = S.ARTIFACTS / "fragments.jsonl"
    if not frag.exists():
        raise SystemExit(
            f"{frag} not found — Phase 0.2 (data preparation) has not run.\n"
            "Run it first:  python -m embeded.data.prepare_data --hf --verify-spec\n"
            "(notebook: the 'Phase 0.2' cell). The throughput test sizes training on\n"
            "real fragment lengths, so it needs the prepared fragments.")
    texts = list(load_fragments().values())[:4000] or ["int x = 1;"] * 4000
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"

    amp = _AMP[(amp_dtype or S.AMP_DTYPE).lower()]
    use_ckpt = S.GRAD_CHECKPOINT if grad_checkpoint is None else grad_checkpoint
    use_amp = dev == "cuda" and amp != "float32"
    amp_torch = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(amp)

    tok = AutoTokenizer.from_pretrained(S.MODEL_ID)
    # mean pooling never touches the pooler head; leaving it out also silences the
    # "newly initialized: pooler" warning that reads like a broken checkpoint
    model = AutoModel.from_pretrained(S.MODEL_ID, add_pooling_layer=False).to(dev)
    if use_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(model.parameters(), lr=S.LR)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp == "float16")

    def enc(ids, mask):
        h = model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        m = mask.unsqueeze(-1).float()
        return F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-6), dim=1)

    base = {"max_len": max_len, "batch": batch, "amp": amp, "grad_checkpoint": use_ckpt, "gpu": gpu}
    rng = np.random.default_rng(S.SEED)
    model.train()
    step = 0
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    limit = minutes * 60
    try:
        while time.time() - t0 < limit:
            idx = rng.integers(0, len(texts), size=(batch, 3))
            flat = [texts[i] for row in idx for i in row]
            e = tok(flat, padding=True, truncation=True, max_length=max_len,
                    return_tensors="pt").to(dev)
            n = e["input_ids"].shape[0] // 3
            with torch.autocast(device_type="cuda", dtype=amp_torch, enabled=use_amp):
                a = enc(e["input_ids"][0:n], e["attention_mask"][0:n])
                p = enc(e["input_ids"][n:2 * n], e["attention_mask"][n:2 * n])
                q = enc(e["input_ids"][2 * n:], e["attention_mask"][2 * n:])
                loss = F.triplet_margin_loss(a, p, q, margin=0.2)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % 20 == 0:
                print(f"  step {step} loss {loss.item():.3f} "
                      f"{step / (time.time() - t0):.2f} batch/s")
    except torch.cuda.OutOfMemoryError:
        rec = {**base, "oom": True, "steps_before_oom": step,
               "note": f"does not fit on {gpu} — skipped; try a smaller batch or --grad-checkpoint"}
        print(json.dumps(rec, indent=2))
        return rec
    finally:
        del model, opt
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    rate = batch * step / elapsed                                 # triples per second
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev == "cuda" else 0.0
    rec = {**base, "steps": step, "triples_per_sec": round(rate, 1),
           "cap_for_budget": int(rate * budget_min * 60),
           "peak_mem_gb": round(peak, 2),
           "note": "1 epoch at cap_for_budget triples ~= budget_min minutes/run"}
    print(json.dumps(rec, indent=2))
    return rec


def suggest(results: list[dict], min_max_len: int, cap_ceiling: int = 90_000,
            within: float = 0.10) -> dict | None:
    """Pick the training batch from a sweep. Pure, so the rule is unit-testable.

    1. drop candidates that OOMed or sit below the MAX_LEN floor fixed by the
       Phase 0.4 token-length measurement;
    2. keep those within `within` (10%) of the best throughput;
    3. of those, take the one with the LOWEST peak memory.

    Step 3 is deliberate: near GPU saturation a bigger batch buys a few percent
    at the cost of the headroom a 40-minute run needs (T4 sweep 2026-09-26:
    512:8 = 13.5 triples/s @ 6.7 GB vs 512:16 = 14.1 @ 11.4 of 14.5 GB).
    `cap_ceiling` keeps the subset at or below CodeXGLUE's own ~10% (~90k) slice.
    """
    eligible = [r for r in results
                if not r.get("oom") and r["max_len"] >= min_max_len and r.get("triples_per_sec")]
    if not eligible:
        return None
    top = max(r["triples_per_sec"] for r in eligible)
    close = [r for r in eligible if r["triples_per_sec"] >= (1 - within) * top]
    best = min(close, key=lambda r: (r.get("peak_mem_gb") or 0, -r["triples_per_sec"]))
    return {"MAX_LEN": best["max_len"], "BATCH": best["batch"],
            "TRAIN_PAIRS_CAP": min(best["cap_for_budget"], cap_ceiling), "EPOCHS": 1,
            "AMP_DTYPE": best["amp"], "GRAD_CHECKPOINT": best["grad_checkpoint"],
            "triples_per_sec": best["triples_per_sec"], "peak_mem_gb": best.get("peak_mem_gb"),
            "top_triples_per_sec": top}


def record_measurements(results: list[dict], best: dict | None, cmd: str) -> None:
    """Upsert a '## Phase 0.5' section in report/measurements.md (same convention
    as prepare_data / fetch_sprime / hardcheck: the block is data, never a regex
    template)."""
    import re
    from embeded import settings as S

    lines = ["## Phase 0.5 — throughput (measured, fixes BATCH / TRAIN_PAIRS_CAP / EPOCHS)", "",
             f"generated by `{cmd}` — version `{S.VERSION}`, "
             f"recipe `AMP_DTYPE={S.AMP_DTYPE}`, `GRAD_CHECKPOINT={S.GRAD_CHECKPOINT}`", "",
             "| max_len:batch (triples) | triples/s | peak GB | cap for one 40-min epoch |",
             "|---|---|---|---|"]
    for r in results:
        if r.get("oom"):
            lines.append(f"| {r['max_len']}:{r['batch']} | OOM on {r['gpu']} | — | — |")
        else:
            lines.append(f"| {r['max_len']}:{r['batch']} | {r['triples_per_sec']} | "
                         f"{r['peak_mem_gb']} | {r['cap_for_budget']:,} |")
    lines.append("")
    if best is None:
        lines.append(f"**No candidate at `MAX_LEN = {S.MAX_LEN}` fits** — rerun with smaller batches "
                     "or `--grad-checkpoint`; do not lower MAX_LEN.")
    else:
        lines.append(f"Suggested: **BATCH = {best['BATCH']}, TRAIN_PAIRS_CAP = {best['TRAIN_PAIRS_CAP']:,}, "
                     f"EPOCHS = 1** (rule: fastest-within-10% with the lowest peak memory; "
                     f"{best['triples_per_sec']} vs top {best['top_triples_per_sec']} triples/s). "
                     f"`settings.py` at run time: BATCH = {S.BATCH}, TRAIN_PAIRS_CAP = {S.TRAIN_PAIRS_CAP:,}, "
                     f"EPOCHS = {S.EPOCHS}.")
    lines.append("")
    block = "\n".join(lines)
    S.REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    if S.REPORT_MD.exists():
        old = S.REPORT_MD.read_text()
        if "## Phase 0.5" in old:
            old = re.sub(r"## Phase 0\.5 — throughput.*?(?=\n## |\Z)", lambda _m: block + "\n", old, flags=re.S)
        else:
            old = old.rstrip("\n") + "\n\n" + block + "\n"
        S.REPORT_MD.write_text(old)
    else:
        S.REPORT_MD.write_text("# Measurements\n\n" + block + "\n")
    print(f"wrote {S.REPORT_MD}")


def main(argv=None):
    from embeded import settings as S

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", nargs="+", default=["512:4", "512:8", "512:16", "512:32"],
                    help="max_len:batch pairs; batch counts triples (3x sequences per step)")
    ap.add_argument("--minutes", type=float, default=1.0, help="timed window per candidate")
    ap.add_argument("--budget", type=float, default=40.0, help="minutes per training run to size the cap for")
    ap.add_argument("--amp", choices=sorted(set(_AMP)), default=None,
                    help=f"autocast dtype (default settings.AMP_DTYPE={S.AMP_DTYPE})")
    ap.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=None,
                    help=f"activation checkpointing (default settings.GRAD_CHECKPOINT={S.GRAD_CHECKPOINT})")
    a = ap.parse_args(argv)

    results = []
    for cand in a.candidates:
        max_len, batch = (int(x) for x in cand.split(":"))
        print(f"== {cand} ==")
        results.append(bench(max_len, batch, a.minutes, budget_min=a.budget,
                             amp_dtype=a.amp, grad_checkpoint=a.grad_checkpoint))
        gc.collect()

    print("\nSUMMARY (triples/s | peak GB):")
    for r in results:
        tag = "OOM" if r.get("oom") else f"{r['triples_per_sec']:>6} | {r['peak_mem_gb']:.1f}"
        print(f"  {r['max_len']}:{r['batch']:<3} {tag}")

    best = suggest(results, S.MAX_LEN)
    cmd = "python -m scripts.phase0_throughput --candidates " + " ".join(a.candidates) + f" --minutes {a.minutes:g}"
    record_measurements(results, best, cmd)
    if best is None:
        print(f"\nNO candidate at max_len >= settings.MAX_LEN={S.MAX_LEN} fits on this GPU. "
              "Re-run with smaller batches (e.g. 512:2 512:4) and/or --grad-checkpoint; "
              "do not lower MAX_LEN — it is fixed by the token-length measurement.")
        return
    print(f"\nSUGGESTED settings.py edits (at MAX_LEN={S.MAX_LEN}: lowest peak memory among candidates "
          f"within 10% of the fastest; {best['triples_per_sec']} triples/s vs top "
          f"{best['top_triples_per_sec']}, peak {best['peak_mem_gb']} GB):")
    for key in ("MAX_LEN", "TRAIN_PAIRS_CAP", "BATCH", "EPOCHS"):
        print(f"  {key} = {best[key]}")
    for key in ("AMP_DTYPE", "GRAD_CHECKPOINT"):
        if best[key] != getattr(S, key):
            print(f"  {key} = {best[key]!r}   # differs from settings — commit this too")


if __name__ == "__main__":
    main()
