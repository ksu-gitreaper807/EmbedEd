"""Phase 2 training entry point (PHASE2_PLAN §3.2, IMPLEMENTATION_PLAN Phase 2.1).

    python -m embeded.train --condition C1 --seed 13            # one of the nine runs
    python -m embeded.train --condition C1 --seed 13 --smoke    # harness check, not a result
    python -m embeded.train --condition C2 --seed 14 --force    # redo a completed run

One objective for all three conditions (SCOPE P1-4): explicit-triplet margin
loss on the mined negative, `max(0, margin + cos(a,neg) - cos(a,pos))`. The
independent variable IS the specific negative, so the loss consumes it directly
instead of meeting it through in-batch composition. Nothing else differs
between conditions: same anchors, same positives, same cap, same k, same
hyperparameters, same code path — only `triples_C{1,2,3}.jsonl` changes.

The measured T4 recipe is used verbatim (settings: MAX_LEN 512, BATCH 8,
TRAIN_PAIRS_CAP 32k, EPOCHS 1, fp16 autocast, no gradient checkpointing).
Anchors are deliberately re-encoded for each of their k negatives: that is what
`scripts/phase0_throughput.py` measured 13.5 triples/s for, and caching would
invalidate the frozen budget.

Resume protocol matches the rest of the repo: a run whose `metrics.json`
matches the current config fingerprint is skipped, and an interrupted run
continues from its checkpoint instead of silently restarting.

torch is imported inside functions so the offline test suite needs no GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

from . import settings as S
from .hardcheck import missing_artifact_message

TRAINABLE = ("C1", "C2", "C3", "C4")   # C4 = filtered hard negatives (v7)
CONFIG_NAME = "run_config.json"
METRICS_NAME = "metrics.json"          # training metrics
EVAL_CONFIG_NAME = "eval_config.json"    # written by embeded.evaluate
EVAL_METRICS_NAME = "eval_metrics.json"  # written by embeded.evaluate
PREDICTIONS_NAME = "predictions.npz"     # scores + labels, so the table is regenerable
CHECKPOINT_NAME = "checkpoint.pt"
LOG_NAME = "train_log.jsonl"


# --------------------------------------------------------------------- plumbing

def run_dir(condition: str, seed: int, *, smoke: bool = False) -> Path:
    """Smoke runs get their OWN directory (`runs/smoke_<condition>_<seed>`): a resumed
    session may legitimately re-run the notebook's smoke cell after pulling finished
    runs from the HF checkpoint, and a smoke file landing in a real run's directory
    clobbers its metrics — the fingerprint check then demands a spurious retrain
    (the 2026-09-28 Phase 3 incident, same class). Smoke can never touch a real run."""
    name = f"smoke_{condition}_{seed}" if smoke else f"{condition}_{seed}"
    return S.ARTIFACTS / S.RUNS_SUBDIR / name


def set_seed(seed: int) -> None:
    """Deterministic initialisation and sampling order (SCOPE P2-12). The seed
    must never change the data SPLIT — it only orders/shuffles what every
    condition already shares."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:            # offline logic tests run without torch
        pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def artifact_hashes() -> dict:
    """Identity of the artifacts this run consumes, so a later table can prove
    all nine runs read the same triples (PHASE2_PLAN §2 item 2)."""
    out = {}
    for name in ("fragments.jsonl", "mining_summary.json", "corpus_emb.meta.json",
                 *(f"triples_{c}.jsonl" for c in TRAINABLE)):
        p = S.ARTIFACTS / name
        if p.exists():
            out[name] = sha256_file(p)
    return out


def require_training_artifacts(condition: str) -> None:
    need = ["fragments.jsonl", f"triples_{condition}.jsonl", "mining_summary.json"]
    for name in need:
        p = S.ARTIFACTS / name
        if not p.exists():
            raise SystemExit(missing_artifact_message(p))
    summary = json.loads((S.ARTIFACTS / "mining_summary.json").read_text())
    run = summary.get("_run") or {}
    if run.get("version") != S.VERSION:
        raise SystemExit(
            f"mining_summary.json was produced by version {run.get('version')!r}, "
            f"settings.VERSION is {S.VERSION!r} — stale artifacts.\n"
            "  Re-mine all three conditions together:  python -m embeded.negatives --force")
    if condition not in summary:
        raise SystemExit(f"{condition} is not in mining_summary.json "
                         f"(have {sorted(k for k in summary if not k.startswith('_'))}) — "
                         "re-mine all three conditions together")


def check_trainable(condition: str) -> None:
    """C0 is a reference evaluation, never a trained condition — say so before
    any artifact lookup, or the message is about a file that must not exist."""
    if condition not in TRAINABLE:
        raise SystemExit(f"{condition} is not trainable — C0 is the untuned baseline: "
                         "`python -m embeded.evaluate --condition C0`")


def load_triples(condition: str) -> list[tuple[int, int, int]]:
    check_trainable(condition)
    p = S.ARTIFACTS / f"triples_{condition}.jsonl"
    if not p.exists():
        raise SystemExit(missing_artifact_message(p))
    out = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out.append((int(r["anchor"]), int(r["positive"]), int(r["negative"])))
    return out


def run_config(condition: str, seed: int, *, smoke: bool = False) -> dict:
    """Every frozen choice that this run is made of. Its fingerprint decides
    whether a checkpoint/metrics file on disk belongs to the current config."""
    cfg = {
        "version": S.VERSION, "condition": condition, "seed": int(seed),
        "smoke": bool(smoke),
        "model_id": S.MODEL_ID, "pooling": S.POOLING, "option": "A (token-only)",
        "loss": S.LOSS, "triplet_margin": S.TRIPLET_MARGIN, "optimizer": "adamw",
        "lr": S.LR, "weight_decay": S.WEIGHT_DECAY, "warmup_steps": S.WARMUP_STEPS,
        "grad_clip": S.GRAD_CLIP, "batch": S.BATCH, "epochs": S.EPOCHS,
        "max_len": S.MAX_LEN, "amp_dtype": S.AMP_DTYPE,
        "grad_checkpoint": S.GRAD_CHECKPOINT, "k_negatives": S.K_NEGATIVES,
        "train_pairs_cap": S.TRAIN_PAIRS_CAP,
        "threshold_policy": S.THRESHOLD_POLICY, "seed_list": list(S.SEEDS),
        "artifacts": artifact_hashes(),
    }
    if smoke:
        cfg.update({"batch": min(4, S.BATCH), "epochs": S.SMOKE_EPOCHS,
                    "max_len": S.SMOKE_MAX_LEN, "warmup_steps": 0,
                    "amp_dtype": "float32", "max_triples": S.SMOKE_TRIPLES})
    return cfg


def fingerprint(cfg: dict) -> str:
    """Stable id of a run configuration. The fingerprint field itself is
    excluded, so `fingerprint(cfg)` is unchanged by storing the result back into
    cfg — otherwise a checkpoint could never match the config that wrote it."""
    payload = {k: v for k, v in cfg.items() if k != "fingerprint"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     default=str).encode("utf-8")).hexdigest()[:16]


def batches(triples: list, batch_size: int, seed: int, epoch: int) -> list[list]:
    """Deterministic per-epoch order: same seed + epoch => same order in every
    condition, so the only difference between C1/C2/C3 stays the negatives."""
    rng = random.Random(f"{seed}#{epoch}")   # str seed: stable across processes,
                                             # unlike hash((seed, epoch))
    order = list(range(len(triples)))
    rng.shuffle(order)
    return [[triples[i] for i in order[s:s + batch_size]]
            for s in range(0, len(order), batch_size)]


def triplet_loss(anchor, positive, negative, margin: float):
    """Explicit-triplet cosine hinge. Inputs are L2-normalised, so a row-wise
    dot product is the cosine; the same similarity the evaluation uses."""
    import torch

    pos = (anchor * positive).sum(-1)
    neg = (anchor * negative).sum(-1)
    return torch.clamp(margin + neg - pos, min=0.0).mean()


def save_checkpoint(path: Path, model, *, epoch: int, step: int, cfg: dict,
                    loss: float | None = None) -> None:
    import torch

    tmp = path.with_suffix(".pt.part")
    torch.save({"model": model.state_dict(), "epoch": epoch, "step": step,
                "fingerprint": fingerprint(cfg), "loss": loss,
                "torch_rng": torch.get_rng_state()}, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, model, cfg: dict) -> dict | None:
    """Reload weights only if the checkpoint belongs to this exact config —
    resuming across a settings change would mix two experiments."""
    import torch

    if not path.exists():
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("fingerprint") != fingerprint(cfg):
        print(f"[warn] {path.name} was written by a different config "
              f"({ck.get('fingerprint')} != {fingerprint(cfg)}) — starting from the "
              "base model instead of resuming")
        return None
    model.load_state_dict(ck["model"])
    return ck


def verify_checkpoint(path: Path, model, cfg: dict) -> dict:
    """Save/reload round-trip (PHASE2_PLAN §2 item 5): the file on disk must
    reproduce the trained weights exactly, or a resumed/evaluated run is not
    the model that was trained. Runs on every run, not only smoke runs — it is
    cheap and it is the difference between a checkpoint and a hope."""
    import torch

    if not path.exists():
        raise SystemExit(f"missing {path} — no checkpoint was written, so there is "
                         "nothing to evaluate or resume")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("fingerprint") != fingerprint(cfg):
        raise SystemExit(f"{path}: fingerprint {ck.get('fingerprint')!r} != "
                         f"{fingerprint(cfg)!r} — the checkpoint does not belong to this config")
    live = model.state_dict()
    bad = [k for k in live if k not in ck["model"] or not torch.equal(live[k].cpu(), ck["model"][k])]
    if bad:
        raise SystemExit(f"{path}: {len(bad)}/{len(live)} tensors differ after reload "
                         f"(first: {bad[0]}) — checkpointing is broken, do not evaluate this run")
    return {"tensors": len(live), "epoch": ck["epoch"], "step": ck["step"]}


# --------------------------------------------------------------------- training

def train(condition: str, seed: int, *, smoke: bool = False, force: bool = False,
          encoder=None, device: str | None = None, verbose: bool = True) -> dict:
    """One run. `encoder` is injectable so the loop is testable without the
    500 MB pretrained checkpoint; production passes None and loads GraphCodeBERT."""
    import torch

    from .encoder import autocast_ctx, forward_embed, load_encoder

    check_trainable(condition)
    require_training_artifacts(condition)
    out = run_dir(condition, seed, smoke=smoke)
    out.mkdir(parents=True, exist_ok=True)
    cfg = run_config(condition, seed, smoke=smoke)
    cfg["fingerprint"] = fingerprint(cfg)
    cfg_path, metrics_path, ckpt_path = (out / CONFIG_NAME, out / METRICS_NAME,
                                         out / CHECKPOINT_NAME)

    if metrics_path.exists() and not force:
        prev = json.loads(metrics_path.read_text())
        if prev.get("fingerprint") == cfg["fingerprint"]:
            print(f"[cache] {condition} seed={seed} already trained for version "
                  f"{S.VERSION} — skipping (--force to redo)")
            return prev
        print(f"[warn] {metrics_path} is from config {prev.get('fingerprint')}, "
              f"current is {cfg['fingerprint']} — retraining (--force silences this)")

    triples = load_triples(condition)
    cap = cfg.get("max_triples", len(triples))
    if cap < len(triples):
        triples = triples[:cap]
    from .data.prepare_data import load_fragments
    frags = load_fragments()

    if encoder is None:
        tok, model, dev = load_encoder(S.MODEL_ID, device)
    else:
        tok, model, dev = encoder
    device = dev
    model.train()
    if cfg["grad_checkpoint"] and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    ck = None if force else load_checkpoint(ckpt_path, model, cfg)
    start_epoch = int(ck["epoch"]) if ck else 0
    step = int(ck["step"]) if ck else 0
    if ck:
        print(f"[resume] {ckpt_path.name}: continuing at epoch {start_epoch}, step {step}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = max(1, cfg["epochs"] * max(1, -(-len(triples) // cfg["batch"])))
    warmup = int(cfg["warmup_steps"])

    def lr_at(s: int) -> float:
        if warmup and s < warmup:
            return (s + 1) / warmup
        left = max(0, total_steps - s)
        return left / max(1, total_steps - warmup)

    amp = cfg["amp_dtype"]
    use_scaler = amp == "float16" and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    set_seed(seed)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    log = open(out / LOG_NAME, "a", encoding="utf-8")
    t0, losses, n_steps = time.time(), [], 0
    for epoch in range(start_epoch, cfg["epochs"]):
        for batch in batches(triples, cfg["batch"], seed, epoch):
            a = [frags[x] for x, _, _ in batch]
            p = [frags[y] for _, y, _ in batch]
            n = [frags[z] for _, _, z in batch]
            with autocast_ctx(device, amp):
                va = forward_embed(model, tok, a, max_len=cfg["max_len"],
                                   device=device, amp_dtype=amp)
                vp = forward_embed(model, tok, p, max_len=cfg["max_len"],
                                   device=device, amp_dtype=amp)
                vn = forward_embed(model, tok, n, max_len=cfg["max_len"],
                                   device=device, amp_dtype=amp)
                loss = triplet_loss(va, vp, vn, cfg["triplet_margin"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            for g in opt.param_groups:
                g["lr"] = cfg["lr"] * lr_at(step)
            step += 1
            n_steps += 1
            losses.append(float(loss.detach()))
            if verbose and step % 25 == 0:
                el = time.time() - t0
                print(f"  {condition} seed={seed} epoch {epoch} step {step}/{total_steps} "
                      f"loss {losses[-1]:.4f} mean {np.mean(losses[-25:]):.4f} "
                      f"{n_steps / el:.1f} triples/s", flush=True)
            log.write(json.dumps({"epoch": epoch, "step": step, "loss": float(loss.detach()),
                                  "triples_per_s": n_steps / max(time.time() - t0, 1e-9)}) + "\n")
        save_checkpoint(ckpt_path, model, epoch=epoch + 1, step=step, cfg=cfg,
                        loss=float(np.mean(losses)) if losses else None)
    log.close()

    metrics = {
        "condition": condition, "seed": int(seed), "version": S.VERSION,
        "fingerprint": cfg["fingerprint"], "smoke": bool(smoke),
        "loss_final_mean": round(float(np.mean(losses)), 6) if losses else None,
        "loss_last": round(losses[-1], 6) if losses else None,
        "steps": step, "triples": len(triples), "epochs": cfg["epochs"],
        "train_seconds": round(time.time() - t0, 1),
        "triples_per_s": round(n_steps / max(time.time() - t0, 1e-9), 2),
        "checkpoint": str(ckpt_path), "device": str(device),
        "checkpoint_reload": verify_checkpoint(ckpt_path, model, cfg),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    if verbose:
        print(json.dumps(metrics, indent=2))
        print(f"next: python -m embeded.evaluate --condition {condition} --seed {seed}")
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", required=True, choices=TRAINABLE)
    ap.add_argument("--seed", type=int, required=True,
                    help=f"one of {list(S.SEEDS)} for the main runs")
    ap.add_argument("--smoke", action="store_true",
                    help=f"{S.SMOKE_TRIPLES} triples, max_len {S.SMOKE_MAX_LEN}: validates "
                         "the harness only, then runs the evaluation path")
    ap.add_argument("--force", action="store_true", help="redo a completed/resumable run")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    if a.seed not in S.SEEDS and not a.smoke:
        print(f"[warn] seed {a.seed} is not in the frozen list {list(S.SEEDS)} "
              "(SCOPE P2-12) — this run is not one of the nine")
    m = train(a.condition, a.seed, smoke=a.smoke, force=a.force,
              device=a.device, verbose=not a.quiet)
    if a.smoke:
        from .evaluate import main as eval_main
        eval_main(["--condition", a.condition, "--seed", str(a.seed), "--smoke",
                   *(["--force"] if a.force else [])])
    return 0 if m.get("steps") else 1


if __name__ == "__main__":
    raise SystemExit(main())
