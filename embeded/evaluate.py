"""Phase 2 evaluation: validation-selected threshold, then test once
(PHASE2_PLAN §3.3, SCOPE P2-14/15).

    python -m embeded.evaluate --condition C0                 # untuned baseline
    python -m embeded.evaluate --condition C1 --seed 13       # one trained run
    python -m embeded.evaluate --results-table                # rebuild the main table

Protocol, frozen:

* score = cosine of mean-pooled embeddings — the same path for C0 and for every
  trained condition (`embeded.encoder`), so the conditions differ only in the
  negatives they were trained on;
* the F1 decision threshold is chosen on the **validation** split per
  condition/model (`settings.THRESHOLD_POLICY`) and then applied unchanged to
  test. Test scores never feed back into it;
* primary metric F1 with precision/recall alongside; secondary MAP@R, which is
  threshold-free;
* predictions, thresholds and configuration are kept so the table can be
  regenerated without re-running anything.

The metric functions are pure numpy and unit-tested; only the encoder needs
torch, and it is injectable (`embed_fn`) so the whole protocol runs offline.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np

from . import settings as S
from .hardcheck import missing_artifact_message
from .train import (CHECKPOINT_NAME, EVAL_CONFIG_NAME, EVAL_METRICS_NAME,
                    PREDICTIONS_NAME, fingerprint, load_checkpoint, run_config,
                    run_dir, set_seed)

CONDITIONS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")
SECTION = "## Phase 2 — main results"
# Published CodeXGLUE fine-tuned-CodeBERT reference for the sanity check
# (SCOPE P2-16, PHASE2_PLAN §3.3): approximate, and measured under a different
# protocol/model — a large gap means check the harness, not tune on test.
SANITY_F1 = 0.95


# --------------------------------------------------------------------- metrics

def precision_recall_f1(scores: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    """Binary F1 at a fixed cosine threshold. Predicted positive = score >= thr."""
    pred = scores >= thr
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"threshold": round(float(thr), 6), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4), "tp": tp, "fp": fp,
            "fn": fn, "n": int(len(labels)), "n_predicted_positive": int(np.sum(pred))}


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Exact F1-maximising threshold over the observed scores (no grid, no
    interpolation): sweeping candidates in descending score order and taking the
    first maximum keeps the choice deterministic and equal to the largest
    threshold that attains it."""
    labels = np.asarray(labels)
    r = int(np.sum(labels == 1))
    if r == 0 or len(scores) == 0:
        return {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    order = np.argsort(-np.asarray(scores), kind="stable")
    lab = labels[order]
    sc = np.asarray(scores)[order]
    tp = np.cumsum(lab == 1)
    fp = np.cumsum(lab == 0)
    prec = tp / (tp + fp)
    rec = tp / r
    denom = prec + rec
    f1 = np.where(denom > 0, 2 * prec * rec / np.where(denom > 0, denom, 1.0), 0.0)
    k = int(np.argmax(f1))
    return {"threshold": float(sc[k]), "f1": float(f1[k]),
            "precision": float(prec[k]), "recall": float(rec[k]),
            "n_candidates": int(len(sc)), "n_positives": r}


def map_at_r(scores: np.ndarray, labels: np.ndarray) -> float:
    """MAP@R over the whole split (threshold-free secondary metric): rank every
    pair by score, R = number of positive pairs in the split,
    AP@R = (1/R) * sum_{k<=R} P(k) * rel(k)."""
    labels = np.asarray(labels)
    r = int(np.sum(labels == 1))
    if r == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="stable")
    lab = labels[order]
    hits = np.cumsum(lab == 1)
    prec_at_k = hits / np.arange(1, len(lab) + 1)
    return float((prec_at_k * (lab == 1))[:r].sum() / r)


def pair_cosines(vec_of: dict[int, np.ndarray], pairs) -> np.ndarray:
    return np.array([float(vec_of[i] @ vec_of[j]) for i, j, _ in pairs])


def subsample(pairs, n: int, seed: int):
    """Deterministic tiny subset for --smoke (stratified so both classes are
    present); never used for a reported number."""
    if n >= len(pairs):
        return list(pairs)
    pos = [p for p in pairs if p[2] == 1]
    neg = [p for p in pairs if p[2] == 0]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos = min(len(pos), max(1, n // 2))
    out = pos[:n_pos] + neg[:n - n_pos]
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------- encoder

def default_embed_fn(condition: str, seed: int, *, smoke: bool = False,
                     device: str | None = None):
    """(embed_fn, device, model_note). C0 = untouched base model; C1/C2/C3 =
    the base model with the run's checkpoint loaded (and refused if the
    checkpoint belongs to a different config)."""
    from .encoder import encode_texts, load_encoder

    ck_path = None
    if condition != "C0":
        # checked BEFORE the model loads: a missing run is a one-line message,
        # not a 500 MB download that ends in "checkpoint not found"
        ck_path = run_dir(condition, seed, smoke=smoke) / CHECKPOINT_NAME
        if not ck_path.exists():
            raise SystemExit(f"missing {ck_path} — train it first: "
                             f"`python -m embeded.train --condition {condition} --seed {seed}`")
    tok, model, dev = load_encoder(S.MODEL_ID, device)
    note = f"{S.MODEL_ID} (untuned)"
    if ck_path is not None:
        cfg = run_config(condition, seed, smoke=smoke)
        ck = load_checkpoint(ck_path, model, cfg)
        if ck is None:
            raise SystemExit(f"{ck_path} was written by a different config — retrain "
                             f"(`python -m embeded.train --condition {condition} "
                             f"--seed {seed} --force`); evaluating it would mix two experiments")
        note = f"{S.MODEL_ID} + {condition} seed {seed} (step {ck['step']})"
    model.eval()
    max_len = S.SMOKE_MAX_LEN if smoke else S.MAX_LEN
    amp = S.AMP_DTYPE if str(dev).startswith("cuda") else "float32"

    def embed_fn(texts):
        return encode_texts(model, tok, texts, max_len=max_len, device=dev,
                            batch_size=S.EVAL_BATCH, amp_dtype=amp)

    return embed_fn, dev, note


# --------------------------------------------------------------------- protocol

def evaluate(condition: str, seed: int = S.SEED, *, smoke: bool = False,
             force: bool = False, embed_fn=None, device: str | None = None,
             verbose: bool = True) -> dict:
    from .data.prepare_data import load_fragments, load_pairs

    if condition not in CONDITIONS:
        raise SystemExit(f"unknown condition {condition!r} — one of {list(CONDITIONS)}")
    frag_path = S.ARTIFACTS / "fragments.jsonl"
    if not frag_path.exists():
        raise SystemExit(missing_artifact_message(frag_path))
    frags = load_fragments()

    out = run_dir(condition, seed, smoke=smoke)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path, pred_path = out / EVAL_METRICS_NAME, out / PREDICTIONS_NAME
    cfg = {"condition": condition, "seed": int(seed), "smoke": bool(smoke),
           "version": S.VERSION, "model_id": S.MODEL_ID, "pooling": S.POOLING,
           "max_len": S.SMOKE_MAX_LEN if smoke else S.MAX_LEN,
           "eval_batch": S.EVAL_BATCH, "threshold_policy": S.THRESHOLD_POLICY,
           "pairs_source": "artifacts/pairs_{valid,test}.tsv"}
    cfg["fingerprint"] = fingerprint(cfg)
    if metrics_path.exists() and not force:
        prev = json.loads(metrics_path.read_text())
        if prev.get("fingerprint") == cfg["fingerprint"]:
            print(f"[cache] {condition} seed={seed} already evaluated for version "
                  f"{S.VERSION} — skipping (--force to redo)")
            return prev

    set_seed(seed)
    pairs = {"valid": load_pairs("valid"), "test": load_pairs("test")}
    if smoke:
        pairs = {s: subsample(v, S.SMOKE_PAIRS, seed) for s, v in pairs.items()}
    ids = sorted({x for v in pairs.values() for p in v for x in p[:2]})
    missing = [i for i in ids if i not in frags]
    if missing:
        raise SystemExit(f"{len(missing)} pair fragments are not in fragments.jsonl "
                         f"(e.g. {missing[:3]}) — stale artifacts, rerun prepare_data")

    if embed_fn is None:
        embed_fn, device, note = default_embed_fn(condition, seed, smoke=smoke,
                                                 device=device)
    else:
        note = "injected embed_fn (test)"
    t0 = time.time()
    vecs = np.asarray(embed_fn([frags[i] for i in ids]), dtype=np.float32)
    vec_of = {i: vecs[k] for k, i in enumerate(ids)}
    scores = {s: pair_cosines(vec_of, pairs[s]) for s in pairs}
    labels = {s: np.array([p[2] for p in pairs[s]], dtype=int) for s in pairs}

    # threshold on VALIDATION only; test is scored once, with it frozen
    sel = best_f1_threshold(scores["valid"], labels["valid"])
    valid = precision_recall_f1(scores["valid"], labels["valid"], sel["threshold"])
    test = precision_recall_f1(scores["test"], labels["test"], sel["threshold"])
    mapr_test = map_at_r(scores["test"], labels["test"])
    mapr_valid = map_at_r(scores["valid"], labels["valid"])

    metrics = {
        "condition": condition, "seed": int(seed), "version": S.VERSION,
        "fingerprint": cfg["fingerprint"], "smoke": bool(smoke), "model": note,
        "threshold_policy": S.THRESHOLD_POLICY,
        "threshold": valid["threshold"],
        "valid": {k: valid[k] for k in ("precision", "recall", "f1", "n")},
        "valid_map_at_r": round(mapr_valid, 4),
        "test_f1": test["f1"], "test_precision": test["precision"],
        "test_recall": test["recall"], "test_map_at_r": round(mapr_test, 4),
        "test": {k: test[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn", "n")},
        "n_fragments_encoded": len(ids),
        "n_pairs": {s: len(pairs[s]) for s in pairs},
        "encode_seconds": round(time.time() - t0, 1),
        "evaluated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / EVAL_CONFIG_NAME).write_text(json.dumps(cfg, indent=2))
    metrics_path.write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        pred_path,
        threshold=np.float32(valid["threshold"]),
        valid_scores=scores["valid"].astype(np.float32),
        valid_labels=labels["valid"],
        test_scores=scores["test"].astype(np.float32),
        test_labels=labels["test"],
    )
    if verbose:
        print(json.dumps(metrics, indent=2))
        if not smoke:
            print(f"sanity target: published CodeXGLUE fine-tuned-CodeBERT F1 ~ {SANITY_F1} "
                  "(different model/protocol). A large gap is a harness check "
                  "(pair order, labels, metric, threshold) — never a licence to tune on test.")
    return metrics


# --------------------------------------------------------------------- table

def write_results_table(runs_root: Path | None = None, *, report_md: Path | None = None,
                        sanity_f1: float = SANITY_F1) -> dict:
    """Rebuild the main-results section from the per-run `eval_metrics.json`
    files: per-seed rows plus mean/spread per condition, C0 kept visibly
    separate from the trained conditions (PHASE2_PLAN §4 item 3)."""
    root = runs_root or (S.ARTIFACTS / S.RUNS_SUBDIR)
    rows, smoke, foreign = [], [], []
    for p in sorted(root.glob("*/" + EVAL_METRICS_NAME)):
        m = json.loads(p.read_text())
        if m.get("smoke"):
            smoke.append(m)
        elif "sprime" in p.parent.name:
            # Phase 3 s' run dirs may reuse C-condition labels in their metrics
            # (observed 2026-10-01: sprime_C1_14 wrote condition "C1", 600-pair
            # unseen split) — the run-dir namespace decides, not the field.
            foreign.append(m)
        elif m.get("condition") in CONDITIONS:
            rows.append(m)
        else:
            # Phase 3 sprime/generalisation result files share the runs root but
            # have a different schema (no test_f1) — they get their own table.
            foreign.append(m)
    by_cond: dict[str, list[dict]] = {}
    for m in rows:
        by_cond.setdefault(m["condition"], []).append(m)

    lines = [SECTION, "",
             "Generated from `artifacts/runs/<condition>_<seed>/eval_metrics.json` by "
             f"`python -m embeded.evaluate --results-table` — version `{S.VERSION}`. "
             f"Threshold policy: *{S.THRESHOLD_POLICY}* (chosen on validation, applied "
             "unchanged to test; test is scored once per run).", "",
             "| condition | seed | F1 | precision | recall | MAP@R | valid threshold | n test pairs |",
             "|---|---|---|---|---|---|---|---|"]
    for c in CONDITIONS:
        for m in sorted(by_cond.get(c, []), key=lambda r: r["seed"]):
            lines.append(f"| {c} | {m['seed']} | {m['test_f1']:.4f} | {m['test_precision']:.4f} | "
                         f"{m['test_recall']:.4f} | {m['test_map_at_r']:.4f} | "
                         f"{m['threshold']:.4f} | {m['test']['n']} |")
    lines += ["", "| condition | runs | mean F1 | sd F1 | mean P | mean R | mean MAP@R |",
              "|---|---|---|---|---|---|---|"]
    agg = {}
    for c in CONDITIONS:
        ms = by_cond.get(c, [])
        if not ms:
            continue
        f1 = np.array([m["test_f1"] for m in ms])
        agg[c] = {"n": len(ms), "mean_f1": round(float(f1.mean()), 4),
                  "sd_f1": round(float(f1.std(ddof=1)) if len(ms) > 1 else 0.0, 4)}
        lines.append(f"| {c} | {len(ms)} | {f1.mean():.4f} | "
                     f"{(f1.std(ddof=1) if len(ms) > 1 else 0.0):.4f} | "
                     f"{np.mean([m['test_precision'] for m in ms]):.4f} | "
                     f"{np.mean([m['test_recall'] for m in ms]):.4f} | "
                     f"{np.mean([m['test_map_at_r'] for m in ms]):.4f} |")
    notes = []
    if "C0" in agg:
        notes.append(f"C0 is the untuned baseline (one evaluation, not a trained condition) "
                     f"at mean F1 {agg['C0']['mean_f1']:.4f}.")
    trained = [c for c in ("C1", "C2", "C3", "C4", "C5", "C6") if c in agg]
    if trained:
        best = max(trained, key=lambda c: agg[c]["mean_f1"])
        notes.append(f"Among the trained conditions the highest mean F1 is {best} "
                     f"({agg[best]['mean_f1']:.4f}); per-seed values above, not the best seed.")
    if rows:
        spread = [m["test_f1"] for m in rows]
        notes.append(f"Sanity reference: published CodeXGLUE fine-tuned-CodeBERT F1 ~ "
                     f"{sanity_f1} under a different model/protocol. Observed F1 range "
                     f"{min(spread):.4f}–{max(spread):.4f}. A large discrepancy is a "
                     "harness/protocol check (metric implementation, pair order, labels, "
                     "threshold handling), not a reason to tune against the test split.")
    if smoke:
        notes.append(f"{len(smoke)} smoke-run result(s) are excluded from this table "
                     "(harness checks, not measurements).")
    if foreign:
        notes.append(f"{len(foreign)} result file(s) with conditions outside "
                     f"{list(CONDITIONS)} (Phase 3 sprime runs) are excluded from "
                     "this table.")
    if not rows:
        notes.append("No completed runs yet — this table is a placeholder.")
    lines += [""] + [f"- {n}" for n in notes] + [""]

    block = "\n".join(lines)
    rep = report_md or S.REPORT_MD
    rep.parent.mkdir(parents=True, exist_ok=True)
    if rep.exists():
        old = rep.read_text()
        if SECTION in old:
            old = re.sub(re.escape(SECTION) + r".*?(?=\n## |\Z)",
                         lambda _m: block.rstrip("\n") + "\n", old, flags=re.S)
        else:
            old = old.rstrip("\n") + "\n\n" + block + "\n"
        rep.write_text(old)
    else:
        rep.write_text("# Measurements\n\n" + block + "\n")
    print(f"wrote {rep} ({len(rows)} runs, {len(smoke)} smoke, "
          f"{len(foreign)} other-condition excluded)")
    return {"runs": len(rows), "aggregate": agg}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", choices=CONDITIONS)
    ap.add_argument("--seed", type=int, default=S.SEED)
    ap.add_argument("--smoke", action="store_true",
                    help=f"{S.SMOKE_PAIRS} pairs per split: validates the eval path only")
    ap.add_argument("--force", action="store_true", help="re-evaluate a completed run")
    ap.add_argument("--device", default=None)
    ap.add_argument("--results-table", action="store_true",
                    help="rebuild the main-results section from saved run metrics and exit")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    if a.results_table:
        write_results_table()
        return 0
    if not a.condition:
        ap.error("--condition is required (or use --results-table)")
    m = evaluate(a.condition, a.seed, smoke=a.smoke, force=a.force,
                 device=a.device, verbose=not a.quiet)
    if not a.smoke:
        write_results_table()
    return 0 if m.get("test", {}).get("n") else 1


if __name__ == "__main__":
    raise SystemExit(main())
