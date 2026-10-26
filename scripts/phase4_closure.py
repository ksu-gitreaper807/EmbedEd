"""Fail-closed checks for the Phase 4 sequencer; no training or remote writes."""
import csv
import hashlib
import json
import re
from pathlib import Path

from embeded import settings as S


def fetch_repair_checkpoint(name):
    """Copy only the requested checkpoint, never restore a shared cached snapshot."""
    import shutil
    from huggingface_hub import hf_hub_download
    from scripts.hf_artifacts import load_repo_config

    if not re.fullmatch(r'C[0-6]_\d+', name):
        raise ValueError(name)
    cfg = load_repo_config()
    if cfg is None:
        raise SystemExit('Configure EMBEDED_HF_REPO_ID before checkpoint repair')
    filename = '/'.join(p for p in (cfg.subdir, 'artifacts', 'runs', name, 'checkpoint.pt') if p)
    source = hf_hub_download(repo_id=cfg.repo_id, repo_type=cfg.repo_type,
                             revision=cfg.revision, filename=filename, token=cfg.token)
    dest = S.ARTIFACTS / S.RUNS_SUBDIR / name / 'checkpoint.pt'
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def audit_mismatches():
    import numpy as np

    root = S.ARTIFACTS / S.RUNS_SUBDIR
    names = ['C0_13'] + [f'C{c}_{s}' for c in range(1, 7) for s in range(13, 19)]
    names = sorted(set(names) | {p.name for p in root.glob('C*')
                                if re.fullmatch(r'C[0-6]_\d+', p.name)})
    missing = [str(root / n / f) for n in names
               for f in ('predictions.npz', 'eval_metrics.json')
               if not (root / n / f).is_file()]
    if missing:
        raise SystemExit('Incomplete benchmark inventory; cannot certify zero mismatches:\n'
                         + '\n'.join(missing))
    out = []
    for name in names:
        metrics = json.loads((root / name / 'eval_metrics.json').read_text())
        with np.load(root / name / 'predictions.npz') as z:
            y, scores = z['test_labels'], z['test_scores']
            threshold = float(np.asarray(z['threshold']).ravel()[0])
        if (y.ndim != 1 or scores.shape != y.shape or not len(y)
                or not np.isfinite(scores).all() or not np.isfinite(threshold)
                or not np.isin(y, [0, 1]).all()):
            raise SystemExit(f'{name}: invalid predictions')
        pred = scores >= threshold
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        stored = float(metrics['test_f1'])
        if not np.isfinite(stored):
            raise SystemExit(f'{name}: non-finite stored F1')
        if abs(stored - f1) > 5e-4:
            out.append((name, stored, round(f1, 4), round(stored - f1, 4)))
    return out


def outcome_marker(cond, result):
    digest = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return f'Phase 4 audit outcome {cond} sha256={digest}'


def verified_audits(log_path):
    """Reject stale scores and require an explicit outcome tied to each current score."""
    from scripts.audit_sample import check_key_matches_triples, score_rows, review_provenance

    log = Path(log_path).read_text()
    audits = {}
    for cond in ('C4', 'C5', 'C6'):
        d = S.ARTIFACTS / f'audit_{cond.lower()}'
        with (d / 'audit_key.csv').open() as f:
            key = list(csv.DictReader(f))
        with (d / 'audit_labels.csv').open() as f:
            rows = list(csv.DictReader(f))
        ids = {r['id'] for r in key}
        if (len(key) != 50 or len(ids) != 50 or len(rows) != 50
                or {r['id'] for r in rows} != ids
                or {r['condition'] for r in key} != {cond}):
            raise SystemExit(f'{cond}: expected 50 unique, matching sheet labels')
        drift = check_key_matches_triples(key)
        if any(v['not_in_current_triples'] for v in drift.values()):
            raise SystemExit(f'{cond}: stale sheet')
        result = score_rows({r['id']: r['label'] for r in rows}, key)[cond]
        stored = json.loads((d / 'audit_result.json').read_text())
        review = review_provenance(d, rows, {cond})
        amendment_path = Path(log_path).resolve().parent / 'report/phase4_ai_audit_protocol.json'
        if amendment_path.exists() and not review:
            raise SystemExit(f'{cond}: active amendment requires AI-review provenance')
        if any(stored.get('_provenance', {}).get(k) != v for k, v in review.items()):
            raise SystemExit(f'{cond}: stale/missing AI-review provenance')
        if stored.get(cond) != result or stored.get('_provenance', {}).get('n_labelled') != 50:
            raise SystemExit(f'{cond}: stale/invalid score; rerun section 3b')
        marker = outcome_marker(cond, result)
        pattern = re.escape(marker) + r'\nbranch taken: ([^\n]+)\nconsequence: ([^\n]+)'
        match = re.search(pattern, log)
        if not match or any('<' in v or 'TODO' in v for v in match.groups()):
            raise SystemExit('Append the reviewed outcome to sanitycheck.md before building:\n'
                             + marker + '\nbranch taken: <decision>\nconsequence: <claim changes>')
        audits[cond] = result
    return audits
