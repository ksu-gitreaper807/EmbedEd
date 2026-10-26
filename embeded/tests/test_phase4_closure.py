"""Closure must never pass on missing evidence or recycled audit results."""
import ast
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from embeded import settings as S
from scripts import phase4_closure as C
from scripts import audit_sample as A

NOTEBOOK = Path(__file__).resolve().parents[2] / 'notebooks/Phase4_Colab.ipynb'


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(S, 'ARTIFACTS', tmp_path)
    root = tmp_path / S.RUNS_SUBDIR
    for name in ['C0_13'] + [f'C{c}_{s}' for c in range(1, 7) for s in range(13, 19)]:
        d = root / name
        d.mkdir(parents=True)
        np.savez(d / 'predictions.npz', test_labels=[0, 1], test_scores=[0., 1.], threshold=.5)
        (d / 'eval_metrics.json').write_text(json.dumps({'test_f1': 1.}))
    return root


def test_complete_inventory_and_extension_mismatch(inventory):
    assert C.audit_mismatches() == []
    (inventory / 'C2_18/eval_metrics.json').write_text('{"test_f1": 0.5}')
    assert C.audit_mismatches() == [('C2_18', .5, 1., -.5)]


def test_missing_predictions_fail_closed(inventory):
    (inventory / 'C3_17/predictions.npz').unlink()
    with pytest.raises(SystemExit, match='Incomplete benchmark inventory'):
        C.audit_mismatches()


def test_empty_inventory_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(S, 'ARTIFACTS', tmp_path)
    with pytest.raises(SystemExit, match='Incomplete benchmark inventory'):
        C.audit_mismatches()


@pytest.mark.parametrize('threshold', [np.float32(.99999988), np.float32(.99999815)])
def test_evaluation_persists_the_threshold_it_actually_scored(fx, monkeypatch, threshold):
    """Six-decimal rounding changes decisions on saturated, adjacent fp32 scores."""
    from embeded import evaluate as E
    from embeded import train as T

    # The shared fixture's validation slice has no positives; give this
    # persistence test an identifiable, non-fallback validation threshold.
    (S.ARTIFACTS / 'pairs_valid.tsv').write_text('0\t1\t1\n0\t8\t0\n')
    lower = np.nextafter(threshold, np.float32(0))
    monkeypatch.setattr(E, 'pair_cosines', lambda _vecs, pairs: np.array(
        [float(threshold if label else lower) for _, _, label in pairs]))
    metrics = E.evaluate('C0', 13, embed_fn=lambda texts: np.ones((len(texts), 2)),
                         verbose=False)
    with np.load(T.run_dir('C0', 13) / 'predictions.npz') as saved:
        selected = E.best_f1_threshold(saved['valid_scores'], saved['valid_labels'])
        assert metrics['threshold'] == selected['threshold'] == float(saved['threshold'])
        for split in ('valid', 'test'):
            replay = E.precision_recall_f1(saved[split + '_scores'], saved[split + '_labels'],
                                         float(saved['threshold']))
            assert replay['f1'] == metrics[split]['f1'] == 1.0


@pytest.fixture
def audits(tmp_path, monkeypatch):
    monkeypatch.setattr(S, 'ARTIFACTS', tmp_path)
    log = tmp_path / 'sanitycheck.md'
    blocks = []
    for cond in ('C4', 'C5', 'C6'):
        d = tmp_path / f'audit_{cond.lower()}'
        d.mkdir()
        key = [dict(id=f'{cond}-{i}', condition=cond, anchor=i, negative=i+100) for i in range(50)]
        rows = [dict(id=r['id'], label='not_clone') for r in key]
        for name, data in [('audit_key.csv', key), ('audit_labels.csv', rows)]:
            with (d / name).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
        triples = [json.dumps({'anchor': i, 'negative': i+100}) for i in range(50)]
        (tmp_path / f'triples_{cond}.jsonl').write_text('\n'.join(triples))
        result = A.score_rows({r['id']: r['label'] for r in rows}, key)[cond]
        (d / 'audit_result.json').write_text(json.dumps({cond: result, '_provenance': {'n_labelled': 50}}))
        blocks.append(C.outcome_marker(cond, result) + '\nbranch taken: reviewed\nconsequence: claims revised')
    log.write_text('\n\n'.join(blocks))
    return log


def test_audit_requires_current_score_and_recorded_branch(audits):
    assert set(C.verified_audits(audits)) == {'C4', 'C5', 'C6'}
    audits.write_text('§9.5 outcome — audit_c5 (scored <UTC date>)\nbranch taken: <decision>')
    with pytest.raises(SystemExit, match='Append the reviewed outcome'):
        C.verified_audits(audits)


def test_audit_rejects_old_result(audits):
    p = S.ARTIFACTS / 'audit_c4/audit_result.json'
    result = json.loads(p.read_text())
    result['_provenance']['n_labelled'] = 100
    p.write_text(json.dumps(result))
    with pytest.raises(SystemExit, match='stale/invalid score'):
        C.verified_audits(audits)


def test_audit_rejects_foreign_labels(audits):
    p = S.ARTIFACTS / 'audit_c4/audit_labels.csv'
    p.write_text(p.read_text().replace('C4-0,', 'FOREIGN,'))
    with pytest.raises(SystemExit, match='unique, matching'):
        C.verified_audits(audits)


def test_active_amendment_refuses_stripped_ai_attribution(audits):
    report = audits.parent / 'report'
    report.mkdir()
    (report / 'phase4_ai_audit_protocol.json').write_text('{}')
    with pytest.raises(SystemExit, match='active amendment requires AI-review provenance'):
        C.verified_audits(audits)


def test_notebook_quarantine_precedes_make_and_preserves_evidence(tmp_path):
    cells = json.loads(NOTEBOOK.read_text())['cells']
    source = next(''.join(c['source']) for c in cells if '# --- 3a.' in ''.join(c['source']))
    # Execute the actual notebook function against the historical 100/50-row failure.
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'quarantine_foreign')
    import time
    env = {'csv': csv, 'time': time}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<quarantine>', 'exec'), env)
    (tmp_path / 'audit_key.csv').write_text('id\nP001\n')
    (tmp_path / 'audit_labels.csv').write_text('id,label\nP001,clone\nP002,clone\n')
    (tmp_path / 'audit_result.json').write_text('{}')
    env['quarantine_foreign'](tmp_path)
    assert not (tmp_path / 'audit_labels.csv').exists()
    assert not (tmp_path / 'audit_result.json').exists()
    assert len(list(tmp_path.glob('*.foreign-*'))) == 2
    assert source.index('    quarantine_foreign(d)') < source.index("rc = sh('scripts.audit_sample', 'make'")


def test_all_notebook_code_compiles():
    for c in json.loads(NOTEBOOK.read_text())['cells']:
        if c['cell_type'] == 'code':
            source = ''.join(line for line in c['source'] if not line.startswith('%'))
            compile(source, str(NOTEBOOK), 'exec')


def test_notebook_checks_precision_and_acceptance_before_sync():
    cells = json.loads(NOTEBOOK.read_text())['cells']
    source = next(''.join(c['source']) for c in cells if '# --- 4b.' in ''.join(c['source']))
    assert source.index("['threshold'] != _probe") < source.index('fetch_repair_checkpoint(name)')
    assert source.index('remaining = audit_mismatches()') < source.index('hf_push()')
    assert source.index('if remaining: raise SystemExit') < source.index('hf_push()')


def test_checkpoint_fetch_does_not_restore_cached_metrics(inventory, tmp_path, monkeypatch):
    import huggingface_hub
    from scripts import hf_artifacts as H

    cached = tmp_path / 'snapshot' / 'artifacts' / 'runs' / 'C2_18'
    cached.mkdir(parents=True)
    (cached / 'checkpoint.pt').write_bytes(b'weights')
    (cached / 'eval_metrics.json').write_text('stale metrics')
    monkeypatch.setattr(H, 'load_repo_config', lambda: H.RepoConfig(
        repo_id='test/data', repo_type='dataset', revision='main', subdir='', private=False, token=None))
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda **kw: str(cached / 'checkpoint.pt'))
    C.fetch_repair_checkpoint('C2_18')
    assert (inventory / 'C2_18/checkpoint.pt').read_bytes() == b'weights'
    assert json.loads((inventory / 'C2_18/eval_metrics.json').read_text()) == {'test_f1': 1.}


def test_table_uses_exact_seeds_and_includes_required_metrics(inventory, audits):
    import time

    for p in inventory.glob('*/eval_metrics.json'):
        cond = p.parent.name.split('_')[0]
        p.write_text(json.dumps(dict(condition=cond, test_f1=1., test_map_at_r=.75, threshold=.5)))
    # The old prefix glob accidentally admitted C1_130 alongside seed 13.
    extra = inventory / 'C1_130'
    extra.mkdir()
    (extra / 'eval_metrics.json').write_text('{"condition": "C1", "test_f1": 0}')
    for c in range(1, 7):
        for seed in range(13, 16):
            d = inventory / f'sprime_C{c}_{seed}'
            d.mkdir()
            (d / 'eval_metrics.json').write_text(json.dumps(dict(
                condition=f'C{c}', f1_seen=.8, f1_unseen=.7 + (seed-13)*.01, threshold=.9999)))
    d = inventory / 'sprime_transfer_C0_13'
    d.mkdir()
    (d / 'eval_metrics.json').write_text('{"f1_seen": 0.7, "f1_unseen": 0.69}')
    (S.ARTIFACTS / 'phase4_decisions.json').write_text('{"decisions": {"layout": "panels"}}')
    cells = json.loads(NOTEBOOK.read_text())['cells']
    source = next(''.join(c['source']) for c in cells if '# --- 5b.' in ''.join(c['source']))
    output = {}
    env = dict(S=S, ROOT=str(audits.parent), REPORT=audits.parent / 'report.md',
               DECISIONS={'layout': 'panels'}, Path=Path, time=time,
               verified_audits=C.verified_audits, audit_mismatches=lambda: [],
               need=lambda *paths: [p.stat() for p in paths],
               replace_or_append=lambda heading, text: output.update(text=text))
    exec(compile(source, '<table>', 'exec'), env)
    text = output['text']
    assert '| C1 | 1.0000 ± 0.0000 | 0.7500 | 0.5000, 0.5000, 0.5000 | 0/3 |' in text
    assert '+0.0900 ± 0.0100 | 3/3 |' in text
    assert 'C0 (untuned anchor, transfer)' in text
    assert 'all-positive F1 baseline' in text
    assert '### Panel C — measured false-negative rates' in text
