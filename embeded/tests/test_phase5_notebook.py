"""Phase 5 is a read/verify/write-up workflow, never an automatic experiment/release."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / 'notebooks/Phase5_Colab.ipynb'


def cell(cell_id):
    return next(''.join(c['source']) for c in json.loads(NOTEBOOK.read_text())['cells']
                if c['id'] == cell_id)


def function(cell_id, name, env):
    tree = ast.parse(cell(cell_id))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<notebook-function>', 'exec'), env)
    return env[name]


def test_phase5_notebook_is_unexecuted_and_compiles():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook['nbformat'] == 4
    assert len({c['id'] for c in notebook['cells']}) == len(notebook['cells'])
    for c in notebook['cells']:
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), str(NOTEBOOK), 'exec')
            assert c['outputs'] == [] and c['execution_count'] is None
    assert notebook['metadata']['embeded_phase5']['publication_automatic'] is False


def test_no_training_or_automatic_remote_writer_and_reuses_existing_table():
    sources = '\n'.join(''.join(c['source']) for c in json.loads(NOTEBOOK.read_text())['cells']
                        if c['cell_type'] == 'code')
    for forbidden in ('git reset', 'reset --hard', 'git push', 'create_commit(',
                      'upload_folder(', 'hf_push(', "'embeded.train'", 'files.upload(',
                      'userdata.get(\'HF_TOKEN\')\nprint(TOKEN)'):
        assert forbidden not in sources
    assert "'# --- 5b.'" in cell('p5-table')
    assert "'eval_metrics.json', 'predictions.npz'" in cell('p5-inputs')
    assert 'range(13, 19)' in cell('p5-inputs')
    assert 'range(13, 16)' in cell('p5-inputs')
    assert 'e3144bea7732c20ab2803782282282be4fede7b4' in cell('p5-config')
    assert 'scripts.phase5_report' in cell('p5-table')
    assert 'scripts.export_confusion_matrices' in cell('p5-table')
    assert 'report/figures/confusion_matrices' in cell('p5-review')
    assert 'report/final/' in cell('p5-review')
    assert "('paper/IEEEtran.cls', 'IEEEtran.cls')" in cell('p5-drafts')
    assert "'draft/IEEEtran.cls'" in cell('p5-review')


def test_copy_guard_preserves_changed_work(tmp_path):
    save = function('p5-config', 'save_new_or_identical', {'Path': Path})
    path = tmp_path / 'draft.txt'
    save(path, b'first')
    save(path, b'first')
    with pytest.raises(SystemExit, match='Preserving changed file'):
        save(path, b'replacement')
    assert path.read_bytes() == b'first'


def test_recheck_covers_working_copies_not_only_snapshot(tmp_path):
    snapshot, work = tmp_path / 'snapshot', tmp_path / 'work'
    name = 'artifacts/triples_C4.jsonl'
    for root in (snapshot, work):
        (root / 'artifacts').mkdir(parents=True)
        (root / name).write_bytes(b'input')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    verify = function('p5-inputs', 'verify_input_copies', dict(
        WORK=work, SNAPSHOT=snapshot, sha=sha,
        INPUT_MANIFEST={'files': {name: sha(snapshot / name)}}))
    verify()
    (work / name).write_bytes(b'changed')
    with pytest.raises(SystemExit, match='Pinned working input changed'):
        verify()


def test_run_all_cannot_export_after_unreviewed_draft(tmp_path):
    (tmp_path / 'phase5_review.json').write_text(json.dumps(
        {'reviewer': '', 'checks': {'paper_visually_reviewed': False}}))
    with pytest.raises(SystemExit, match='incomplete editorial review'):
        exec(compile(cell('p5-export'), '<export>', 'exec'), dict(
            WORK=tmp_path, json=json, REVIEW_FIELDS=['paper_visually_reviewed']))
    assert not list(tmp_path.glob('*.zip'))


def test_changed_reviewed_document_blocks_export(tmp_path):
    (tmp_path / 'draft.txt').write_text('revised')
    (tmp_path / 'phase5_review.json').write_text(json.dumps(dict(
        reviewer='test reviewer', checks={'report_reviewed': True},
        reviewed_sha256={'draft.txt': 'stale hash'})))
    with pytest.raises(SystemExit, match='reviewed content has changed'):
        exec(compile(cell('p5-export'), '<export>', 'exec'), dict(
            WORK=tmp_path, json=json, REVIEW_FIELDS=['report_reviewed'],
            RELEASE_DOCS=['draft.txt'], need=lambda *args: None,
            sha=lambda path: hashlib.sha256(path.read_bytes()).hexdigest()))


def test_review_and_integrity_gates_precede_archive_creation():
    source = cell('p5-export')
    archive = source.index("with zipfile.ZipFile(package, 'x'")
    for gate in ('incomplete editorial review', 'reviewed content has changed',
                 'verify_input_copies()', 'verified_audits(', 'again.returncode'):
        assert source.index(gate) < archive
    assert 'No training' in (ROOT / 'PHASE5_PLAN.md').read_text()
