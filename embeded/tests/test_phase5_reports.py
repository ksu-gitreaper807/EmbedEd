"""Small offline guards for the pinned Phase 5 snapshot/report path."""
import hashlib
import json

import pytest

from scripts.phase5_report import INPUT_REVISION, input_identity
from scripts.phase5_snapshot import matches, selected


def test_report_requires_a_verified_manifest(tmp_path):
    with pytest.raises(SystemExit, match='manifest is required'):
        input_identity(tmp_path)


def test_report_rejects_wrong_revision(tmp_path):
    (tmp_path / 'input_manifest.json').write_text(json.dumps(
        dict(repo='Kusshal/Embed', revision='different', files={})))
    with pytest.raises(SystemExit, match='Wrong pinned input identity'):
        input_identity(tmp_path)


@pytest.mark.parametrize('name,key', [('snapshot_manifest.json', 'sha256'),
                                     ('input_manifest.json', 'files')])
def test_report_accepts_exact_identity(tmp_path, name, key):
    (tmp_path / name).write_text(json.dumps(
        dict(repo='Kusshal/Embed', revision=INPUT_REVISION, **{key: {'path': 'hash'}})))
    assert input_identity(tmp_path) == {'path': 'hash'}


def test_snapshot_scope_omits_weights_and_redundant_archives():
    for name in ('artifacts/runs/C1_13/checkpoint.pt', 'artifacts/embeddings.npy',
                 'artifacts/sprime/raw.java', 'artifacts/v5-backup/report.md', 'raw.zip'):
        assert not selected(name)
    assert selected('artifacts/runs/C1_13/predictions.npz')
    assert selected('report/measurements.md')


@pytest.mark.parametrize('lfs', [False, True])
def test_download_hash_is_checked_not_just_length(tmp_path, lfs):
    path = tmp_path / 'file'
    data = b'correct'
    path.write_bytes(data)
    entry = {'size': len(data), 'oid': hashlib.sha1(b'blob 7\0' + data).hexdigest()}
    if lfs:
        entry['lfs'] = {'oid': hashlib.sha256(data).hexdigest()}
    assert matches(path, entry)
    path.write_bytes(b'changed')
    assert not matches(path, entry)
