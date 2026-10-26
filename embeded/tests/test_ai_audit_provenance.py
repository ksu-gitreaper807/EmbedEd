"""Amended audits must not lose attribution or accept changed frozen labels."""
import csv
import hashlib
import json

import pytest

from scripts import audit_sample as A


def test_ai_provenance_and_freeze(tmp_path, monkeypatch):
    d = tmp_path / 'audit_c4'
    d.mkdir()
    rows = [dict(id=f'p{i}', label='not_clone', note=f'Function difference {i}',
                 reviewer_type='ai', amendment_id='P4-AI-AUDIT-2026-10-23')
            for i in range(50)]
    with (d / 'audit_labels.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (d / 'audit_pairs.md').write_text('frozen code sheet')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps(dict(
        amendment_id='P4-AI-AUDIT-2026-10-23',
        sheet_sha256={'audit_c4/audit_pairs.md': sha(d / 'audit_pairs.md')})))
    freeze = tmp_path / 'freeze.json'
    freeze.write_text(json.dumps(dict(amendment_id='P4-AI-AUDIT-2026-10-23',
        frozen_utc='2026-10-23T11:38:33Z', label_sha256={'C4': sha(d / 'audit_labels.csv')})))
    monkeypatch.setenv('EMBEDED_AUDIT_PROTOCOL', str(protocol))
    monkeypatch.setenv('EMBEDED_AUDIT_FREEZE', str(freeze))
    got = A.review_provenance(d, rows, {'C4'})
    assert got['reviewer_type'] == 'ai'
    assert got['independent_human_validation'] is False
    assert got['label_sha256'] == sha(d / 'audit_labels.csv')
    (d / 'audit_labels.csv').write_text((d / 'audit_labels.csv').read_text() + '\n')
    with pytest.raises(SystemExit, match='frozen labels'):
        A.review_provenance(d, rows, {'C4'})


def test_legacy_review_and_missing_ai_rationale(tmp_path):
    assert A.review_provenance(tmp_path, [{'id': 'old', 'label': 'clone'}], {'C2'}) == {}
    with pytest.raises(SystemExit, match='50.*rationale'):
        A.review_provenance(tmp_path, [{'reviewer_type': 'ai'}], {'C4'})


def test_ai_report_discloses_limitations(tmp_path, monkeypatch):
    from embeded import settings as S
    monkeypatch.setattr(S, 'REPORT_MD', tmp_path / 'report.md')
    result = A.score_rows({'p': 'clone'}, [{'id': 'p', 'condition': 'C4'}])
    result['_provenance'] = dict(reviewer_type='ai', amendment_id='P4-AI-AUDIT-2026-10-23')
    A.write_section(result)
    text = S.REPORT_MD.read_text()
    assert 'AI-reviewed; not independently human-validated' in text
    assert 'P4-AI-AUDIT-2026-10-23' in text
    assert 'sampling uncertainty only' in text
    assert '(blind;' not in text
