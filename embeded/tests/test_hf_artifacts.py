"""Offline tests for the HF artifact sync helpers."""
from pathlib import Path

from scripts import hf_artifacts as H


def test_allow_patterns_root_and_subdir():
    assert H.allow_patterns() == [
        "artifacts/*",
        "artifacts/**",
        "report/*",
        "report/**",
    ]
    assert H.allow_patterns("alice/run1") == [
        "alice/run1/artifacts/*",
        "alice/run1/artifacts/**",
        "alice/run1/report/*",
        "alice/run1/report/**",
    ]


def test_stage_upload_tree_and_restore_download_tree(tmp_path: Path):
    artifacts = tmp_path / "local-artifacts"
    report = tmp_path / "report" / "measurements.md"
    (artifacts / "nested").mkdir(parents=True)
    (artifacts / "manifest.json").write_text('{"ok": true}')
    (artifacts / "nested" / "triples_C1.jsonl").write_text('{"row": 1}\n')
    report.parent.mkdir(parents=True)
    report.write_text("# Measurements\n")

    stage = tmp_path / "stage"
    staged = H.stage_upload_tree(stage, artifacts, report, include_report=True)
    assert staged == [
        "artifacts/manifest.json",
        "artifacts/nested/triples_C1.jsonl",
        "report/measurements.md",
        "sync_manifest.json",
    ]
    assert (stage / "sync_manifest.json").is_file()

    restored_artifacts = tmp_path / "restored-artifacts"
    restored_report = tmp_path / "restored" / "measurements.md"
    restored = H.restore_download_tree(stage, restored_artifacts, restored_report)
    assert restored == [
        "artifacts/manifest.json",
        "artifacts/nested/triples_C1.jsonl",
        "report/measurements.md",
    ]
    assert (restored_artifacts / "manifest.json").read_text() == '{"ok": true}'
    assert (restored_artifacts / "nested" / "triples_C1.jsonl").read_text() == '{"row": 1}\n'
    assert restored_report.read_text() == "# Measurements\n"


def test_load_repo_config_optional(monkeypatch):
    monkeypatch.delenv("EMBEDED_HF_REPO_ID", raising=False)
    assert H.load_repo_config(required=False) is None


def test_load_repo_config_reads_env(monkeypatch):
    monkeypatch.setenv("EMBEDED_HF_REPO_ID", "user/embeded-artifacts")
    monkeypatch.setenv("EMBEDED_HF_REPO_TYPE", "dataset")
    monkeypatch.setenv("EMBEDED_HF_REVISION", "main")
    monkeypatch.setenv("EMBEDED_HF_SUBDIR", "alice/phase01")
    monkeypatch.setenv("EMBEDED_HF_PRIVATE", "true")
    monkeypatch.setenv("HF_TOKEN", "secret")
    cfg = H.load_repo_config()
    assert cfg.repo_id == "user/embeded-artifacts"
    assert cfg.repo_type == "dataset"
    assert cfg.revision == "main"
    assert cfg.subdir == "alice/phase01"
    assert cfg.private is True
    assert cfg.token == "secret"
