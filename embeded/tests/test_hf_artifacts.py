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


def test_is_transient_classifies_network_vs_permanent():
    # classes are matched by NAME (see scripts.hf_artifacts) — mirror the real
    # requests/urllib3/huggingface_hub names here; the offline suite has none
    class ChunkedEncodingError(Exception):
        pass

    class ProtocolError(Exception):
        pass

    class HfHubHTTPError(Exception):
        def __init__(self, status_code: int):
            self.response = type("Resp", (), {"status_code": status_code})()

    assert H.is_transient(ChunkedEncodingError("Response ended prematurely"))
    assert H.is_transient(ProtocolError("Connection aborted"))
    assert H.is_transient(ConnectionError("cut"))          # builtin / requests
    assert H.is_transient(TimeoutError("slow"))            # builtin
    assert H.is_transient(HfHubHTTPError(503))
    assert H.is_transient(HfHubHTTPError(429))
    assert not H.is_transient(HfHubHTTPError(401))         # auth: show at once
    assert not H.is_transient(HfHubHTTPError(404))
    assert not H.is_transient(RuntimeError("bad config"))


def test_run_with_retries_recovers_then_returns(monkeypatch):
    monkeypatch.setattr(H.time, "sleep", lambda _s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("cut")
        return "ok"

    assert H.run_with_retries(flaky, label="pull") == "ok"
    assert len(calls) == 3


def test_run_with_retries_permanent_error_raises_immediately(monkeypatch):
    monkeypatch.setattr(H.time, "sleep", lambda _s: None)
    calls = []

    def broken():
        calls.append(1)
        raise RuntimeError("bad config")

    try:
        H.run_with_retries(broken)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError to propagate")
    assert len(calls) == 1


def test_run_with_retries_exhausts_attempts(monkeypatch):
    monkeypatch.setattr(H.time, "sleep", lambda _s: None)
    calls = []

    def down():
        calls.append(1)
        raise ConnectionError("cut")

    try:
        H.run_with_retries(down)
    except ConnectionError:
        pass
    else:
        raise AssertionError("expected ConnectionError after retries")
    assert len(calls) == H.SYNC_RETRIES + 1


def test_stage_upload_tree_skips_redownloadable_data(tmp_path: Path, capsys):
    """The s' replication package (138 MB + ~2,000 files) is re-fetchable by
    DOI and MD5-verified; checkpointing it made every push a 481 MB upload."""
    artifacts = tmp_path / "art"
    (artifacts / "sprime" / "bcb_v2_sampled_bf").mkdir(parents=True)
    (artifacts / "sprime" / "bcb_v2_sampled_bf" / "ast.pkl").write_bytes(b"x" * 8)
    (artifacts / "sprime" / "ASE25.zip").write_bytes(b"z" * 8)
    (artifacts / "triples_C1.jsonl").write_text("{}\n")
    (artifacts / "audit").mkdir(parents=True)
    (artifacts / "audit" / "audit_key.csv").write_text("id\n")
    report = tmp_path / "report.md"
    report.write_text("# Measurements\n")

    staged = H.stage_upload_tree(tmp_path / "stage", artifacts, report,
                                 include_report=True)
    assert "artifacts/triples_C1.jsonl" in staged
    assert "artifacts/audit/audit_key.csv" in staged          # the audit sheet IS checkpointed
    assert not [f for f in staged if "sprime" in f]           # the s' package is not
    import json as _json
    meta = _json.loads((tmp_path / "stage" / "sync_manifest.json").read_text())
    assert meta["excluded_files"] == 2 and "sprime/*" in meta["excluded_patterns"]
    assert "not checkpointing 2 file(s)" in capsys.readouterr().out

    # an explicit override wins, including "exclude nothing"
    staged = H.stage_upload_tree(tmp_path / "stage2", artifacts, report, exclude=())
    assert any("ASE25.zip" in f for f in staged)
    staged = H.stage_upload_tree(tmp_path / "stage3", artifacts, report,
                                 exclude=("triples_*",))
    assert not [f for f in staged if "triples" in f] and any("sprime" in f for f in staged)


def test_exclude_patterns_env_override(monkeypatch):
    assert H.exclude_patterns() == H.DEFAULT_EXCLUDE
    monkeypatch.setenv("EMBEDED_HF_EXCLUDE", "sprime/*, runs/*/checkpoint.pt")
    assert H.exclude_patterns() == ("sprime/*", "runs/*/checkpoint.pt")
    monkeypatch.setenv("EMBEDED_HF_EXCLUDE", "")
    assert H.exclude_patterns() == ()
