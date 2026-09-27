"""Offline tests for scripts.colab_git (the Colab → GitHub push route).

Nothing here touches the network: `resolve_user` is stubbed, and the "remote" is a local bare
repository. What is exercised for real is the part that failed on the VM — git's own credential
lookup and the pull --rebase → push sequence.
"""
import os
import subprocess
from pathlib import Path

import pytest

from scripts import colab_git as G


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)
    assert proc.returncode == 0, f"git {' '.join(args)}: {proc.stderr}"
    return proc.stdout.strip()


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))
    return home_dir


@pytest.fixture()
def fresh_clone(remote_and_clone, tmp_path: Path) -> Path:
    """A clone with no git identity — the state of a VM that has never committed."""
    bare, _, _ = remote_and_clone
    clone = tmp_path / "fresh-clone"
    subprocess.run(["git", "clone", "-q", str(bare), str(clone)], capture_output=True, check=True)
    return clone


@pytest.fixture()
def remote_and_clone(tmp_path: Path):
    """A bare 'GitHub' plus a clone that shares history with it, both with an identity set."""
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "--initial-branch=main", str(bare)],
                   capture_output=True, check=True)
    work = tmp_path / "upstream-work"
    work.mkdir()
    _git(work, "init", "-q", "--initial-branch=main")
    _git(work, "config", "user.email", "owner@example.com")
    _git(work, "config", "user.name", "owner")
    (work / "report").mkdir()
    (work / "report" / "measurements.md").write_text("# base\n", encoding="utf-8")
    _git(work, "add", "-A")
    _git(work, "commit", "-qm", "base")
    _git(work, "remote", "add", "origin", str(bare))
    _git(work, "push", "-q", "origin", "main")

    clone = tmp_path / "vm-clone"
    subprocess.run(["git", "clone", "-q", str(bare), str(clone)], capture_output=True, check=True)
    _git(clone, "config", "user.email", "vm@example.com")       # the VM already committed once
    _git(clone, "config", "user.name", "vm")
    return bare, work, clone


# --------------------------------------------------------------------------- redaction

def test_redact_masks_known_token_urls_and_key_value_lines():
    token = "ghp_0123456789abcdefghijABCDEF"
    text = (f"fatal: {token} in https://me:{token}@github.com/o/r.git\n"
            "password=whatever-was-filled-in\n"
            "Authorization: Bearer github_pat_11ABCDEFG0abcdefghijklmn\n"
            "nothing sensitive here\n")
    out = G.redact(text, token)
    assert token not in out and "whatever-was-filled-in" not in out
    assert "github_pat_11ABCDEFG0abcdefghijklmn" not in out
    assert "https://***@github.com/o/r.git" in out
    assert "password=***" in out
    assert "nothing sensitive here" in out


def test_redact_masks_a_token_it_does_not_know():
    """A token baked into .git/config by an earlier attempt must not survive either."""
    out = G.redact("url = https://user:ghp_0123456789abcdefghij@github.com/o/r.git", None)
    assert "ghp_0123456789abcdefghij" not in out


def test_read_secret_falls_back_to_env_and_returns_none(monkeypatch: pytest.MonkeyPatch):
    for name in G.TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    assert G.read_secret() is None
    monkeypatch.setenv("GITHUB_PAT", "  value  ")
    assert G.read_secret() == "value"


# --------------------------------------------------------------------------- credentials

def test_write_credentials_is_0600_and_keeps_other_hosts(home: Path):
    path = home / ".git-credentials"
    path.write_text("https://someone:pw@huggingface.co\n", encoding="utf-8")
    os.chmod(path, 0o644)

    written = G.write_credentials("alice", "tok", path=path)

    assert written == path
    assert oct(path.stat().st_mode & 0o777) == "0o600"          # tightened, not left 0644
    assert path.read_text(encoding="utf-8").splitlines() == [
        "https://someone:pw@huggingface.co",
        "https://alice:tok@github.com",
    ]
    G.write_credentials("alice", "tok2", path=path)              # re-run must not duplicate
    assert path.read_text(encoding="utf-8").count("github.com") == 1
    assert "tok2" in path.read_text(encoding="utf-8")


def test_configure_wires_the_store_helper_and_git_actually_returns_the_token(
        home: Path, fresh_clone, monkeypatch: pytest.MonkeyPatch):
    clone = fresh_clone
    monkeypatch.setattr(G, "resolve_user",
                        lambda token, api=G.API_USER, timeout=20:
                        {"login": "alice", "id": 4242, "scopes": "repo", "api_ok": True})

    info = G.configure(clone, "tok-abc")

    assert info["username"] == "alice"
    assert info["mode"] == "0o600"
    assert info["identity"] == "set from the token (alice)"
    assert _git(clone, "config", "--get", "user.email") == "4242+alice@users.noreply.github.com"
    # git stores an empty `helper =` first: with no other helper configured it simply reads
    # back as ["store"], and when a global helper exists it cancels it (next test).
    assert _git(clone, "config", "--get-all", "credential.helper").splitlines() == ["store"]
    # the real check: git's own lookup finds the token through the helper
    assert info["helper_wired"] is True and info["stored_matches_token"] is True
    assert "tok-abc" not in _git(clone, "remote", "-v")          # never in .git/config


def test_configure_cancels_a_helper_inherited_from_the_global_config(
        home: Path, fresh_clone, monkeypatch: pytest.MonkeyPatch):
    """Colab images ship system git config; a stale helper there must not be consulted first."""
    (home / ".gitconfig").write_text("[credential]\n\thelper = cache\n", encoding="utf-8")
    monkeypatch.setattr(G, "resolve_user",
                        lambda token, api=G.API_USER, timeout=20:
                        {"login": "alice", "id": 4242, "scopes": "repo", "api_ok": True})

    info = G.configure(fresh_clone, "tok-abc")

    assert _git(fresh_clone, "config", "--get-all", "credential.helper").splitlines() == [
        "cache", "", "store"]
    assert info["helper_wired"] is True and info["stored_matches_token"] is True


def test_configure_falls_back_when_the_api_is_unreachable(
        home: Path, remote_and_clone, monkeypatch: pytest.MonkeyPatch):
    _, _, clone = remote_and_clone
    _git(clone, "config", "user.email", "kept@example.com")       # an existing identity is kept
    monkeypatch.setattr(G, "resolve_user",
                        lambda token, api=G.API_USER, timeout=20:
                        {"login": "x-access-token", "id": None, "scopes": "", "api_ok": False})

    info = G.configure(clone, "tok-abc")

    assert info["username"] == "x-access-token"
    assert info["identity"] == "pre-existing"
    assert info["stored_matches_token"] is True


def test_configure_says_so_when_the_identity_is_a_guess(home: Path, fresh_clone,
                                                        monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(G, "resolve_user",
                        lambda token, api=G.API_USER, timeout=20:
                        {"login": "x-access-token", "id": None, "scopes": "", "api_ok": False})

    info = G.configure(fresh_clone, "tok-abc")

    assert info["identity"].startswith("GUESSED")
    assert "not be attributed" in info["identity"]


# --------------------------------------------------------------------------- push

def test_sync_and_push_rebases_and_pushes_to_a_diverged_remote(home: Path, remote_and_clone):
    bare, work, clone = remote_and_clone
    # someone else pushed first
    (work / "report" / "measurements.md").write_text("# upstream change\n", encoding="utf-8")
    _git(work, "commit", "-qam", "upstream change")
    _git(work, "push", "-q", "origin", "main")
    # ... and the VM has its own local commit, exactly the reported situation
    (clone / "report" / "audit_separability.txt").write_text("table\n", encoding="utf-8")
    _git(clone, "add", "report/audit_separability.txt")
    _git(clone, "commit", "-qm", "audit separability")

    result = G.sync_and_push(clone, token=None)

    assert result["ok"] is True, result
    assert result["remote_sha"] == result["local_sha"]
    assert _git(bare, "log", "--oneline", "-1") == _git(clone, "log", "--oneline", "-1")
    upstream = _git(bare, "show", "HEAD~1:report/measurements.md")
    assert upstream == "# upstream change"                        # rebase kept the upstream work


def test_sync_and_push_can_commit_paths_first(home: Path, remote_and_clone):
    bare, _, clone = remote_and_clone
    (clone / "report" / "measurements.md").write_text("# appended section\n", encoding="utf-8")

    result = G.sync_and_push(clone, token=None, commit="report: audit section",
                             paths=["report/measurements.md"])

    assert result["ok"] is True, result
    assert "# appended section" in _git(bare, "show", "HEAD:report/measurements.md")


def test_sync_and_push_refuses_a_dirty_worktree_instead_of_guessing(home: Path, remote_and_clone):
    _, _, clone = remote_and_clone
    (clone / "report" / "measurements.md").write_text("# unstaged\n", encoding="utf-8")

    result = G.sync_and_push(clone, token=None)

    assert result["ok"] is False and result["stage"] == "worktree"
    assert "report/measurements.md" in result["message"]
    assert _git(clone, "status", "--porcelain") != ""             # left exactly as it was


def test_sync_and_push_reports_a_missing_identity_instead_of_raising(home: Path, fresh_clone):
    """A VM with no git identity gets a readable FAIL, not a traceback (reproduced here)."""
    (fresh_clone / "report" / "measurements.md").write_text("# appended\n", encoding="utf-8")

    result = G.sync_and_push(fresh_clone, token=None, commit="report: audit",
                             paths=["report/measurements.md"])

    assert result["ok"] is False and result["stage"] == "worktree"
    assert "identity" in result["message"]


def test_sync_and_push_ignores_untracked_files(home: Path, remote_and_clone):
    """An untracked stray does not block pull --rebase or push — measured, not assumed."""
    _, _, clone = remote_and_clone
    (clone / "audit_labels.csv").write_text("id,label,note\n", encoding="utf-8")

    result = G.sync_and_push(clone, token=None)

    assert result["ok"] is True, result
    assert any("leaving untracked: audit_labels.csv" in line for line in result["lines"])
    assert (clone / "audit_labels.csv").is_file()                 # untouched, still untracked


def test_diagnose_does_not_blame_untracked_files_for_exit_128(home: Path, remote_and_clone):
    _, _, clone = remote_and_clone
    (clone / "audit_labels.csv").write_text("id,label,note\n", encoding="utf-8")

    lines = "\n".join(G.diagnose(clone, "main", token=None))

    assert "untracked only — does not block a rebase" in lines
    assert "cannot pull with rebase" not in lines
    assert "no known blocker found" in lines


def test_sync_and_push_will_not_stage_the_audit_key(home: Path, remote_and_clone):
    _, _, clone = remote_and_clone
    (clone / "report" / "audit_key.csv").write_text("id,label\n", encoding="utf-8")
    with pytest.raises(ValueError, match="audit key"):
        G.sync_and_push(clone, token=None, commit="x", paths=["report/audit_key.csv"])
    assert _git(clone, "status", "--porcelain") != ""


# --------------------------------------------------------------------------- diagnostics

def test_diagnose_names_the_dirty_worktree_as_the_cause_of_exit_128(home: Path, remote_and_clone):
    _, _, clone = remote_and_clone
    (clone / "report" / "measurements.md").write_text("# unstaged\n", encoding="utf-8")

    lines = "\n".join(G.diagnose(clone, "main", token=None))

    assert "worktree: DIRTY" in lines
    assert "cannot pull with rebase: You have unstaged changes" in lines
    assert "commit or stash it" in lines
    assert "ahead/behind vs origin/main: ahead 0, behind 0" in lines


def test_known_causes_cover_the_documented_exit_128_paths():
    assert G._cause("error: cannot pull with rebase: You have unstaged changes.")
    assert "unset" in G._cause("fatal: empty ident name (for <>) not allowed")
    assert "credential helper" in G._cause("fatal: could not read Username for 'https://github.com'")
    assert "Contents: Read and write" in G._cause("remote: 403 forbidden")
