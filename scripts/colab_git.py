"""Push a Colab VM's local commits back to GitHub using a token from Colab Secrets.

    python -m scripts.colab_git diagnose                  # why did git fail? nothing secret printed
    python -m scripts.colab_git push                      # configure, pull --rebase, push
    python -m scripts.colab_git push --ref arena/... --commit "report: audit separability" --paths report

Why this exists
---------------
`git push` from a fresh Colab VM dies with **exit 128**: there is no credential helper
configured and no terminal for git to prompt on, so the interactive path fails. The fix is a
personal access token, but the token must never appear in cell source, in `argv`, in an
environment dump, or in a notebook output — notebook outputs are saved to Drive and are the
first thing that gets pasted into chat or a PR.

So this module owns the whole handshake:

* the token is read from Colab's **Secrets** panel (`GITHUB_TOKEN` by default) or, failing
  that, from the environment — it is never an argument on the command line;
* it is written to a mode-``0600`` ``~/.git-credentials`` and git's ``store`` helper is pointed
  at it, which keeps it out of ``.git/config`` (so ``git remote -v`` cannot leak it) and out of
  every child process's ``argv``;
* every byte of git output is redacted before it leaves this module, so a traceback or a
  ``--verbose`` flag cannot print it.

Fetching this repo needs no token at all — it is public. Only ``push`` does.

Run ``diagnose`` first. Exit 128 has several distinct causes (dirty worktree, missing identity,
expired token, insufficient scope) and they have different fixes; ``diagnose`` prints the
command that failed, its stderr, and which of those it looks like.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib import error, parse, request

TOKEN_NAMES = ("GITHUB_TOKEN", "GITHUB_PAT", "GH_TOKEN")
HOST = "github.com"
MASK = "***"
API_USER = "https://api.github.com/user"

# Anything that looks like a credential is masked whether or not we know the exact token:
# a token accidentally baked into the remote URL by an earlier attempt must not be printed.
_PATTERNS = (
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9]{16,}\b"), MASK),
    (re.compile(r"(://)[^/@\s]*(@)"), r"\1" + MASK + r"\2"),
    (re.compile(r"(?im)^((?:password|token|secret|authorization)=).*$"), r"\1" + MASK),
)

# Exit 128 from `git pull --rebase` / `git push` has a short list of causes. Each string below
# was produced by running git here, not copied from a blog post.
KNOWN_CAUSES = (
    ("cannot pull with rebase: You have unstaged changes",
     "the worktree is dirty; commit or stash it (see the porcelain lines above)"),
    ("cannot pull with rebase: You have unmerged files",
     "a previous rebase stopped on a conflict; `git rebase --abort` and retry"),
    ("there is already a rebase-merge directory",
     "an interrupted rebase is still in progress; `git rebase --abort`"),
    ("empty ident name",
     "git identity is unset; set user.name / user.email (push does this for you)"),
    ("could not read Username",
     "no credential helper and no terminal; store the PAT (push does this for you)"),
    ("Authentication failed",
     "the PAT is invalid or expired; create a new one"),
    ("Repository not found",
     "wrong URL, or the repo is private and the PAT cannot see it"),
    ("403", "the PAT lacks permission — a fine-grained token needs Contents: Read and write "
            "on this repository"),
    ("insufficient scope",
     "classic PAT is missing the `repo` scope"),
)

FORBIDDEN_PATHS = ("audit_key",)  # the blind audit key must never be staged, committed, or pushed


class GitError(RuntimeError):
    """A git command failed. The message is already redacted."""

    def __init__(self, argv, code: int, out: str, err: str):
        self.argv, self.code, self.out, self.err = argv, code, out, err
        detail = "\n".join(part for part in (out, err) if part)
        super().__init__(f"`{' '.join(argv)}` exited {code}\n{detail}".strip())


def redact(text: str, token: str | None = None) -> str:
    """Mask the token (if known) plus anything that merely looks like a credential."""
    if not text:
        return ""
    if token:
        text = text.replace(token, MASK)
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def read_secret(names: tuple[str, ...] = TOKEN_NAMES) -> str | None:
    """First hit in Colab Secrets, then in the environment. Returns ``None`` rather than raising."""
    try:                                                # only importable inside Colab
        from google.colab import userdata               # type: ignore[import-not-found]
    except Exception:                                   # noqa: BLE001 - not on Colab, or unset
        userdata = None
    for name in names:
        if userdata is not None:
            try:
                value = userdata.get(name)
            except Exception:                           # noqa: BLE001 - SecretNotFound
                value = None
            if value:
                return value.strip()
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return None


class Git:
    """`git` in one directory, with every output redacted and no interactive prompt possible."""

    def __init__(self, repo: str | Path, token: str | None = None):
        self.repo = str(repo)
        self.token = token

    def run(self, *args: str, check: bool = True, stdin: str | None = None,
            timeout: int = 600) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.update(GIT_TERMINAL_PROMPT="0", GIT_PAGER="cat", GIT_ASKPASS="echo", PAGER="cat")
        proc = subprocess.run(["git", *args], cwd=self.repo, input=stdin,
                              capture_output=True, text=True, env=env, timeout=timeout)
        cp = subprocess.CompletedProcess(
            tuple(redact(a, self.token) for a in ("git", *args)),
            proc.returncode,
            redact(proc.stdout, self.token).strip(),
            redact(proc.stderr, self.token).strip(),
        )
        if check and proc.returncode:
            raise GitError(cp.args, cp.returncode, cp.stdout, cp.stderr)
        return cp

    def get(self, *args: str) -> str:
        return self.run(*args, check=False).stdout


def credential_path() -> Path:
    return Path(os.environ.get("HOME") or Path.home()) / ".git-credentials"


def _lines_for_other_hosts(path: Path, host: str) -> list[str]:
    """Keep entries for every host except the one we are about to rewrite."""
    if not path.is_file():
        return []
    kept = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            found = parse.urlparse(line).hostname
        except ValueError:                              # noqa: PERF203 - unreadable line
            found = None
        if found != host:
            kept.append(line)
    return kept


def write_credentials(user: str, token: str, host: str = HOST,
                      path: Path | None = None) -> Path:
    """Write the credential line with mode 0600, preserving entries for other hosts."""
    path = path or credential_path()
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    lines = _lines_for_other_hosts(path, host) + [f"https://{user}:{token}@{host}"]
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    os.chmod(path, 0o600)                               # also tightens a pre-existing file
    return path


def resolve_user(token: str, api: str = API_USER, timeout: int = 20) -> dict:
    """Who is this token? Used for the credential username and the commit identity.

    Falls back to ``x-access-token`` (which GitHub accepts in the username slot) when the API
    is unreachable — a fresh VM may have no route out, and the push is still worth trying.
    """
    req = request.Request(api, headers={"Authorization": f"Bearer {token}",
                                        "Accept": "application/vnd.github+json",
                                        "User-Agent": "embeded-colab-git"})
    try:
        with request.urlopen(req, timeout=timeout) as resp:      # noqa: S310 - fixed https URL
            body = json.loads(resp.read().decode("utf-8"))
            scopes = (resp.headers.get("X-OAuth-Scopes") or "").strip()
        return {"login": body.get("login") or "x-access-token",
                "id": body.get("id"), "scopes": scopes, "api_ok": True}
    except (error.URLError, error.HTTPError, OSError, ValueError, TimeoutError):
        return {"login": "x-access-token", "id": None, "scopes": "", "api_ok": False}


def configure(repo: str | Path, token: str, host: str = HOST) -> dict:
    """Store the token, point git's store helper at it, and set an identity if there is none."""
    info = resolve_user(token)
    user = info["login"]
    path = write_credentials(user, token, host)
    git = Git(repo, token)
    git.run("config", "--local", "--replace-all", "credential.helper", "")   # reset any system helper
    git.run("config", "--local", "--add", "credential.helper", "store")
    identity = "pre-existing"
    if not git.get("config", "--get", "user.email"):
        if info["id"]:
            git.run("config", "--local", "user.name", user)
            git.run("config", "--local", "user.email", f"{info['id']}+{user}@users.noreply.github.com")
            identity = f"set from the token ({user})"
        else:
            git.run("config", "--local", "user.name", user)
            git.run("config", "--local", "user.email", f"{user}@users.noreply.github.com")
            identity = ("GUESSED — api.github.com was unreachable, so the commit will not be "
                        "attributed to you; set git config user.name/user.email and amend if that "
                        "matters")
    return {"username": user, "credential_file": str(path),
            "mode": oct(path.stat().st_mode & 0o777), "identity": identity,
            "token_scopes": info["scopes"] or "(fine-grained, or API unreachable)",
            "api_ok": info["api_ok"], **verify_credentials(repo, token, host)}


def verify_credentials(repo: str | Path, token: str, host: str = HOST) -> dict:
    """Ask git itself for the github.com credential. Never prints the answer, only whether it works."""
    git = Git(repo, token)
    cp = git.run("credential", "fill", check=False, stdin=f"protocol=https\nhost={host}\n\n")
    stored = ""
    path = credential_path()
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                parts = parse.urlparse(line.strip())
            except ValueError:
                continue
            if parts.hostname == host:
                stored = parse.unquote(parts.password or "")
    return {"helper_wired": cp.returncode == 0 and f"password={MASK}" in cp.stdout,
            "stored_matches_token": stored == token}


def _cause(text: str) -> str:
    for needle, advice in KNOWN_CAUSES:
        if needle.lower() in text.lower():
            return advice
    return "no known cause matched — read the stderr above"


def diagnose(repo: str | Path = ".", ref: str | None = None, token: str | None = None) -> list[str]:
    """Read-only inspection. Returns printable lines; never raises, never prints a credential."""
    git = Git(repo, token)
    lines = [f"repo: {git.repo}"]
    for label, args in (
        ("branch", ("rev-parse", "--abbrev-ref", "HEAD")),
        ("head", ("rev-parse", "--short", "HEAD")),
        ("identity", ("config", "--get", "user.email")),
        ("helpers", ("config", "--get-all", "credential.helper")),
        ("remote", ("remote", "-v")),
    ):
        lines.append(f"{label}: {git.get(*args) or '(unset)'}")
    for name in ("rebase-merge", "rebase-apply"):
        if (Path(git.repo) / ".git" / name).exists():
            lines.append(f"warning: .git/{name} exists — an interrupted rebase is in progress")
    status = git.get("status", "--porcelain=v1")
    lines.append(f"worktree: {'clean' if not status else 'DIRTY'}")
    lines.extend(f"  {line}" for line in status.splitlines())
    path = credential_path()
    lines.append(f"credential file: {path} "
                 f"{'exists, mode ' + oct(path.stat().st_mode & 0o777) if path.is_file() else 'absent'}")
    lines.append(f"log: {git.get('log', '--oneline', '-3').replace(chr(10), ' | ') or '(no commits)'}")

    target = ref or git.get("rev-parse", "--abbrev-ref", "HEAD")
    fetch = git.run("fetch", "origin", target, check=False)
    lines.append(f"fetch origin {target}: exit {fetch.returncode}")
    lines.extend(f"  {line}" for line in (fetch.stdout + "\n" + fetch.stderr).splitlines() if line)
    if fetch.returncode == 0:
        counts = git.get("rev-list", "--left-right", "--count", f"FETCH_HEAD...HEAD")
        if counts:
            behind, ahead = (counts.split() + ["0", "0"])[:2]
            lines.append(f"ahead/behind vs origin/{target}: ahead {ahead}, behind {behind}")

    failures: list[tuple[str, int, str]] = []
    if fetch.returncode:
        failures.append((f"fetch origin {target}", fetch.returncode,
                         f"{fetch.stdout}\n{fetch.stderr}"))
    if status:
        # the same 128 git itself would print, so the advice is matched against git's own words
        failures.append(("pull --rebase", 128,
                         "error: cannot pull with rebase: You have unstaged changes.\n"
                         "error: please commit or stash them."))
    for label, code, blob in failures:
        lines.append(f"likely cause of `{label}` exiting {code}: {_cause(blob)}")
        lines.extend(f"  git said: {line}" for line in blob.strip().splitlines() if line.strip())
    if not failures:
        lines.append("no known blocker found — `push` should work")
    return lines


def _guard_paths(paths: list[str]) -> None:
    bad = [p for p in paths if any(token in p for token in FORBIDDEN_PATHS)]
    if bad:
        raise ValueError(f"refusing to stage {bad}: the blind audit key must never be committed")


def sync_and_push(repo: str | Path = ".", ref: str | None = None, token: str | None = None,
                  commit: str | None = None, paths: list[str] | None = None,
                  dry_run: bool = False) -> dict:
    """Stage/commit (optional) → pull --rebase → push. Returns a report; raises only on misuse."""
    out: dict = {"ok": False, "stage": "start", "lines": []}
    git = Git(repo, token)
    target = ref or git.get("rev-parse", "--abbrev-ref", "HEAD")
    out["ref"] = target

    if token:
        out["stage"] = "credentials"
        out["credentials"] = configure(repo, token)
        out["lines"] += [f"username: {out['credentials']['username']}",
                         f"credential file: {out['credentials']['credential_file']} "
                         f"(mode {out['credentials']['mode']})",
                         f"store helper wired: {out['credentials']['helper_wired']}, "
                         f"stored token matches: {out['credentials']['stored_matches_token']}",
                         f"scopes: {out['credentials']['token_scopes']}",
                         f"identity: {out['credentials']['identity']}"]
        if not (out["credentials"]["helper_wired"] and out["credentials"]["stored_matches_token"]):
            out["message"] = "the credential helper did not pick the token up; nothing pushed"
            return out

    out["stage"] = "worktree"
    try:
        if paths:
            _guard_paths(paths)
            for path in paths:
                git.run("add", "--", path)
        if commit:
            staged = git.get("diff", "--cached", "--name-only")
            if staged:
                git.run("commit", "-m", commit)
                out["lines"].append(f"committed: {commit}\n  " + staged.replace("\n", "\n  "))
            else:
                out["lines"].append("nothing staged; skipping the commit")
    except GitError as exc:
        blob = f"{exc.out}\n{exc.err}"
        out["message"] = f"{exc}; likely cause: {_cause(blob)}"
        return out
    dirty = git.get("status", "--porcelain=v1")
    if dirty:
        out["message"] = ("the worktree is dirty — commit or stash it first "
                          "(or pass --commit MSG --paths ...). Untracked/unstaged:\n  "
                          + dirty.replace("\n", "\n  "))
        out["lines"] += [f"  {line}" for line in dirty.splitlines()]
        return out

    out["stage"] = "pull --rebase"
    pull = git.run("pull", "--rebase", "origin", target, check=False)
    out["lines"] += [f"pull --rebase origin {target}: exit {pull.returncode}"]
    out["lines"] += [f"  {line}" for line in (pull.stdout + "\n" + pull.stderr).splitlines() if line]
    if pull.returncode:
        out["message"] = f"pull --rebase failed: {_cause(pull.stdout + pull.stderr)}"
        if "conflict" in (pull.stdout + pull.stderr).lower():
            out["lines"].append("resolve the conflict, then `git rebase --continue`; "
                                "or `git rebase --abort` to get back to where you were")
        return out

    out["stage"] = "push"
    push = git.run("push", *(("--dry-run",) if dry_run else ()), "origin", f"HEAD:{target}",
                   check=False)
    out["lines"] += [f"push origin HEAD:{target}{' (dry run)' if dry_run else ''}: "
                     f"exit {push.returncode}"]
    out["lines"] += [f"  {line}" for line in (push.stdout + "\n" + push.stderr).splitlines() if line]
    if push.returncode:
        out["message"] = f"push failed: {_cause(push.stdout + push.stderr)}"
        return out

    out["ok"] = True
    if not dry_run:
        remote = git.get("ls-remote", "origin", target).split()
        out["remote_sha"] = remote[0][:7] if remote else "?"
        out["local_sha"] = git.get("rev-parse", "--short", "HEAD")
        out["message"] = (f"pushed; origin/{target} is now {out['remote_sha']} "
                          f"(local {out['local_sha']}) — revoke the token when the run is done")
    else:
        out["message"] = "dry run: nothing was pushed"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("diagnose", "push"))
    ap.add_argument("--repo", default=".", help="path to the git clone (default: cwd)")
    ap.add_argument("--ref", default=None, help="branch to sync (default: current branch)")
    ap.add_argument("--token-name", default=TOKEN_NAMES[0],
                    help="Colab Secret / env var holding the PAT (default: GITHUB_TOKEN)")
    ap.add_argument("--commit", default=None, help="commit the staged paths with this message")
    ap.add_argument("--paths", nargs="*", default=None,
                    help="paths to `git add` before committing (never audit_key*)")
    ap.add_argument("--dry-run", action="store_true", help="push --dry-run")
    args = ap.parse_args(argv)

    token = read_secret((args.token_name, *TOKEN_NAMES))
    if args.command == "diagnose":
        for line in diagnose(args.repo, args.ref, token):
            print(line)
        if not token:
            print("no token found: push will fail on a VM with no credential helper. "
                  f"Put the PAT in Colab Secrets as {args.token_name}.")
        return 0

    if not token:
        print(f"no token in Colab Secrets or the environment "
              f"(looked for {args.token_name}, {', '.join(TOKEN_NAMES)}); "
              "nothing was pushed. Diagnostics for the current state:")
        for line in diagnose(args.repo, args.ref, token):
            print("  " + line)
        return 1
    result = sync_and_push(args.repo, args.ref, token, args.commit, args.paths, args.dry_run)
    for line in result["lines"]:
        print(line)
    print(("ok  " if result["ok"] else "FAIL") + f" [{result['stage']}] {result['message']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
