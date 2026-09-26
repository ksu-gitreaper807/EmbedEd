"""The three pin sources must agree exactly: settings.PINNED ==
embeded/requirements.txt == the notebook's install cell.

'change both together' is a human promise until it is a test. Two past failures
motivate this: torch was once pinned only in the notebook cell and drifted to a
version with no wheels for current Colab's Python (2.3.1 vs py3.13, 2026-09);
then it was hard-pinned (2.6.0), which installed but forced a ~3 GB re-download
every session and desynced Colab's matched torchvision. Final policy: torch is
deliberately UNpinned everywhere (Colab's preinstalled build is GPU-matched) —
so this suite also rejects any `torch==` pin sneaking back into any source.
"""
import json
import re
from pathlib import Path

from embeded import settings as S

ROOT = Path(__file__).resolve().parents[2]
REQS = ROOT / "embeded" / "requirements.txt"
NOTEBOOK = ROOT / "notebooks" / "Phase0_Phase1_Colab.ipynb"

_PIN = re.compile(r"^\s*([A-Za-z0-9_.-]+)==([^\s#;]+)\s*$", re.M)          # one per line
_PIN_ANYWHERE = re.compile(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!*]+)")  # pip cell: several per line


def _pinned() -> dict:
    return dict(p.split("==", 1) for p in S.PINNED)


def _pins(text: str, anywhere: bool = False) -> dict:
    rx = _PIN_ANYWHERE if anywhere else _PIN
    return dict(rx.findall(text))


def _pip_cell() -> str:
    nb = json.loads(NOTEBOOK.read_text())
    cells = [
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code" and "%pip install" in "".join(c["source"])
    ]
    assert len(cells) == 1, "expected exactly one %pip install cell"
    return cells[0]


def test_requirements_mirror_pinned_exactly():
    assert _pins(REQS.read_text()) == _pinned()


def test_notebook_install_cell_mirror_pinned_exactly():
    assert _pins(_pip_cell(), anywhere=True) == _pinned()


def test_torch_stays_on_the_platform_build():
    # torch must not be pinned anywhere — see module docstring for the two
    # failure modes (drifted notebook-only pin; wasteful hard pin) this guards.
    # \d after == so prose like "never add torch== to this cell" doesn't trip it.
    assert "torch" not in _pinned()
    assert not re.search(r"\btorch==\d", _pip_cell())
    assert not re.search(r"^torch==\d", REQS.read_text(), re.M)
