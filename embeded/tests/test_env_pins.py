"""The three pin sources must agree exactly: settings.PINNED ==
embeded/requirements.txt == the notebook's install cell.

'change both together' is a human promise until it is a test. torch was once
pinned only in the notebook cell and drifted to a version with no wheels for
current Colab's Python (2.3.1 vs py3.13, 2026-09) — a session died on
`pip install` before this test existed. Keep every pin in all three places.
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


def test_requirements_mirror_pinned_exactly():
    assert _pins(REQS.read_text()) == _pinned()


def test_notebook_install_cell_mirror_pinned_exactly():
    nb = json.loads(NOTEBOOK.read_text())
    cells = [
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code" and "%pip install" in "".join(c["source"])
    ]
    assert len(cells) == 1, "expected exactly one %pip install cell"
    assert _pins(cells[0], anywhere=True) == _pinned()


def test_torch_is_pinned_like_the_rest():
    # torch must live in PINNED itself — a notebook-only pin is how it drifted.
    assert "torch" in _pinned()
