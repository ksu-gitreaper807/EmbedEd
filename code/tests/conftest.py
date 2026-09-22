"""Shared fixture: tmp artifacts + generated mini dataset, fully offline."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def fx(tmp_path, monkeypatch):
    from code.fixtures import make_fixture
    from code import settings as S

    fdir = tmp_path / "fixture"
    make_fixture.main(fdir)
    meta = json.loads((fdir / "fixture_meta.json").read_text())

    monkeypatch.setattr(S, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(S, "REPORT_MD", tmp_path / "report" / "measurements.md")

    from code.data import prepare_data as P
    P.main(["--dir", str(fdir)])

    class NS: pass
    n = NS()
    n.fdir, n.meta, n.tmp = fdir, meta, tmp_path
    n.clones = {k: set(v) for k, v in P.labeled_clones().items()}
    n.corpus = P.corpus_ids()
    n.frags = P.load_fragments()
    n.P = P
    return n
