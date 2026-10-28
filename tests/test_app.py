"""The dashboard renders every tab without exceptions on the committed results."""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "reports" / "tables" / "results.json").exists(),
                    reason="run the pipeline first")
def test_dashboard_renders():
    at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 6
    assert at.metric[0].value.endswith("%")
