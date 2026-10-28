"""Smoke tests on the real-data sample extract.

The sample is not committed (Basketball-Reference terms); `python -m
bbs.pipeline` writes it to data/sample/ on its first full run. These tests
skip until then.
"""

import pandas as pd
import pytest

from bbs import analysis, features, pipeline

pytestmark = pytest.mark.skipif(
    not pipeline.SAMPLE_FILE.exists(),
    reason="sample not generated yet; run `python -m bbs.pipeline` once")


@pytest.fixture(scope="module")
def sample_table():
    ps = pd.read_csv(pipeline.SAMPLE_FILE)
    return ps, features.build_model_table(ps, pipeline.SAMPLE_DESIGN)


def test_sample_has_required_columns_and_unique_keys(sample_table):
    ps, _ = sample_table
    assert set(pipeline.SAMPLE_COLUMNS) <= set(ps.columns)
    assert not ps.duplicated(["season", "player_id"]).any()
    assert ps["season"].min() == pipeline.SAMPLE_FIRST_SEASON


def test_sample_model_table_is_sane(sample_table):
    _, mt = sample_table
    assert len(mt) > 1000
    assert mt["age"].max() <= pipeline.SAMPLE_DESIGN.max_age
    assert mt["weight"].between(0, 1).all()
    obs = mt[mt["observed"] == 1]
    assert obs["bpm_next"].notna().all()
    assert (obs["mp_next"] >= pipeline.SAMPLE_DESIGN.min_mp_next).all()


def test_interpretive_features_respect_collinearity_limit(sample_table):
    _, mt = sample_table
    kept = [f for f in analysis.INTERPRETIVE_FEATURES if f not in analysis.COLLINEAR_DROP]
    _, _, r = analysis.max_abs_correlation(mt, kept)
    assert r < analysis.COLLINEARITY_LIMIT


def test_analysis_tables_run_on_sample(sample_table):
    _, mt = sample_table
    rtm = analysis.regression_to_mean(mt)
    assert {"mean_delta", "ci_low", "ci_high", "minutes"} <= set(rtm.columns)
    assert (rtm["ci_low"] <= rtm["mean_delta"]).all()
    sig = analysis.signal_regression(mt)
    assert sig["ci_low"].le(sig["coef_per_sd"]).all() and sig.attrs["n"] > 500


@pytest.mark.slow
def test_full_sample_pipeline_runs():
    res = pipeline.run(sample=True)
    assert res["mode"] == "sample" and res["n_test_rows"] > 0
