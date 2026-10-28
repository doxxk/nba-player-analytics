import numpy as np
import pandas as pd
import pytest

from bbs import models
from bbs.config import StudyDesign
from conftest import synthetic_model_table


def test_weighted_metrics():
    assert models.weighted_rmse([1, 2], [1, 4], [1, 1]) == pytest.approx(np.sqrt(2))
    assert models.weighted_rmse([1, 2], [1, 4], [1, 0]) == 0.0
    assert np.isnan(models.weighted_rmse([1], [2], [0]))
    assert models.weighted_mae([0, 0], [1, -3]) == 2.0


def test_top_decile_hit_rate():
    pred = np.arange(20)
    actual = np.where(pred >= 18, 2.0, 0.0)
    assert models.top_decile_hit_rate(pred, actual, threshold=1.5) == 1.0
    assert np.isnan(models.top_decile_hit_rate(pred[:5], actual[:5]))
    assert np.isnan(models.top_decile_hit_rate(np.zeros(20), actual))


def test_persistence_and_age_curve():
    df = synthetic_model_table()
    assert np.allclose(models.Persistence().predict(df), df["bpm"])
    ac = models.AgeCurve().fit(df)
    assert ac.predict(df).mean() == pytest.approx(df["bpm_next"].mean(), abs=0.05)


def test_marcel_single_season_with_no_regression_is_identity():
    df = pd.DataFrame({"bpm": [2.0, -1.0], "mp": [1000, 500], "bpm_lag1": [np.nan, np.nan],
                       "mp_lag1": [np.nan, np.nan], "bpm_lag2": [np.nan] * 2,
                       "mp_lag2": [np.nan] * 2, "age": [22, 22]})
    m = models.Marcel(prior=0.0)
    assert np.allclose(m._regressed(df, k=0), [2.0, -1.0])
    # With k minutes of a zero prior, a 1,000-minute season is pulled halfway.
    assert m._regressed(df, k=1000)[0] == pytest.approx(1.0)


def test_marcel_weights_history_by_minutes():
    df = pd.DataFrame({"bpm": [3.0], "mp": [1000], "bpm_lag1": [-1.0], "mp_lag1": [1250],
                       "bpm_lag2": [np.nan], "mp_lag2": [np.nan], "age": [22]})
    m = models.Marcel(prior=0.0)
    # weights 5/4 on minutes: 1000*1 and 1250*0.8 = equal effective minutes
    assert m._regressed(df, k=0)[0] == pytest.approx(1.0)


def test_marcel_beats_persistence_on_noisy_data():
    df = synthetic_model_table(seed=3)
    train, test = df[df.season < 2016], df[df.season >= 2016]
    mar = models.Marcel().fit(train)
    assert mar.k in mar.k_grid
    rmse_m = models.weighted_rmse(test.bpm_next, mar.predict(test))
    rmse_p = models.weighted_rmse(test.bpm_next, models.Persistence().predict(test))
    assert rmse_m < rmse_p


class _Recorder:
    """A fake model that records which seasons it was trained on."""

    name = "Recorder"
    seen: list = []

    def fit(self, df):
        _Recorder.seen.append((df["season"].max(), df["observed"].min()))
        return self

    def predict(self, df):
        return np.zeros(len(df))


def test_rolling_origin_never_trains_on_test_or_later_seasons():
    df = synthetic_model_table()
    df.loc[df.sample(frac=0.1, random_state=0).index, "observed"] = 0
    _Recorder.seen = []
    design = StudyDesign(first_test_season=2014, last_test_season=2019)
    preds = models.rolling_origin(df, lambda: [_Recorder()], design)
    years = sorted(preds["season"].unique())
    assert years == list(range(2014, 2020))
    for (max_train_season, min_observed), year in zip(_Recorder.seen, years):
        assert max_train_season < year
        assert min_observed == 1
    assert (preds["bpm_next"].notna()).all()


def test_paired_bootstrap_identical_models_have_zero_difference():
    df = synthetic_model_table()
    base = df.assign(pred=df["bpm"])
    preds = pd.concat([base.assign(model="A"), base.assign(model="B")])
    res = models.paired_bootstrap(preds, "A", "B", n_boot=200)
    assert res["rmse_diff"] == pytest.approx(0.0)
    assert res["ci_low"] == pytest.approx(0.0) and res["ci_high"] == pytest.approx(0.0)


def test_paired_bootstrap_detects_better_model():
    df = synthetic_model_table()
    good = df.assign(pred=df["bpm_next"] + 0.1, model="Good")
    bad = df.assign(pred=df["bpm"], model="Bad")
    res = models.paired_bootstrap(pd.concat([good, bad]), "Good", "Bad", n_boot=300)
    assert res["ci_high"] < 0 and res["p_model_better"] == 1.0


def test_online_conformal_uses_only_past_residuals():
    rows = []
    for s, err in [(2010, 1.0), (2011, 1.0), (2012, 5.0)]:
        for i in range(120):
            rows.append({"model": "M", "season": s, "pred": 0.0, "bpm_next": err})
    out = models.online_conformal(pd.DataFrame(rows), "M", level=0.8)
    assert sorted(out["season"].unique()) == [2011, 2012]
    # 2012's interval is built from 2010-2011 errors only, so it misses 2012.
    assert out.loc[out.season == 2012, "half_width"].iloc[0] == pytest.approx(1.0)
    assert out.loc[out.season == 2012, "covered"].sum() == 0


def test_ridge_wrapper_fits_and_predicts_with_missing_values():
    df = synthetic_model_table()
    df["usg_percent"] = np.where(df.index % 7 == 0, np.nan, 20.0)
    feats = ["marcel_pred", "age", "bpm", "mp", "bpm_lag1", "usg_percent"]
    m = models.SklearnModel("Ridge", feats, "ridge", 10.0).fit(df)
    p = m.predict(df)
    assert p.shape == (len(df),) and np.isfinite(p).all()


def test_ipw_weights_are_bounded():
    df = synthetic_model_table()
    df["observed"] = (np.random.default_rng(0).random(len(df)) > 0.2).astype(int)
    for c in models.IPW_FEATURES:
        if c not in df:
            df[c] = 1.0
    out = models.add_ipw(df)
    assert out["ipw"].between(1, 10).all()
    assert out["p_observed"].between(0, 1).all()


def test_marcel_refuses_insufficient_data():
    df = synthetic_model_table().head(5)
    with pytest.raises(ValueError, match="at least 20"):
        models.Marcel().fit(df)


def test_tune_refuses_too_few_seasons():
    df = synthetic_model_table(seasons=range(2010, 2014))
    with pytest.raises(ValueError, match="at least 5 pre-test seasons"):
        models.tune(df, StudyDesign(tuning_cutoff=2013), features=["marcel_pred", "bpm"])
