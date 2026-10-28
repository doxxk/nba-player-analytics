import numpy as np
import pandas as pd
import pytest

from bbs import features
from bbs.config import StudyDesign
from bbs.data import clean


def test_safe_divide_handles_zero_and_missing_denominators():
    out = features.safe_divide([1, 2, 3, 4], [2, 0, np.nan, 4])
    assert out[0] == 0.5 and out[3] == 1.0
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_true_shooting_known_value_and_zero_attempts():
    # 30 points on 20 FGA and 10 FTA: 30 / (2 * 24.4)
    assert features.true_shooting(30, 20, 10) == pytest.approx(30 / 48.8)
    assert np.isnan(features.true_shooting(0, 0, 0))


def test_beta_prior_recovers_simulated_population():
    rng = np.random.default_rng(1)
    p0, k = 0.75, 40.0
    talent = rng.beta(p0 * k, (1 - p0) * k, size=4000)
    attempts = rng.integers(50, 400, size=4000)
    made = rng.binomial(attempts, talent)
    est_p0, est_k = features.estimate_beta_prior(made, attempts)
    assert est_p0 == pytest.approx(p0, abs=0.01)
    assert est_k == pytest.approx(k, rel=0.25)


def test_beta_prior_needs_enough_players():
    with pytest.raises(ValueError):
        features.estimate_beta_prior([1, 2], [10, 10])


def test_beta_prior_with_no_true_talent_variance_is_infinite():
    # Every player shoots exactly the league rate: observed spread < binomial noise.
    p0, k = features.estimate_beta_prior([5] * 50, [10] * 50)
    assert p0 == 0.5 and np.isinf(k)


def test_shrink_rate_edge_cases():
    assert features.shrink_rate(0, 0, 0.75, 20) == pytest.approx(0.75)
    assert features.shrink_rate(np.nan, np.nan, 0.75, 20) == pytest.approx(0.75)
    assert features.shrink_rate(10, 10, 0.75, np.inf) == pytest.approx(0.75)
    # Large samples barely move.
    assert features.shrink_rate(900, 1000, 0.75, 20) == pytest.approx(0.897, abs=0.01)


def _model_table(raw_tables, **design):
    ps = clean.build_player_seasons(raw_tables, first_season=2020)
    ps = features.add_era_adjusted(ps, raw_tables["team_summaries"])
    ps, _ = features.add_shrunk_shooting(ps, prior_seasons_max=2023)
    d = StudyDesign(**{"max_age": 30, "min_mp": 500, "first_base_season": 2020, **design})
    return features.build_model_table(ps, d), ps


def test_lags_only_use_consecutive_seasons(raw_tables):
    mt, _ = _model_table(raw_tables)
    gamma22 = mt.loc[(mt.player_id == "gamma") & (mt.season == 2022)].iloc[0]
    assert np.isnan(gamma22["bpm_lag1"])  # skipped 2021, so no t-1 value
    assert gamma22["bpm_lag2"] == 0.0  # 2020 is t-2
    alpha22 = mt.loc[(mt.player_id == "alpha") & (mt.season == 2022)].iloc[0]
    assert alpha22["bpm_lag1"] == 0.5 and alpha22["bpm_change"] == pytest.approx(1.0)


def test_outcome_and_observed_flag(raw_tables):
    mt, _ = _model_table(raw_tables, min_mp_next=250)
    a21 = mt.loc[(mt.player_id == "alpha") & (mt.season == 2021)].iloc[0]
    assert a21["bpm_next"] == 1.5 and a21["observed"] == 1
    assert a21["delta_bpm"] == pytest.approx(1.0)
    # gamma has no 2021 season, so the 2020 row has no usable outcome
    g20 = mt.loc[(mt.player_id == "gamma") & (mt.season == 2020)].iloc[0]
    assert g20["observed"] == 0 and g20["mp_next"] == 0 and g20["weight"] == 0


def test_features_do_not_depend_on_next_season(raw_tables):
    """Changing season t+1 must not change any feature of season t (no leakage)."""
    mt1, _ = _model_table(raw_tables)
    altered = {k: v.copy() for k, v in raw_tables.items()}
    adv = altered["advanced"]
    mask = (adv.player_id == "alpha") & (adv.season == 2023)
    adv.loc[mask, ["bpm", "ts_percent", "usg_percent"]] = [99.0, 0.99, 99.0]
    mt2, _ = _model_table(altered)
    row = lambda m: m.loc[(m.player_id == "alpha") & (m.season == 2022)].iloc[0]  # noqa: E731
    cols = [c for c in features.ALL_FEATURES if c in mt1.columns]
    pd.testing.assert_series_equal(row(mt1)[cols], row(mt2)[cols])
    assert row(mt2)["bpm_next"] == 99.0


def test_eligibility_filters(raw_tables):
    mt, _ = _model_table(raw_tables, min_mp=1000)
    assert (mt["mp"] >= 1000).all() and len(mt) > 0
    young, _ = _model_table(raw_tables, max_age=22)
    assert young["age"].max() <= 22 and "gamma" not in set(young["player_id"])
    no_covid, _ = _model_table(raw_tables, exclude_outcome_seasons=(2022,))
    assert 2022 not in set(no_covid["outcome_season"])


def test_era_adjustment_is_relative_to_league(raw_tables):
    _, ps = _model_table(raw_tables)
    assert ps["ts_rel"].iloc[0] == pytest.approx(0.56 - 0.57)
    assert ps["x3p_ar_rel"].iloc[0] == pytest.approx(0.35 - 0.38)
