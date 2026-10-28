"""Exploratory statistics and the interpretive (inference) model.

The predictive models in ``models.py`` answer "how well can we forecast?".
This module answers "what is going on?": how much apparent improvement is
regression to the mean, how development varies with age, who disappears from
the sample, and which signals move the forecast once age and track record are
accounted for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from bbs.features import FEATURE_GROUPS
from bbs.models import Marcel, add_ipw


def _wmean_ci(x: pd.Series, w: pd.Series) -> tuple[float, float, float, int]:
    """Weighted mean with a normal-approximation 95% CI (Kish effective n)."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    if len(x) < 2 or w.sum() == 0:
        return float("nan"), float("nan"), float("nan"), len(x)
    m = np.average(x, weights=w)
    var = np.average((x - m) ** 2, weights=w)
    n_eff = w.sum() ** 2 / np.sum(w ** 2)
    se = np.sqrt(var / n_eff)
    return float(m), float(m - 1.96 * se), float(m + 1.96 * se), len(x)


def regression_to_mean(mt: pd.DataFrame, bins=(-10, -4, -2, 0, 2, 4, 15)) -> pd.DataFrame:
    """Average next-season change by this-season BPM, split by minutes played.

    If changes were pure development, the lines would be flat. A downward slope
    means extreme seasons were partly luck, and the slope is steeper for
    low-minute players because their BPM is noisier.
    """
    obs = mt.loc[mt["observed"] == 1].copy()
    obs["bpm_bin"] = pd.cut(obs["bpm"], bins=list(bins))
    obs["minutes"] = np.where(obs["mp"] < 1500, "500-1,499 MP", "1,500+ MP")
    rows = []
    for (b, m), g in obs.groupby(["bpm_bin", "minutes"], observed=True):
        mean, lo, hi, n = _wmean_ci(g["delta_bpm"], g["weight"])
        rows.append({"bpm_bin": str(b), "bin_mid": float(g["bpm"].mean()), "minutes": m,
                     "mean_delta": mean, "ci_low": lo, "ci_high": hi, "n": n})
    return pd.DataFrame(rows)


def age_curve(mt: pd.DataFrame) -> pd.DataFrame:
    """Mean next-season BPM change and dropout rate by age in the base season."""
    rows = []
    for age, g in mt.groupby(mt["age"].clip(lower=19).astype(int)):
        obs = g.loc[g["observed"] == 1]
        mean, lo, hi, n = _wmean_ci(obs["delta_bpm"], obs["weight"])
        rows.append({"age": int(age), "mean_delta": mean, "ci_low": lo, "ci_high": hi,
                     "n_observed": n, "n_total": len(g),
                     "dropout_rate": 1 - g["observed"].mean()})
    return pd.DataFrame(rows)


def attrition_by_projection(mt: pd.DataFrame, train_max_season: int | None = None
                            ) -> pd.DataFrame:
    """Dropout rate by quintile of Marcel's projected next-season BPM.

    Shows that the players with no usable outcome are not missing at random:
    those projected to be worst vanish most often, so models trained on
    survivors see an over-optimistic picture of low-end players.
    """
    seasons = mt["season"] <= (train_max_season or mt["season"].max())
    pool = mt.loc[seasons & (mt["season"] < mt["season"].max())].copy()
    marcel = Marcel().fit(pool.loc[pool["observed"] == 1])
    pool["marcel_pred"] = marcel.predict(pool)
    pool["quintile"] = pd.qcut(pool["marcel_pred"], 5, labels=[
        "Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
    out = pool.groupby("quintile", observed=True).agg(
        n=("observed", "size"), dropout_rate=("observed", lambda s: 1 - s.mean()),
        mean_projection=("marcel_pred", "mean"), mean_mp=("mp", "mean")).reset_index()
    return out


INTERPRETIVE_FEATURES = (
    ["bpm_change", "gs_share"]
    + FEATURE_GROUPS["box_rates"] + FEATURE_GROUPS["shot_profile"]
    + FEATURE_GROUPS["on_off"] + FEATURE_GROUPS["pedigree"] + FEATURE_GROUPS["team_context"]
)
# One member of each strongly correlated pair (|r| >= 0.75 in the full sample)
# is left out of the interpretive regression so coefficients stay readable:
# 3PA per 100 and shot distance duplicate era-adjusted 3PA rate and rim share;
# raw on-court +/- mostly measures team quality (r = 0.80 with team SRS), so
# the team-relative on/off differential is used instead; weight tracks height;
# rim-attempt share is the hub of the shot-profile cluster (r = 0.74-0.79 with
# dunk share, offensive rebound rate, and 3PA rate since 2014), so it and dunk
# share are left out. All of these stay in the predictive models, where ridge
# handles collinearity. `max_abs_correlation` checks the rule on loaded data.
COLLINEAR_DROP = ["x3pa_per_100_poss", "avg_dist_fga", "on_court_pm", "weight_lb",
                  "percent_dunks_of_fga", "percent_fga_from_x0_3_range"]
COLLINEARITY_LIMIT = 0.75


def max_abs_correlation(mt: pd.DataFrame, features: list[str]) -> tuple[str, str, float]:
    """The most correlated pair among ``features`` (to verify the drop rule)."""
    c = mt[features].corr().abs()
    c = c.where(np.triu(np.ones(c.shape, dtype=bool), 1)).stack()
    (a, b), r = c.idxmax(), c.max()
    return a, b, float(r)


def signal_regression(mt: pd.DataFrame, features: list[str] | None = None,
                      use_ipw: bool = False) -> pd.DataFrame:
    """Which signals predict next-season BPM *beyond* the Marcel projection?

    Fits weighted least squares of (actual - Marcel projection) on standardized
    features, with standard errors clustered by player. Each coefficient is the
    extra BPM expected next season per one-standard-deviation difference in the
    feature, holding the others fixed. These are predictive associations, not
    causal effects. With ``use_ipw`` the weights are multiplied by inverse
    probabilities of remaining in the sample (survivorship check).
    """
    features = features or [f for f in INTERPRETIVE_FEATURES if f not in COLLINEAR_DROP]
    pool = mt.loc[mt["season"] < mt["season"].max()]
    if use_ipw:
        pool = add_ipw(pool)
    obs = pool.loc[pool["observed"] == 1].copy()
    w = obs["weight"] * (obs["ipw"] if use_ipw else 1.0)
    marcel = Marcel().fit(obs)
    obs["resid"] = obs["bpm_next"] - marcel.predict(obs)
    X = obs[features].astype(float)
    X = X.fillna(X.median())
    sd = X.std(ddof=0).replace(0, 1)
    Xz = sm.add_constant((X - X.mean()) / sd)
    fit = sm.WLS(obs["resid"], Xz, weights=w).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(obs["player_id"])[0]})
    ci = fit.conf_int()
    res = pd.DataFrame({
        "feature": features,
        "coef_per_sd": fit.params[features].to_numpy(),
        "ci_low": ci.loc[features, 0].to_numpy(),
        "ci_high": ci.loc[features, 1].to_numpy(),
        "p_value": fit.pvalues[features].to_numpy(),
        "feature_sd": sd[features].to_numpy(),
    })
    res["group"] = res["feature"].map(
        {f: g for g, cols in FEATURE_GROUPS.items() for f in cols}).fillna("core")
    res.attrs["n"] = int(len(obs))
    res.attrs["r2"] = float(fit.rsquared)
    return res.sort_values("coef_per_sd", key=np.abs, ascending=False).reset_index(drop=True)


def naive_most_improved(mt: pd.DataFrame, season: int, top: int = 10) -> pd.DataFrame:
    """The raw 'biggest BPM jump' list for one outcome season, with context.

    Shows how often the leaderboard is driven by a poor, low-minute base season
    (regression to the mean) rather than a durable skill change.
    """
    obs = mt.loc[(mt["observed"] == 1) & (mt["outcome_season"] == season)]
    cols = ["player", "age", "mp", "bpm", "bpm_next", "delta_bpm", "mp_next"]
    return obs.nlargest(top, "delta_bpm")[cols].reset_index(drop=True)
