"""Metric calculations and the modeling table.

One row of the modeling table is a young player in base season ``t``. Features
use only information available at the end of season ``t``; the outcome is the
player's Box Plus/Minus in season ``t+1``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bbs import config
from bbs.config import StudyDesign


# --------------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------------

def safe_divide(num, den):
    """Element-wise num/den that returns NaN (not inf) when den is 0 or missing."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num, den).shape, np.nan)
    np.divide(num, den, out=out, where=(den != 0) & ~np.isnan(den))
    return out


def true_shooting(pts, fga, fta):
    """TS% = PTS / (2 * (FGA + 0.44 * FTA))."""
    return safe_divide(pts, 2 * (np.asarray(fga, float) + 0.44 * np.asarray(fta, float)))


def estimate_beta_prior(made, attempts, min_attempts: int = 1) -> tuple[float, float]:
    """Method-of-moments beta-binomial prior for a shooting percentage.

    Returns ``(p0, k)``: the population mean and the prior strength in attempts.
    Shrinking with ``(made + k*p0) / (attempts + k)`` is the empirical-Bayes
    posterior mean. ``k`` is large when true talent varies little relative to
    binomial noise.
    """
    made = np.asarray(made, float)
    att = np.asarray(attempts, float)
    keep = att >= min_attempts
    made, att = made[keep], att[keep]
    if len(att) < 3:
        raise ValueError("Need at least 3 players with attempts to estimate a prior.")
    p = made / att
    p0 = made.sum() / att.sum()
    # Observed variance of rates = true-talent variance + mean binomial variance.
    w = att / att.sum()
    var_obs = np.sum(w * (p - p0) ** 2)
    var_noise = np.sum(w * p0 * (1 - p0) / att)
    var_true = var_obs - var_noise
    if var_true <= 0:
        return float(p0), float("inf")
    k = p0 * (1 - p0) / var_true - 1
    return float(p0), float(max(k, 0.0))


def shrink_rate(made, attempts, p0: float, k: float):
    """Empirical-Bayes shrunken percentage; equals ``p0`` with zero attempts."""
    made = np.asarray(made, float)
    att = np.asarray(attempts, float)
    if np.isinf(k):
        return np.full(np.broadcast(made, att).shape, p0)
    return (np.nan_to_num(made) + k * p0) / (np.nan_to_num(att) + k)


# --------------------------------------------------------------------------
# Modeling table
# --------------------------------------------------------------------------

LAG_COLS = ["bpm", "obpm", "dbpm", "mp", "usg_percent", "ts_rel", "ft_pct_shrunk"]

FEATURE_GROUPS: dict[str, list[str]] = {
    # What any projection system knows: age, track record, playing time.
    "core": ["marcel_pred", "age", "experience", "bpm", "obpm", "dbpm", "mp",
             "gs_share", "bpm_lag1", "mp_lag1", "has_lag1", "bpm_change"],
    # Per-possession box-score rates (era-adjusted where the league has drifted).
    "box_rates": ["usg_percent", "ts_rel", "ast_percent", "tov_percent", "orb_percent",
                  "drb_percent", "stl_percent", "blk_percent", "x3p_ar_rel", "f_tr_rel",
                  "ft_pct_shrunk", "x3p_pct_shrunk", "x3pa_per_100_poss", "usg_change"],
    # How shots are created and where they come from.
    "shot_profile": ["avg_dist_fga", "percent_fga_from_x0_3_range",
                     "fg_percent_from_x0_3_range", "percent_assisted_x2p_fg",
                     "percent_dunks_of_fga"],
    # Team point differential with the player on/off the floor (play-by-play).
    "on_off": ["on_court_pm", "net_on_off"],
    # Information a scout has before the box score: size and draft slot.
    "pedigree": ["log_draft_pick", "undrafted", "height_in", "weight_lb"],
    # Who the player played with.
    "team_context": ["team_srs", "traded"],
}
ALL_FEATURES = [f for cols in FEATURE_GROUPS.values() for f in cols]


def add_era_adjusted(ps: pd.DataFrame, team_summaries: pd.DataFrame) -> pd.DataFrame:
    """Express TS%, 3PA rate, and FT rate relative to that season's league average."""
    lg = (team_summaries.loc[(team_summaries["team"] == "League Average")
                             & (team_summaries["lg"] == "NBA"),
                             ["season", "ts_percent", "x3p_ar", "f_tr"]]
          .rename(columns={"ts_percent": "lg_ts", "x3p_ar": "lg_3par", "f_tr": "lg_ftr"}))
    out = ps.merge(lg, on="season", how="left", validate="many_to_one")
    out["ts_rel"] = out["ts_percent"] - out["lg_ts"]
    out["x3p_ar_rel"] = out["x3p_ar"] - out["lg_3par"]
    out["f_tr_rel"] = out["f_tr"] - out["lg_ftr"]
    return out.drop(columns=["lg_ts", "lg_3par", "lg_ftr"])


def add_shrunk_shooting(ps: pd.DataFrame, prior_seasons_max: int) -> tuple[pd.DataFrame, dict]:
    """Shrink FT% and 3P% toward priors estimated on early seasons only.

    Priors come from seasons <= ``prior_seasons_max`` so no information from the
    test period leaks into the features.
    """
    early = ps.loc[ps["season"] <= prior_seasons_max]
    priors = {}
    out = ps.copy()
    for name, made, att in [("ft", "ft", "fta"), ("x3p", "x3p", "x3pa")]:
        p0, k = estimate_beta_prior(early[made].fillna(0), early[att].fillna(0),
                                    min_attempts=10)
        priors[name] = {"p0": p0, "k": k}
        out[f"{name}_pct_shrunk"] = shrink_rate(out[made], out[att], p0, k)
    return out, priors


def _lagged(ps: pd.DataFrame, lag: int) -> pd.DataFrame:
    cols = [c for c in LAG_COLS if c in ps.columns]
    lagged = ps[["player_id", "season"] + cols].copy()
    lagged["season"] = lagged["season"] + lag
    return lagged.rename(columns={c: f"{c}_lag{lag}" for c in cols})


def build_model_table(ps: pd.DataFrame, design: StudyDesign = config.DEFAULT_DESIGN
                      ) -> pd.DataFrame:
    """Create one row per eligible base season with features and next-season outcome.

    Rows are kept even when the player does not reach ``min_mp_next`` minutes in
    ``t+1``; ``observed`` marks whether the outcome is usable. That lets the
    survivorship analysis see who dropped out.

    ``ps`` must already carry the era-adjusted and shrunk shooting columns
    (see :func:`add_era_adjusted` and :func:`add_shrunk_shooting`).
    """
    df = ps.copy()
    df["gs_share"] = safe_divide(df["gs"], df["g"])
    df["on_court_pm"] = df["on_court_plus_minus_per_100_poss"].clip(-20, 20)
    df["net_on_off"] = df["net_plus_minus_per_100_poss"].clip(-20, 20)
    df["log_draft_pick"] = np.log(df["draft_pick"])

    for lag in (1, 2):
        df = df.merge(_lagged(ps, lag), on=["player_id", "season"], how="left",
                      validate="one_to_one")
    df["has_lag1"] = df["bpm_lag1"].notna().astype(int)
    df["bpm_change"] = df["bpm"] - df["bpm_lag1"]
    df["usg_change"] = df["usg_percent"] - df["usg_percent_lag1"]
    df["experience"] = df["experience"].fillna(1)

    nxt = ps[["player_id", "season", "bpm", "mp", "obpm", "dbpm", "age"]].copy()
    nxt["season"] -= 1
    nxt = nxt.rename(columns={"bpm": "bpm_next", "mp": "mp_next", "obpm": "obpm_next",
                              "dbpm": "dbpm_next", "age": "age_next"})
    df = df.merge(nxt, on=["player_id", "season"], how="left", validate="one_to_one")
    df["mp_next"] = df["mp_next"].fillna(0)
    df["observed"] = (df["mp_next"] >= design.min_mp_next).astype(int)
    df["delta_bpm"] = df["bpm_next"] - df["bpm"]

    eligible = ((df["age"] <= design.max_age) & (df["mp"] >= design.min_mp)
                & (df["season"] >= design.first_base_season))
    df = df.loc[eligible].copy()
    df["outcome_season"] = df["season"] + 1
    if design.exclude_outcome_seasons:
        df = df.loc[~df["outcome_season"].isin(design.exclude_outcome_seasons)]
    # Outcome weight: next-season BPM over 2,000 minutes is far more reliable
    # than over 300, so errors are weighted by (capped) next-season minutes.
    df["weight"] = (df["mp_next"].clip(upper=2000) / 2000).where(df["observed"] == 1, 0.0)
    return df.reset_index(drop=True)
