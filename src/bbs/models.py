"""Baselines, predictive models, and season-based validation.

The question is not "can a model predict next-season BPM?" (persistence alone
does that reasonably well) but "does anything beat a disciplined projection
that already knows age and regression to the mean?" The Marcel-style baseline
below is that disciplined projection, so every model is judged against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from bbs import config
from bbs.config import StudyDesign
from bbs.features import ALL_FEATURES, FEATURE_GROUPS


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def weighted_rmse(y, pred, w=None) -> float:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    if w.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.sum(w * (y - pred) ** 2) / w.sum()))


def weighted_mae(y, pred, w=None) -> float:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    if w.sum() == 0:
        return float("nan")
    return float(np.sum(w * np.abs(y - pred)) / w.sum())


def top_decile_hit_rate(pred_delta, actual_delta,
                        threshold: float = config.IMPROVEMENT_THRESHOLD) -> float:
    """Share of the top 10% predicted improvers who actually improved by ``threshold``."""
    pred_delta, actual_delta = np.asarray(pred_delta, float), np.asarray(actual_delta, float)
    # A model that predicts the same change for everyone has no top decile.
    if len(pred_delta) < 10 or np.ptp(pred_delta) == 0:
        return float("nan")
    cut = np.quantile(pred_delta, 0.9)
    top = pred_delta >= cut
    return float(np.mean(actual_delta[top] >= threshold))


def summarize(df: pd.DataFrame, pred_col: str = "pred") -> dict:
    """Standard out-of-sample metrics for one model's predictions."""
    pred_delta = df[pred_col] - df["bpm"]
    return {
        "n": int(len(df)),
        "rmse_w": weighted_rmse(df["bpm_next"], df[pred_col], df["weight"]),
        "mae_w": weighted_mae(df["bpm_next"], df[pred_col], df["weight"]),
        "rmse": weighted_rmse(df["bpm_next"], df[pred_col]),
        # Undefined for persistence, whose predicted change is always zero.
        "spearman_delta": float(stats.spearmanr(pred_delta, df["delta_bpm"]).statistic)
        if len(df) > 2 and pred_delta.nunique() > 1 else float("nan"),
        "top10_hit_rate": top_decile_hit_rate(pred_delta, df["delta_bpm"]),
    }


def paired_bootstrap(df: pd.DataFrame, model: str, reference: str, n_boot: int = 2000,
                     seed: int = config.RANDOM_SEED) -> dict:
    """Player-clustered bootstrap of the weighted-RMSE difference (model - reference).

    Resampling players rather than rows respects the fact that the same player
    appears in several test seasons.
    """
    a = df.loc[df["model"] == model].set_index(["player_id", "season"])
    b = df.loc[df["model"] == reference].set_index(["player_id", "season"])
    j = a[["pred", "bpm_next", "weight"]].join(b[["pred"]], rsuffix="_ref", how="inner")
    j = j.reset_index()
    if j.empty:
        raise ValueError("No overlapping predictions to compare.")
    j["se_m"] = j["weight"] * (j["bpm_next"] - j["pred"]) ** 2
    j["se_r"] = j["weight"] * (j["bpm_next"] - j["pred_ref"]) ** 2
    g = j.groupby("player_id")[["se_m", "se_r", "weight"]].sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    vals = g.to_numpy()
    sm, sr, sw = (vals[idx, k].sum(axis=1) for k in range(3))
    diffs = np.sqrt(sm / sw) - np.sqrt(sr / sw)
    point = np.sqrt(g["se_m"].sum() / g["weight"].sum()) - np.sqrt(g["se_r"].sum() / g["weight"].sum())
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"model": model, "reference": reference, "rmse_diff": float(point),
            "ci_low": float(lo), "ci_high": float(hi),
            "p_model_better": float(np.mean(diffs < 0)), "n_players": int(len(g))}


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

def _age_key(age: pd.Series) -> pd.Series:
    """Pool the rare 19- and 20-year-old seasons into one bucket."""
    return age.clip(lower=20).astype(int)


@dataclass
class Persistence:
    """Next season looks exactly like this season."""

    name: str = "Persistence"

    def fit(self, df: pd.DataFrame) -> "Persistence":
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["bpm"].to_numpy(float)


@dataclass
class AgeCurve:
    """This season plus the average one-year change for players of this age."""

    name: str = "Age curve"
    deltas: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "AgeCurve":
        d = df.assign(_k=_age_key(df["age"]))
        self.deltas = (d.groupby("_k").apply(
            lambda g: np.average(g["delta_bpm"], weights=g["weight"]), include_groups=False)
            .to_dict())
        self._overall = float(np.average(df["delta_bpm"], weights=df["weight"]))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        adj = _age_key(df["age"]).map(self.deltas).fillna(self._overall)
        return df["bpm"].to_numpy(float) + adj.to_numpy(float)


@dataclass
class Marcel:
    """Marcel-style projection: minutes-weighted history, regressed, plus aging.

    Following Tom Tango's Marcel system for baseball, the last three seasons are
    weighted 5/4/3 by minutes and blended with ``k`` minutes of a league-average
    prior. A per-age adjustment then adds typical development. ``k`` is chosen
    by grid search on the training data only.
    """

    name: str = "Marcel"
    season_weights: tuple[float, float, float] = (5.0, 4.0, 3.0)
    k_grid: tuple[float, ...] = (0, 250, 500, 750, 1000, 1500, 2000, 3000, 4000)
    k: float = 0.0
    prior: float = 0.0
    age_adj: dict = field(default_factory=dict)

    def _regressed(self, df: pd.DataFrame, k: float) -> np.ndarray:
        num = np.full(len(df), k * self.prior)
        den = np.full(len(df), float(k))
        for w, suffix in zip(self.season_weights, ["", "_lag1", "_lag2"]):
            mp = df[f"mp{suffix}"].fillna(0).to_numpy(float) * w / self.season_weights[0]
            bpm = df[f"bpm{suffix}"].fillna(0).to_numpy(float)
            num += mp * bpm
            den += mp
        return num / np.where(den == 0, 1, den)

    def _fit_age(self, df: pd.DataFrame, reg: np.ndarray) -> tuple[dict, float]:
        resid = df["bpm_next"].to_numpy(float) - reg
        d = pd.DataFrame({"k": _age_key(df["age"]).to_numpy(), "r": resid, "w": df["weight"]})
        adj = d.groupby("k").apply(lambda g: np.average(g["r"], weights=g["w"]),
                                   include_groups=False).to_dict()
        overall = float(np.average(d["r"], weights=d["w"]))
        return adj, overall

    def fit(self, df: pd.DataFrame) -> "Marcel":
        if len(df) < 20 or df["mp"].sum() <= 0 or df["weight"].sum() <= 0:
            raise ValueError(f"Marcel needs at least 20 observed rows to fit; got {len(df)}.")
        self.prior = float(np.average(df["bpm"], weights=df["mp"]))
        best = None
        for k in self.k_grid:
            reg = self._regressed(df, k)
            adj, overall = self._fit_age(df, reg)
            pred = reg + _age_key(df["age"]).map(adj).fillna(overall).to_numpy(float)
            err = weighted_rmse(df["bpm_next"], pred, df["weight"])
            if best is None or err < best[0]:
                best = (err, k, adj, overall)
        _, self.k, self.age_adj, self._overall = best
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        reg = self._regressed(df, self.k)
        return reg + _age_key(df["age"]).map(self.age_adj).fillna(self._overall).to_numpy(float)


# --------------------------------------------------------------------------
# Learned models
# --------------------------------------------------------------------------

RIDGE_ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
GBM_GRID = (
    {"learning_rate": 0.03, "max_depth": 3, "max_iter": 300, "min_samples_leaf": 40},
    {"learning_rate": 0.05, "max_depth": 2, "max_iter": 300, "min_samples_leaf": 60},
    {"learning_rate": 0.05, "max_depth": 4, "max_iter": 200, "min_samples_leaf": 80},
)


def make_ridge(features: list[str], alpha: float) -> Pipeline:
    pre = ColumnTransformer([("num", make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()), features)])
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha))])


def make_gbm(features: list[str], params: dict) -> Pipeline:
    pre = ColumnTransformer([("num", "passthrough", features)])
    return Pipeline([("pre", pre), ("gbm", HistGradientBoostingRegressor(
        l2_regularization=1.0, random_state=config.RANDOM_SEED, **params))])


@dataclass
class SklearnModel:
    """Wrap a scikit-learn pipeline that also receives the Marcel projection."""

    name: str
    features: list[str]
    kind: str  # "ridge" or "gbm"
    hyper: object = None
    use_ipw: bool = False

    def fit(self, df: pd.DataFrame) -> "SklearnModel":
        self.marcel = Marcel().fit(df)
        X = df.assign(marcel_pred=self.marcel.predict(df))
        w = df["weight"].to_numpy(float)
        if self.use_ipw:
            w = w * df["ipw"].to_numpy(float)
        self.pipe = (make_ridge(self.features, self.hyper) if self.kind == "ridge"
                     else make_gbm(self.features, self.hyper))
        self.pipe.fit(X[self.features], df["bpm_next"], **{
            f"{self.pipe.steps[-1][0]}__sample_weight": w})
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df.assign(marcel_pred=self.marcel.predict(df))
        return self.pipe.predict(X[self.features])


def tune(df: pd.DataFrame, design: StudyDesign, features: list[str] = ALL_FEATURES) -> dict:
    """Pick ridge alpha and GBM settings using only the pre-test era.

    The last quarter of pre-test seasons (at least two) is the validation
    block; earlier seasons train. The test period (base seasons >=
    ``first_test_season``) is never touched.
    """
    pool = df.loc[(df["season"] <= design.tuning_cutoff) & (df["observed"] == 1)]
    seasons = sorted(pool["season"].unique())
    if len(seasons) < 5:
        raise ValueError(f"Tuning needs at least 5 pre-test seasons; found {len(seasons)} "
                         f"(base seasons <= {design.tuning_cutoff}).")
    n_val = max(2, len(seasons) // 4)
    split = seasons[-n_val - 1]
    tr, va = pool.loc[pool["season"] <= split], pool.loc[pool["season"] > split]

    def score(model):
        model.fit(tr)
        return weighted_rmse(va["bpm_next"], model.predict(va), va["weight"])

    ridge = {a: score(SklearnModel("r", features, "ridge", a)) for a in RIDGE_ALPHAS}
    gbm = {i: score(SklearnModel("g", features, "gbm", p)) for i, p in enumerate(GBM_GRID)}
    best_gbm = min(gbm, key=gbm.get)
    return {"ridge_alpha": min(ridge, key=ridge.get), "gbm_params": GBM_GRID[best_gbm],
            "ridge_scores": ridge, "gbm_scores": {str(GBM_GRID[k]): v for k, v in gbm.items()}}


def standard_models(features: list[str], tuned: dict, use_ipw: bool = False) -> list:
    return [
        Persistence(),
        AgeCurve(),
        Marcel(),
        SklearnModel("Ridge", features, "ridge", tuned["ridge_alpha"], use_ipw),
        SklearnModel("Gradient boosting", features, "gbm", tuned["gbm_params"], use_ipw),
    ]


# --------------------------------------------------------------------------
# Survivorship correction
# --------------------------------------------------------------------------

IPW_FEATURES = ["bpm", "mp", "age", "experience", "log_draft_pick", "gs_share", "usg_percent"]


def add_ipw(train: pd.DataFrame, target: pd.DataFrame | None = None) -> pd.DataFrame:
    """Inverse-probability weights for reaching the next-season minutes threshold.

    Players who fall out of the rotation have no usable outcome, and they are
    disproportionately those who would have declined. Weighting observed rows by
    1 / P(observed | season-t profile) re-balances the training sample toward
    the kind of player who tends to vanish. Weights are capped at 10.
    """
    target = train if target is None else target
    clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        LogisticRegression(max_iter=1000))
    clf.fit(train[IPW_FEATURES], train["observed"])
    p = clf.predict_proba(target[IPW_FEATURES])[:, 1]
    return target.assign(p_observed=p, ipw=np.clip(1 / np.clip(p, 1e-3, None), 1, 10))


# --------------------------------------------------------------------------
# Rolling-origin evaluation
# --------------------------------------------------------------------------

ID_COLS = ["player", "player_id", "season", "age", "team", "pos_group", "mp", "bpm",
           "bpm_next", "delta_bpm", "mp_next", "weight"]


def rolling_origin(df: pd.DataFrame, models_factory, design: StudyDesign) -> pd.DataFrame:
    """Season-by-season out-of-sample predictions.

    For each test base season Y, every model is refit on base seasons < Y. All
    of those outcomes (seasons <= Y) are known at the end of season Y, so this
    mirrors exactly what a team could have done in real time.
    """
    out = []
    for year in range(design.first_test_season, design.last_test_season + 1):
        train = df.loc[(df["season"] < year) & (df["observed"] == 1)]
        test = df.loc[(df["season"] == year) & (df["observed"] == 1)]
        if test.empty or len(train) < 200:
            continue
        if design.use_ipw:
            train = add_ipw(df.loc[df["season"] < year]).loc[lambda d: d["observed"] == 1]
        for model in models_factory():
            model.fit(train)
            out.append(test[ID_COLS].assign(model=model.name, pred=model.predict(test)))
    return pd.concat(out, ignore_index=True)


def metrics_by_model(preds: pd.DataFrame) -> pd.DataFrame:
    rows = [{"model": m, **summarize(g)} for m, g in preds.groupby("model", sort=False)]
    return pd.DataFrame(rows)


def metrics_by_season(preds: pd.DataFrame) -> pd.DataFrame:
    rows = [{"model": m, "season": s, **summarize(g)}
            for (m, s), g in preds.groupby(["model", "season"], sort=False)]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Prediction intervals
# --------------------------------------------------------------------------

def online_conformal(preds: pd.DataFrame, model: str, level: float = 0.8) -> pd.DataFrame:
    """Split-conformal intervals that only use residuals from earlier test seasons.

    For test season Y the half-width is the ``level`` quantile of absolute
    out-of-sample errors from seasons < Y, so coverage is an honest forecast
    of how often the interval would have contained the truth.
    """
    p = preds.loc[preds["model"] == model].copy()
    p["abs_err"] = (p["bpm_next"] - p["pred"]).abs()
    rows = []
    for year in sorted(p["season"].unique()):
        hist = p.loc[p["season"] < year, "abs_err"]
        if len(hist) < 100:
            continue
        q = float(np.quantile(hist, level))
        cur = p.loc[p["season"] == year].assign(half_width=q)
        cur["covered"] = (cur["abs_err"] <= q).astype(int)
        rows.append(cur)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def interval_half_width(preds: pd.DataFrame, model: str, level: float = 0.8) -> float:
    """Half-width from all available out-of-sample residuals (for live forecasts)."""
    p = preds.loc[preds["model"] == model]
    return float(np.quantile((p["bpm_next"] - p["pred"]).abs(), level))


# --------------------------------------------------------------------------
# Feature-group ablation
# --------------------------------------------------------------------------

def ablation(df: pd.DataFrame, design: StudyDesign, alpha: float) -> pd.DataFrame:
    """Ridge with the core features plus one group at a time, and with everything.

    Answers "does this kind of information add anything a projection system
    doesn't already know?" The comparison is against core-only ridge, using the
    same rolling-origin splits.
    """
    sets = {"Core only": FEATURE_GROUPS["core"]}
    for g, cols in FEATURE_GROUPS.items():
        if g != "core":
            sets[f"Core + {g}"] = FEATURE_GROUPS["core"] + cols
    sets["All features"] = ALL_FEATURES

    def factory():
        return [SklearnModel(name, cols, "ridge", alpha) for name, cols in sets.items()]

    return rolling_origin(df, factory, design)
