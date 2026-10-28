"""Beyond the Box Score: interactive dashboard.

Run with ``streamlit run app.py``. Reads the result tables written by
``python -m bbs.pipeline`` (committed under reports/tables). Player history for
the comparison view comes from the locally generated data, so that tab needs
one pipeline run; every other tab works from the committed tables alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bbs.features import ALL_FEATURES  # noqa: E402
from bbs.viz import FEATURE_LABELS, GROUP_LABELS  # noqa: E402

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b5b3ab"
INK_2, GRID = "#52514e", "#e1e0d9"
SERIES = [BLUE, ORANGE, AQUA]  # comparison view is capped at three players

TABLE_DIRS = [ROOT / "reports" / "tables", ROOT / "outputs" / "sample" / "tables"]
# Generated locally by the pipeline; not committed (Basketball-Reference terms).
HISTORY_FILES = [ROOT / "data" / "processed" / "player_seasons.parquet",
                 ROOT / "data" / "sample" / "player_seasons_sample.csv.gz"]

st.set_page_config(page_title="Beyond the Box Score", page_icon="🏀", layout="wide")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

@st.cache_data
def load_tables() -> tuple[dict[str, pd.DataFrame], dict, Path] | None:
    for d in TABLE_DIRS:
        if (d / "results.json").exists():
            names = ["model_metrics", "bootstrap_vs_marcel", "model_metrics_by_season",
                     "ablation", "signal_regression", "sensitivity", "forecast_next_season",
                     "age_curve", "regression_to_mean",
                     "attrition_by_projection", "interval_coverage_by_minutes",
                     "largest_misses"]
            tables = {n: pd.read_csv(d / f"{n}.csv") for n in names if (d / f"{n}.csv").exists()}
            return tables, json.loads((d / "results.json").read_text()), d
    return None


@st.cache_data
def load_history() -> pd.DataFrame | None:
    for f in HISTORY_FILES:
        if f.exists():
            return pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
    return None


def base_layout(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif",
                                size=13),
                      legend=dict(orientation="h", y=1.08, x=0), hoverlabel=dict(
                          bgcolor="white"), **kw)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="#c3c2b7")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#c3c2b7")
    return fig


def season_label(s: int) -> str:
    return f"{s - 1}-{str(s)[2:]}"


loaded = load_tables()
if loaded is None:
    st.error("No results found. Run `python -m bbs.pipeline` (or `python -m bbs.pipeline "
             "--sample` offline) to generate reports/tables, then reload this page.")
    st.stop()
T, R, table_dir = loaded
history = load_history()
fc_season = R["forecast_base_season"] + 1

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("Beyond the Box Score")
st.markdown("##### Which young NBA players will improve next season, and which signals "
            "predict it beyond age and regression to the mean?")
if R.get("mode") == "sample":
    st.warning("Showing results from the bundled sample (2012-2026). Run the full "
               "pipeline for the published numbers.")

rv = R["ridge_vs_marcel"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Error reduction vs. Marcel", f"{R['ridge_skill_pct']:.1f}%",
          help="Minutes-weighted RMSE of next-season BPM, out of sample.")
c2.metric("95% CI on RMSE gain", f"{-rv['ci_high']:.3f} to {-rv['ci_low']:.3f} BPM",
          help="Player-clustered bootstrap.")
c3.metric("80% interval coverage", f"{R['interval_80_coverage'] * 100:.0f}%",
          help="Conformal intervals built only from earlier seasons' errors.")
c4.metric("Test player-seasons", f"{R['n_test_rows']:,}",
          help=f"Outcome seasons {season_label(R['test_seasons'][0])} to "
               f"{season_label(R['test_seasons'][1])}.")

tab_sum, tab_fc, tab_cmp, tab_val, tab_sig, tab_meth = st.tabs(
    ["Summary", f"{season_label(fc_season)} forecast", "Compare players", "Validation",
     "What predicts growth", "Methods & limits"])

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

with tab_sum:
    m = T["model_metrics"].set_index("model")
    st.markdown(f"""
**What we did.** For every player aged 24 or younger with 500+ minutes since 2000-01, we
forecast next-season Box Plus/Minus (BPM) and test each season using only data that
existed at the time. The benchmark is **Marcel**: a disciplined projection that
already knows age, minutes, and regression to the mean.

**What we found.**
- **Most "improvement" is regression to the mean.** Young players coming off a
  -4 BPM season improve by about 2 points on average, while +4 players stay flat.
  A raw "biggest jump" list mostly rediscovers bad luck reversing.
- **Box-score rates add a small, real edge.** A ridge model cuts Marcel's error by
  {R['ridge_skill_pct']:.1f}% (95% CI {-rv['ci_high']:.3f} to {-rv['ci_low']:.3f} BPM), and its
  top-decile picks improved by 1.5+ BPM {m.loc['Ridge', 'top10_hit_rate'] * 100:.0f}% of the
  time vs. {m.loc['Marcel', 'top10_hit_rate'] * 100:.0f}% for Marcel.
- **Bet on volume and touch, discount hot shooting.** Given the same projection,
  higher usage, free-throw rate, free-throw % and turnover rate (a proxy for
  ball-handling load) predict more growth; above-average true shooting predicts less.
- **On/off plus-minus, team quality, draft slot and height add nothing measurable**
  once the box score is known. Gradient boosting does not beat the linear model.

**What this is not.** BPM is a box-score *estimate* of impact, not measured on-court
impact, and every finding here is predictive association rather than cause and effect.
""")
    rtm = T["regression_to_mean"]
    fig = go.Figure()
    for label, color in [("1,500+ MP", BLUE), ("500-1,499 MP", ORANGE)]:
        d = rtm[rtm["minutes"] == label].sort_values("bin_mid")
        fig.add_trace(go.Scatter(
            x=d["bpm_bin"], y=d["mean_delta"], mode="lines+markers", name=label,
            line=dict(color=color, width=2), marker=dict(size=9),
            error_y=dict(type="data", symmetric=False, array=d["ci_high"] - d["mean_delta"],
                         arrayminus=d["mean_delta"] - d["ci_low"], thickness=1.5),
            customdata=d[["n"]], hovertemplate="BPM %{x}<br>Mean change %{y:+.2f}"
                                                "<br>n = %{customdata[0]}<extra>%{fullData.name}</extra>"))
    fig.add_hline(y=0, line_color="#c3c2b7")
    st.subheader("Bad seasons bounce back, great ones don't repeat")
    st.plotly_chart(base_layout(fig, xaxis_title="BPM in base season",
                                yaxis_title="Mean BPM change next season"),
                    width="stretch")
    st.caption("Shows: average next-season change by current BPM, 95% CIs. Doesn't show: "
               "players who fell out of the rotation (about 9% of the sample).")

# --------------------------------------------------------------------------
# Forecast
# --------------------------------------------------------------------------

with tab_fc:
    fc = T["forecast_next_season"].copy()
    st.markdown(f"Projections for **{season_label(fc_season)}**, made from "
                f"{season_label(fc_season - 1)} data before the season began. "
                "**Model minus Marcel** is the part of the forecast that comes from box-score "
                "signals rather than regression to the mean. It is the column to scout from.")
    f1, f2, f3, f4 = st.columns([2, 2, 1.4, 1.6])
    teams = sorted(fc["team"].unique())
    pick_team = f1.multiselect("Team", teams, placeholder="All teams")
    pick_pos = f2.multiselect("Position group", sorted(fc["pos_group"].unique()),
                              placeholder="All positions")
    min_mp = f3.slider("Min. minutes last season", 500, 2500, 1000, 100)
    sort_by = f4.selectbox("Rank by", ["Model minus Marcel", "Projected BPM",
                                       "Projected change"])
    view = fc[fc["mp"] >= min_mp]
    if pick_team:
        view = view[view["team"].isin(pick_team)]
    if pick_pos:
        view = view[view["pos_group"].isin(pick_pos)]
    key = {"Model minus Marcel": "model_minus_marcel", "Projected BPM": "pred",
           "Projected change": "pred_delta"}[sort_by]
    view = view.sort_values(key, ascending=False)
    if sort_by == "Projected change":
        st.info("Ranking by projected change mostly surfaces players coming off poor, "
                "low-minute seasons: that's regression to the mean, not a skill signal.")
    if view.empty:
        st.warning("No players match these filters.")
    else:
        top = view.head(20).iloc[::-1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.concat([top["pred_low"], top["pred_high"]]),
            y=pd.concat([top["player"], top["player"]]), mode="markers",
            marker=dict(size=0.1, color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False))
        for _, r in top.iterrows():
            fig.add_shape(type="line", x0=r["pred_low"], x1=r["pred_high"], y0=r["player"],
                          y1=r["player"], line=dict(color="#cde2fb", width=9), layer="below")
        fig.add_trace(go.Scatter(x=top["marcel_pred"], y=top["player"], mode="markers",
                                 name="Marcel projection", marker=dict(color=GRAY, size=10),
                                 hovertemplate="%{y}: Marcel %{x:.1f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=top["pred"], y=top["player"], mode="markers", name="Model projection",
            marker=dict(color=BLUE, size=11), customdata=top[["bpm", "pred_low", "pred_high",
                                                              "team", "age"]],
            hovertemplate="%{y} (%{customdata[3]}, %{customdata[4]:.0f})<br>"
                          "Last season %{customdata[0]:.1f}<br>Projection %{x:.1f} "
                          "[%{customdata[1]:.1f}, %{customdata[2]:.1f}]<extra></extra>"))
        st.plotly_chart(base_layout(fig, height=max(320, 26 * len(top) + 80),
                                    xaxis_title="Projected BPM (band = 80% interval)"),
                        width="stretch")
        show = view.rename(columns={
            "player": "Player", "team": "Team", "age": "Age", "pos_group": "Pos", "mp": "MP",
            "bpm": "BPM (last)", "marcel_pred": "Marcel", "pred": "Model",
            "model_minus_marcel": "Model - Marcel", "pred_low": "80% low",
            "pred_high": "80% high"})
        cols = ["Player", "Team", "Age", "Pos", "MP", "BPM (last)", "Marcel", "Model",
                "Model - Marcel", "80% low", "80% high"]
        st.dataframe(show[cols].round(2), hide_index=True, width="stretch")
    st.caption(f"80% intervals are +/-{R['interval_80_half_width']:.1f} BPM, calibrated on "
               "past errors. They are too narrow for players whose minutes collapse "
               "(see Validation). Team '2TM' = played for two teams.")

# --------------------------------------------------------------------------
# Compare players
# --------------------------------------------------------------------------

with tab_cmp:
    if history is None:
        st.info("Player histories aren't bundled with the repository. Run "
                "`python -m bbs.pipeline` once to download the data, then reload this page.")
    else:
        names = sorted(fc["player"].unique())
        default = [n for n in ["Stephon Castle", "Keyonte George", "Alex Sarr"] if n in names]
        chosen = st.multiselect("Choose up to three players from the forecast pool", names,
                                default=default, max_selections=3)
        if chosen:
            fig = go.Figure()
            for name, color in zip(chosen, SERIES):
                pid = fc.loc[fc["player"] == name, "player_id"].iloc[0]
                h = history[history["player_id"] == pid].sort_values("season")
                f = fc[fc["player_id"] == pid].iloc[0]
                fig.add_trace(go.Scatter(
                    x=h["season"], y=h["bpm"], mode="lines+markers", name=name,
                    line=dict(color=color, width=2), marker=dict(size=8),
                    customdata=h[["mp", "age"]],
                    hovertemplate="%{x}: BPM %{y:.1f} (%{customdata[0]:.0f} MP, age "
                                  "%{customdata[1]:.0f})<extra>" + name + "</extra>"))
                fig.add_trace(go.Scatter(
                    x=[fc_season], y=[f["pred"]], mode="markers", showlegend=False,
                    marker=dict(color=color, size=11, symbol="diamond-open", line=dict(width=2)),
                    error_y=dict(type="data", array=[f["pred_high"] - f["pred"]],
                                 color=color, thickness=1.5),
                    hovertemplate=f"{name} {season_label(fc_season)} projection: "
                                  "%{y:.1f}<extra></extra>"))
            fig.add_hline(y=0, line_color="#c3c2b7")
            ticks = list(range(int(history["season"].min()), fc_season + 1))
            fig.update_xaxes(tickmode="array", tickvals=ticks,
                             ticktext=[season_label(t) for t in ticks])
            st.plotly_chart(base_layout(fig, xaxis_title="Season (diamond = projection "
                                        "with 80% interval)",
                                        yaxis_title="Box Plus/Minus"), width="stretch")

            pool = fc[fc["mp"] >= 500]
            sig_cols = {"usg_percent": "Usage", "f_tr_rel": "FT rate (vs lg)",
                        "ft_pct_shrunk": "FT% (shrunk)", "ts_rel": "TS% (vs lg)",
                        "gs_share": "Start share"}
            rows = []
            for name in chosen:
                r = pool[pool["player"] == name]
                if r.empty:
                    continue
                for c, lab in sig_cols.items():
                    z = (r[c].iloc[0] - pool[c].mean()) / pool[c].std()
                    rows.append({"player": name, "signal": lab, "z": z})
            if rows:
                z = pd.DataFrame(rows)
                fig = go.Figure()
                for name, color in zip(chosen, SERIES):
                    d = z[z["player"] == name]
                    fig.add_trace(go.Bar(x=d["signal"], y=d["z"], name=name, marker_color=color,
                                         hovertemplate="%{x}: %{y:+.2f} SD<extra>" + name +
                                                       "</extra>"))
                fig.add_hline(y=0, line_color="#c3c2b7")
                st.subheader("Growth signals vs. other young players")
                st.plotly_chart(base_layout(fig, height=330, barmode="group", bargap=0.3,
                                            yaxis_title="SD from young-player mean"),
                                width="stretch")
                st.caption("Above zero on usage, FT rate and FT% is associated with growth; "
                           "above zero on TS% is associated with some regression.")

# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

with tab_val:
    met = T["model_metrics"].copy()
    met["vs Marcel"] = met["model"].map(T["bootstrap_vs_marcel"].set_index("model").apply(
        lambda r: f"{r['rmse_diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]", axis=1))
    st.dataframe(met.rename(columns={
        "model": "Model", "n": "n", "rmse_w": "RMSE (wtd)", "mae_w": "MAE (wtd)",
        "rmse": "RMSE", "spearman_delta": "Rank corr. of change",
        "top10_hit_rate": "Top-10% hit rate"}).round(3), hide_index=True, width="stretch")
    st.caption("Rolling origin: each test season is predicted by models refit on earlier "
               "seasons only. Hit rate = share of top-decile predicted improvers who gained "
               "1.5+ BPM.")
    a, b = st.columns(2)
    bs = T["model_metrics_by_season"].pivot(index="season", columns="model", values="rmse_w")
    skill = (1 - bs["Ridge"] / bs["Marcel"]) * 100
    fig = go.Figure(go.Bar(x=[season_label(s + 1) for s in skill.index], y=skill,
                           marker_color=[BLUE if v > 0 else ORANGE for v in skill],
                           hovertemplate="%{x}: %{y:+.1f}%<extra></extra>"))
    a.subheader("Error reduction vs. Marcel by season")
    a.plotly_chart(base_layout(fig, height=330, yaxis_title="%"), width="stretch")
    abl = T["ablation"][T["ablation"]["model"] != "Core only"].copy()
    abl["label"] = abl["model"].map(GROUP_LABELS)
    fig = go.Figure(go.Scatter(
        x=abl["rmse_diff"], y=abl["label"], mode="markers", marker=dict(color="#0b0b0b", size=9),
        error_x=dict(type="data", symmetric=False, array=abl["ci_high"] - abl["rmse_diff"],
                     arrayminus=abl["rmse_diff"] - abl["ci_low"], color=BLUE, thickness=2),
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color="#c3c2b7")
    b.subheader("Information added to the core model")
    b.plotly_chart(base_layout(fig, height=330, xaxis_title="RMSE change (negative = helps)"),
                   width="stretch")
    cov = T["interval_coverage_by_minutes"]
    st.subheader("Where the intervals fail")
    st.dataframe(cov.rename(columns={"mp_bin": "Next-season MP", "mean": "80% coverage",
                                     "size": "n"}).round(3), hide_index=True)
    st.caption("Intervals are too narrow for players whose role shrinks (250-999 MP next "
               "season). Minutes changes from injury or role are the biggest blind spot.")
    st.subheader("Biggest misses")
    st.dataframe(T["largest_misses"][["kind", "player", "season", "age", "bpm", "pred",
                                      "bpm_next", "mp_next"]].rename(columns={
        "kind": "", "season": "Base season", "bpm": "BPM", "pred": "Forecast",
        "bpm_next": "Actual", "mp_next": "Next MP"}).round(2), hide_index=True,
                 width="stretch")

# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------

with tab_sig:
    sig = T["signal_regression"].copy()
    sig["label"] = sig["feature"].map(FEATURE_LABELS).fillna(sig["feature"])
    sig = sig.iloc[::-1]
    strong = (sig["ci_low"] > 0) | (sig["ci_high"] < 0)
    fig = go.Figure(go.Scatter(
        x=sig["coef_per_sd"], y=sig["label"], mode="markers",
        marker=dict(color=["#0b0b0b"] * len(sig), size=9),
        error_x=dict(type="data", symmetric=False, array=sig["ci_high"] - sig["coef_per_sd"],
                     arrayminus=sig["coef_per_sd"] - sig["ci_low"], color=GRAY, thickness=2),
        customdata=sig[["p_value"]], name="Estimate (95% CI)",
        hovertemplate="%{y}: %{x:+.3f} BPM per SD (p = %{customdata[0]:.3f})<extra></extra>"))
    fig.add_trace(go.Scatter(x=sig.loc[strong, "coef_per_sd"], y=sig.loc[strong, "label"],
                             mode="markers", marker=dict(color=BLUE, size=11),
                             name="95% CI excludes zero", hoverinfo="skip"))
    fig.add_vline(x=0, line_color="#c3c2b7")
    st.subheader("Bet on volume and touch, discount hot shooting")
    st.plotly_chart(base_layout(fig, height=640,
                                xaxis_title="Extra next-season BPM per 1 SD, beyond Marcel"),
                    width="stretch")
    st.caption("Weighted least squares of (actual - Marcel) on standardized season-t "
               "features, errors clustered by player. Effects are small (0.1-0.2 BPM per SD) "
               "and associational. Strongly collinear features are omitted here but kept in "
               "the forecasting model.")
    sens = T["sensitivity"]
    st.subheader("Do the conclusions survive different choices?")
    show = sens[["variant", "n_test", "rmse_diff", "ci_low", "ci_high", "coef_ts_rel",
                 "coef_ft_pct_shrunk", "coef_f_tr_rel", "coef_usg_percent"]].rename(columns={
        "variant": "Variant", "n_test": "n", "rmse_diff": "RMSE vs Marcel", "ci_low": "CI low",
        "ci_high": "CI high", "coef_ts_rel": "TS% coef", "coef_ft_pct_shrunk": "FT% coef",
        "coef_f_tr_rel": "FT rate coef", "coef_usg_percent": "Usage coef"})
    st.dataframe(show.round(3), hide_index=True, width="stretch")

# --------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------

with tab_meth:
    st.markdown(f"""
**Data.** Basketball-Reference season tables (advanced, totals, per-100, shooting,
play-by-play, draft, team summaries) via the public
[sumitrodatta/bball-reference-datasets](https://github.com/sumitrodatta/bball-reference-datasets)
mirror, pinned to commit `{R.get('source_commit', '76a70b4')[:7]}`. NBA seasons 1996-97 to 2025-26.
Traded players use Basketball-Reference's combined row; team context is minutes-weighted.

**Population.** Base seasons 2000-01 onward, age <= 24 (Basketball-Reference age on
Feb 1), 500+ MP. An outcome counts if the player logs 250+ MP the next season.
Errors are weighted by next-season minutes (capped at 2,000), since BPM over few
minutes is noise.

**Baselines.** Persistence (same BPM again); age curve (BPM + average change at
that age); **Marcel** (last three seasons weighted 5/4/3 by minutes, regressed
toward the mean with k = {R['marcel_k_full']:.0f} minutes, plus a fitted age adjustment).

**Models.** Ridge regression (alpha = {R['tuned']['ridge_alpha']}) and histogram gradient boosting
on {len(ALL_FEATURES)} features, tuned only on seasons before 2016. Rolling-origin
test over {R['n_test_rows']:,} player-seasons; bootstrap CIs resample players.

**Limitations.**
- BPM is a box-score estimate; it under-rates defense that doesn't show up in steals,
  blocks and rebounds. No tracking, lineup or RAPM data was used.
- {R['dropout_rate'] * 100:.0f}% of player-seasons have no usable outcome, and they skew toward
  players projected to decline. Inverse-probability weighting leaves results unchanged,
  but it can only correct for what the model observes.
- Injuries, role changes and trades drive the largest misses and are invisible here.
- The effects are associations. Handing a player more usage will not by itself make that player improve.
- Data comes from Basketball-Reference; check Sports Reference's terms before reuse.
""")
    st.caption(f"Results loaded from {table_dir.relative_to(ROOT)}.")
