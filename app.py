"""Beyond the Box Score: interactive dashboard.

Run with ``streamlit run app.py``. Reads the result tables written by
``python -m bbs.pipeline`` (committed under reports/tables). Player history for
the comparison view comes from the locally generated data, so that tab needs
one pipeline run; every other tab works from the committed tables alone.

The first four tabs are written for basketball fans; the last two keep the
technical detail. Every number shown comes from the result tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bbs.config import IMPROVEMENT_THRESHOLD  # noqa: E402
from bbs.features import ALL_FEATURES  # noqa: E402
from bbs.viz import FEATURE_LABELS, GROUP_LABELS  # noqa: E402

# One accent (blue). Orange only ever means the opposite: cooling off, or the
# model is skeptical. Gray is context. All three read on light and dark.
BLUE, ORANGE, GRAY = "#2a78d6", "#e8703a", "#8e95a3"
AQUA = "#1baf7a"  # third player in the comparison view only
SERIES = [BLUE, ORANGE, AQUA]
BAND = "rgba(42, 120, 214, 0.22)"
ZERO = "rgba(128, 128, 128, 0.55)"

TABLE_DIRS = [ROOT / "reports" / "tables", ROOT / "outputs" / "sample" / "tables"]
# Generated locally by the pipeline; not committed (Basketball-Reference terms).
HISTORY_FILES = [ROOT / "data" / "processed" / "player_seasons.parquet",
                 ROOT / "data" / "sample" / "player_seasons_sample.csv.gz"]

CARD_MIN_MINUTES = 1000

st.set_page_config(page_title="Beyond the Box Score", page_icon="🏀", layout="wide")


# --------------------------------------------------------------------------
# Data loading and helpers
# --------------------------------------------------------------------------

@st.cache_data
def load_tables() -> tuple[dict[str, pd.DataFrame], dict, Path] | None:
    for d in TABLE_DIRS:
        if (d / "results.json").exists():
            names = ["model_metrics", "bootstrap_vs_marcel", "model_metrics_by_season",
                     "ablation", "signal_regression", "sensitivity", "forecast_next_season",
                     "age_curve", "regression_to_mean", "naive_most_improved",
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


def style(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    """Shared chart layout. Colors for text and grid come from the Streamlit theme."""
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=36, b=8),
                      legend=dict(orientation="h", y=1.1, x=0, title=None), **kw)
    return fig


def season_label(s: int) -> str:
    return f"{s - 1}-{str(s)[2:]}"


def team_label(t: str) -> str:
    return f"{t[0]} teams" if t.endswith("TM") else t


def tier(bpm: float) -> str:
    """Basketball-Reference's rough guide to what a BPM level means."""
    for cut, name in [(8, "MVP level"), (6, "All-NBA level"), (4, "All-Star level"),
                      (2, "good starter"), (0, "solid starter"), (-2, "bench player")]:
        if bpm >= cut:
            return name
    return "fringe rotation player"


def signed(v: float, d: int = 1) -> str:
    return f"{v:+.{d}f}".replace("-", "−")


loaded = load_tables()
if loaded is None:
    st.error("No results found. Run `python -m bbs.pipeline` to generate them, then reload "
             "this page.")
    st.stop()
T, R, table_dir = loaded
history = load_history()
fc_season = R["forecast_base_season"] + 1
FC = season_label(fc_season)
LAST = season_label(fc_season - 1)

m = T["model_metrics"].set_index("model")
by_season = T["model_metrics_by_season"].pivot(index="season", columns="model", values="rmse_w")
skill_by_season = (1 - by_season["Ridge"] / by_season["Marcel"]) * 100
seasons_won = int((skill_by_season > 0).sum())
n_seasons = len(skill_by_season)
hit_ours, hit_std = m.loc["Ridge", "top10_hit_rate"], m.loc["Marcel", "top10_hit_rate"]
hit_base = R["base_rate_improve"]

# --------------------------------------------------------------------------
# Header: the answer first
# --------------------------------------------------------------------------

st.title("Beyond the Box Score")
st.markdown("#### Which young NBA players are about to get better, and how can you tell?")
st.markdown(
    "**Most one-year jumps by young players are luck evening out.** The players who keep "
    "improving are the ones carrying a big share of the offense and getting to the "
    "free-throw line, not the ones riding a hot shooting streak.")
if R.get("mode") == "sample":
    st.warning("Showing results from the local sample (2012-2026). Run the full pipeline "
               "for the published numbers.")

h1, h2, h3 = st.columns(3, gap="medium")
h1.metric("Of our top picks made a real jump", f"{hit_ours * 100:.0f}%", border=True,
          help=f"From {season_label(R['test_seasons'][0])} to "
               f"{season_label(R['test_seasons'][1])}: of the 10% of young players our model "
               f"liked most each year, {hit_ours * 100:.0f}% improved their impact rating by "
               f"{IMPROVEMENT_THRESHOLD:g}+ points. A standard projection's top picks: "
               f"{hit_std * 100:.0f}%. All young players: {hit_base * 100:.0f}%.")
h2.metric("Seasons we beat a standard projection", f"{seasons_won} of {n_seasons}",
          border=True,
          help="Each season was predicted using only information available before it "
               "started, then checked against what actually happened.")
h3.metric("Young-player seasons studied", f"{R['n_model_rows']:,}", border=True,
          help="Every player aged 24 or younger with 500+ minutes, 2000-01 to 2025-26.")

tabs = st.tabs(["The answer", "Who's about to get better", "Why: the signs of growth",
                "Compare players", "How we tested this (for stats nerds)",
                "Methods & data (for stats nerds)"])

# --------------------------------------------------------------------------
# Tab 1: The answer
# --------------------------------------------------------------------------

rtm = T["regression_to_mean"]
age = T["age_curve"].set_index("age")
naive = T.get("naive_most_improved")
bounce = rtm.loc[rtm["bin_mid"] < -4, "mean_delta"].mean()

with tabs[0]:
    st.info(
        "**What's an impact rating?** It's Box Plus/Minus (BPM): a box-score estimate of how "
        "many points a player adds per 100 possessions compared with an average player. "
        "**0 is average, +2 is a good starter, +4 is All-Star level, −2 is a bench player.** "
        "(Rough guide from Basketball-Reference.) It's built from the box score, so it "
        "misses some defense.")

    st.subheader("What we learned")
    w1, w2, w3 = st.columns(3, gap="large")
    with w1:
        st.markdown("##### Big jumps are mostly luck")
        jumps = ""
        if naive is not None and len(naive):
            jumps = (f" {int((naive['bpm'] < 0).sum())} of the {len(naive)} biggest jumps "
                     f"in {LAST} came from players coming off a below-average year.")
        st.markdown(f"Players coming off a terrible season (−4 or worse) improved about "
                    f"**{bounce:.0f} points** the next year on average, just from luck "
                    f"evening out.{jumps}")
    with w2:
        st.markdown("##### Real growth shows in the offense they carry")
        st.markdown("Players who handle lots of plays, get to the line, shoot free throws "
                    "well and start games tend to keep growing. **Hot shooting usually "
                    "cools off.**")
    with w3:
        st.markdown("##### Most growth happens before 22")
        st.markdown(f"Teenagers gained about **{age.loc[[19, 20], 'mean_delta'].mean():.1f} "
                    f"points** a year on average. By 23-24 it's about "
                    f"**{age.loc[[23, 24], 'mean_delta'].mean():.1f}**. Draft slot, height "
                    "and team quality added nothing once we knew the box score.")

    st.divider()
    st.subheader("Bad seasons bounce back. Great years in limited minutes don't last.")
    bin_names = {"(-10, -4]": "−4 or worse", "(-4, -2]": "−4 to −2", "(-2, 0]": "−2 to 0",
                 "(0, 2]": "0 to +2", "(2, 4]": "+2 to +4", "(4, 15]": "+4 or better"}
    fig = go.Figure()
    for key, name, color in [("1,500+ MP", "Played a lot (1,500+ minutes)", BLUE),
                             ("500-1,499 MP", "Played less (500–1,499 minutes)", ORANGE)]:
        d = rtm[rtm["minutes"] == key].sort_values("bin_mid")
        fig.add_trace(go.Scatter(
            x=d["bpm_bin"].map(bin_names), y=d["mean_delta"], mode="lines+markers", name=name,
            line=dict(color=color, width=3), marker=dict(size=10),
            error_y=dict(type="data", symmetric=False, array=d["ci_high"] - d["mean_delta"],
                         arrayminus=d["mean_delta"] - d["ci_low"], thickness=1.2, width=0),
            customdata=d[["n"]],
            hovertemplate="This season %{x}<br>Average change next season: %{y:+.1f}"
                          "<br>%{customdata[0]} players<extra></extra>"))
    fig.add_hline(y=0, line_color=ZERO)
    st.plotly_chart(style(fig, xaxis_title="Impact rating this season",
                          yaxis_title="Average change next season"), width="stretch")
    st.caption("Each point is the average next-season change for young players who started "
               "at that level; the thin lines show how sure we are. If improvement were pure "
               "skill, both lines would be flat.")

# --------------------------------------------------------------------------
# Tab 2: Who's about to get better
# --------------------------------------------------------------------------

fc = T["forecast_next_season"].copy()
pool = fc[fc["mp"] >= 500]
pct = {c: pool[c].rank(pct=True) for c in
       ["usg_percent", "f_tr_rel", "ft_pct_shrunk", "ts_rel", "gs_share"]}


def reasons(idx: int, rising: bool) -> str:
    """One plain sentence from the player's most notable growth signals.

    Compares the player with the other young players in this forecast on five of
    the model's inputs. It describes the profile; the projection itself uses many
    more inputs.
    """
    p = {c: s.get(idx, np.nan) for c, s in pct.items()}
    found: list[tuple[float, str]] = []
    if rising:
        if p["usg_percent"] >= 0.7:
            found.append((p["usg_percent"], f"handles more of the offense than "
                                            f"{p['usg_percent'] * 100:.0f}% of young players"))
        if p["f_tr_rel"] >= 0.7:
            found.append((p["f_tr_rel"], "gets to the free-throw line more than most"))
        if p["ft_pct_shrunk"] >= 0.7:
            found.append((p["ft_pct_shrunk"], "is a strong free-throw shooter"))
        if p["gs_share"] >= 0.75:
            found.append((p["gs_share"] - 0.05, "is already a starter"))
        if p["ts_rel"] <= 0.35:
            found.append((1 - p["ts_rel"], "isn't relying on hot shooting"))
        tail = "the profile that tends to keep improving"
    else:
        if p["ts_rel"] >= 0.7:
            found.append((p["ts_rel"], "scored more efficiently than "
                                       f"{p['ts_rel'] * 100:.0f}% of young players, which "
                                       "usually cools off"))
        if p["usg_percent"] <= 0.3:
            found.append((1 - p["usg_percent"], "has a small role in the offense"))
        if p["f_tr_rel"] <= 0.3:
            found.append((1 - p["f_tr_rel"], "rarely gets to the free-throw line"))
        if p["ft_pct_shrunk"] <= 0.3:
            found.append((1 - p["ft_pct_shrunk"], "is a weak free-throw shooter"))
        if p["gs_share"] <= 0.25:
            found.append((1 - p["gs_share"] - 0.05, "mostly comes off the bench"))
        tail = "so we expect less growth than a standard projection does"
    found.sort(key=lambda t: -t[0])
    picked = [s for _, s in found[:2]]
    if not picked:
        return ("No single signal stands out; the mix of inputs points "
                f"{'up' if rising else 'down'} a little.")
    body = picked[0] if len(picked) == 1 else f"{picked[0]} and {picked[1]}"
    return f"{body[0].upper()}{body[1:]}, {tail}."


def card(row: pd.Series, rising: bool) -> None:
    with st.container(border=True):
        st.markdown(f"**{row['player']}**  \n"
                    f":gray[{team_label(row['team'])} · age {int(row['age'])} · "
                    f"{row['pos_group']}]")
        st.metric(f"Projected {FC} rating", signed(row["pred"]),
                  # ASCII sign: Streamlit reads it to pick the arrow direction.
                  delta=f"{row['model_minus_marcel']:+.1f} vs. standard projection",
                  help=f"{LAST}: {signed(row['bpm'])}. Standard projection: "
                       f"{signed(row['marcel_pred'])}. Likely range: "
                       f"{signed(row['pred_low'])} to {signed(row['pred_high'])}.")
        st.caption(f"Last season {signed(row['bpm'])} ({tier(row['bpm'])}) → projected "
                   f"{tier(row['pred'])}. {reasons(row.name, rising)}")


def card_grid(rows: pd.DataFrame, rising: bool, per_row: int = 3) -> None:
    for start in range(0, len(rows), per_row):
        cols = st.columns(per_row, gap="medium")
        for col, (_, row) in zip(cols, rows.iloc[start:start + per_row].iterrows()):
            with col:
                card(row, rising)


with tabs[1]:
    st.markdown(f"Projections for **{FC}**, made from {LAST} stats before the season "
                f"started. A **standard projection** assumes everyone drifts back toward "
                f"normal and young players improve with age. These cards show where our "
                f"model disagrees with it. (Impact rating: 0 = average, +4 ≈ All-Star.)")
    eligible = fc[fc["mp"] >= CARD_MIN_MINUTES]
    st.subheader("Rising: projected to improve, and our model is more bullish than a "
                 "standard projection")
    rising = eligible[(eligible["pred_delta"] > 0) & (eligible["model_minus_marcel"] > 0)]
    card_grid(rising.nlargest(6, "model_minus_marcel"), rising=True)
    st.subheader("Skeptical: our model expects less than a standard projection")
    card_grid(eligible.nsmallest(4, "model_minus_marcel"), rising=False, per_row=4)
    st.caption(f"Players aged 24 or younger with {CARD_MIN_MINUTES:,}+ minutes in {LAST}. "
               "The gaps are smaller than the likely ranges, so treat these as players to "
               "watch, not sure things. Sentences describe each player's most notable "
               "growth signals; the projection uses many more inputs.")

    st.divider()
    st.subheader("Every young player")
    f1, f2, f3, f4 = st.columns([2, 2, 1.6, 2.2], gap="medium")
    pick_team = f1.multiselect("Team", sorted(fc["team"].unique()), placeholder="All teams",
                               format_func=team_label)
    pick_pos = f2.multiselect("Position", sorted(fc["pos_group"].unique()),
                              placeholder="All positions")
    min_mp = f3.slider("Minutes last season (at least)", 500, 2500, 1000, 100)
    rank_opts = {"Most likely to beat expectations": "model_minus_marcel",
                 "Best projected rating": "pred", "Biggest projected jump": "pred_delta"}
    sort_by = f4.selectbox("Sort by", list(rank_opts))
    view = fc[fc["mp"] >= min_mp]
    if pick_team:
        view = view[view["team"].isin(pick_team)]
    if pick_pos:
        view = view[view["pos_group"].isin(pick_pos)]
    view = view.sort_values(rank_opts[sort_by], ascending=False)
    if sort_by == "Biggest projected jump":
        st.info("The biggest projected jumps are mostly players bouncing back from a rough, "
                "low-minute season. That's luck evening out, not a sign of new skill.")
    if view.empty:
        st.warning("No players match these filters. Try lowering the minutes or clearing "
                   "a filter.")
    else:
        top = view.head(15).iloc[::-1]
        st.markdown(f"##### Where the top {len(top)} are headed in {FC}")
        fig = go.Figure()
        for _, r in top.iterrows():
            fig.add_shape(type="line", x0=r["pred_low"], x1=r["pred_high"], y0=r["player"],
                          y1=r["player"], line=dict(color=BAND, width=10), layer="below")
        fig.add_trace(go.Scatter(x=top["marcel_pred"], y=top["player"], mode="markers",
                                 name="Standard projection", marker=dict(color=GRAY, size=10),
                                 hovertemplate="%{y}: standard projection %{x:+.1f}"
                                               "<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=top["pred"], y=top["player"], mode="markers", name="Our projection",
            marker=dict(color=BLUE, size=12),
            customdata=np.c_[top["bpm"], top["pred_low"], top["pred_high"],
                             top["team"].map(team_label), top["age"]],
            hovertemplate="%{y} (%{customdata[3]}, age %{customdata[4]:.0f})<br>"
                          "Last season %{customdata[0]:+.1f}<br>Our projection %{x:+.1f}"
                          "<br>Likely range %{customdata[1]:+.1f} to %{customdata[2]:+.1f}"
                          "<extra></extra>"))
        fig.add_vline(x=0, line_color=ZERO)
        st.plotly_chart(style(fig, height=max(340, 30 * len(top) + 90),
                              xaxis_title="Projected impact rating (shaded = likely range)"),
                        width="stretch")
        table = pd.DataFrame({
            "Player": view["player"], "Team": view["team"].map(team_label),
            "Age": view["age"].astype(int), "Position": view["pos_group"],
            "Minutes": view["mp"].astype(int), f"{LAST} rating": view["bpm"],
            "Standard projection": view["marcel_pred"], "Our projection": view["pred"],
            "Ours vs. standard": view["model_minus_marcel"],
            "Likely range": [f"{signed(a)} to {signed(b)}"
                             for a, b in zip(view["pred_low"], view["pred_high"])]})
        num = st.column_config.NumberColumn
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            f"{LAST} rating": num(format="%+.1f"), "Standard projection": num(format="%+.1f"),
            "Our projection": num(format="%+.1f"), "Ours vs. standard": num(format="%+.1f"),
            "Minutes": num(format="%d")})
    st.caption(f"About 8 in 10 players land inside their likely range "
               f"({R['interval_80_coverage'] * 100:.0f}% in testing). It's less reliable "
               "for players whose minutes drop sharply, often because of injury.")

# --------------------------------------------------------------------------
# Tab 3: Why
# --------------------------------------------------------------------------

FAN_SIGNALS = [
    ("ts_rel", "Scoring efficiency (hot shooting)",
     "Very efficient scoring this year tends to cool off next year."),
    ("ft_pct_shrunk", "Free-throw shooting",
     "Good touch at the line often shows up later as better shooting overall."),
    ("f_tr_rel", "Getting to the free-throw line", "A sign of attacking the rim and creating."),
    ("usg_percent", "Share of the offense they handle",
     "Players trusted with lots of plays tend to grow into the role."),
    ("tov_percent", "Turnovers",
     "For young players, turnovers mostly mean they have the ball a lot."),
    ("gs_share", "How often they started", "The coaching staff's own vote of confidence."),
]
FAN_NULLS = [("log_draft_pick", "Draft position"), ("height_in", "Height"),
             ("team_srs", "How good their team was"),
             ("net_on_off", "Team's scoring margin with them on vs. off the court")]

with tabs[2]:
    st.subheader("Hot shooters usually cool off. Players who carry the offense keep growing.")
    st.markdown("We compared each player's next season with what a standard projection "
                "expected, then asked which of this season's stats explained the "
                "difference.")
    sig = T["signal_regression"].set_index("feature")
    rows = [(lab, sig.loc[f, "coef_per_sd"], BLUE if sig.loc[f, "coef_per_sd"] > 0 else ORANGE)
            for f, lab, _ in FAN_SIGNALS]
    rows += [(lab, sig.loc[f, "coef_per_sd"], GRAY) for f, lab in FAN_NULLS]
    d = pd.DataFrame(rows, columns=["label", "effect", "color"]).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["effect"], y=d["label"], orientation="h", marker_color=d["color"],
        hovertemplate="%{y}: %{x:+.2f} rating points next season<extra></extra>"))
    fig.add_vline(x=0, line_color=ZERO)
    fig.add_annotation(x=0.21, y=len(d) - 0.4, text="More growth →", showarrow=False,
                       font=dict(color=BLUE), xanchor="right")
    fig.add_annotation(x=-0.21, y=len(d) - 0.4, text="← Cools off", showarrow=False,
                       font=dict(color=ORANGE), xanchor="left")
    st.plotly_chart(style(fig, height=440, showlegend=False,
                          xaxis=dict(range=[-0.24, 0.24], title="Extra impact rating next "
                                     "season vs. an average young player")),
                    width="stretch")
    st.caption("Blue and orange: clear patterns. Gray: no clear effect once the box score is "
               "known. Each bar compares a player around the top sixth on that stat with an "
               "average young player, after accounting for age and track record.")

    st.markdown("##### What each signal means")
    for f, lab, why in FAN_SIGNALS:
        up = sig.loc[f, "coef_per_sd"] > 0
        mark = ":blue[**▲ more growth**]" if up else ":orange[**▼ cools off**]"
        st.markdown(f"**{lab}** · {mark}  \n{why}")
    st.markdown("**Draft position, height, team quality, on/off numbers** · :gray[**no clear "
                "effect**]  \nNothing extra once you know the box score.")
    st.info("These are patterns, not causes. Handing a player more shots won't by itself make "
            "them better. The effects are also small: they nudge a projection, they don't "
            "replace watching the games.")

# --------------------------------------------------------------------------
# Tab 4: Compare players
# --------------------------------------------------------------------------

with tabs[3]:
    if history is None:
        st.info("Player histories aren't included in the repository (data licensing). Run "
                "`python -m bbs.pipeline` once to download the data, then reload this page.")
    else:
        names = sorted(fc["player"].unique())
        default = [n for n in ["Stephon Castle", "Keyonte George", "Kon Knueppel"]
                   if n in names]
        chosen = st.multiselect("Pick up to three young players", names, default=default,
                                max_selections=3)
        if chosen:
            st.subheader("How their impact rating has moved, and where we project it next")
            fig = go.Figure()
            for name, color in zip(chosen, SERIES):
                pid = fc.loc[fc["player"] == name, "player_id"].iloc[0]
                h = history[history["player_id"] == pid].sort_values("season")
                f = fc[fc["player_id"] == pid].iloc[0]
                fig.add_trace(go.Scatter(
                    x=h["season"], y=h["bpm"], mode="lines+markers", name=name,
                    line=dict(color=color, width=3), marker=dict(size=9),
                    customdata=h[["mp", "age"]],
                    hovertemplate="%{y:+.1f} (%{customdata[0]:.0f} minutes, age "
                                  "%{customdata[1]:.0f})<extra>" + name + "</extra>"))
                fig.add_trace(go.Scatter(
                    x=[fc_season], y=[f["pred"]], mode="markers", showlegend=False,
                    marker=dict(color=color, size=13, symbol="diamond-open", line=dict(width=2.5)),
                    error_y=dict(type="data", symmetric=False,
                                 array=[f["pred_high"] - f["pred"]],
                                 arrayminus=[f["pred"] - f["pred_low"]], color=color,
                                 thickness=1.5, width=0),
                    hovertemplate=f"{name}, {FC} projection: %{{y:+.1f}}<extra></extra>"))
            fig.add_hline(y=0, line_color=ZERO)
            ticks = list(range(int(history.loc[history["player_id"].isin(
                fc.loc[fc["player"].isin(chosen), "player_id"]), "season"].min()),
                fc_season + 1))
            fig.update_xaxes(tickmode="array", tickvals=ticks,
                             ticktext=[season_label(t) for t in ticks])
            st.plotly_chart(style(fig, xaxis_title="Season (diamond = our projection and "
                                  "likely range)", yaxis_title="Impact rating"),
                            width="stretch")

            signals = {"usg_percent": "Share of the offense", "f_tr_rel": "Gets to the line",
                       "ft_pct_shrunk": "Free-throw shooting", "gs_share": "Starts",
                       "ts_rel": "Scoring efficiency"}
            rows = []
            for name in chosen:
                r = pool[pool["player"] == name]
                if r.empty:
                    continue
                for c, lab in signals.items():
                    rows.append({"player": name, "signal": lab,
                                 "z": (r[c].iloc[0] - pool[c].mean()) / pool[c].std()})
            if rows:
                z = pd.DataFrame(rows)
                st.subheader("Who has the signs of growth?")
                fig = go.Figure()
                for name, color in zip(chosen, SERIES):
                    d = z[z["player"] == name]
                    fig.add_trace(go.Bar(x=d["signal"], y=d["z"], name=name, marker_color=color,
                                         hovertemplate="%{x}: %{y:+.1f} vs. a typical young "
                                                       "player<extra>" + name + "</extra>"))
                fig.add_hline(y=0, line_color=ZERO)
                st.plotly_chart(style(fig, height=340, barmode="group", bargap=0.3,
                                      yaxis_title="Above or below a typical young player"),
                                width="stretch")
                st.caption("0 is a typical young player this season. Above zero on the first "
                           "four is a good sign for growth; above zero on scoring efficiency "
                           "tends to mean some cooling off.")

# --------------------------------------------------------------------------
# Tab 5: How we tested this
# --------------------------------------------------------------------------

with tabs[4]:
    st.markdown("Rolling-origin test: each season from "
                f"{season_label(R['test_seasons'][0])} to {season_label(R['test_seasons'][1])} "
                "was predicted by models refit only on earlier seasons. **Marcel** is the "
                "standard projection (5/4/3 minutes-weighted history, regressed to the mean, "
                "plus aging). BPM = Box Plus/Minus.")
    met = m.reset_index()
    met["vs Marcel [95% CI]"] = met["model"].map(
        T["bootstrap_vs_marcel"].set_index("model").apply(
            lambda r: f"{r['rmse_diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]", axis=1))
    met = met.round(3)
    for c in ["spearman_delta", "top10_hit_rate"]:  # undefined for persistence
        met[c] = met[c].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
    st.dataframe(met.rename(columns={
        "model": "Model", "rmse_w": "RMSE (minutes-weighted)", "mae_w": "MAE (weighted)",
        "rmse": "RMSE", "spearman_delta": "Rank corr. of change",
        "top10_hit_rate": "Top-10% hit rate"}).round(3), hide_index=True, width="stretch")
    st.caption(f"Hit rate = share of each model's top-decile predicted improvers who gained "
               f"{IMPROVEMENT_THRESHOLD:g}+ BPM (base rate {hit_base * 100:.0f}%). Bootstrap "
               "CIs resample players.")

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown(f"##### Our model beat the standard projection in {seasons_won} of "
                    f"{n_seasons} seasons")
        fig = go.Figure(go.Bar(
            x=[season_label(s + 1) for s in skill_by_season.index], y=skill_by_season,
            marker_color=[BLUE if v > 0 else ORANGE for v in skill_by_season],
            hovertemplate="%{x}: %{y:+.1f}% vs. Marcel<extra></extra>"))
        fig.add_hline(y=0, line_color=ZERO)
        st.plotly_chart(style(fig, height=320, yaxis_title="RMSE reduction vs. Marcel (%)"),
                        width="stretch")
    with b:
        st.markdown("##### Only box-score rates add real information")
        abl = T["ablation"][T["ablation"]["model"] != "Core only"].copy()
        abl["label"] = abl["model"].map(GROUP_LABELS)
        fig = go.Figure(go.Scatter(
            x=abl["rmse_diff"], y=abl["label"], mode="markers",
            marker=dict(color=BLUE, size=10),
            error_x=dict(type="data", symmetric=False, array=abl["ci_high"] - abl["rmse_diff"],
                         arrayminus=abl["rmse_diff"] - abl["ci_low"], color=GRAY,
                         thickness=2, width=0),
            hovertemplate="%{y}: %{x:+.3f} RMSE<extra></extra>"))
        fig.add_vline(x=0, line_color=ZERO)
        st.plotly_chart(style(fig, height=320, xaxis_title="RMSE change when added to the "
                              "core model (negative = helps)"), width="stretch")

    st.markdown("##### Signal regression: effects beyond Marcel, with 95% CIs")
    sig_all = T["signal_regression"].copy()
    sig_all["label"] = sig_all["feature"].map(FEATURE_LABELS).fillna(sig_all["feature"])
    sig_all = sig_all.iloc[::-1]
    strong = (sig_all["ci_low"] > 0) | (sig_all["ci_high"] < 0)
    fig = go.Figure(go.Scatter(
        x=sig_all["coef_per_sd"], y=sig_all["label"], mode="markers",
        marker=dict(color=np.where(strong, BLUE, GRAY), size=9),
        error_x=dict(type="data", symmetric=False,
                     array=sig_all["ci_high"] - sig_all["coef_per_sd"],
                     arrayminus=sig_all["coef_per_sd"] - sig_all["ci_low"], color=GRAY,
                     thickness=1.5, width=0),
        customdata=sig_all[["p_value"]],
        hovertemplate="%{y}: %{x:+.3f} BPM per SD (p = %{customdata[0]:.3f})<extra></extra>"))
    fig.add_vline(x=0, line_color=ZERO)
    st.plotly_chart(style(fig, height=620, showlegend=False,
                          xaxis_title="Extra next-season BPM per 1 SD, beyond Marcel"),
                    width="stretch")
    st.caption(f"Weighted least squares of (actual − Marcel) on standardized features, errors "
               f"clustered by player, n = {R['n_observed']:,}. Blue = 95% CI excludes zero. "
               "Collinear features (|r| ≥ 0.75) omitted here but kept in the forecasting "
               "model. Associations, not causal effects.")

    st.markdown("##### Do the conclusions survive different choices?")
    sens = T["sensitivity"]
    st.dataframe(sens[["variant", "n_test", "rmse_diff", "ci_low", "ci_high", "coef_ts_rel",
                       "coef_ft_pct_shrunk", "coef_f_tr_rel", "coef_usg_percent"]].rename(
        columns={"variant": "Variant", "n_test": "n", "rmse_diff": "RMSE vs Marcel",
                 "ci_low": "CI low", "ci_high": "CI high", "coef_ts_rel": "TS% coef",
                 "coef_ft_pct_shrunk": "FT% coef", "coef_f_tr_rel": "FT rate coef",
                 "coef_usg_percent": "Usage coef"}).round(3), hide_index=True, width="stretch")

    st.markdown("##### Where the 80% intervals fail")
    cov = T["interval_coverage_by_minutes"]
    st.dataframe(cov.rename(columns={"mp_bin": "Next-season MP", "mean": "Coverage",
                                     "size": "n"}).round(3), hide_index=True)
    st.caption("Too narrow when a player's minutes collapse (injury, lost role).")
    st.markdown("##### Biggest misses")
    st.dataframe(T["largest_misses"][["kind", "player", "season", "age", "bpm", "pred",
                                      "bpm_next", "mp_next"]].rename(columns={
        "kind": "", "player": "Player", "season": "Base season", "age": "Age", "bpm": "BPM",
        "pred": "Forecast", "bpm_next": "Actual", "mp_next": "Next MP"}).round(1),
        hide_index=True, width="stretch")

# --------------------------------------------------------------------------
# Tab 6: Methods & data
# --------------------------------------------------------------------------

with tabs[5]:
    rv = R["ridge_vs_marcel"]
    t1, t2, t3, t4 = st.columns(4, gap="medium")
    t1.metric("RMSE reduction vs. Marcel", f"{R['ridge_skill_pct']:.1f}%", border=True,
              help="Minutes-weighted RMSE of next-season BPM, out of sample.")
    t2.metric("95% CI on RMSE gain", f"{-rv['ci_high']:.3f} to {-rv['ci_low']:.3f}",
              border=True, help="BPM points; player-clustered paired bootstrap.")
    t3.metric("80% interval coverage", f"{R['interval_80_coverage'] * 100:.1f}%", border=True,
              help="Online split-conformal intervals built only from earlier seasons' errors.")
    t4.metric("Test player-seasons", f"{R['n_test_rows']:,}", border=True,
              help=f"Outcome seasons {season_label(R['test_seasons'][0])} to "
                   f"{season_label(R['test_seasons'][1])}.")
    st.markdown(f"""
**Data.** Basketball-Reference season tables (advanced, totals, per-100, shooting,
play-by-play, draft, team summaries) via the public
[sumitrodatta/bball-reference-datasets](https://github.com/sumitrodatta/bball-reference-datasets)
mirror, pinned to commit `{R.get('source_commit', '76a70b4')[:7]}`. NBA seasons 1996-97 to 2025-26.
Traded players use Basketball-Reference's combined row; team context is minutes-weighted.
No datasets are redistributed with this project.

**Population.** Base seasons 2000-01 onward, age ≤ 24 (Basketball-Reference age on
Feb 1), 500+ MP. An outcome counts if the player logs 250+ MP the next season.
Errors are weighted by next-season minutes (capped at 2,000), since BPM over few
minutes is noise.

**Baselines.** Persistence (same BPM again); age curve (BPM + average change at
that age); **Marcel** (last three seasons weighted 5/4/3 by minutes, regressed
toward the mean with k = {R['marcel_k_full']:.0f} minutes, plus a fitted age adjustment).

**Models.** Ridge regression (alpha = {R['tuned']['ridge_alpha']}) and histogram gradient boosting
on {len(ALL_FEATURES)} features, tuned only on seasons before 2016. Rolling-origin
test over {R['n_test_rows']:,} player-seasons; bootstrap CIs resample players.

**Impact-rating tiers** on the fan tabs use Basketball-Reference's rough BPM guide
(+8 MVP, +6 All-NBA, +4 All-Star, +2 good starter, 0 solid starter, −2 bench).

**Limitations.**
- BPM is a box-score estimate; it under-rates defense that doesn't show up in steals,
  blocks and rebounds. No tracking, lineup or RAPM data was used.
- {R['dropout_rate'] * 100:.0f}% of player-seasons have no usable outcome, and they skew toward
  players projected to decline. Inverse-probability weighting leaves results unchanged,
  but it can only correct for what the model observes.
- Injuries, role changes and trades drive the largest misses and are invisible here.
- The effects are associations. Handing a player more usage will not by itself make that
  player improve.
""")
    st.caption(f"Results loaded from {table_dir.relative_to(ROOT)}.")
