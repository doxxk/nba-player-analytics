"""Static report figures (matplotlib).

Each function answers one question and is named for it. Colors follow a
validated colorblind-safe palette: slot 1 (blue) for the series of interest,
slot 2 (orange) only when a second series is genuinely needed, and gray for
context.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b5b3ab"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

SOURCE_NOTE = "Data: Basketball-Reference via sumitrodatta/bball-reference-datasets."

FEATURE_LABELS = {
    "ts_rel": "True shooting % (vs. league)",
    "f_tr_rel": "Free-throw rate (vs. league)",
    "ft_pct_shrunk": "Free-throw % (shrunk)",
    "usg_percent": "Usage rate",
    "tov_percent": "Turnover rate",
    "gs_share": "Share of games started",
    "bpm_change": "BPM change from prior season",
    "ast_percent": "Assist rate",
    "log_draft_pick": "Draft pick (log; higher = later)",
    "undrafted": "Undrafted",
    "orb_percent": "Offensive rebound rate",
    "drb_percent": "Defensive rebound rate",
    "percent_fga_from_x0_3_range": "Share of shots at the rim (0-3 ft)",
    "fg_percent_from_x0_3_range": "FG% at the rim",
    "x3p_pct_shrunk": "3P% (shrunk)",
    "percent_dunks_of_fga": "Dunk share of FGA",
    "traded": "Traded mid-season",
    "stl_percent": "Steal rate",
    "blk_percent": "Block rate",
    "team_srs": "Team strength (SRS)",
    "x3p_ar_rel": "3PA rate (vs. league)",
    "net_on_off": "On/off net rating",
    "height_in": "Height",
    "usg_change": "Usage change from prior season",
    "percent_assisted_x2p_fg": "Share of 2s assisted",
}

GROUP_LABELS = {
    "Core + box_rates": "Box-score rates",
    "Core + shot_profile": "Shot profile",
    "Core + on_off": "On/off plus-minus",
    "Core + pedigree": "Draft slot & size",
    "Core + team_context": "Team context",
    "All features": "All features",
}


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.labelcolor": INK_2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "legend.frameon": False,
        "legend.fontsize": 9, "figure.dpi": 110, "savefig.dpi": 160,
    })


def _finish(fig, path: Path, subtitle: str | None = None, note: str = SOURCE_NOTE) -> Path:
    """Lay out title, subtitle, and source note without overlapping the plot.

    Single-panel charts get their axes title promoted to a figure-level
    headline; multi-panel charts keep per-panel titles under the subtitle.
    """
    top = 0.97
    if len(fig.axes) == 1 and fig.axes[0].get_title(loc="left"):
        ax = fig.axes[0]
        fig.text(0.01, top, ax.get_title(loc="left"), ha="left", va="top",
                 fontsize=13, fontweight="bold", color=INK)
        ax.set_title("", loc="left")
        top -= 0.055
    if subtitle:
        fig.text(0.01, top, subtitle, ha="left", va="top", color=INK_2, fontsize=9.5,
                 wrap=True)
        top -= 0.06
    fig.tight_layout(rect=(0, 0.045, 1, top))
    fig.text(0.01, 0.005, note, ha="left", va="bottom", color=MUTED, fontsize=7.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path


def _zero_line(ax, axis="y"):
    (ax.axhline if axis == "y" else ax.axvline)(0, color=AXIS, linewidth=1, zorder=1)


def regression_to_mean(rtm: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, color, offset in [("1,500+ MP", BLUE, -0.12), ("500-1,499 MP", ORANGE, 0.12)]:
        d = rtm.loc[rtm["minutes"] == label].sort_values("bin_mid")
        x = np.arange(len(d)) + offset
        ax.errorbar(x, d["mean_delta"], yerr=[d["mean_delta"] - d["ci_low"],
                    d["ci_high"] - d["mean_delta"]], fmt="o-", color=color, ms=6, lw=2,
                    capsize=0, elinewidth=1.5, label=f"{label} in base season")
    bins = rtm.drop_duplicates("bpm_bin").sort_values("bin_mid")["bpm_bin"]
    ax.set_xticks(np.arange(len(bins)), [b.replace("(", "").replace("]", "")
                                          .replace(", ", " to ") for b in bins])
    _zero_line(ax)
    ax.set_xlabel("BPM in base season (bins)")
    ax.set_ylabel("Mean change in BPM next season")
    ax.set_title("Bad seasons bounce back, great ones don't repeat")
    ax.legend(loc="upper right")
    return _finish(fig, path, "Young players (age <= 24, 500+ MP), 2000-01 to 2025-26. "
                   "Bars: 95% CI. A flat line would mean no regression to the mean.")


def age_and_attrition(age: pd.DataFrame, attr: pd.DataFrame, path: Path) -> Path:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4))
    a1.errorbar(age["age"], age["mean_delta"], yerr=[age["mean_delta"] - age["ci_low"],
                age["ci_high"] - age["mean_delta"]], fmt="o-", color=BLUE, ms=6, lw=2,
                capsize=0, elinewidth=1.5)
    _zero_line(a1)
    a1.set_xlabel("Age in base season")
    a1.set_ylabel("Mean BPM change next season")
    a1.set_title("Development slows sharply after 22")
    a1.set_xticks(age["age"])
    a2.bar(np.arange(len(attr)), attr["dropout_rate"] * 100, color=BLUE, width=0.62)
    a2.set_xticks(np.arange(len(attr)), attr["quintile"].astype(str), fontsize=8.5)
    for i, v in enumerate(attr["dropout_rate"] * 100):
        a2.text(i, v + 0.4, f"{v:.0f}%", ha="center", color=INK_2, fontsize=9)
    a2.set_ylabel("Fell below 250 MP next season (%)")
    a2.set_xlabel("Projected next-season BPM (Marcel quintile)")
    a2.set_title("Who drops out is not random")
    a2.grid(axis="x", visible=False)
    ratio = attr["dropout_rate"].iloc[0] / max(attr["dropout_rate"].iloc[-1], 1e-9)
    return _finish(fig, path, "Left: survivors only, so the true curve is somewhat lower. "
                   f"Right: players projected worst are {ratio:.1f}x likelier to lose "
                   "their role than those projected best.")


def model_comparison(metrics: pd.DataFrame, boots: pd.DataFrame, path: Path) -> Path:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    m = metrics.iloc[::-1]
    colors = [BLUE if n == "Ridge" else GRAY for n in m["model"]]
    a1.barh(m["model"], m["rmse_w"], color=colors, height=0.6)
    for i, v in enumerate(m["rmse_w"]):
        a1.text(v + 0.01, i, f"{v:.3f}", va="center", color=INK_2, fontsize=9)
    a1.set_xlim(1.4, m["rmse_w"].max() * 1.06)
    a1.set_xlabel("Minutes-weighted RMSE, next-season BPM (lower is better)")
    a1.set_title("Out-of-sample error, 2016-17 to 2025-26")
    a1.grid(axis="y", visible=False)
    b = boots.iloc[::-1]
    y = np.arange(len(b))
    a2.hlines(y, b["ci_low"], b["ci_high"], color=[BLUE if n == "Ridge" else GRAY
                                                   for n in b["model"]], lw=2.5)
    a2.plot(b["rmse_diff"], y, "o", color=INK, ms=6)
    a2.set_yticks(y, b["model"])
    _zero_line(a2, "x")
    a2.set_xlabel("RMSE minus Marcel's (negative = better)")
    a2.set_title("Gain over Marcel, 95% CI")
    a2.grid(axis="y", visible=False)
    return _finish(fig, path, "Rolling-origin test: each season predicted by models refit "
                   "on earlier seasons only. CIs: player-clustered bootstrap.")


def skill_by_season(by_season: pd.DataFrame, path: Path) -> Path:
    piv = by_season.pivot(index="season", columns="model", values="rmse_w")
    skill = (1 - piv["Ridge"] / piv["Marcel"]) * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    x = skill.index + 1  # label by outcome season
    ax.bar(x, skill, color=[BLUE if v > 0 else ORANGE for v in skill], width=0.6)
    for xi, v in zip(x, skill):
        ax.text(xi, v + (0.15 if v >= 0 else -0.15), f"{v:+.1f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK_2)
    _zero_line(ax)
    ax.set_xticks(x, [f"{s-1}-{str(s)[2:]}" for s in x], rotation=0, fontsize=8)
    ax.set_ylabel("Error reduction vs. Marcel (%)")
    ax.set_xlabel("Season being predicted")
    wins = int((skill > 0).sum())
    ax.set_title(f"The edge is small: Ridge beat Marcel in {wins} of {len(skill)} seasons")
    ax.grid(axis="x", visible=False)
    return _finish(fig, path, "Ridge RMSE improvement over the Marcel baseline, by test "
                   "season. Orange = the model did worse that year.")


def ablation(abl: pd.DataFrame, path: Path) -> Path:
    d = abl.loc[abl["model"] != "Core only"].copy()
    d["label"] = d["model"].map(GROUP_LABELS)
    d = d.iloc[::-1]
    y = np.arange(len(d))
    sig = d["ci_high"] < 0
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hlines(y, d["ci_low"], d["ci_high"], color=np.where(sig, BLUE, GRAY), lw=2.5)
    ax.plot(d["rmse_diff"], y, "o", color=INK, ms=6)
    ax.set_yticks(y, d["label"])
    _zero_line(ax, "x")
    ax.set_xlabel("Change in RMSE when added to the core model (negative = helps)")
    ax.set_title("What information actually improves the forecast?")
    ax.grid(axis="y", visible=False)
    return _finish(fig, path, "Core = Marcel projection, age, experience, BPM history, "
                   "minutes. Blue = 95% CI excludes zero.")


def signals(sig: pd.DataFrame, path: Path, top: int = 14) -> Path:
    d = sig.head(top).iloc[::-1].copy()
    d["label"] = d["feature"].map(FEATURE_LABELS).fillna(d["feature"])
    y = np.arange(len(d))
    strong = (d["ci_low"] > 0) | (d["ci_high"] < 0)
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.hlines(y, d["ci_low"], d["ci_high"], color=np.where(strong, BLUE, GRAY), lw=2.5)
    ax.plot(d["coef_per_sd"], y, "o", color=INK, ms=6)
    ax.set_yticks(y, d["label"])
    _zero_line(ax, "x")
    ax.set_xlabel("Extra next-season BPM per 1 SD, beyond the Marcel projection")
    ax.set_title("Bet on volume and touch, discount hot shooting")
    ax.grid(axis="y", visible=False)
    return _finish(fig, path, f"Weighted least squares on Marcel residuals, n = "
                   f"{sig.attrs.get('n', 0):,} player-seasons; 95% CIs clustered by player. "
                   "Associations, not causal effects.")


def calibration(preds: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5))
    lim = [-3, 3.5]
    ax.plot(lim, lim, color=AXIS, lw=1)
    for model, color in [("Marcel", GRAY), ("Ridge", BLUE)]:
        p = preds.loc[preds["model"] == model].copy()
        p["pred_delta"] = p["pred"] - p["bpm"]
        p["decile"] = pd.qcut(p["pred_delta"], 10, labels=False)
        g = p.groupby("decile").apply(lambda d: pd.Series({
            "pred": np.average(d["pred_delta"], weights=d["weight"]),
            "actual": np.average(d["delta_bpm"], weights=d["weight"])}), include_groups=False)
        ax.plot(g["pred"], g["actual"], "o-", color=color, lw=2, ms=6, label=model)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Predicted BPM change (decile mean)")
    ax.set_ylabel("Actual BPM change")
    ax.set_title("Predicted improvement shows up on the court")
    ax.legend(loc="upper left")
    return _finish(fig, path, "Test seasons 2016-17 to 2025-26. Points on the diagonal "
                   "are perfectly calibrated.")


def forecast(fc: pd.DataFrame, path: Path, top: int = 14, min_mp: int = 1000) -> Path:
    """Where the model's signals disagree most with the standard projection.

    Ranking by projected *change* would just surface last year's worst
    low-minute seasons (regression to the mean). The model-vs-Marcel gap is the
    part of the forecast that comes from box-score signals, not luck reversal.
    """
    d = fc.loc[fc["mp"] >= min_mp].nlargest(top, "model_minus_marcel").iloc[::-1]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8.8, 6))
    ax.hlines(y, d["pred_low"], d["pred_high"], color="#cde2fb", lw=7, zorder=1)
    ax.hlines(y, d["marcel_pred"], d["pred"], color=BLUE, lw=1.5, zorder=2)
    ax.plot(d["marcel_pred"], y, "o", color=GRAY, ms=7, zorder=3, label="Marcel projection")
    ax.plot(d["pred"], y, "o", color=BLUE, ms=8, zorder=3, label="Model projection")
    ax.set_yticks(y, [f"{p} ({t}, {int(a)})" for p, t, a in zip(d["player"], d["team"], d["age"])])
    ax.set_xlabel("Projected 2026-27 BPM (light band = 80% prediction interval)")
    ax.set_title("2026-27: young players the box-score signals like more than Marcel")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    return _finish(fig, path, f"Age <= 24 with {min_mp:,}+ MP in 2025-26, ranked by model "
                   "minus Marcel. Gaps are smaller than the intervals: leads, not verdicts.")
