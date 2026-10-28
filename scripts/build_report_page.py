"""Build reports/report.html from the pipeline result tables.

Usage: python scripts/build_report_page.py . reports/report.html
"""
import html
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
T = ROOT / "reports" / "tables"
R = json.loads((T / "results.json").read_text())
sys.path.insert(0, str(ROOT / "src"))
from bbs.viz import FEATURE_LABELS  # noqa: E402

E = html.escape


def lin(d0, d1, r0, r1):
    return lambda v: r0 + (v - d0) / (d1 - d0) * (r1 - r0)


def fmt(v, d=2, sign=False):
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace("-", "−")


def svg(w, h, body, label):
    return (f'<div class="chart"><svg viewBox="0 0 {w} {h}" role="img" aria-label="{E(label)}" '
            f'preserveAspectRatio="xMidYMid meet">{body}</svg></div>')


# ---------------------------------------------------------------- chart: regression to mean
def chart_rtm():
    d = pd.read_csv(T / "regression_to_mean.csv")
    bins = d.drop_duplicates("bpm_bin").sort_values("bin_mid")["bpm_bin"].tolist()
    W, H, L, Rr, Tp, B = 720, 340, 52, 130, 16, 46
    x = lambda i: L + (i + 0.5) * (W - L - Rr) / len(bins)  # noqa: E731
    y = lin(-3, 2.6, H - B, Tp)
    o = []
    for t in [-3, -2, -1, 0, 1, 2]:
        o.append(f'<line class="{"axis" if t == 0 else "grid"}" x1="{L}" x2="{W - Rr}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        o.append(f'<text class="tick" x="{L - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{fmt(t, 0, t != 0)}</text>')
    for i, b in enumerate(bins):
        lab = b.strip("(]").replace(", ", " to ").replace("-", "−")
        o.append(f'<text class="tick" x="{x(i):.1f}" y="{H - B + 20}" text-anchor="middle">{lab}</text>')
    o.append(f'<text class="axlab" x="{(L + W - Rr) / 2}" y="{H - 6}" text-anchor="middle">BPM in the base season</text>')
    o.append(f'<text class="axlab" transform="translate(14 {(Tp + H - B) / 2}) rotate(-90)" text-anchor="middle">Change in BPM next season</text>')
    for key, cls, off, name in [("1,500+ MP", "s1", -6, "1,500+ minutes"), ("500-1,499 MP", "s2", 6, "500–1,499 minutes")]:
        g = d[d["minutes"] == key].set_index("bpm_bin").loc[bins]
        pts = [(x(i) + off, y(v)) for i, v in enumerate(g["mean_delta"])]
        o.append(f'<polyline class="line {cls}" points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}"/>')
        for (px, py), (_, r) in zip(pts, g.iterrows()):
            o.append(f'<g class="pt"><title>{name}, BPM {E(r.name)}: {fmt(r.mean_delta, 2, True)} '
                     f'(95% CI {fmt(r.ci_low, 2, True)} to {fmt(r.ci_high, 2, True)}), n = {r.n}</title>'
                     f'<line class="whisk {cls}" x1="{px:.1f}" x2="{px:.1f}" y1="{y(r.ci_low):.1f}" y2="{y(r.ci_high):.1f}"/>'
                     f'<circle class="dot {cls}" cx="{px:.1f}" cy="{py:.1f}" r="5"/>'
                     f'<rect class="hit" x="{px - 14:.1f}" y="{y(r.ci_high) - 6:.1f}" width="28" height="{y(r.ci_low) - y(r.ci_high) + 12:.1f}"/></g>')
        lx, ly = pts[-1]
        o.append(f'<text class="dlab" x="{lx + 12:.1f}" y="{ly + 4:.1f}">{name}</text>')
    return svg(W, H, "".join(o), "Mean next-season BPM change by base-season BPM, two minute groups")


# ---------------------------------------------------------------- chart: gain vs marcel
def chart_gain():
    b = pd.read_csv(T / "bootstrap_vs_marcel.csv").set_index("model")
    order = ["Persistence", "Age curve", "Gradient boosting", "Ridge"]
    W, H, L, Rr, Tp, B = 720, 230, 150, 24, 14, 50
    x = lin(-0.12, 0.28, L, W - Rr)
    rows = len(order)
    y = lambda i: Tp + (i + 0.5) * (H - Tp - B) / rows  # noqa: E731
    o = []
    for t in [-0.1, 0, 0.1, 0.2]:
        o.append(f'<line class="{"axis" if t == 0 else "grid"}" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{Tp}" y2="{H - B}"/>')
        o.append(f'<text class="tick" x="{x(t):.1f}" y="{H - B + 18}" text-anchor="middle">{fmt(t, 1, t != 0)}</text>')
    o.append(f'<text class="tick em" x="{x(0) + 6:.1f}" y="{Tp + 10}">Marcel</text>')
    o.append(f'<text class="axlab" x="{(L + W - Rr) / 2}" y="{H - 8}" text-anchor="middle">RMSE minus Marcel’s, in BPM (left of zero = more accurate)</text>')
    for i, m in enumerate(order):
        r = b.loc[m]
        cls = "s1" if m == "Ridge" else "mute"
        o.append(f'<g class="pt"><title>{m}: {fmt(r.rmse_diff, 3, True)} (95% CI {fmt(r.ci_low, 3, True)} to {fmt(r.ci_high, 3, True)})</title>'
                 f'<text class="rowlab{" em" if m == "Ridge" else ""}" x="{L - 12}" y="{y(i) + 4:.1f}" text-anchor="end">{m}</text>'
                 f'<line class="ci {cls}" x1="{x(r.ci_low):.1f}" x2="{x(r.ci_high):.1f}" y1="{y(i):.1f}" y2="{y(i):.1f}"/>'
                 f'<circle class="dot ink" cx="{x(r.rmse_diff):.1f}" cy="{y(i):.1f}" r="5"/>'
                 f'<text class="vlab" x="{x(r.ci_high) + 8:.1f}" y="{y(i) + 4:.1f}">{fmt(r.rmse_diff, 3, True)}</text>'
                 f'<rect class="hit" x="{L}" y="{y(i) - 14:.1f}" width="{W - Rr - L}" height="28"/></g>')
    return svg(W, H, "".join(o), "RMSE difference versus Marcel with 95% confidence intervals")


# ---------------------------------------------------------------- chart: skill by season
def chart_skill():
    d = pd.read_csv(T / "model_metrics_by_season.csv").pivot(index="season", columns="model", values="rmse_w")
    s = (1 - d["Ridge"] / d["Marcel"]) * 100
    W, H, L, Rr, Tp, B = 720, 250, 44, 10, 22, 40
    n = len(s)
    bw = (W - L - Rr) / n
    y = lin(-4, 8, H - B, Tp)
    o = []
    for t in [-4, 0, 4, 8]:
        o.append(f'<line class="{"axis" if t == 0 else "grid"}" x1="{L}" x2="{W - Rr}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        o.append(f'<text class="tick" x="{L - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{fmt(t, 0, t != 0)}%</text>')
    for i, (season, v) in enumerate(s.items()):
        cx = L + (i + 0.5) * bw
        top, bot = (y(v), y(0)) if v >= 0 else (y(0), y(v))
        cls = "s1" if v >= 0 else "s2"
        lab = f"{season}-{str(season + 1)[2:]}"
        o.append(f'<g class="pt"><title>{lab}: {fmt(v, 1, True)}% vs. Marcel</title>'
                 f'<rect class="bar {cls}" x="{cx - bw * 0.32:.1f}" y="{top:.1f}" width="{bw * 0.64:.1f}" height="{max(bot - top, 1):.1f}" rx="2"/>'
                 f'<text class="vlab" x="{cx:.1f}" y="{(top - 6) if v >= 0 else (bot + 14):.1f}" text-anchor="middle">{fmt(v, 1, True)}</text>'
                 f'<text class="tick" x="{cx:.1f}" y="{H - B + 18}" text-anchor="middle">{lab}</text>'
                 f'<rect class="hit" x="{cx - bw / 2:.1f}" y="{Tp}" width="{bw:.1f}" height="{H - B - Tp}"/></g>')
    return svg(W, H, "".join(o), "Ridge error reduction versus Marcel by season predicted")


# ---------------------------------------------------------------- chart: signals
def chart_signals():
    d = pd.read_csv(T / "signal_regression.csv").head(12)
    W, rowh, L, Rr, Tp, B = 720, 30, 250, 20, 10, 46
    H = Tp + B + rowh * len(d)
    x = lin(-0.35, 0.25, L, W - Rr)
    o = []
    for t in [-0.3, -0.2, -0.1, 0, 0.1, 0.2]:
        o.append(f'<line class="{"axis" if abs(t) < 1e-9 else "grid"}" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{Tp}" y2="{H - B}"/>')
        o.append(f'<text class="tick" x="{x(t):.1f}" y="{H - B + 18}" text-anchor="middle">{fmt(t, 1, abs(t) > 1e-9)}</text>')
    o.append(f'<text class="axlab" x="{(L + W - Rr) / 2}" y="{H - 8}" text-anchor="middle">Extra next-season BPM per 1 SD, beyond the Marcel projection</text>')
    for i, r in enumerate(d.itertuples()):
        cy = Tp + (i + 0.5) * rowh
        strong = r.ci_low > 0 or r.ci_high < 0
        cls = ("s1" if r.coef_per_sd > 0 else "s2") if strong else "mute"
        lab = FEATURE_LABELS.get(r.feature, r.feature)
        o.append(f'<g class="pt"><title>{E(lab)}: {fmt(r.coef_per_sd, 3, True)} (95% CI {fmt(r.ci_low, 3, True)} to {fmt(r.ci_high, 3, True)}, p = {r.p_value:.3f})</title>'
                 f'<text class="rowlab{"" if strong else " dim"}" x="{L - 12}" y="{cy + 4:.1f}" text-anchor="end">{E(lab)}</text>'
                 f'<line class="ci {cls}" x1="{x(r.ci_low):.1f}" x2="{x(r.ci_high):.1f}" y1="{cy:.1f}" y2="{cy:.1f}"/>'
                 f'<circle class="dot ink" cx="{x(r.coef_per_sd):.1f}" cy="{cy:.1f}" r="4.5"/>'
                 f'<rect class="hit" x="0" y="{cy - rowh / 2:.1f}" width="{W}" height="{rowh}"/></g>')
    return svg(W, H, "".join(o), "Standardized effects of season signals on next-season BPM beyond Marcel")


# ---------------------------------------------------------------- chart: forecast
def chart_forecast(fc):
    d = fc[fc["mp"] >= 1000].nlargest(10, "model_minus_marcel")
    W, rowh, L, Rr, Tp, B = 720, 32, 200, 20, 10, 46
    H = Tp + B + rowh * len(d)
    x = lin(-4.5, 12.5, L, W - Rr)
    o = []
    for t in [-4, 0, 4, 8, 12]:
        o.append(f'<line class="{"axis" if t == 0 else "grid"}" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{Tp}" y2="{H - B}"/>')
        o.append(f'<text class="tick" x="{x(t):.1f}" y="{H - B + 18}" text-anchor="middle">{fmt(t, 0, t != 0)}</text>')
    o.append(f'<text class="axlab" x="{(L + W - Rr) / 2}" y="{H - 8}" text-anchor="middle">Projected 2026-27 BPM · band = 80% prediction interval</text>')
    for i, r in enumerate(d.itertuples()):
        cy = Tp + (i + 0.5) * rowh
        o.append(f'<g class="pt"><title>{E(r.player)} ({r.team}, {int(r.age)}): Marcel {fmt(r.marcel_pred, 1)}, model {fmt(r.pred, 1)} '
                 f'[{fmt(r.pred_low, 1)}, {fmt(r.pred_high, 1)}]; 2025-26 BPM {fmt(r.bpm, 1)}</title>'
                 f'<text class="rowlab" x="{L - 12}" y="{cy + 4:.1f}" text-anchor="end">{E(r.player)} <tspan class="dim">{r.team}</tspan></text>'
                 f'<line class="band" x1="{x(r.pred_low):.1f}" x2="{x(r.pred_high):.1f}" y1="{cy:.1f}" y2="{cy:.1f}"/>'
                 f'<line class="link" x1="{x(r.marcel_pred):.1f}" x2="{x(r.pred):.1f}" y1="{cy:.1f}" y2="{cy:.1f}"/>'
                 f'<circle class="dot mutefill" cx="{x(r.marcel_pred):.1f}" cy="{cy:.1f}" r="5"/>'
                 f'<circle class="dot s1" cx="{x(r.pred):.1f}" cy="{cy:.1f}" r="6"/>'
                 f'<rect class="hit" x="0" y="{cy - rowh / 2:.1f}" width="{W}" height="{rowh}"/></g>')
    return svg(W, H, "".join(o), "2026-27 projections: Marcel versus model, with intervals")


# ---------------------------------------------------------------- tables
m = pd.read_csv(T / "model_metrics.csv").set_index("model")
b = pd.read_csv(T / "bootstrap_vs_marcel.csv").set_index("model")
sens = pd.read_csv(T / "sensitivity.csv")
fc = pd.read_csv(T / "forecast_next_season.csv")
age = pd.read_csv(T / "age_curve.csv").set_index("age")
attr = pd.read_csv(T / "attrition_by_projection.csv")
cov = pd.read_csv(T / "interval_coverage_by_minutes.csv")
abl = pd.read_csv(T / "ablation.csv").set_index("model")
sig = pd.read_csv(T / "signal_regression.csv").set_index("feature")
naive = pd.read_csv(T / "naive_most_improved.csv")

rv = R["ridge_vs_marcel"]
skill = R["ridge_skill_pct"]
wins = int(((1 - pd.read_csv(T / "model_metrics_by_season.csv").pivot(index="season", columns="model", values="rmse_w").pipe(lambda p: p["Ridge"] / p["Marcel"])) > 0).sum())


def pct(v, d=0):
    return f"{v * 100:.{d}f}%"


model_rows = ""
for name in ["Persistence", "Age curve", "Marcel", "Ridge", "Gradient boosting"]:
    r = m.loc[name]
    vs = "—" if name == "Marcel" else (f"{fmt(b.loc[name].rmse_diff, 3, True)} <span class='dim'>[{fmt(b.loc[name].ci_low, 3, True)}, {fmt(b.loc[name].ci_high, 3, True)}]</span>")
    hit = "—" if pd.isna(r.top10_hit_rate) else pct(r.top10_hit_rate)
    rc = "—" if pd.isna(r.spearman_delta) else fmt(r.spearman_delta, 2)
    model_rows += (f"<tr{' class=hl' if name == 'Ridge' else ''}><th scope=row>{name}</th><td>{r.rmse_w:.3f}</td>"
                   f"<td>{rc}</td><td>{hit}</td><td>{vs}</td></tr>")

sens_rows = ""
for r in sens.itertuples():
    sens_rows += (f"<tr><th scope=row>{E(r.variant)}</th><td>{r.n_test:,}</td>"
                  f"<td>{fmt(r.rmse_diff, 3, True)} <span class='dim'>[{fmt(r.ci_low, 3, True)}, {fmt(r.ci_high, 3, True)}]</span></td>"
                  f"<td>{fmt(r.coef_ts_rel, 2, True)}</td><td>{fmt(r.coef_ft_pct_shrunk, 2, True)}</td>"
                  f"<td>{fmt(r.coef_f_tr_rel, 2, True)}</td><td>{fmt(r.coef_usg_percent, 2, True)}</td></tr>")

fc_top = fc[fc["mp"] >= 1000].nlargest(7, "model_minus_marcel")
fc_rows = "".join(
    f"<tr><th scope=row>{E(r.player)}</th><td>{r.team}</td><td>{int(r.age)}</td><td>{fmt(r.bpm, 1)}</td>"
    f"<td>{fmt(r.marcel_pred, 1)}</td><td><b>{fmt(r.pred, 1)}</b></td><td>{fmt(r.model_minus_marcel, 1, True)}</td>"
    f"<td class=dim>{fmt(r.pred_low, 1)} to {fmt(r.pred_high, 1)}</td></tr>" for r in fc_top.itertuples())
kk = fc[fc["player"] == "Kon Knueppel"].iloc[0]
cs = fc[fc["player"] == "Stephon Castle"].iloc[0]
kg = fc[fc["player"] == "Keyonte George"].iloc[0]

S = lambda f: sig.loc[f]  # noqa: E731
sig_rows = ""
for f, read in [("ts_rel", "BPM built on hot shooting regresses. Efficiency is the noisiest part of the box score."),
                ("ft_pct_shrunk", "Touch. Shooting growth tends to show up at the line first."),
                ("f_tr_rel", "Rim pressure and creation."),
                ("usg_percent", "Players trusted with volume tend to grow into it."),
                ("tov_percent", "At a given usage, turnovers mark ball-handling load, not just carelessness."),
                ("gs_share", "The coaching staff's revealed belief carries information.")]:
    r = S(f)
    sig_rows += (f"<tr><th scope=row>{E(FEATURE_LABELS[f])}</th><td class='{'neg' if r.coef_per_sd < 0 else 'pos'}'>{fmt(r.coef_per_sd, 2, True)}</td>"
                 f"<td class=dim>{fmt(r.ci_low, 2, True)} to {fmt(r.ci_high, 2, True)}</td><td class=read>{read}</td></tr>")

page = f"""<title>Beyond the Box Score</title>
<meta name="claude:viewport-fit-sides" content="cover">
<meta name="description" content="Forecasting young NBA player development beyond age and regression to the mean.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a single reading column (~68ch) with charts and stat tables breaking slightly wider; condensed scoreboard type for headings and numbers, serif for argument. */
:root {{
  --bg: #f6f7f9; --surface: #ffffff; --fg: #14171c; --fg-2: #4a5160; --muted: #7d8494;
  --rule: #dde1e8; --grid: #e6e9ef; --axis: #b9bfca;
  --s1: #2a78d6; --s1-soft: #cfe0f6; --s2: #eb6834; --mute: #b3b8c2; --hl: #edf3fc;
  --display: "Barlow Condensed", "Arial Narrow", "Roboto Condensed", sans-serif;
  --body: "Source Serif 4", Georgia, "Times New Roman", serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #0f1115; --surface: #171a20; --fg: #eceef2; --fg-2: #b4bac6; --muted: #8a91a0;
  --rule: #2a2f38; --grid: #232830; --axis: #3d4450;
  --s1: #4b93ea; --s1-soft: #1f3a5e; --s2: #e0703f; --mute: #5c6372; --hl: #18263a; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #0f1115; --surface: #171a20; --fg: #eceef2; --fg-2: #b4bac6; --muted: #8a91a0;
  --rule: #2a2f38; --grid: #232830; --axis: #3d4450;
  --s1: #4b93ea; --s1-soft: #1f3a5e; --s2: #e0703f; --mute: #5c6372; --hl: #18263a; color-scheme: dark }}
* {{ box-sizing: border-box }}
body {{ background: var(--bg); color: var(--fg); font: 400 18px/1.62 var(--body);
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
main {{ max-width: 860px; margin: 0 auto; padding-block: 48px 72px; display: grid; gap: 0 }}
.col {{ max-width: 68ch }}
header.hero {{ padding-block: 8px 28px; border-bottom: 1px solid var(--rule); display: grid; gap: 14px }}
.kicker {{ font: 600 14px/1 var(--display); letter-spacing: .14em; text-transform: uppercase; color: var(--s1) }}
h1 {{ font: 700 clamp(44px, 9vw, 84px)/.92 var(--display); letter-spacing: -.01em; margin: 0; text-wrap: balance; text-transform: uppercase }}
.dek {{ font-size: 21px; line-height: 1.5; color: var(--fg-2); margin: 0; max-width: 62ch; text-wrap: pretty }}
.byline {{ font: 400 13px/1.5 var(--mono); color: var(--muted) }}
.statline {{ margin-top: 30px; border-top: 3px solid var(--fg); border-bottom: 1px solid var(--rule); overflow-x: auto }}
.statline table {{ width: 100%; border-collapse: collapse; min-width: 560px }}
.statline th {{ font: 600 12px/1.2 var(--display); letter-spacing: .1em; text-transform: uppercase; color: var(--muted); text-align: left; padding: 10px 12px 4px 0; vertical-align: bottom }}
.statline td {{ font: 600 34px/1.1 var(--display); padding: 0 12px 12px 0; font-variant-numeric: tabular-nums; vertical-align: top }}
.statline td small {{ display: block; font: 400 12.5px/1.35 var(--mono); color: var(--fg-2); margin-top: 4px; max-width: 22ch }}
section {{ padding-top: 56px; display: grid; gap: 18px }}
h2 {{ font: 700 clamp(28px, 5vw, 38px)/1.02 var(--display); text-transform: uppercase; letter-spacing: .005em; margin: 0; text-wrap: balance }}
h2 .n {{ color: var(--muted); font-weight: 500; margin-right: .3em }}
h3 {{ font: 600 22px/1.2 var(--display); text-transform: uppercase; letter-spacing: .03em; margin: 10px 0 0 }}
p {{ margin: 0; text-wrap: pretty }}
.col > * + * {{ margin-top: 14px }}
b, strong {{ font-weight: 600 }}
ul {{ margin: 0; padding-left: 1.2em; display: grid; gap: 8px }}
figure {{ margin: 6px 0 0; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 18px 18px 14px; display: grid; gap: 10px }}
figure .ft {{ font: 600 19px/1.15 var(--display); text-transform: uppercase; letter-spacing: .02em }}
figcaption {{ font: 400 13px/1.5 var(--mono); color: var(--fg-2) }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font: 400 13px/1.3 var(--mono); color: var(--fg-2) }}
.legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px }}
.legend .band {{ width: 22px; height: 8px; border-radius: 2px; background: var(--s1-soft) }}
svg {{ width: 100%; height: auto; display: block; overflow: visible }}
.chart {{ overflow-x: auto; min-width: 0 }}
@media (max-width: 640px) {{ .chart svg {{ min-width: 600px }} }}
svg text {{ font-family: var(--mono) }}
.grid {{ stroke: var(--grid); stroke-width: 1 }}
.axis {{ stroke: var(--axis); stroke-width: 1.2 }}
.tick {{ font-size: 11.5px; fill: var(--muted) }}
.tick.em {{ fill: var(--fg-2) }}
.axlab {{ font-size: 12px; fill: var(--fg-2) }}
.rowlab {{ font-size: 13px; fill: var(--fg) }}
.rowlab.em {{ font-weight: 500 }}
.rowlab.dim, .dim {{ fill: var(--muted); color: var(--muted) }}
.dlab {{ font-size: 12.5px; fill: var(--fg-2) }}
.vlab {{ font-size: 11.5px; fill: var(--fg-2) }}
.line {{ fill: none; stroke-width: 2.2; stroke-linejoin: round }}
.line.s1, .whisk.s1, .ci.s1 {{ stroke: var(--s1) }}
.line.s2, .whisk.s2, .ci.s2 {{ stroke: var(--s2) }}
.ci.mute {{ stroke: var(--mute) }}
.whisk {{ stroke-width: 1.6 }}
.ci {{ stroke-width: 3; stroke-linecap: round }}
.band {{ stroke: var(--s1-soft); stroke-width: 10; stroke-linecap: round }}
.link {{ stroke: var(--s1); stroke-width: 1.6 }}
.dot {{ stroke: var(--surface); stroke-width: 2 }}
.dot.s1 {{ fill: var(--s1) }} .dot.s2 {{ fill: var(--s2) }} .dot.ink {{ fill: var(--fg) }} .dot.mutefill {{ fill: var(--mute) }}
.bar.s1 {{ fill: var(--s1) }} .bar.s2 {{ fill: var(--s2) }}
.hit {{ fill: transparent }}
.pt:hover .dot {{ r: 7 }} .pt:hover .rowlab {{ fill: var(--s1) }} .pt:hover .bar {{ opacity: .85 }}
.tbl {{ overflow-x: auto; border-top: 2px solid var(--fg); }}
.tbl table {{ width: 100%; border-collapse: collapse; font: 400 14px/1.4 var(--mono); font-variant-numeric: tabular-nums; min-width: 520px }}
.tbl th, .tbl td {{ padding: 9px 12px 9px 0; border-bottom: 1px solid var(--rule); text-align: right; vertical-align: top }}
.tbl thead th {{ font: 600 12.5px/1.2 var(--display); letter-spacing: .08em; text-transform: uppercase; color: var(--muted) }}
.tbl th[scope=row], .tbl thead th:first-child {{ text-align: left; font-weight: 500 }}
.tbl tr.hl td, .tbl tr.hl th {{ background: var(--hl) }}
.tbl td.read {{ text-align: left; font-family: var(--body); font-size: 15px; color: var(--fg-2); min-width: 24ch }}
.tbl td.pos {{ color: var(--s1); font-weight: 500 }} .tbl td.neg {{ color: var(--s2); font-weight: 500 }}
.note {{ font: 400 13px/1.5 var(--mono); color: var(--fg-2) }}
.callout {{ border-left: 3px solid var(--s2); padding: 4px 0 4px 16px; max-width: 66ch }}
.callout p + p {{ margin-top: 8px }}
code {{ font: 400 .86em var(--mono); background: var(--surface); border: 1px solid var(--rule); border-radius: 4px; padding: 1px 5px }}
a {{ color: var(--s1); text-underline-offset: 3px }}
a:focus-visible {{ outline: 2px solid var(--s1); outline-offset: 2px; border-radius: 2px }}
footer {{ margin-top: 64px; padding-top: 18px; border-top: 1px solid var(--rule); font: 400 13px/1.6 var(--mono); color: var(--muted); display: grid; gap: 8px }}
@media (max-width: 560px) {{ body {{ font-size: 17px }} .dek {{ font-size: 18.5px }} figure {{ padding: 12px 10px 10px }} main {{ padding-block: 28px 56px }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important }} }}
</style>

<main>
<header class="hero">
  <div class="kicker">NBA player development · Research</div>
  <h1>Beyond the Box Score</h1>
  <p class="dek">Most one-year leaps by young NBA players are regression to the mean. Once you account for that, the box score's real growth signal is offensive load and touch, not shooting efficiency.</p>
  <div class="byline">Cyrus Skomra · Data through the 2025-26 season · Forecast for 2026-27 made before tip-off</div>
  <div class="statline"><table>
    <thead><tr><th scope="col">Sample</th><th scope="col">Edge vs. Marcel</th><th scope="col">Top-decile hit rate</th><th scope="col">Interval coverage</th></tr></thead>
    <tbody><tr>
      <td>{R['n_model_rows']:,}<small>young player-seasons, 2000-01 to 2025-26</small></td>
      <td>{skill:.1f}%<small>lower error; 95% CI {fmt(-rv['ci_high'], 3)} to {fmt(-rv['ci_low'], 3)} BPM</small></td>
      <td>{pct(m.loc['Ridge'].top10_hit_rate)}<small>gained 1.5+ BPM, vs. {pct(m.loc['Marcel'].top10_hit_rate)} for Marcel, {pct(R['base_rate_improve'])} base rate</small></td>
      <td>{pct(R['interval_80_coverage'])}<small>of outcomes inside the 80% intervals</small></td>
    </tr></tbody></table></div>
</header>

<section>
  <h2><span class="n">01</span>The question</h2>
  <div class="col">
    <p>Rookie-scale extensions, trades for young players and minutes on a rebuilding roster are all bets on development. The question is always the same: <b>is this player going to get better?</b></p>
    <p>The obvious approach is to study last year's most-improved players and look for what they had in common. That mostly rediscovers luck. So I asked a harder version: <b>once a disciplined projection already accounts for age and regression to the mean, what else predicts growth?</b></p>
    <p>The sample is every player aged 24 or younger with at least 500 minutes since 2000-01. The outcome is next-season Box Plus/Minus (BPM), a box-score estimate of points per 100 possessions above average. Each season was forecast using only information that existed at the time.</p>
  </div>
</section>

<section>
  <h2><span class="n">02</span>Most “improvement” is luck reversing</h2>
  <div class="col"><p>If one-year changes were pure development, the average change would be flat across starting points. It isn't. Players coming off a season of −4 BPM or worse gained about <b>two points</b> the next year, however many minutes they played. Players at +4 or better over 1,500+ minutes held steady. The same level in fewer minutes gave back almost two points.</p></div>
  <figure>
    <div class="ft">Bad seasons bounce back, great ones don't repeat</div>
    <div class="legend"><span><i style="background:var(--s1)"></i>1,500+ minutes in base season</span><span><i style="background:var(--s2)"></i>500–1,499 minutes</span></div>
    {chart_rtm()}
    <figcaption>Mean next-season BPM change by base-season BPM, with 95% CIs. Young players, 2000-01 to 2025-26. Hover a point for n.</figcaption>
  </figure>
  <div class="col"><p>Of the ten biggest BPM jumps from 2024-25 to 2025-26, <b>{int((naive.bpm < 0).sum())} started from a negative season</b> (average {fmt(naive.bpm.mean(), 1)}). Some, like {E(naive.player.iloc[0])}, are probably real growth. A raw leaderboard can't tell which.</p>
  <p>Development is also front-loaded: about <b>{fmt(age.loc[[19, 20], 'mean_delta'].mean(), 1, True)} BPM a year at 19-20</b>, falling to {fmt(age.loc[22, 'mean_delta'], 1, True)} at 22 and {fmt(age.loc[[23, 24], 'mean_delta'].mean(), 1, True)} at 23-24. Survivors flatter that curve. Players projected in the bottom fifth fall out of the rotation {pct(attr.dropout_rate.iloc[0])} of the time, against {pct(attr.dropout_rate.iloc[-1])} in the top fifth, so every result here was re-checked with inverse-probability weighting.</p></div>
</section>

<section>
  <h2><span class="n">03</span>The bar to clear</h2>
  <div class="col"><p>A model has to beat a disciplined projection, not a strawman. <b>Marcel</b>, adapted from Tom Tango's baseball system, weights the last three seasons 5/4/3 by minutes. It blends them with {R['marcel_k_full']:.0f} minutes of league-average prior and adds a fitted age adjustment. It already knows about age and regression to the mean.</p>
  <p>Each season from 2016-17 to 2025-26 was predicted by models refit only on earlier seasons. That gives {R['n_test_rows']:,} out-of-sample player-seasons, exactly what a front office could have done in real time. Hyperparameters were tuned on pre-2016 data only, and confidence intervals resample players, not rows.</p></div>
  <div class="tbl"><table>
    <thead><tr><th scope="col">Model</th><th scope="col">Weighted RMSE</th><th scope="col">Rank corr.</th><th scope="col">Top-10% hit</th><th scope="col">vs. Marcel [95% CI]</th></tr></thead>
    <tbody>{model_rows}</tbody></table></div>
  <p class="note">Rank corr. = Spearman correlation of predicted vs. actual BPM change. Top-10% hit = share of each model's top-decile predicted improvers who gained 1.5+ BPM (base rate {pct(R['base_rate_improve'])}). Persistence predicts zero change for everyone, so neither applies.</p>
  <figure>
    <div class="ft">Ridge regression is the only model clearly better than Marcel</div>
    {chart_gain()}
    <figcaption>Paired, player-clustered bootstrap (2,000 resamples). Gradient boosting's interval crosses zero: no evidence of nonlinearity worth paying for.</figcaption>
  </figure>
  <figure>
    <div class="ft">The edge is small and steady: better in {wins} of 10 seasons</div>
    <div class="legend"><span><i style="background:var(--s1);border-radius:2px"></i>Ridge more accurate</span><span><i style="background:var(--s2);border-radius:2px"></i>Marcel more accurate</span></div>
    {chart_skill()}
    <figcaption>Ridge RMSE reduction vs. Marcel, by season predicted. The two misses came first, when the model had the least training history.</figcaption>
  </figure>
</section>

<section>
  <h2><span class="n">04</span>Bet on volume and touch, discount hot shooting</h2>
  <div class="col"><p>Adding feature groups one at a time showed where the edge comes from. <b>Per-possession box-score rates</b> were the only group that clearly helped ({fmt(abl.loc['Core + box_rates'].rmse_diff, 3, True)} RMSE, CI {fmt(abl.loc['Core + box_rates'].ci_low, 3, True)} to {fmt(abl.loc['Core + box_rates'].ci_high, 3, True)}). On/off plus-minus, shot location, draft slot, height and team quality added nothing distinguishable from zero once the box score was known.</p>
  <p>To see which rates matter, I regressed the gap between actual BPM and Marcel's projection on standardized season signals, with errors clustered by player.</p></div>
  <figure>
    <div class="ft">What predicts growth beyond the projection</div>
    <div class="legend"><span><i style="background:var(--s1)"></i>Predicts more growth (95% CI excludes zero)</span><span><i style="background:var(--s2)"></i>Predicts regression</span><span><i style="background:var(--mute)"></i>Not distinguishable from zero</span></div>
    {chart_signals()}
    <figcaption>Weighted least squares on Marcel residuals, n = {R['n_observed']:,}. Collinear pairs (|r| ≥ 0.75) removed. Predictive associations, not causal effects.</figcaption>
  </figure>
  <div class="tbl"><table>
    <thead><tr><th scope="col">Signal</th><th scope="col">BPM per SD</th><th scope="col">95% CI</th><th scope="col" style="text-align:left">Basketball reading</th></tr></thead>
    <tbody>{sig_rows}</tbody></table></div>
  <div class="col"><h3>What a front office could do with it</h3>
  <ul>
    <li><b>Discount efficiency-driven breakouts.</b> A young role player whose BPM rests on above-league true shooting at low usage is likelier to regress than one with the same BPM built on creation.</li>
    <li><b>Read free-throw percentage as a development signal,</b> not just a free-throw signal.</li>
    <li><b>Don't over-penalize turnovers</b> in young players carrying a big offensive load.</li>
  </ul>
  <p>The effects are modest: 0.1 to 0.2 BPM per standard deviation. Together they explain {pct(R['signal_regression_r2'], 1)} of what Marcel misses. This sharpens priors and points film study; it doesn't replace scouting.</p></div>
</section>

<section>
  <h2><span class="n">05</span>2026-27: where the model disagrees with Marcel</h2>
  <div class="col"><p>Ranking players by projected change just lists last season's worst low-minute years, which is regression to the mean again. The useful number is <b>model minus Marcel</b>, the part of the forecast that comes from box-score signals.</p></div>
  <figure>
    <div class="ft">Young players the signals like more than a standard projection</div>
    <div class="legend"><span><i style="background:var(--mute)"></i>Marcel projection</span><span><i style="background:var(--s1)"></i>Model projection</span><span><i class="band"></i>80% interval</span></div>
    {chart_forecast(fc)}
    <figcaption>Age ≤ 24 with 1,000+ minutes in 2025-26, ranked by model minus Marcel. Team “2TM” = played for two teams.</figcaption>
  </figure>
  <div class="tbl"><table>
    <thead><tr><th scope="col">Player</th><th scope="col">Team</th><th scope="col">Age</th><th scope="col">2025-26</th><th scope="col">Marcel</th><th scope="col">Model</th><th scope="col">Gap</th><th scope="col">80% interval</th></tr></thead>
    <tbody>{fc_rows}</tbody></table></div>
  <div class="col"><p>The common thread is load. {E(cs.player)} ({cs.usg_percent:.0f}% usage, free-throw rate {fmt(cs.f_tr_rel, 2, True)} over league) and {E(kg.player)} ({kg.usg_percent:.0f}% usage, {kg.ft_pct_shrunk * 100:.0f}% shrunk FT%) carry big roles at modest efficiency, the profile the signals favor.</p></div>
  <div class="callout"><p><b>The counterexample.</b> {E(kk.player)} posted {fmt(kk.bpm, 1, True)} BPM as a rookie. Marcel, which sees only BPM and age, projects {fmt(kk.marcel_pred, 1, True)}. The model projects {fmt(kk.pred, 1, True)}, because his season leaned on above-league true shooting ({fmt(kk.ts_rel * 100, 0, True)} points) with a below-average free-throw rate at {kk.usg_percent:.0f}% usage.</p>
  <p>That isn't a call that he's overrated. It's a flag that his profile resembles shooting-driven seasons that tend not to build further, and the first thing I'd check on film.</p></div>
  <p class="note">The gaps (about 1 to 1.7 BPM) are smaller than the ±{R['interval_80_half_width']:.1f} BPM intervals. Treat them as scouting leads, not verdicts. This forecast will be scored against 2026-27 results.</p>
</section>

<section>
  <h2><span class="n">06</span>Does it hold up?</h2>
  <div class="col"><p>Each variant re-runs the full season-by-season comparison and the signal regression. The ridge edge holds in all eight, and it is largest for the youngest players. Every key signal keeps its sign in every variant.</p></div>
  <div class="tbl"><table>
    <thead><tr><th scope="col">Variant</th><th scope="col">n</th><th scope="col">vs. Marcel [95% CI]</th><th scope="col">TS%</th><th scope="col">FT%</th><th scope="col">FT rate</th><th scope="col">Usage</th></tr></thead>
    <tbody>{sens_rows}</tbody></table></div>
  <div class="col"><h3>Where it fails</h3>
  <ul>
    <li><b>Role changes and injuries.</b> The biggest upside misses were breakouts tied to new roles: Victor Oladipo after the 2017 trade to Indiana, Giannis Antetokounmpo in 2016-17, Ja Morant in 2021-22. The biggest downside misses were mostly injury-shortened seasons. The box score can't see either coming.</li>
    <li><b>Intervals undercover when minutes collapse:</b> {pct(cov['mean'].iloc[0])} coverage for players who end up with 250–999 minutes, against {pct(cov['mean'].iloc[1])} and {pct(cov['mean'].iloc[2])} in the higher-minute bins.</li>
    <li><b>BPM's blind spots carry through.</b> It under-rates defense that doesn't produce steals, blocks or rebounds, and it isn't RAPM or tracking-based impact.</li>
    <li><b>No market context.</b> Nothing here speaks to salary or contract value.</li>
  </ul>
  <h3>What would make it better</h3>
  <ul>
    <li>A multi-year RAPM target, so the model isn't only predicting a box-score formula.</li>
    <li>Tracking data (drives, touches, matchups) and injury and role data for the largest misses.</li>
    <li>A separate minutes model: “will he improve?” and “will he play?” are different questions.</li>
  </ul></div>
</section>

<footer>
  <div>Method: Python (pandas, scikit-learn, statsmodels). Rolling-origin validation, player-clustered bootstrap, online split-conformal intervals, empirical-Bayes shrinkage of FT% and 3P%. The whole analysis regenerates from one command, <code>python -m bbs.pipeline</code>, with 44 automated tests. Code: <a href="https://github.com/doxxk/nba-player-analytics" target="_blank" rel="noopener">github.com/doxxk/nba-player-analytics</a>.</div>
  <div>Data: <a href="https://www.basketball-reference.com" target="_blank" rel="noopener">Basketball-Reference</a> (Sports Reference LLC), via the public <a href="https://github.com/sumitrodatta/bball-reference-datasets" target="_blank" rel="noopener">sumitrodatta/bball-reference-datasets</a> mirror at commit {R['source_commit'][:7]}. BPM 2.0 by Daniel Myers; Marcel method by Tom Tango.</div>
</footer>
</main>
"""
OUT.write_text(page)
print(OUT, len(page))
