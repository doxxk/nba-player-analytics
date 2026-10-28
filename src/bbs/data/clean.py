"""Turn raw Basketball-Reference tables into one row per player-season.

Basketball-Reference lists a traded player once per team plus a combined row
(team code ``2TM``, ``3TM``, ...). We keep the combined row for player
statistics and use the per-team rows only to build a minutes-weighted team
context. Every filter is logged so exclusions are auditable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bbs import config

MULTI_TEAM = re.compile(r"^\dTM$|^TOT$")

ADVANCED_COLS = [
    "per", "ts_percent", "x3p_ar", "f_tr", "orb_percent", "drb_percent", "trb_percent",
    "ast_percent", "stl_percent", "blk_percent", "tov_percent", "usg_percent",
    "ows", "dws", "ws", "ws_48", "obpm", "dbpm", "bpm", "vorp",
]
TOTALS_COLS = ["fg", "fga", "x3p", "x3pa", "x2p", "x2pa", "ft", "fta", "orb", "drb",
               "ast", "stl", "blk", "tov", "pts"]
PER100_COLS = ["x3pa_per_100_poss", "fta_per_100_poss", "ast_per_100_poss", "o_rtg", "d_rtg"]
SHOOTING_COLS = [
    "avg_dist_fga", "percent_fga_from_x0_3_range", "fg_percent_from_x0_3_range",
    "percent_fga_from_x16_3p_range", "percent_assisted_x2p_fg",
    "percent_assisted_x3p_fg", "percent_corner_3s_of_3pa", "percent_dunks_of_fga",
]
PBP_COLS = ["pg_percent", "sg_percent", "sf_percent", "pf_percent", "c_percent",
            "on_court_plus_minus_per_100_poss", "net_plus_minus_per_100_poss",
            "points_generated_by_assists"]


@dataclass
class CleaningLog:
    steps: list[dict] = field(default_factory=list)

    def record(self, step: str, before: int, after: int, note: str = "") -> None:
        self.steps.append({"step": step, "rows_before": before, "rows_after": after,
                           "rows_removed": before - after, "note": note})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


def is_multi_team(team: pd.Series) -> pd.Series:
    return team.astype(str).str.match(MULTI_TEAM)


def collapse_traded(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per (season, player_id): the combined row if one exists."""
    df = df.copy()
    df["_multi"] = is_multi_team(df["team"])
    df = df.sort_values(["season", "player_id", "_multi"], ascending=[True, True, False])
    out = df.drop_duplicates(["season", "player_id"], keep="first").drop(columns="_multi")
    return out.reset_index(drop=True)


def team_context(advanced: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Minutes-weighted team SRS and net rating for each player-season.

    Uses per-team stint rows (not the combined row), so a player traded from a
    good team to a bad one gets a blend of the two.
    """
    stints = advanced.loc[~is_multi_team(advanced["team"]),
                          ["season", "player_id", "team", "mp"]]
    t = teams.loc[teams["abbreviation"].notna(), ["season", "abbreviation", "srs", "n_rtg"]]
    m = stints.merge(t, left_on=["season", "team"], right_on=["season", "abbreviation"],
                     how="left")
    m["w"] = m["mp"].clip(lower=1)

    def wavg(g: pd.DataFrame) -> pd.Series:
        ok = g["srs"].notna()
        if not ok.any():
            return pd.Series({"team_srs": np.nan, "team_net_rtg": np.nan, "n_teams": len(g)})
        w = g.loc[ok, "w"]
        return pd.Series({
            "team_srs": np.average(g.loc[ok, "srs"], weights=w),
            "team_net_rtg": np.average(g.loc[ok, "n_rtg"], weights=w),
            "n_teams": len(g),
        })

    return (m.groupby(["season", "player_id"])[["srs", "n_rtg", "w"]]
             .apply(wavg).reset_index())


def draft_positions(draft: pd.DataFrame) -> pd.DataFrame:
    """One NBA draft pick per player; for players drafted twice, the later draft."""
    d = draft.loc[(draft["lg"] == "NBA") & draft["player_id"].notna()]
    d = d.sort_values("season").drop_duplicates("player_id", keep="last")
    return d[["player_id", "overall_pick", "season"]].rename(
        columns={"season": "draft_year"})


def build_player_seasons(tables: dict[str, pd.DataFrame],
                         first_season: int = config.FIRST_SEASON,
                         log: CleaningLog | None = None) -> pd.DataFrame:
    """Merge all per-player tables into one row per NBA player-season."""
    log = log or CleaningLog()
    adv = tables["advanced"]
    n0 = len(adv)
    adv = adv.loc[(adv["lg"] == "NBA") & (adv["season"] >= first_season)]
    log.record("keep NBA seasons >= %d" % first_season, n0, len(adv),
               "drops ABA/BAA and seasons before shot-location data")

    n0 = len(adv)
    adv = adv.drop_duplicates(["season", "player_id", "team"])
    log.record("drop exact duplicate (season, player, team) rows", n0, len(adv))

    ctx = team_context(adv, tables["team_summaries"])
    n0 = len(adv)
    base = collapse_traded(adv)
    log.record("collapse traded players to combined row", n0, len(base),
               "per-team stints retained only for team context")

    keys = ["season", "player_id", "team"]
    base = base[["season", "player", "player_id", "age", "team", "pos", "g", "gs", "mp"]
                + ADVANCED_COLS]

    for name, cols in [("totals", TOTALS_COLS), ("per100", PER100_COLS),
                       ("shooting", SHOOTING_COLS), ("pbp", PBP_COLS)]:
        t = tables[name]
        t = t.loc[(t["lg"] == "NBA") & (t["season"] >= first_season)]
        t = t.drop_duplicates(keys)[keys + [c for c in cols if c in t.columns]]
        before = len(base)
        base = base.merge(t, on=keys, how="left", validate="one_to_one")
        assert len(base) == before

    exp = (tables["season_info"].loc[lambda d: d["lg"] == "NBA"]
           .groupby(["season", "player_id"], as_index=False)["experience"].max())
    base = base.merge(exp, on=["season", "player_id"], how="left", validate="one_to_one")
    base = base.merge(ctx, on=["season", "player_id"], how="left", validate="one_to_one")

    career = tables["career_info"].drop_duplicates("player_id")[["player_id", "ht_in_in", "wt"]]
    base = base.merge(career.rename(columns={"ht_in_in": "height_in", "wt": "weight_lb"}),
                      on="player_id", how="left", validate="many_to_one")
    base = base.merge(draft_positions(tables["draft"]), on="player_id", how="left",
                      validate="many_to_one")
    # A pick only counts if it happened before this season (no look-ahead).
    future_pick = base["draft_year"] >= base["season"]
    base.loc[future_pick, ["overall_pick", "draft_year"]] = np.nan
    base["undrafted"] = base["overall_pick"].isna().astype(int)
    base["draft_pick"] = base["overall_pick"].fillna(config.UNDRAFTED_PICK)

    base["traded"] = (base["n_teams"].fillna(1) > 1).astype(int)
    base["pos_group"] = base["pos"].map(position_group)
    return base.sort_values(["player_id", "season"]).reset_index(drop=True)


def position_group(pos: str | float) -> str:
    """Collapse Basketball-Reference positions into Guard / Wing / Big."""
    if not isinstance(pos, str) or not pos:
        return "Unknown"
    first = pos.split("-")[0].strip().upper()
    return {"PG": "Guard", "SG": "Guard", "G": "Guard", "SF": "Wing", "GF": "Wing",
            "F": "Wing", "PF": "Big", "FC": "Big", "C": "Big"}.get(first, "Unknown")


def season_completeness(team_summaries: pd.DataFrame, season: int) -> float:
    """Share of an 82-game schedule completed in ``season`` (league average)."""
    t = team_summaries.loc[(team_summaries["season"] == season)
                           & (team_summaries["lg"] == "NBA")
                           & team_summaries["abbreviation"].notna()]
    if t.empty:
        return 0.0
    return float(((t["w"] + t["l"]) / 82).mean())
