"""Expected columns for each raw table, and a validator with useful errors."""

from __future__ import annotations

import pandas as pd

KEY = ["season", "lg", "player", "player_id", "age", "team", "pos", "g", "gs", "mp"]

REQUIRED_COLUMNS: dict[str, list[str]] = {
    "advanced": KEY + [
        "per", "ts_percent", "x3p_ar", "f_tr", "orb_percent", "drb_percent",
        "trb_percent", "ast_percent", "stl_percent", "blk_percent", "tov_percent",
        "usg_percent", "ows", "dws", "ws", "ws_48", "obpm", "dbpm", "bpm", "vorp",
    ],
    "totals": KEY + ["fg", "fga", "x3p", "x3pa", "x2p", "x2pa", "ft", "fta", "orb", "drb",
                     "ast", "stl", "blk", "tov", "pts"],
    "per100": KEY + ["x3pa_per_100_poss", "fta_per_100_poss", "ast_per_100_poss", "o_rtg",
                     "d_rtg"],
    "shooting": KEY + [
        "avg_dist_fga", "percent_fga_from_x0_3_range", "fg_percent_from_x0_3_range",
        "percent_fga_from_x16_3p_range", "percent_assisted_x2p_fg", "percent_assisted_x3p_fg",
        "percent_corner_3s_of_3pa", "percent_dunks_of_fga",
    ],
    "pbp": KEY + ["pg_percent", "sg_percent", "sf_percent", "pf_percent", "c_percent",
                  "on_court_plus_minus_per_100_poss", "net_plus_minus_per_100_poss",
                  "points_generated_by_assists"],
    "season_info": ["season", "lg", "player_id", "team", "experience"],
    "career_info": ["player_id", "ht_in_in", "wt", "birth_date"],
    "draft": ["season", "lg", "overall_pick", "player_id"],
    "team_summaries": ["season", "lg", "team", "abbreviation", "srs", "n_rtg", "w", "l",
                       "ts_percent", "x3p_ar", "f_tr"],
}


class SchemaError(ValueError):
    """Raised when a raw table is missing expected columns or is empty."""


def validate(name: str, df: pd.DataFrame) -> None:
    """Check that ``df`` has the columns this project relies on."""
    if df.empty:
        raise SchemaError(f"Table '{name}' is empty.")
    missing = [c for c in REQUIRED_COLUMNS.get(name, []) if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Table '{name}' is missing columns {missing}. The upstream dataset may "
            "have changed format; check config.SOURCE_COMMIT."
        )


def validate_all(tables: dict[str, pd.DataFrame]) -> None:
    absent = [n for n in REQUIRED_COLUMNS if n not in tables]
    if absent:
        raise SchemaError(f"Missing tables: {absent}")
    for name, df in tables.items():
        validate(name, df)
