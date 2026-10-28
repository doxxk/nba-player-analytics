"""Small synthetic fixtures that mimic the Basketball-Reference table layout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bbs.data.schema import REQUIRED_COLUMNS


def _player_row(season, pid, team, age, mp, bpm, **extra):
    row = {c: 0.0 for c in REQUIRED_COLUMNS["advanced"]}
    row.update({"season": season, "lg": "NBA", "player": pid.title(), "player_id": pid,
                "age": age, "team": team, "pos": "SG", "g": 60, "gs": 30, "mp": mp,
                "bpm": bpm, "obpm": bpm / 2, "dbpm": bpm / 2, "ts_percent": 0.56,
                "x3p_ar": 0.35, "f_tr": 0.25, "usg_percent": 20.0})
    row.update(extra)
    return row


@pytest.fixture
def raw_tables() -> dict[str, pd.DataFrame]:
    """Three players over four seasons, including a traded player and a draftee."""
    adv = pd.DataFrame([
        _player_row(2020, "alpha", "AAA", 21, 1500, -1.0),
        _player_row(2021, "alpha", "AAA", 22, 1800, 0.5),
        _player_row(2022, "alpha", "AAA", 23, 2000, 1.5),
        _player_row(2023, "alpha", "AAA", 24, 2100, 2.0),
        _player_row(2021, "beta", "2TM", 20, 1000, -3.0),
        _player_row(2021, "beta", "AAA", 20, 400, -4.0),
        _player_row(2021, "beta", "BBB", 20, 600, -2.0),
        _player_row(2022, "beta", "BBB", 21, 1200, -1.0),
        _player_row(2020, "gamma", "BBB", 23, 900, 0.0),
        _player_row(2022, "gamma", "BBB", 25, 700, 1.0),  # skipped 2021
    ])
    keys = ["season", "lg", "player", "player_id", "age", "team", "pos", "g", "gs", "mp"]

    def table(name, **defaults):
        t = adv[keys].copy()
        for c in REQUIRED_COLUMNS[name]:
            if c not in t:
                t[c] = defaults.get(c, 1.0)
        return t

    totals = table("totals", fga=500, fta=100, ft=75, x3pa=200, x3p=70, pts=600)
    season_info = adv[["season", "lg", "player_id", "team"]].assign(experience=1)
    career = pd.DataFrame({"player_id": ["alpha", "beta", "gamma"], "ht_in_in": [78, 80, 82],
                           "wt": [200, 210, 240], "birth_date": ["1999-01-01"] * 3})
    draft = pd.DataFrame({"season": [2019, 2020, 2021], "lg": ["NBA"] * 3,
                          "overall_pick": [5, 40, 12], "player_id": ["alpha", "beta", "beta"]})
    teams = pd.DataFrame([
        {"season": s, "lg": "NBA", "team": n, "abbreviation": a, "srs": srs, "n_rtg": srs,
         "w": 41, "l": 41, "ts_percent": 0.57, "x3p_ar": 0.38, "f_tr": 0.25}
        for s in range(2020, 2024)
        for n, a, srs in [("Team A", "AAA", 4.0), ("Team B", "BBB", -2.0),
                          ("League Average", np.nan, 0.0)]
    ])
    return {"advanced": adv, "totals": totals, "per100": table("per100"),
            "shooting": table("shooting"), "pbp": table("pbp"), "season_info": season_info,
            "career_info": career, "draft": draft, "team_summaries": teams}


def synthetic_model_table(n_players: int = 120, seasons=range(2005, 2021), seed: int = 0
                          ) -> pd.DataFrame:
    """A model-table-shaped frame where next-season BPM depends on a latent talent."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_players):
        talent = rng.normal(-1, 2)
        start = rng.integers(seasons.start, seasons.stop - 3)
        prev = None
        for i, s in enumerate(range(start, min(start + 4, seasons.stop))):
            mp = float(rng.integers(500, 2500))
            bpm = talent + 0.4 * i + rng.normal(0, 3000 / mp ** 0.5 / 20)
            rows.append({"player": f"P{p}", "player_id": f"p{p}", "season": s,
                         "age": 20 + i, "team": "AAA", "pos_group": "Wing", "mp": mp,
                         "bpm": bpm, "bpm_lag1": prev[0] if prev else np.nan,
                         "mp_lag1": prev[1] if prev else np.nan, "bpm_lag2": np.nan,
                         "mp_lag2": np.nan, "talent": talent})
            prev = (bpm, mp)
    df = pd.DataFrame(rows)
    # Next season reflects latent talent plus growth, not this season's noise.
    df["bpm_next"] = df["talent"] + 0.4 * (df["age"] - 19) + rng.normal(0, 1.0, len(df))
    df["delta_bpm"] = df["bpm_next"] - df["bpm"]
    df["mp_next"] = 1500.0
    df["weight"] = 0.75
    df["observed"] = 1
    return df
