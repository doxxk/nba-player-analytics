"""Season-by-season BPM history for the dashboard's player comparison.

Built directly from the pinned Advanced table, so the dashboard works where the
pipeline has never run (for example Streamlit Community Cloud). It applies the
same filters and traded-player handling as ``clean.build_player_seasons``, so
the values match the pipeline's player-season table exactly.
"""

from __future__ import annotations

import io

import pandas as pd
import requests

from bbs import config
from bbs.data import clean
from bbs.data.acquire import DataUnavailableError

# Only what the comparison view plots: identity, season, age, minutes, BPM.
HISTORY_COLUMNS = ["season", "lg", "player", "player_id", "age", "team", "mp", "bpm"]


def advanced_url() -> str:
    return config.SOURCE_URL.format(name=requests.utils.quote(config.SOURCE_FILES["advanced"]))


def fetch_advanced(url: str | None = None, timeout: int = 60) -> pd.DataFrame:
    """Download the pinned Advanced table, keeping only the history columns."""
    url = url or advanced_url()
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException as e:
        raise DataUnavailableError(f"Could not download player history from {url} ({e}).")
    return pd.read_csv(io.BytesIO(r.content), usecols=HISTORY_COLUMNS)


def build_history(advanced: pd.DataFrame, player_ids=None,
                  first_season: int = config.FIRST_SEASON) -> pd.DataFrame:
    """One row per NBA player-season (traded players use the combined row).

    ``player_ids`` limits the result to those players, which keeps the cached
    table small.
    """
    missing = [c for c in HISTORY_COLUMNS if c not in advanced.columns]
    if missing:
        raise DataUnavailableError(f"Advanced table is missing columns {missing}.")
    adv = advanced.loc[(advanced["lg"] == "NBA") & (advanced["season"] >= first_season)]
    adv = adv.drop_duplicates(["season", "player_id", "team"])
    out = clean.collapse_traded(adv)
    if player_ids is not None:
        out = out.loc[out["player_id"].isin(set(player_ids))]
    cols = ["season", "player", "player_id", "age", "team", "mp", "bpm"]
    return out[cols].sort_values(["player_id", "season"]).reset_index(drop=True)
