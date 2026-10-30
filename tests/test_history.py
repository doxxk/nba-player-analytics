"""The dashboard's downloaded player history must match the pipeline exactly."""

from pathlib import Path

import pandas as pd
import pytest
import requests

from bbs import config
from bbs.data import clean, history
from bbs.data.acquire import DataUnavailableError

COLS = ["season", "player", "player_id", "age", "team", "mp", "bpm"]


def _key(df):
    return df[COLS].sort_values(["player_id", "season"]).reset_index(drop=True)


def test_matches_pipeline_player_seasons(raw_tables):
    built = history.build_history(raw_tables["advanced"], first_season=2020)
    pipeline = clean.build_player_seasons(raw_tables, first_season=2020)
    pd.testing.assert_frame_equal(_key(built), _key(pipeline), check_dtype=False)


def test_traded_player_uses_combined_row(raw_tables):
    h = history.build_history(raw_tables["advanced"], first_season=2020)
    beta = h[(h.player_id == "beta") & (h.season == 2021)]
    assert len(beta) == 1 and beta.iloc[0]["team"] == "2TM" and beta.iloc[0]["mp"] == 1000
    assert not h.duplicated(["season", "player_id"]).any()


def test_filters_to_requested_players(raw_tables):
    h = history.build_history(raw_tables["advanced"], player_ids=("alpha",), first_season=2020)
    assert set(h["player_id"]) == {"alpha"} and len(h) == 4
    assert list(h.columns) == COLS


def test_missing_columns_raise(raw_tables):
    with pytest.raises(DataUnavailableError, match="bpm"):
        history.build_history(raw_tables["advanced"].drop(columns=["bpm"]))


def test_download_failure_raises(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(history.requests, "get", boom)
    with pytest.raises(DataUnavailableError, match="player history"):
        history.fetch_advanced("https://example.invalid/Advanced.csv")


def test_fetch_reads_only_history_columns(monkeypatch, raw_tables):
    body = raw_tables["advanced"].to_csv(index=False).encode()

    class Resp:
        content = body

        def raise_for_status(self):
            pass

    monkeypatch.setattr(history.requests, "get", lambda *a, **k: Resp())
    df = history.fetch_advanced("https://example.invalid/Advanced.csv")
    assert set(df.columns) == set(history.HISTORY_COLUMNS)


def test_url_points_at_pinned_commit():
    url = history.advanced_url()
    assert config.SOURCE_COMMIT in url and url.endswith("Advanced.csv")


RAW = config.RAW_DIR / "advanced.csv"
PROCESSED = config.PROCESSED_DIR / "player_seasons.parquet"


@pytest.mark.skipif(not (RAW.exists() and PROCESSED.exists()),
                    reason="needs one local pipeline run")
def test_real_data_matches_pipeline_output():
    adv = pd.read_csv(RAW, usecols=history.HISTORY_COLUMNS)
    built = history.build_history(adv)
    pipeline = pd.read_parquet(PROCESSED)
    pd.testing.assert_frame_equal(_key(built), _key(pipeline), check_dtype=False)
