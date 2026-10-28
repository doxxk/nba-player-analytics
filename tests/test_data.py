import numpy as np
import pandas as pd
import pytest
import requests

from bbs.data import acquire, clean, schema


def test_collapse_traded_keeps_combined_row(raw_tables):
    out = clean.collapse_traded(raw_tables["advanced"])
    beta = out.loc[(out.player_id == "beta") & (out.season == 2021)]
    assert len(beta) == 1 and beta.iloc[0]["team"] == "2TM" and beta.iloc[0]["mp"] == 1000
    assert not out.duplicated(["season", "player_id"]).any()


def test_team_context_is_minutes_weighted(raw_tables):
    ctx = clean.team_context(raw_tables["advanced"], raw_tables["team_summaries"])
    beta = ctx.loc[(ctx.player_id == "beta") & (ctx.season == 2021)].iloc[0]
    # 400 MP on a +4 team and 600 MP on a -2 team
    assert beta["team_srs"] == pytest.approx((400 * 4 + 600 * -2) / 1000)
    assert beta["n_teams"] == 2


def test_build_player_seasons_one_row_per_player_season(raw_tables):
    log = clean.CleaningLog()
    ps = clean.build_player_seasons(raw_tables, first_season=2020, log=log)
    assert not ps.duplicated(["season", "player_id"]).any()
    assert len(ps) == 8
    assert ps.loc[(ps.player_id == "beta") & (ps.season == 2021), "traded"].item() == 1
    steps = log.to_frame()
    assert {"rows_before", "rows_after", "rows_removed"} <= set(steps.columns)


def test_draft_pick_has_no_look_ahead(raw_tables):
    ps = clean.build_player_seasons(raw_tables, first_season=2020)
    # beta was drafted in 2020 (pick 40) and again in 2021 (pick 12); the later
    # draft is kept, so seasons up to and including 2021 must not see it.
    b = ps.loc[ps.player_id == "beta"].set_index("season")
    assert np.isnan(b.loc[2021, "overall_pick"]) and b.loc[2021, "undrafted"] == 1
    assert b.loc[2022, "draft_pick"] == 12
    a = ps.loc[ps.player_id == "alpha"].set_index("season")
    assert (a["draft_pick"] == 5).all()


@pytest.mark.parametrize("pos,group", [("PG", "Guard"), ("SF-PF", "Wing"), ("C", "Big"),
                                       ("FC", "Big"), (np.nan, "Unknown"), ("", "Unknown")])
def test_position_group(pos, group):
    assert clean.position_group(pos) == group


def test_season_completeness(raw_tables):
    assert clean.season_completeness(raw_tables["team_summaries"], 2021) == 1.0
    assert clean.season_completeness(raw_tables["team_summaries"], 1990) == 0.0


def test_schema_rejects_missing_columns(raw_tables):
    bad = raw_tables["advanced"].drop(columns=["bpm"])
    with pytest.raises(schema.SchemaError, match="bpm"):
        schema.validate("advanced", bad)
    with pytest.raises(schema.SchemaError, match="empty"):
        schema.validate("advanced", pd.DataFrame())
    schema.validate_all(raw_tables)


def test_download_failure_raises_informative_error(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(acquire.requests, "get", boom)
    monkeypatch.setattr(acquire.time, "sleep", lambda s: None)
    with pytest.raises(acquire.DataUnavailableError, match="--sample"):
        acquire.download_file("https://example.invalid/x.csv", tmp_path / "x.csv", retries=2)
    assert not (tmp_path / "x.csv").exists()
    assert not (tmp_path / "x.csv.part").exists()


def test_load_raw_offline_reports_missing_files(tmp_path):
    with pytest.raises(acquire.DataUnavailableError, match="missing"):
        acquire.load_raw(raw_dir=tmp_path, download=False)
