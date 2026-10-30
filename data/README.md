# Data

## Source

| | |
|---|---|
| Origin | [Basketball-Reference](https://www.basketball-reference.com) season tables (Sports Reference LLC) |
| Mirror used | [sumitrodatta/bball-reference-datasets](https://github.com/sumitrodatta/bball-reference-datasets), CSV exports of Basketball-Reference pages |
| Pinned commit | `76a70b41ad1c13948f25c62c921ed822e5db7f0e` (committed 2026-04-13, after the 2025-26 regular season ended) |
| Retrieval | `python -m bbs.pipeline` downloads 9 CSVs (~24 MB) over HTTPS from `raw.githubusercontent.com` at that commit and caches them in `data/raw/`. `data/raw/manifest.json` records each file's URL, size, SHA-256, and retrieval time. |
| Seasons used | NBA 1996-97 through 2025-26 (shot-location and play-by-play tables begin in 1996-97). ABA/BAA rows are dropped. |

### Why this source

- **NBA Stats (stats.nba.com)** has richer data (tracking, lineups) but no public
  terms for programmatic use, and it blocks many cloud networks. It was unreachable
  from the build environment, and I did not try to get around that.
- **Basketball-Reference** directly: it publishes a bot-traffic limit (about 20 requests
  per minute) and discourages bulk scraping. It was also unreachable from the
  build environment.
- **The mirror** is a public, versioned GitHub repository that is also the source
  of the widely used Kaggle "NBA Stats (1947-present)" dataset. Pinning a commit makes
  every run identical. Spot checks against known values (complete 82-game 2025-26
  team records, multi-team rows) matched.

### Tables used

| File | Used for |
|---|---|
| Advanced.csv | BPM (outcome), usage, TS%, rebound/assist/steal/block/turnover rates |
| Player Totals.csv | FT and 3P makes/attempts for shrunken percentages |
| Per 100 Poss.csv | 3PA per 100 possessions |
| Player Shooting.csv | shot distance, rim attempts, % assisted, dunks |
| Player Play By Play.csv | on-court and on/off plus-minus per 100 possessions |
| Player Season Info.csv | years of experience |
| Player Career Info.csv | height, weight |
| Draft Pick History.csv | draft slot (later draft kept for players drafted twice) |
| Team Summaries.csv | team SRS (teammate quality) and league-average shooting |

## Limitations of the data

- **BPM is a box-score estimate** of points per 100 possessions above average
  (Daniel Myers' BPM 2.0). It does not measure defense well, and it is not RAPM or
  tracking-based impact.
- **Ages** are Basketball-Reference's convention (age on February 1 of the season).
- **Shortened seasons**: 1998-99 (50 games), 2011-12 (66), 2019-20 (~72 plus bubble)
  and 2020-21 (72). Fewer minutes mean noisier stats; a sensitivity check drops the
  COVID-affected outcome seasons.
- **Traded players** appear once per team plus a combined row (`2TM`, `3TM`, ...).
  The analysis uses the combined row; team context is the minutes-weighted average
  of each stint's team.
- **Upstream changes**: the mirror is maintained by one person. If a future commit
  changes column names, `bbs.data.schema` fails loudly instead of producing silent
  errors.

## What is committed here

**No Basketball-Reference data is committed.** `raw/`, `processed/` and
`sample/*.csv.gz` are git-ignored, and the pipeline regenerates all of them. The
first full run writes `sample/player_seasons_sample.csv.gz`, a 2011-12 to 2025-26
extract used for fast offline re-runs (`python -m bbs.pipeline --sample`) and tests.

The dashboard's player-comparison view downloads `Advanced.csv` from the same pinned commit at run time when no local pipeline output exists (`bbs.data.history`), keeping only season, age, minutes and BPM for the forecast players, in memory. Nothing is written to disk.

The repository does contain the analysis outputs in `reports/`: aggregate tables,
figures, the report, the 2026-27 forecast table, and a few player-season examples
cited in the report. These quote a limited number of Basketball-Reference values
(such as a player's BPM and minutes) as context for original analysis, with attribution.

**Licensing note.** The statistics are compiled by Sports Reference LLC and used
here for non-commercial research with attribution. Sports Reference's
[data use policy](https://www.sports-reference.com/data_use.html) asks that people
not build websites or tools on scraped data without permission, so this repository
redistributes no datasets. Anyone reproducing the analysis downloads the source
tables themselves. See also the [terms of use](https://www.sports-reference.com/termsofuse.html).
