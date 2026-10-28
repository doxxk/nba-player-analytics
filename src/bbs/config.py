"""Project-wide constants: data source, paths, and study-design parameters.

Every analytical choice that a reviewer might want to change lives here so the
sensitivity analysis and the main analysis read from one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# Basketball-Reference season tables, mirrored as CSVs by Sumitro Datta.
# Pinned to a commit so every run sees identical inputs.
SOURCE_REPO = "sumitrodatta/bball-reference-datasets"
SOURCE_COMMIT = "76a70b41ad1c13948f25c62c921ed822e5db7f0e"
SOURCE_URL = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_COMMIT}/Data/{{name}}"

# Local file stem -> upstream file name.
SOURCE_FILES = {
    "advanced": "Advanced.csv",
    "totals": "Player Totals.csv",
    "per100": "Per 100 Poss.csv",
    "shooting": "Player Shooting.csv",
    "pbp": "Player Play By Play.csv",
    "season_info": "Player Season Info.csv",
    "career_info": "Player Career Info.csv",
    "draft": "Draft Pick History.csv",
    "team_summaries": "Team Summaries.csv",
}

# Shooting-location and play-by-play tables begin in 1996-97.
FIRST_SEASON = 1997
# Seasons are labelled by the year they end (2026 = 2025-26).
LAST_SEASON = 2026

UNDRAFTED_PICK = 61  # one past the last pick of a two-round draft


@dataclass(frozen=True)
class StudyDesign:
    """Population, outcome, and validation settings for one analysis run."""

    max_age: int = 24
    min_mp: int = 500  # minutes in the base season t
    min_mp_next: int = 250  # minutes in t+1 for the outcome to count as observed
    first_base_season: int = 2000
    # Rolling-origin test: for each base season Y in this range, train on
    # base seasons < Y (whose outcomes are all known by the end of season Y).
    first_test_season: int = 2016
    last_test_season: int = 2025
    # Hyperparameters are tuned only on base seasons <= this value.
    tuning_cutoff: int = 2015
    exclude_outcome_seasons: tuple[int, ...] = field(default_factory=tuple)
    use_ipw: bool = False

    def with_(self, **kwargs) -> "StudyDesign":
        return replace(self, **kwargs)


DEFAULT_DESIGN = StudyDesign()

# A "meaningful improvement" for hit-rate metrics, in BPM points.
IMPROVEMENT_THRESHOLD = 1.5

RANDOM_SEED = 2027
