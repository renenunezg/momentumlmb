from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

CACHE_DIR = REPO_ROOT / "cache"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
STATIC_DIR = REPO_ROOT / "backend" / "data_static"

# The current independent-league regime starts in 2021 (2020 was cancelled).
FIRST_SEASON = 2021
# Every design choice is made inside the development seasons; the holdout is
# read once (see AGENTS.md) and never tuned on.
DEVELOPMENT_SEASONS = (2021, 2022, 2023, 2024, 2025)
HOLDOUT_SEASON = 2026
