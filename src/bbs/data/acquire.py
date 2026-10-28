"""Download and cache the raw Basketball-Reference season tables.

Files are fetched once from a pinned commit of a public GitHub mirror and cached
in ``data/raw/``. A manifest records the URL, size, SHA-256, and retrieval time
of each file so a run can be audited later.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from bbs import config


class DataUnavailableError(RuntimeError):
    """Raised when raw data is neither cached nor downloadable."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, retries: int = 4, timeout: int = 60) -> None:
    """Download ``url`` to ``dest`` with exponential backoff.

    Writes to a temporary file first so an interrupted download never leaves a
    truncated CSV in the cache.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(url, timeout=timeout, stream=True) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            return
        except requests.RequestException as e:
            last_error = e
            time.sleep(2 ** (attempt + 1))
    tmp.unlink(missing_ok=True)
    raise DataUnavailableError(
        f"Could not download {url} after {retries} attempts ({last_error}). "
        "Run with --sample to use the bundled sample data instead."
    )


def fetch_raw(raw_dir: Path = config.RAW_DIR, force: bool = False) -> dict[str, Path]:
    """Ensure every source table is cached locally; return stem -> path."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    paths = {}
    for stem, name in config.SOURCE_FILES.items():
        path = raw_dir / f"{stem}.csv"
        url = config.SOURCE_URL.format(name=requests.utils.quote(name))
        if force or not path.exists():
            download_file(url, path)
            manifest[stem] = {
                "source_file": name,
                "url": url,
                "commit": config.SOURCE_COMMIT,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        paths[stem] = path
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return paths


def load_raw(raw_dir: Path = config.RAW_DIR, download: bool = True) -> dict[str, pd.DataFrame]:
    """Load all raw tables, downloading missing ones if ``download`` is True."""
    if download:
        paths = fetch_raw(raw_dir)
    else:
        paths = {s: raw_dir / f"{s}.csv" for s in config.SOURCE_FILES}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise DataUnavailableError(
                f"Raw files missing and downloads disabled: {missing}. "
                "Run `python -m bbs.pipeline` with network access, or use --sample."
            )
    return {stem: pd.read_csv(path, low_memory=False) for stem, path in paths.items()}
