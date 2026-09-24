"""Download and load the Amazon Books reviews."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

from . import config as C

log = logging.getLogger(__name__)

KEEP_COLUMNS = ["rating", "title", "text", "asin", "parent_asin", "user_id",
                "timestamp", "helpful_vote", "verified_purchase"]


def _count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def download_reviews(n: int = C.N_REVIEWS, url: str = C.HF_BOOKS_URL,
                     dest: Path = C.RAW_PATH, force: bool = False) -> Path:
    """Stream the Books reviews file and keep the first `n` lines.

    The full file is tens of gigabytes, so it is read line by line and the connection
    is closed as soon as `n` reviews have been written. Set HF_TOKEN if you are rate limited.
    """
    if dest.exists() and not force and _count_lines(dest) >= n:
        log.info("Using cached reviews at %s", dest)
        return dest

    import requests

    headers = {}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    written = 0
    log.info("Streaming the first %s reviews from %s", f"{n:,}", url)
    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with tmp.open("w", encoding="utf-8") as out:
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                out.write(raw_line.decode("utf-8") + "\n")
                written += 1
                if written >= n:
                    break
    tmp.replace(dest)
    log.info("Saved %s reviews to %s", f"{written:,}", dest)
    return dest


def read_reviews(path: Path, n: int | None = None) -> pd.DataFrame:
    """Read a JSONL (Amazon Reviews 2023 format) or CSV file into a DataFrame."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, nrows=n)
    else:
        rows = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
                if n and len(rows) >= n:
                    break
        df = pd.DataFrame(rows)

    missing = {"rating", "text"} - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")
    df = df[[c for c in KEEP_COLUMNS if c in df.columns]].copy()

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["text"] = df["text"].astype("string")
    if "timestamp" in df.columns:
        # Amazon Reviews 2023 stores Unix time in milliseconds
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        df["review_date"] = pd.to_datetime(ts, unit="ms", errors="coerce")
    if "helpful_vote" in df.columns:
        df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0).astype(int)
    return df.reset_index(drop=True)


def anonymise(df: pd.DataFrame) -> pd.DataFrame:
    """Drop reviewer identifiers once profiling is done; nothing downstream needs them."""
    return df.drop(columns=[c for c in ("user_id",) if c in df.columns])
