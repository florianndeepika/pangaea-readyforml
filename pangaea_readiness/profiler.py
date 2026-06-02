import pandas as pd
import numpy as np


def profile_dataset(fetch_result: dict) -> dict:
    """
    Generate a statistical profile of a fetched PANGAEA dataset.

    Args:
        fetch_result: dict returned by fetcher.fetch_dataset()

    Returns:
        dict with profile metrics used by the scorer
    """
    df = fetch_result.get("data", pd.DataFrame())

    if df.empty:
        return _empty_profile(fetch_result)

    profile = {}

    # ── Shape ────────────────────────────────────────────────────────────────
    profile["row_count"] = len(df)
    profile["col_count"] = len(df.columns)

    # ── Missing values ───────────────────────────────────────────────────────
    missing_per_col = df.isnull().mean()
    profile["missing_pct_overall"] = float(df.isnull().mean().mean())
    profile["cols_high_missing"] = int((missing_per_col > 0.5).sum())
    profile["cols_complete"] = int((missing_per_col == 0).sum())

    # ── Column types ─────────────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()

    profile["numeric_cols"] = numeric_cols
    profile["datetime_cols"] = datetime_cols
    profile["text_cols"] = text_cols
    profile["numeric_ratio"] = (
        len(numeric_cols) / len(df.columns) if len(df.columns) else 0.0
    )

    # ── Spatial keys ─────────────────────────────────────────────────────────
    profile["has_lat_lon"] = _detect_lat_lon(df)

    # ── Temporal keys ────────────────────────────────────────────────────────
    profile["has_datetime"] = len(datetime_cols) > 0 or _detect_datetime_cols(df)

    # ── Duplicates ───────────────────────────────────────────────────────────
    profile["duplicate_row_pct"] = float(df.duplicated().mean())

    # ── Variance (detect dead columns) ───────────────────────────────────────
    if len(numeric_cols) > 0:
        zero_var = (df[numeric_cols].std() == 0).sum()
        profile["zero_variance_cols"] = int(zero_var)
    else:
        profile["zero_variance_cols"] = 0

    # ── Metadata richness ────────────────────────────────────────────────────
    profile["has_title"] = bool(str(fetch_result.get("title") or "").strip())
    profile["has_abstract"] = bool(str(fetch_result.get("abstract") or "").strip())
    profile["has_authors"] = len(fetch_result.get("authors", [])) > 0
    profile["has_license"] = fetch_result.get("license", "unknown") != "unknown"
    profile["is_embargoed"] = fetch_result.get("is_embargoed", False)

    return profile


# ── helpers ──────────────────────────────────────────────────────────────────

def _empty_profile(fetch_result: dict) -> dict:
    """Return a zeroed profile for empty/embargoed datasets."""
    return {
        "row_count": 0, "col_count": 0,
        "missing_pct_overall": 1.0, "cols_high_missing": 0,
        "cols_complete": 0, "numeric_cols": [], "datetime_cols": [],
        "text_cols": [], "numeric_ratio": 0.0, "has_lat_lon": False,
        "has_datetime": False, "duplicate_row_pct": 0.0,
        "zero_variance_cols": 0, "has_title": False,
        "has_abstract": False, "has_authors": False,
        "has_license": False,
        "is_embargoed": fetch_result.get("is_embargoed", False),
    }


def _detect_lat_lon(df: pd.DataFrame) -> bool:
    """Check if dataframe has latitude/longitude columns."""
    lat_keywords = ["lat", "latitude"]
    lon_keywords = ["lon", "lng", "longitude"]
    cols_lower = [c.lower() for c in df.columns]
    has_lat = any(any(k in c for k in lat_keywords) for c in cols_lower)
    has_lon = any(any(k in c for k in lon_keywords) for c in cols_lower)
    return has_lat and has_lon


def _detect_datetime_cols(df: pd.DataFrame) -> bool:
    """Check if any object columns look like dates."""
    date_keywords = ["date", "time", "year", "month", "day"]
    cols_lower = [c.lower() for c in df.columns]
    return any(any(k in c for k in date_keywords) for c in cols_lower)