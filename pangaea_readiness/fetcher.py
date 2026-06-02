import time
import pangaeapy.pandataset as pan
import pandas as pd


class PanFetchError(Exception):
    """Raised when a PANGAEA dataset cannot be fetched."""
    pass


def fetch_dataset(doi_or_id: str, retry: int = 2, delay: float = 2.0) -> dict:
    """
    Fetch a PANGAEA dataset by DOI or numeric ID.

    Args:
        doi_or_id: Full DOI string e.g. '10.1594/PANGAEA.787140'
                   or just the numeric ID e.g. '787140'
        retry:     Number of retry attempts on failure
        delay:     Seconds to wait between requests (be a good citizen)

    Returns:
        dict with keys:
            - id: numeric PANGAEA ID
            - title: dataset title
            - abstract: dataset abstract
            - authors: list of author names
            - columns: list of column names
            - data: pandas DataFrame
            - citation: citation string
            - license: license info
            - is_embargoed: bool
            - error: error message if failed (None if success)
    """
    try:
        dataset_id = _parse_id(doi_or_id)
    except PanFetchError as exc:
        return {
            "id": None,
            "doi": doi_or_id,
            "title": None,
            "abstract": None,
            "authors": [],
            "columns": [],
            "data": pd.DataFrame(),
            "citation": None,
            "license": None,
            "is_embargoed": False,
            "error": str(exc),
        }

    result = {
        "id": dataset_id,
        "doi": doi_or_id,
        "title": None,
        "abstract": None,
        "authors": [],
        "columns": [],
        "data": pd.DataFrame(),
        "citation": None,
        "license": None,
        "is_embargoed": False,
        "error": None,
    }

    for attempt in range(retry + 1):
        try:
            ds = pan.PanDataSet(dataset_id)

            # Check for embargo
            if ds.data is None or ds.data.empty:
                result["is_embargoed"] = True
                result["error"] = "Dataset is embargoed or empty"
                return result

            result["title"] = ds.title or ""
            result["abstract"] = ds.abstract or ""
            result["authors"] = _parse_authors(ds)
            result["columns"] = list(ds.data.columns)
            result["data"] = ds.data
            result["citation"] = ds.citation or ""
            result["license"] = _parse_license(ds)
            result["error"] = None
            return result

        except Exception as e:
            if attempt < retry:
                time.sleep(delay)
            else:
                result["error"] = str(e)
                return result

    return result


def fetch_batch(doi_list: list, delay: float = 2.0) -> list:
    """
    Fetch multiple PANGAEA datasets with rate limiting.

    Args:
        doi_list: List of DOIs or numeric IDs
        delay:    Seconds between requests

    Returns:
        List of result dicts from fetch_dataset()
    """
    results = []
    for i, doi in enumerate(doi_list):
        print(f"Fetching {i+1}/{len(doi_list)}: {doi}")
        result = fetch_dataset(doi)
        results.append(result)
        if i < len(doi_list) - 1:
            time.sleep(delay)
    return results


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_id(doi_or_id: str) -> int:
    """Extract numeric ID from DOI string or return as int."""
    doi_or_id = str(doi_or_id).strip()
    if not doi_or_id:
        raise PanFetchError("DOI or dataset ID is empty")
    try:
        if "PANGAEA." in doi_or_id.upper():
            return int(doi_or_id.split(".")[-1])
        return int(doi_or_id)
    except ValueError as exc:
        raise PanFetchError(f"Could not parse PANGAEA ID from: {doi_or_id}") from exc


def _parse_authors(ds) -> list:
    """Safely extract author names."""
    try:
        return [a.fullname for a in ds.authors] if ds.authors else []
    except Exception:
        return []


def _parse_license(ds) -> str:
    """Safely extract license info."""
    try:
        return ds.licence or "unknown"
    except Exception:
        return "unknown"