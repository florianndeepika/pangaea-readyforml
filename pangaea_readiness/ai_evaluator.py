import os
from openai import OpenAI
from dataclasses import dataclass, field


@dataclass
class AIEvaluation:
    domain: str
    ml_tasks: list = field(default_factory=list)
    ds_tasks: list = field(default_factory=list)
    suitability: str = ""
    confidence: str = ""
    red_flags: list = field(default_factory=list)
    opportunities: list = field(default_factory=list)
    raw_response: str = ""
    error: str = ""


# SOTA model available on SAIA
SAIA_MODEL = "mistral-large-3-675b-instruct-2512"
SAIA_BASE_URL = "https://chat-ai.academiccloud.de/v1"


def ai_evaluate(fetch_result: dict, profile: dict) -> AIEvaluation:
    """
    Use SAIA SOTA model to intelligently evaluate a PANGAEA dataset.

    NOTE: fetch_result must come from fetcher.fetch_dataset()
    which uses pangaeapy under the hood — no other retrieval method used.

    Args:
        fetch_result: dict from fetcher.fetch_dataset() via pangaeapy
        profile:      dict from profiler.profile_dataset()

    Returns:
        AIEvaluation dataclass
    """

    # Guard — pangaeapy returned nothing useful
    if fetch_result.get("is_embargoed") or fetch_result.get("error"):
        return AIEvaluation(
            domain="unknown",
            suitability="Dataset is restricted or empty — cannot evaluate.",
            confidence="low",
            error=fetch_result.get("error", "Unknown error")
        )

    api_key = os.environ.get("SAIA_API_KEY", "").strip()
    if not api_key:
        return AIEvaluation(
            domain="unknown",
            suitability="AI evaluation skipped — SAIA_API_KEY is not set.",
            confidence="low",
            error="SAIA_API_KEY is not set.",
        )

    context = _build_context_from_pangaeapy(fetch_result, profile)

    try:
        client = OpenAI(api_key=api_key, base_url=SAIA_BASE_URL)

        response = client.chat.completions.create(
            model=SAIA_MODEL,
            max_tokens=1000,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert data scientist specialising in "
                        "earth and environmental science datasets. "
                        "You evaluate datasets for ML and DS suitability "
                        "based on their metadata and statistical properties."
                    )
                },
                {
                    "role": "user",
                    "content": f"""Evaluate this PANGAEA dataset for ML and DS suitability.
Respond ONLY in this exact format with no extra text:

DOMAIN: <scientific domain in 3-5 words>
CONFIDENCE: <high|medium|low>
SUITABILITY: <2-3 sentences explaining if this dataset is suitable for ML/DS and why>
ML_TASKS: <comma-separated list of suitable ML tasks, or 'none'>
DS_TASKS: <comma-separated list of suitable DS/analysis tasks, or 'none'>
RED_FLAGS: <comma-separated list of concerns, or 'none'>
OPPORTUNITIES: <comma-separated list of interesting use cases, or 'none'>

Dataset context (retrieved via pangaeapy):
{context}"""
                }
            ]
        )

        raw = response.choices[0].message.content
        if not raw or not raw.strip():
            return AIEvaluation(
                domain="unknown",
                suitability="AI evaluation returned an empty response.",
                confidence="low",
                error="Empty model response",
            )
        return _parse_response(raw)

    except Exception as e:
        return AIEvaluation(
            domain="unknown",
            suitability="AI evaluation failed.",
            confidence="low",
            error=str(e)
        )


def _build_context_from_pangaeapy(fetch_result: dict, profile: dict) -> str:
    """
    Build context string purely from pangaeapy retrieved data.

    pangaeapy provides:
    - ds.title        → fetch_result['title']
    - ds.abstract     → fetch_result['abstract']
    - ds.authors      → fetch_result['authors']
    - ds.citation     → fetch_result['citation']
    - ds.licence      → fetch_result['license']
    - ds.data         → profiled into profile dict
    - ds.data.columns → fetch_result['columns']
    """
    cols = fetch_result.get("columns", [])
    col_preview = cols[:15]
    if len(cols) > 15:
        col_preview.append(f"... and {len(cols) - 15} more")

    authors = fetch_result.get("authors", [])
    author_str = ", ".join(authors[:3]) if authors else "N/A"
    if len(authors) > 3:
        author_str += f" et al. ({len(authors)} total)"

    abstract = fetch_result.get("abstract", "") or ""
    abstract_preview = abstract[:600] if abstract else "N/A"

    return f"""
Title     : {fetch_result.get('title', 'N/A')}
Authors   : {author_str}
License   : {fetch_result.get('license', 'unknown')}
Citation  : {fetch_result.get('citation', 'N/A')[:200] if fetch_result.get('citation') else 'N/A'}

Abstract  : {abstract_preview}

Columns ({len(cols)} total):
  {', '.join(col_preview)}

Statistical profile (from pangaeapy data):
  Rows              : {profile.get('row_count', 0)}
  Columns           : {profile.get('col_count', 0)}
  Numeric ratio     : {profile.get('numeric_ratio', 0):.0%}
  Missing values    : {profile.get('missing_pct_overall', 0):.1%}
  High-missing cols : {profile.get('cols_high_missing', 0)}
  Complete cols     : {profile.get('cols_complete', 0)}
  Has lat/lon       : {profile.get('has_lat_lon', False)}
  Has datetime      : {profile.get('has_datetime', False)}
  Duplicate rows    : {profile.get('duplicate_row_pct', 0):.1%}
  Zero variance cols: {profile.get('zero_variance_cols', 0)}
""".strip()


def _parse_response(raw: str) -> AIEvaluation:
    """Parse SAIA model structured response into AIEvaluation."""
    if not raw or not raw.strip():
        return AIEvaluation(
            domain="unknown",
            suitability="AI evaluation returned an empty response.",
            confidence="low",
            error="Empty model response",
        )

    lines = raw.strip().splitlines()
    parsed = {}

    for line in lines:
        if ":" in line:
            key, _, value = line.partition(":")
            parsed[key.strip()] = value.strip()

    def parse_list(val: str) -> list:
        if not val or val.lower() == "none":
            return []
        return [item.strip() for item in val.split(",") if item.strip()]

    return AIEvaluation(
        domain=parsed.get("DOMAIN", "unknown"),
        confidence=parsed.get("CONFIDENCE", "low"),
        suitability=parsed.get("SUITABILITY", ""),
        ml_tasks=parse_list(parsed.get("ML_TASKS", "")),
        ds_tasks=parse_list(parsed.get("DS_TASKS", "")),
        red_flags=parse_list(parsed.get("RED_FLAGS", "")),
        opportunities=parse_list(parsed.get("OPPORTUNITIES", "")),
        raw_response=raw
    )