# PANGAEA Readiness — Technical Documentation

This document describes the architecture, data flow, and implementation of the `pangaea_readiness` library and its Streamlit front end.

## Purpose

The system evaluates [PANGAEA](https://www.pangaea.de/) earth-science datasets for suitability in **machine learning** and **data science** workflows. It combines:

1. **Automated retrieval** via `pangaeapy`
2. **Statistical profiling** of tabular data
3. **Rule-based scoring** (0–100) with interpretable criteria
4. **Optional AI assessment** via SAIA (OpenAI-compatible API + Mistral model)

---

## High-level workflow

```mermaid
flowchart LR
    A[DOI or numeric ID] --> B[fetcher.fetch_dataset]
    B --> C[profiler.profile_dataset]
    C --> D[scorer.score_dataset]
    B --> E[ai_evaluator.ai_evaluate]
    C --> E
    D --> F[EvaluationResult]
    E --> F
    F --> G[CLI / notebooks / Streamlit app]
```

### Pipeline entry point

```python
from pangaea_readiness import evaluate

result = evaluate("10.1594/PANGAEA.787140", use_ai=True)
print(result.score)      # 0–100
print(result.flag)       # e.g. "⚠️ Needs Preprocessing"
print(result.summary())  # one-line summary
```

`evaluate()` orchestrates fetch → profile → score → optional AI, and returns a single `EvaluationResult` object with convenience properties for UIs.

---

## Repository layout

| Path | Role |
|------|------|
| `pangaea_readiness/fetcher.py` | Download dataset metadata and table via `pangaeapy` |
| `pangaea_readiness/profiler.py` | Compute statistical / structural profile |
| `pangaea_readiness/scorer.py` | Rule-based 0–100 score and ML task hints |
| `pangaea_readiness/ai_evaluator.py` | SAIA LLM interpretation of metadata + profile |
| `pangaea_readiness/__init__.py` | Public API: `evaluate`, dataclasses, re-exports |
| `app/main.py` | Streamlit UI |
| `notebooks/` | Manual integration scripts |
| `pyproject.toml` | Package metadata and dependencies |

---

## Module reference

### 1. Fetcher (`fetcher.py`)

**Input:** DOI string (`10.1594/PANGAEA.787140`) or numeric ID (`787140`).

**Output:** Normalized `dict`:

| Key | Type | Description |
|-----|------|-------------|
| `id` | `int` | Numeric PANGAEA ID |
| `doi` | `str` | Original input |
| `title`, `abstract`, `citation` | `str` | Metadata from `pangaeapy` |
| `authors` | `list[str]` | Author names |
| `columns` | `list[str]` | DataFrame column names |
| `data` | `pd.DataFrame` | Tabular data (empty if restricted) |
| `license` | `str` | Licence string or `"unknown"` |
| `is_embargoed` | `bool` | `True` if no usable table |
| `error` | `str \| None` | Set on parse/network/API failure |

**Behavior:**

- Parses ID with `_parse_id()`; invalid input sets `error` and returns (no exception to caller).
- Retries failed `PanDataSet` calls (`retry` default 2, `delay` 2s between attempts).
- Treats empty/`None` data as embargoed/restricted.
- `fetch_batch()` loops DOIs with rate limiting and stdout progress.

**Exception:** `PanFetchError` — used internally for parse errors; surfaced as `result["error"]`.

---

### 2. Profiler (`profiler.py`)

**Input:** Fetch result dict.

**Output:** Profile dict consumed by the scorer and AI context builder.

| Metric | Meaning |
|--------|---------|
| `row_count`, `col_count` | Table shape |
| `missing_pct_overall` | Mean null fraction across cells |
| `cols_high_missing` | Columns with >50% missing |
| `numeric_cols`, `datetime_cols`, `text_cols` | Dtype buckets |
| `numeric_ratio` | Share of numeric columns |
| `has_lat_lon` | Heuristic on column names |
| `has_datetime` | Dtype or name heuristic |
| `duplicate_row_pct` | Duplicate row fraction |
| `zero_variance_cols` | Numeric columns with std = 0 |
| `has_title`, `has_abstract`, `has_authors`, `has_license` | Metadata richness |
| `is_embargoed` | Passed through from fetch |

Empty tables use `_empty_profile()` with safe defaults (e.g. `missing_pct_overall = 1.0`).

---

### 3. Scorer (`scorer.py`)

**Input:** Profile dict.

**Output:** `ScoreResult` dataclass:

- `score` (0–100)
- `flag` — verdict label
- `ml_tasks` — rule-suggested tasks
- `breakdown` — per-criterion scores
- `warnings`, `recommendations`

#### Scoring rubric (5 × 20 points)

| Criterion | Max | What it measures |
|-----------|-----|------------------|
| `data_quantity` | 20 | Row/column counts |
| `data_quality` | 20 | Missing values, duplicates, zero-variance cols |
| `structure` | 20 | Numeric ratio, lat/lon, datetime presence |
| `metadata` | 20 | Title, abstract, authors, license |
| `ml_potential` | 20 | Heuristic ML task fit |

#### Flags

| Total score | Flag |
|-------------|------|
| Embargoed | `🔒 Restricted` (score forced to 0) |
| ≥ 75 | `✅ ML Ready` |
| ≥ 50 | `⚠️ Needs Preprocessing` |
| < 50 | `❌ Not Suitable` |

#### Rule-suggested ML tasks

Examples: `time_series_forecasting`, `regression`, `geospatial_analysis`, `anomaly_detection`, `clustering` — assigned by simple thresholds on profile fields.

---

### 4. AI evaluator (`ai_evaluator.py`)

**Input:** Fetch dict + profile dict (must originate from `pangaeapy` fetch).

**Output:** `AIEvaluation` dataclass.

**Provider:**

- Base URL: `https://chat-ai.academiccloud.de/v1` (SAIA)
- Model: `mistral-large-3-675b-instruct-2512`
- Client: `openai.OpenAI` with `SAIA_API_KEY`

**Prompt design:** System + user message with structured context (title, abstract, columns, profile stats). Model must reply in a fixed line-oriented format (`DOMAIN:`, `CONFIDENCE:`, etc.) parsed by `_parse_response()`.

**Guards:**

- Skips API if dataset embargoed or fetch errored
- Skips API if `SAIA_API_KEY` missing
- Handles empty API responses without crashing

---

### 5. Package API (`__init__.py`)

#### `EvaluationResult`

Aggregates `fetch`, `profile`, `rule_score`, optional `ai`, and `notes`.

Properties mirror Streamlit needs: `score`, `flag`, `shape`, `title`, `breakdown`, `warnings`, `recommendations`, `ai_*`, `fetch_error`.

#### `evaluate(doi_or_id, use_ai=False)`

Single-call pipeline. Appends fetch/AI errors to `notes` for debugging without raising.

---

## Streamlit application (`app/main.py`)

1. Loads `.env` via `python-dotenv`
2. Adds project root to `sys.path` (so `pangaea_readiness` resolves when run as `streamlit run app/main.py`)
3. User enters DOI or picks an example dataset
4. Calls `evaluate(doi, use_ai=...)`
5. Renders verdict, metrics, breakdown, warnings, recommendations, AI block

Run:

```bash
uv sync
# set SAIA_API_KEY in .env for AI
uv run streamlit run app/main.py
```

Install with app extras:

```bash
uv sync --extra app
```

---

## Configuration

| Variable | Required | Used by |
|----------|----------|---------|
| `SAIA_API_KEY` | For AI only | `ai_evaluator.py` |

Store in `.env` at project root (gitignored). The Streamlit app and notebook scripts call `load_dotenv()`.

---

## Dependencies

Declared in `pyproject.toml`:

- `pangaeapy`, `pandas`, `numpy` — data access and analysis
- `openai` — SAIA-compatible chat API
- `python-dotenv` — env loading in apps/notebooks
- Optional `streamlit` (`[project.optional-dependencies] app`)
- Optional `pytest`, `jupyter` (`dev`)

Install editable:

```bash
uv pip install -e .
uv pip install -e ".[app,dev]"
```

---

## Error handling philosophy

| Layer | Strategy |
|-------|----------|
| Fetcher | Return dict with `error` / `is_embargoed`; avoid raising to callers |
| Profiler | Safe defaults on empty data |
| Scorer | Always returns `ScoreResult` |
| AI | Return `AIEvaluation` with `error` field set |
| `evaluate()` | Never raises for fetch/AI failures; records `notes` |
| Streamlit | `try/except` around `evaluate()` for unexpected failures |

---

## Known limitations

1. **Heuristic scoring** — Rules are generic; domain-specific nuance relies on AI or human review.
2. **Column name detection** — Lat/lon and datetime detection use keywords, not full geospatial parsing.
3. **Embargo vs collection** — `pangaeapy` may return empty data for collections; user may need a child dataset DOI.
4. **No automated tests yet** — `tests/` is empty; scoring and parsing are good candidates for unit tests.
5. **`PanFetchError`** — Defined and used for parse errors; network failures remain in `result["error"]` strings.
6. **`requirements.txt`** — Convenience list; `pyproject.toml` is the source of truth for packaging.

---

## Extension points

- Add `reporter.py` for JSON/HTML export
- Add unit tests for `_parse_id`, `_parse_response`, `_verdict`, scoring thresholds
- Cache `fetch_dataset` results by DOI in Streamlit (`@st.cache_data`)
- Combine rule score + AI confidence into a single composite score
- Support batch evaluation UI over `fetch_batch()`

---

## Code review summary (library health)

| Area | Status | Notes |
|------|--------|-------|
| Module structure | Good | Clear separation fetch → profile → score → AI |
| Public API | Good | `evaluate()` + `EvaluationResult` properties |
| Type hints | Partial | Dict-heavy contracts; dataclasses for results |
| Error handling | Good after fixes | API key guard, empty response guard, parse ID errors |
| Dependencies | Aligned | `openai` + SAIA; `pyproject` optional `app` extra |
| Tests | Missing | Recommend pytest for scorer and AI parser |
| Dead code | Minor | Root `main.py` is placeholder; `PanFetchError` rarely raised to callers |

Overall the library is **coherent and usable** for a hobby/research MVP; the main gaps are automated tests and stricter typed contracts for fetch/profile dicts.
