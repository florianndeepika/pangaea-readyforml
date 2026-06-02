# PANGAEA ML Readiness Checker

`pangaea-readyforml` is a beginner-friendly tool that evaluates [PANGAEA](https://www.pangaea.de/) earth science datasets for machine-learning readiness.

It combines rule-based checks and optional AI analysis to produce:
- a readiness score
- a readiness flag
- a short explanation of strengths and risks

## What this project includes

- **Python package**: evaluate a dataset DOI in code
- **Streamlit app**: simple UI for interactive checks
- **FastAPI service**: API endpoints and auto-generated docs

## Quick start

### 1) Install dependencies

```bash
uv sync --extra app --extra api
```

### 2) (Optional) Enable AI evaluation

Create a `.env` file in the project root and add:

```bash
SAIA_API_KEY=your_api_key_here
```

### 3) Run the app or API

#### Streamlit UI

```bash
uv run streamlit run app/streamlit_main.py
```

#### FastAPI server

```bash
uv run uvicorn api.main:app --reload
```

API docs will be available at:

`http://127.0.0.1:8000/docs`

## Python usage example

```python
from pangaea_readiness import evaluate

result = evaluate("10.1594/PANGAEA.787140", use_ai=True)
print(result.score, result.flag)
print(result.summary())
```

## Project documentation

For architecture, workflow, scoring rubric, and API details, see:

`docs/TECHNICAL.md`
