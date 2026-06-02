# PANGAEA ML Readiness Checker

Evaluate [PANGAEA](https://www.pangaea.de/) datasets for machine learning and data science suitability.

## Quick start

```bash
uv sync --extra app --extra api
# Add SAIA_API_KEY to .env for AI evaluation

# Streamlit UI
uv run streamlit run app/streamlit_main.py

# REST API (docs at http://127.0.0.1:8000/docs)
uv run uvicorn api.main:app --reload
```

```python
from pangaea_readiness import evaluate

result = evaluate("10.1594/PANGAEA.787140", use_ai=True)
print(result.score, result.flag)
print(result.summary())
```

## Documentation

See **[docs/TECHNICAL.md](docs/TECHNICAL.md)** for architecture, pipeline workflow, scoring rubric, and API reference.
