from pathlib import Path
import sys

import streamlit as st
from dotenv import load_dotenv

# Ensure project root is importable when Streamlit runs from `app/`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pangaea_readiness import evaluate

load_dotenv()

FLAG_ALERTS = {
    "✅ ML Ready": "success",
    "⚠️ Needs Preprocessing": "warning",
    "❌ Not Suitable": "error",
    "🔒 Restricted": "error",
}


def _show_verdict(flag: str) -> None:
    alert_type = FLAG_ALERTS.get(flag, "info")
    getattr(st, alert_type)(f"**Verdict:** {flag}")


def _show_ai_section(result) -> None:
    st.subheader("AI Evaluation")

    if result.ai_error:
        st.warning(f"AI evaluation unavailable: {result.ai_error}")
        return

    domain = (result.ai_domain or "").strip()
    if domain in ("", "n/a", "unknown") and not result.ai_suitability:
        st.info(
            "AI ran but could not infer a clear domain. "
            "Check your SAIA API key or try another dataset."
        )
        return

    ai_col1, ai_col2 = st.columns(2)
    ai_col1.metric("Scientific domain", domain if domain not in ("n/a",) else "Unknown")
    ai_col2.metric("Confidence", (result.ai_confidence or "low").title())

    if result.ai_suitability:
        st.markdown(f"**Assessment:** {result.ai_suitability}")

    if result.ai_ml_tasks:
        st.markdown("**ML tasks (AI):**")
        for t in result.ai_ml_tasks:
            st.markdown(f"- {t.replace('_', ' ').title()}")

    if result.ai_ds_tasks:
        st.markdown("**DS tasks (AI):**")
        for t in result.ai_ds_tasks:
            st.markdown(f"- {t.replace('_', ' ').title()}")

    col_rf, col_op = st.columns(2)

    with col_rf:
        if result.ai_red_flags:
            st.markdown("**Red flags**")
            for f in result.ai_red_flags:
                st.error(f)

    with col_op:
        if result.ai_opportunities:
            st.markdown("**Opportunities**")
            for o in result.ai_opportunities:
                st.success(o)


# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="PANGAEA ML Readiness Checker",
    page_icon="🌊",
    layout="wide",
)

# ── Header ───────────────────────────────────────────────
st.title("PANGAEA ML Readiness Checker")
st.markdown(
    "Evaluate any PANGAEA dataset for **machine learning** and "
    "**data science** suitability."
)
st.divider()

# ── Input ────────────────────────────────────────────────
input_col, toggle_col = st.columns([3, 1])

with input_col:
    doi_input = st.text_input(
        label="PANGAEA DOI or numeric ID",
        placeholder="e.g. 10.1594/PANGAEA.787140 or 787140",
        help="Find DOIs at [pangaea.de](https://www.pangaea.de/)",
    )

with toggle_col:
    st.write("")  # align toggle with text input
    use_ai = st.toggle(
        "AI evaluation",
        value=True,
        help="Uses SAIA/Mistral for metadata-aware assessment",
    )

evaluate_btn = st.button(
    "Evaluate dataset",
    type="primary",
    use_container_width=True,
    disabled=not doi_input.strip(),
)

# ── Evaluation ───────────────────────────────────────────
if evaluate_btn and doi_input.strip():

    with st.spinner("Evaluating dataset (fetch, profile, score)…"):
        try:
            result = evaluate(doi_input.strip(), use_ai=use_ai)
        except Exception as e:
            st.error(f"Failed to evaluate: {e}")
            st.stop()

    if result.flag == "🔒 Restricted":
        st.error(
            "This dataset is restricted or is a collection. "
            "Try a child dataset DOI instead."
        )
        st.stop()

    st.subheader("Dataset")
    st.markdown(f"**Title:** {result.title}")
    st.markdown(f"**DOI:** `{result.doi}`")

    _show_verdict(result.flag)

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Rows", f"{result.shape[0]:,}")
    info_col2.metric("Columns", result.shape[1])
    info_col3.metric("Score", f"{result.score}/100")

    st.divider()

    st.subheader("Score breakdown")
    for criterion, detail in result.breakdown.items():
        label = criterion.replace("_", " ").title()
        score = detail["score"]
        max_s = detail["max"]
        pct = score / max_s
        verdict = detail["verdict"]
        detail_text = detail["detail"]

        col_a, col_b = st.columns([2, 3])
        with col_a:
            st.markdown(f"**{verdict} {label}** — {score}/{max_s}")
        with col_b:
            st.progress(pct, text=detail_text)

    st.divider()

    if result.ml_tasks:
        st.subheader("Suggested ML tasks")
        task_cols = st.columns(min(len(result.ml_tasks), 4))
        for i, task in enumerate(result.ml_tasks):
            task_cols[i % len(task_cols)].markdown(
                f"- {task.replace('_', ' ').title()}"
            )

    if result.warnings:
        st.subheader("Warnings")
        for w in result.warnings:
            st.warning(w)

    if result.recommendations:
        st.subheader("Recommendations")
        for r in result.recommendations:
            st.info(r)

    if use_ai:
        st.divider()
        _show_ai_section(result)

else:
    st.info("Enter a DOI above, then click **Evaluate dataset**.")

# ── Footer ───────────────────────────────────────────────
st.divider()
st.caption(
    "Data via [pangaeapy](https://github.com/pangaea-data-publisher/pangaeapy) "
    "| AI via SAIA/Mistral | Built for open science"
)
