from dataclasses import dataclass, field


@dataclass
class ScoreResult:
    score: int                          # 0–100
    flag: str                           # ✅ ML Ready / ⚠️ Needs Work / ❌ Not Suitable
    ml_tasks: list = field(default_factory=list)
    breakdown: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)


def score_dataset(profile: dict) -> ScoreResult:
    """
    Score a PANGAEA dataset profile for ML/DS readiness.

    Args:
        profile: dict returned by profiler.profile_dataset()

    Returns:
        ScoreResult with score, flag, breakdown, warnings
    """
    breakdown = {}
    warnings = []
    recommendations = []

    # ── 1. Data Quantity (max 20) ─────────────────────────────────────────
    q_score = 0
    row_count = profile.get("row_count", 0)
    col_count = profile.get("col_count", 0)

    if row_count >= 1000:
        q_score += 12
    elif row_count >= 500:
        q_score += 9
    elif row_count >= 100:
        q_score += 6
    elif row_count >= 50:
        q_score += 3
    else:
        warnings.append(f"Very few rows ({row_count}) — limited ML utility")

    if col_count >= 10:
        q_score += 8
    elif col_count >= 5:
        q_score += 5
    elif col_count >= 2:
        q_score += 2
    else:
        warnings.append("Very few columns — limited feature space")

    breakdown["data_quantity"] = {
        "score": q_score, "max": 20,
        "verdict": _verdict(q_score, 20),
        "detail": f"{row_count} rows, {col_count} columns"
    }

    # ── 2. Data Quality (max 20) ──────────────────────────────────────────
    dq_score = 0
    missing = profile.get("missing_pct_overall", 1.0)
    dup_pct = profile.get("duplicate_row_pct", 0.0)
    zero_var = profile.get("zero_variance_cols", 0)

    if missing <= 0.05:
        dq_score += 10
    elif missing <= 0.15:
        dq_score += 7
    elif missing <= 0.30:
        dq_score += 4
    else:
        warnings.append(f"High missing values ({missing:.0%}) — imputation needed")

    if dup_pct == 0:
        dq_score += 5
    elif dup_pct <= 0.05:
        dq_score += 3
    else:
        warnings.append(f"Duplicate rows detected ({dup_pct:.0%})")

    if zero_var == 0:
        dq_score += 5
    elif zero_var <= 2:
        dq_score += 3
        warnings.append(f"{zero_var} zero-variance column(s) — consider dropping")
    else:
        warnings.append(f"{zero_var} zero-variance columns — significant cleaning needed")

    breakdown["data_quality"] = {
        "score": dq_score, "max": 20,
        "verdict": _verdict(dq_score, 20),
        "detail": f"Missing: {missing:.0%}, Duplicates: {dup_pct:.0%}"
    }

    # ── 3. Structure & Format (max 20) ────────────────────────────────────
    st_score = 0
    numeric_ratio = profile.get("numeric_ratio", 0.0)
    has_lat_lon = profile.get("has_lat_lon", False)
    has_datetime = profile.get("has_datetime", False)

    if numeric_ratio >= 0.8:
        st_score += 12
    elif numeric_ratio >= 0.5:
        st_score += 8
    elif numeric_ratio >= 0.3:
        st_score += 4
    else:
        warnings.append("Low numeric ratio — mostly text data, NLP preprocessing needed")
        recommendations.append("Consider text vectorization if NLP task is intended")

    if has_lat_lon:
        st_score += 4
    if has_datetime:
        st_score += 4

    breakdown["structure"] = {
        "score": st_score, "max": 20,
        "verdict": _verdict(st_score, 20),
        "detail": f"Numeric ratio: {numeric_ratio:.0%}, Lat/Lon: {has_lat_lon}, Datetime: {has_datetime}"
    }

    # ── 4. Metadata Quality (max 20) ──────────────────────────────────────
    md_score = 0

    if profile.get("has_title"):
        md_score += 5
    else:
        warnings.append("No title — dataset may be poorly documented")

    if profile.get("has_abstract"):
        md_score += 7
    else:
        recommendations.append("Abstract missing — harder to auto-detect ML task type")

    if profile.get("has_authors"):
        md_score += 4

    if profile.get("has_license"):
        md_score += 4
    else:
        warnings.append("No license info — verify usage rights before use")

    breakdown["metadata"] = {
        "score": md_score, "max": 20,
        "verdict": _verdict(md_score, 20),
        "detail": f"Title: {profile.get('has_title')}, Abstract: {profile.get('has_abstract')}, License: {profile.get('has_license')}"
    }

    # ── 5. ML Potential (max 20) ──────────────────────────────────────────
    ml_score = 0
    ml_tasks = []

    # Time series potential
    if has_datetime and len(profile.get("numeric_cols", [])) >= 2:
        ml_score += 5
        ml_tasks.append("time_series_forecasting")

    # Regression potential
    if len(profile.get("numeric_cols", [])) >= 3:
        ml_score += 5
        ml_tasks.append("regression")

    # Spatial/geospatial potential
    if has_lat_lon:
        ml_score += 4
        ml_tasks.append("geospatial_analysis")

    # Anomaly detection potential
    if row_count >= 500 and numeric_ratio >= 0.5:
        ml_score += 3
        ml_tasks.append("anomaly_detection")

    # Clustering potential
    if len(profile.get("numeric_cols", [])) >= 4:
        ml_score += 3
        ml_tasks.append("clustering")

    breakdown["ml_potential"] = {
        "score": ml_score, "max": 20,
        "verdict": _verdict(ml_score, 20),
        "detail": f"Suggested tasks: {', '.join(ml_tasks) if ml_tasks else 'none detected'}"
    }

    # ── Final Score & Flag ────────────────────────────────────────────────
    total = sum(v["score"] for v in breakdown.values())

    if profile.get("is_embargoed"):
        flag = "🔒 Restricted"
        total = 0
    elif total >= 75:
        flag = "✅ ML Ready"
    elif total >= 50:
        flag = "⚠️ Needs Preprocessing"
    else:
        flag = "❌ Not Suitable"

    # General recommendations
    if total >= 75 and not recommendations:
        recommendations.append("Dataset looks strong — validate domain meaning of columns before modeling")
    if missing > 0.05:
        recommendations.append("Impute missing values before training (try median/KNN imputation)")
    if zero_var > 0:
        recommendations.append("Drop zero-variance columns before modeling")

    return ScoreResult(
        score=total,
        flag=flag,
        ml_tasks=ml_tasks,
        breakdown=breakdown,
        warnings=warnings,
        recommendations=recommendations
    )


# ── helpers ──────────────────────────────────────────────────────────────────

def _verdict(score: int, max_score: int) -> str:
    ratio = score / max_score
    if ratio >= 0.75:
        return "✅"
    elif ratio >= 0.50:
        return "⚠️"
    else:
        return "❌"