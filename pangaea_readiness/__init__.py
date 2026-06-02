"""Utilities for fetching and profiling PANGAEA datasets."""

from dataclasses import dataclass, field

from .ai_evaluator import AIEvaluation, ai_evaluate
from .fetcher import PanFetchError, fetch_batch, fetch_dataset
from .profiler import profile_dataset
from .scorer import ScoreResult, score_dataset


@dataclass
class EvaluationResult:
    doi: str
    fetch: dict
    profile: dict
    rule_score: ScoreResult
    ai: AIEvaluation | None = None
    notes: list = field(default_factory=list)

    @property
    def score(self) -> int:
        return self.rule_score.score

    @property
    def flag(self) -> str:
        return self.rule_score.flag

    @property
    def ai_domain(self) -> str:
        if not self.ai:
            return "n/a"
        return self.ai.domain or "unknown"

    @property
    def shape(self):
        data = self.fetch.get("data")
        return data.shape if data is not None else (0, 0)

    @property
    def title(self) -> str:
        return self.fetch.get("title") or ""

    @property
    def breakdown(self) -> dict:
        return self.rule_score.breakdown

    @property
    def ml_tasks(self) -> list:
        return self.rule_score.ml_tasks

    @property
    def warnings(self) -> list:
        return self.rule_score.warnings

    @property
    def recommendations(self) -> list:
        return self.rule_score.recommendations

    @property
    def ai_confidence(self) -> str:
        if not self.ai:
            return "n/a"
        return self.ai.confidence or "low"

    @property
    def ai_suitability(self) -> str:
        return self.ai.suitability if self.ai else ""

    @property
    def ai_ml_tasks(self) -> list:
        return self.ai.ml_tasks if self.ai else []

    @property
    def ai_ds_tasks(self) -> list:
        return self.ai.ds_tasks if self.ai else []

    @property
    def ai_red_flags(self) -> list:
        return self.ai.red_flags if self.ai else []

    @property
    def ai_opportunities(self) -> list:
        return self.ai.opportunities if self.ai else []

    @property
    def ai_error(self) -> str:
        return self.ai.error if self.ai else ""

    @property
    def fetch_error(self) -> str | None:
        return self.fetch.get("error")

    def summary(self) -> str:
        title = self.fetch.get("title") or "unknown title"
        shape = self.shape
        parts = [
            f"DOI: {self.doi}",
            f"Title: {title}",
            f"Shape: {shape}",
            f"Score: {self.score}/100",
            f"Flag: {self.flag}",
        ]
        if self.ai:
            parts.append(f"AI Domain: {self.ai_domain}")
        if self.notes:
            parts.append(f"Notes: {'; '.join(self.notes)}")
        return " | ".join(parts)


def evaluate(doi_or_id: str, use_ai: bool = False) -> EvaluationResult:
    """Run full dataset evaluation pipeline for one DOI or dataset ID."""
    fetch_result = fetch_dataset(doi_or_id)
    profile_result = profile_dataset(fetch_result)
    score_result = score_dataset(profile_result)

    ai_result = None
    notes = []
    if fetch_result.get("error"):
        notes.append(f"Fetch: {fetch_result['error']}")
    if use_ai:
        ai_result = ai_evaluate(fetch_result, profile_result)
        if ai_result.error:
            notes.append(f"AI evaluation error: {ai_result.error}")

    return EvaluationResult(
        doi=str(doi_or_id),
        fetch=fetch_result,
        profile=profile_result,
        rule_score=score_result,
        ai=ai_result,
        notes=notes,
    )


__all__ = [
    "PanFetchError",
    "AIEvaluation",
    "ScoreResult",
    "EvaluationResult",
    "fetch_dataset",
    "fetch_batch",
    "profile_dataset",
    "score_dataset",
    "ai_evaluate",
    "evaluate",
]
