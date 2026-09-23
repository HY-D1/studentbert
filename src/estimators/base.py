"""Common result type for transferability estimators (MRAP Section P).

Every estimator returns an EstimatorResult. The score direction is part of the record rather than
a convention, because a ranking helper that silently assumed "higher is better" would invert any
estimator whose natural score is a loss.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

# Target-information access regimes (MRAP Section I). An estimator declares exactly one.
ACCESS_REGIMES = {
    "U": "unlabelled target: inputs only, no outcome labels as supervision",
    "L": "limited-labelled target: a fixed, predefined number of labelled examples",
    "F": "labelled frozen features: target labels, no candidate fine-tuning",
}
DIRECTIONS = ("higher_is_better", "lower_is_better")


@dataclass
class EstimatorResult:
    estimator: str
    access_regime: str
    candidate: str
    target: str
    seed: int
    score: float
    score_direction: str
    n_target_examples: int
    n_target_labels: int
    feature_extraction_time: float
    scoring_time: float
    peak_memory_mb: float | None
    uncertainty_signals: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score_direction not in DIRECTIONS:
            raise ValueError(f"score_direction must be one of {DIRECTIONS}, "
                             f"got {self.score_direction!r}")
        if self.access_regime not in ACCESS_REGIMES:
            raise ValueError(f"access_regime must be one of {sorted(ACCESS_REGIMES)}, "
                             f"got {self.access_regime!r}")

    def oriented(self) -> float:
        """Score on a higher-is-better scale."""
        return self.score if self.score_direction == "higher_is_better" else -self.score

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class TransferabilityEstimator(Protocol):
    name: str
    access_regime: str

    def score(self, candidate: str | None, target_dataset: str, *,
              seed: int) -> EstimatorResult: ...


def rank_candidates(results: list[EstimatorResult]) -> list[EstimatorResult]:
    """Best first. Refuses to rank results that come from different estimators, targets, seeds
    or regimes, because such a ranking compares information budgets rather than candidates."""
    if not results:
        return []
    keys = {(r.estimator, r.target, r.seed, r.access_regime) for r in results}
    if len(keys) != 1:
        raise ValueError(f"cannot rank across estimator/target/seed/regime: {sorted(keys)}")
    return sorted(results, key=lambda r: r.oriented(), reverse=True)
