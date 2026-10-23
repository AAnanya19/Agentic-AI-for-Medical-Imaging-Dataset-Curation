"""Fusion of classical IQA metrics with embedding anomaly scores.

Stage 4 exists to test an assumption the fusion literature makes implicitly:
that low-level statistical IQA metrics and deep-feature anomaly detection
capture *complementary* information, so combining them beats either alone.
Fusion-IQA (IPICE '24) and Leonardi et al. (Sensors 2021) both build combined
systems without ever reporting a per-distortion breakdown of which paradigm
contributes what.

On this project's data that assumption does not hold. Paired bootstrap over
1,681 matched ChestX-ray14 val+test rows found classical IQA significantly
better on 7 of 9 defects, no significant difference on 2 (jpeg_compression,
occlusion_patch), and embeddings significantly better on **none**. So the
expected outcome here is that fusion does not significantly beat IQA alone -
and demonstrating that carefully is the contribution, not a disappointment.

Three strategies are implemented, deliberately spanning a complexity range so
"fusion didn't help" cannot be dismissed as a consequence of choosing a weak
combiner:

  * **z-score weighted sum** - standardise each score against clean-validation
    statistics, then average. No learned parameters, fully interpretable.
  * **rank fusion** - convert each score to a percentile rank within the
    validation distribution and average. Immune to scale and to monotone
    transformations, so it cannot be defeated by one score having a skewed
    distribution.
  * **logistic regression** - learns per-score weights. If complementary
    information exists, a learned linear combiner is well placed to find it,
    and its coefficients are themselves reportable output.

All three are fitted on the **validation** split and reported on **test**.
Fitting on test would make any observed fusion benefit unfalsifiable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError as error:  # pragma: no cover - environment guard
    raise ImportError("scikit-learn is required for pipeline.fusion") from error


class FusionStrategy:
    """Common interface. Fit on validation, score anything, higher = worse.

    ``fit`` takes the *oriented* score matrix (every column already arranged so
    that higher = more degraded) plus binary labels. Orientation is the
    caller's responsibility - see pipeline.significance.orient_scores - because
    this module cannot know which direction a raw IQA metric means.
    """

    name: str = "base"

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "FusionStrategy":
        raise NotImplementedError

    def score(self, scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"strategy": self.name}


class ZScoreWeightedSum(FusionStrategy):
    """Standardise each score, then take an unweighted mean.

    Standardisation statistics come from the *clean* validation rows only, not
    all validation rows. Using all rows would let the degraded images inflate
    the standard deviation of whichever score detects them best, shrinking that
    score's contribution - i.e. the most informative signal would be
    systematically down-weighted.
    """

    name = "zscore_mean"

    def __init__(self) -> None:
        self._scaler: StandardScaler | None = None
        self._n_fitted = 0

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "ZScoreWeightedSum":
        clean_rows = scores[labels == 0]
        if len(clean_rows) < 2:
            raise ValueError(
                "zscore_mean needs at least 2 clean validation rows to "
                f"standardise against, got {len(clean_rows)}"
            )
        self._scaler = StandardScaler().fit(clean_rows)
        self._n_fitted = len(clean_rows)
        return self

    def score(self, scores: np.ndarray) -> np.ndarray:
        if self._scaler is None:
            raise RuntimeError("zscore_mean must be fitted before scoring")
        return self._scaler.transform(scores).mean(axis=1)

    def describe(self) -> dict[str, Any]:
        return {
            "strategy": self.name,
            "standardised_against": "clean validation rows only",
            "n_clean_validation_rows": self._n_fitted,
        }


class RankFusion(FusionStrategy):
    """Average each score's percentile rank within the clean-validation set.

    Rank transformation is invariant to any monotone rescaling of an individual
    score, so this strategy cannot be handicapped by one input having a heavily
    skewed distribution - which matters here because the score families are
    very differently distributed (Laplacian variance spans orders of magnitude;
    CLIP-IQA is bounded in roughly [0, 1]).
    """

    name = "rank_mean"

    def __init__(self) -> None:
        self._reference: np.ndarray | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "RankFusion":
        clean_rows = scores[labels == 0]
        if len(clean_rows) < 2:
            raise ValueError(
                f"rank_mean needs at least 2 clean validation rows, got {len(clean_rows)}"
            )
        # Sorted clean values per column form the reference distribution that
        # test scores are ranked against.
        self._reference = np.sort(clean_rows, axis=0)
        return self

    def score(self, scores: np.ndarray) -> np.ndarray:
        if self._reference is None:
            raise RuntimeError("rank_mean must be fitted before scoring")
        n_reference = self._reference.shape[0]
        percentiles = np.empty_like(scores, dtype=np.float64)
        for column in range(scores.shape[1]):
            positions = np.searchsorted(self._reference[:, column], scores[:, column])
            percentiles[:, column] = positions / n_reference
        return percentiles.mean(axis=1)

    def describe(self) -> dict[str, Any]:
        return {
            "strategy": self.name,
            "reference": "clean validation rows only",
            "n_reference_rows": (
                int(self._reference.shape[0]) if self._reference is not None else 0
            ),
        }


class LogisticFusion(FusionStrategy):
    """Learned linear combination via logistic regression.

    The strongest of the three: if complementary information exists between the
    two paradigms, a learned combiner should exploit it. That makes it the most
    informative strategy for a *negative* result - if even a fitted linear model
    given both score families cannot beat IQA alone, the complementarity is not
    there to be found.

    ``class_weight="balanced"`` because clean and degraded rows are not
    balanced (roughly 30/70 here) and the objective should not be dominated by
    whichever class is larger.
    """

    name = "logistic"

    def __init__(self, random_state: int = 42, max_iter: int = 2000) -> None:
        self.random_state = random_state
        self.max_iter = max_iter
        self._scaler: StandardScaler | None = None
        self._model: LogisticRegression | None = None
        self._feature_names: list[str] | None = None

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "LogisticFusion":
        if len(np.unique(labels)) < 2:
            raise ValueError("logistic fusion needs both clean and degraded rows to fit")
        # Standardise on all fitting rows here (not clean-only): logistic
        # regression is fitting a decision boundary between the two classes, so
        # it needs the scale of the full fitting distribution.
        self._scaler = StandardScaler().fit(scores)
        self._model = LogisticRegression(
            random_state=self.random_state,
            max_iter=self.max_iter,
            class_weight="balanced",
        ).fit(self._scaler.transform(scores), labels)
        self._feature_names = feature_names
        return self

    def score(self, scores: np.ndarray) -> np.ndarray:
        if self._model is None or self._scaler is None:
            raise RuntimeError("logistic fusion must be fitted before scoring")
        # Probability of the degraded class, so higher = worse, consistent with
        # every other scorer in the project.
        return self._model.predict_proba(self._scaler.transform(scores))[:, 1]

    def describe(self) -> dict[str, Any]:
        if self._model is None:
            return {"strategy": self.name, "fitted": False}
        coefficients = self._model.coef_[0]
        weights: dict[str, float] | list[float]
        if self._feature_names and len(self._feature_names) == len(coefficients):
            # Reported because the learned weights are themselves a result:
            # they show which score families the combiner actually relies on.
            weights = {
                name: float(value)
                for name, value in zip(self._feature_names, coefficients)
            }
        else:
            weights = [float(value) for value in coefficients]
        return {
            "strategy": self.name,
            "fitted": True,
            "coefficients": weights,
            "intercept": float(self._model.intercept_[0]),
            "class_weight": "balanced",
            "n_iterations": int(self._model.n_iter_[0]),
            "converged": bool(self._model.n_iter_[0] < self.max_iter),
        }


FUSION_STRATEGIES: dict[str, type[FusionStrategy]] = {
    "zscore_mean": ZScoreWeightedSum,
    "rank_mean": RankFusion,
    "logistic": LogisticFusion,
}


def available_strategies() -> tuple[str, ...]:
    return tuple(FUSION_STRATEGIES)


@dataclass
class FusionSuite:
    """Every fusion strategy, all fitted on the same validation rows."""

    random_state: int = 42
    strategies: dict[str, FusionStrategy] = field(default_factory=dict, init=False)

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "FusionSuite":
        self.strategies = {}
        for name, factory in FUSION_STRATEGIES.items():
            strategy = (
                factory(random_state=self.random_state)
                if name == "logistic"
                else factory()
            )
            if name == "logistic":
                strategy.fit(scores, labels, feature_names=feature_names)
            else:
                strategy.fit(scores, labels)
            self.strategies[name] = strategy
        return self

    def score_all(self, scores: np.ndarray) -> dict[str, np.ndarray]:
        if not self.strategies:
            raise RuntimeError("FusionSuite must be fitted before scoring")
        return {
            name: strategy.score(scores) for name, strategy in self.strategies.items()
        }

    def describe(self) -> dict[str, Any]:
        return {name: strategy.describe() for name, strategy in self.strategies.items()}