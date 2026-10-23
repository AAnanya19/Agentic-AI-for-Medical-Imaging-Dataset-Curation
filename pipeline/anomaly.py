"""Anomaly detectors over frozen embeddings, fitted on clean images only.

All three detectors follow the same premise: model what *normal* looks like from
clean training images, then score how far a new image departs from it. Nothing
is ever fitted on degraded data, so the detectors never see the defect types
they are later asked to detect. Every detector returns a score where **higher =
more anomalous**, so downstream evaluation needs no per-detector sign handling.

Features are standardised (z-scored) using clean-training statistics before any
detector runs, so all three operate in the same space and no single
high-variance embedding dimension dominates a distance.

  * **kNN distance** - mean distance to the k nearest clean neighbours. The
    non-parametric baseline; this is the mechanism underlying PatchCore-style
    detectors.
  * **Mahalanobis distance** - distance under the clean features' covariance,
    which accounts for correlations between dimensions that Euclidean distance
    ignores. **Uses Ledoit-Wolf shrinkage, which is not optional here.** The
    failure mode is worse than an error: with ISIC's 383 clean training images
    and 512-d ResNet-18 features (n/d = 0.75), the empirical covariance has a
    measured condition number of ~8.6e18. NumPy will still "successfully"
    invert it and return finite numbers, so the distances look plausible while
    being dominated by estimation noise amplified across ~19 orders of
    magnitude. ChestX-ray14 (1178 clean train, n/d = 2.3) is far better
    conditioned at ~2e1. Shrinkage pulls the estimate toward a scaled identity
    by an analytically chosen amount, and the chosen coefficient is reported so
    the degree of regularisation is visible rather than hidden.
  * **Isolation Forest** - isolates points via random axis-aligned splits;
    non-parametric and makes no Gaussianity assumption, unlike Mahalanobis.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from sklearn.covariance import LedoitWolf
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from scipy.linalg import solve_triangular as scipy_solve_triangular
except ImportError as error:  # pragma: no cover - environment guard
    raise ImportError(
        "scikit-learn and scipy are required for pipeline.anomaly"
    ) from error


DEFAULT_KNN_NEIGHBOURS = (1, 5, 10, 20)


class AnomalyDetector:
    """Common interface: fit on clean features, score anything, higher = worse."""

    name: str = "base"

    def fit(self, clean_features: np.ndarray) -> "AnomalyDetector":
        raise NotImplementedError

    def score(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"detector": self.name}


class KNNDistance(AnomalyDetector):
    """Mean Euclidean distance to the k nearest clean training embeddings."""

    def __init__(self, n_neighbours: int = 5) -> None:
        self.name = f"knn_k{n_neighbours}"
        self.n_neighbours = n_neighbours
        self._index: NearestNeighbors | None = None
        self._n_fitted = 0

    def fit(self, clean_features: np.ndarray) -> "KNNDistance":
        if len(clean_features) < self.n_neighbours:
            raise ValueError(
                f"{self.name} needs at least {self.n_neighbours} clean training "
                f"samples, got {len(clean_features)}"
            )
        self._index = NearestNeighbors(n_neighbors=self.n_neighbours).fit(clean_features)
        self._n_fitted = len(clean_features)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self._index is None:
            raise RuntimeError(f"{self.name} must be fitted before scoring")
        distances, _ = self._index.kneighbors(features, n_neighbors=self.n_neighbours)
        return distances.mean(axis=1)

    def describe(self) -> dict[str, Any]:
        return {
            "detector": self.name,
            "n_neighbours": self.n_neighbours,
            "n_clean_fitted": self._n_fitted,
        }


class MahalanobisDistance(AnomalyDetector):
    """Mahalanobis distance to the clean mean, with Ledoit-Wolf shrunk covariance.

    See the module docstring for the measured condition numbers that make
    shrinkage mandatory rather than merely advisable. ``describe()`` reports the
    chosen shrinkage coefficient and the samples-per-dimension ratio, and fit()
    warns when that ratio is low enough that an unshrunk estimate would have
    been untrustworthy - so a reader can tell how much regularisation was
    actually applied on their data.
    """

    name = "mahalanobis"

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None
        self._cholesky: np.ndarray | None = None
        self._fallback_precision: np.ndarray | None = None
        self._shrinkage: float | None = None
        self._n_fitted = 0

    def fit(self, clean_features: np.ndarray) -> "MahalanobisDistance":
        n_samples, n_dimensions = clean_features.shape
        if n_samples < 2:
            raise ValueError("mahalanobis needs at least 2 clean training samples")
        if n_samples <= n_dimensions:
            warnings.warn(
                f"mahalanobis fitted on {n_samples} samples in {n_dimensions} "
                f"dimensions (n/d={n_samples / n_dimensions:.2f}). The unshrunk "
                "covariance would be rank-deficient here; Ledoit-Wolf shrinkage "
                "is carrying the estimate. Distances remain usable but the "
                "covariance structure is heavily regularised.",
                RuntimeWarning,
                stacklevel=2,
            )
        # store_precision=False: the covariance is factorised below rather than
        # inverted. Explicitly inverting it (via scipy's pinvh) was observed to
        # emit divide-by-zero / overflow / invalid-value RuntimeWarnings on some
        # scipy versions even for well-conditioned inputs, and an explicit
        # inverse is never the numerically preferable route to a Mahalanobis
        # distance anyway.
        estimator = LedoitWolf(store_precision=False, assume_centered=False)
        estimator.fit(clean_features)
        self._mean = estimator.location_
        self._shrinkage = float(estimator.shrinkage_)
        self._n_fitted = n_samples
        self._n_dimensions = n_dimensions

        # Cholesky factor of the shrunk covariance. Distances are then obtained
        # by triangular solve, which is both faster and better conditioned than
        # forming an inverse: cond(L) = sqrt(cond(C)).
        covariance = np.asarray(estimator.covariance_, dtype=np.float64)
        try:
            self._cholesky = np.linalg.cholesky(covariance)
            self._fallback_precision = None
        except np.linalg.LinAlgError:
            # Shrinkage should guarantee positive-definiteness, so reaching here
            # is unexpected; fall back to a pseudo-inverse and say so rather
            # than failing the run.
            warnings.warn(
                "shrunk covariance was not positive definite; falling back to a "
                "pseudo-inverse for Mahalanobis distance",
                RuntimeWarning,
                stacklevel=2,
            )
            self._cholesky = None
            self._fallback_precision = np.linalg.pinv(covariance)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self._mean is None or (
            self._cholesky is None and self._fallback_precision is None
        ):
            raise RuntimeError("mahalanobis must be fitted before scoring")
        centred = np.asarray(features, dtype=np.float64) - self._mean
        if self._cholesky is not None:
            # Solve L y = (x - mu) per row; then d^2 = ||y||^2, since
            # (x-mu)' C^-1 (x-mu) = (x-mu)' (L L')^-1 (x-mu) = ||L^-1 (x-mu)||^2.
            solved = scipy_solve_triangular(
                self._cholesky, centred.T, lower=True, check_finite=False
            )
            squared = np.sum(solved**2, axis=0)
        else:
            squared = np.einsum(
                "ij,jk,ik->i", centred, self._fallback_precision, centred
            )
        return np.sqrt(np.maximum(squared, 0.0))

    def describe(self) -> dict[str, Any]:
        return {
            "detector": self.name,
            "covariance_estimator": "LedoitWolf",
            "solver": (
                "cholesky_triangular_solve" if self._cholesky is not None
                else "pseudo_inverse_fallback"
            ),
            "shrinkage_coefficient": self._shrinkage,
            "n_clean_fitted": self._n_fitted,
            "n_dimensions": getattr(self, "_n_dimensions", None),
            "samples_per_dimension": (
                self._n_fitted / getattr(self, "_n_dimensions", 1)
                if self._n_fitted else None
            ),
        }


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest over clean features; sign-flipped so higher = worse."""

    name = "isolation_forest"

    def __init__(self, n_estimators: int = 200, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model: IsolationForest | None = None
        self._n_fitted = 0

    def fit(self, clean_features: np.ndarray) -> "IsolationForestDetector":
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            # The training set is defined as clean, so no contamination is
            # expected within it; 'auto' would assume some and shift the
            # internal offset accordingly.
            contamination="auto",
        ).fit(clean_features)
        self._n_fitted = len(clean_features)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("isolation_forest must be fitted before scoring")
        # score_samples returns higher = more normal, so negate for our
        # higher = more anomalous convention.
        return -self._model.score_samples(features)

    def describe(self) -> dict[str, Any]:
        return {
            "detector": self.name,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "n_clean_fitted": self._n_fitted,
        }


@dataclass
class DetectorSuite:
    """A standardiser plus every detector, all fitted on the same clean features.

    Standardising once here (rather than inside each detector) guarantees all
    detectors see identical inputs, so a performance difference between them is
    attributable to the detector and not to differing preprocessing.
    """

    knn_neighbours: tuple[int, ...] = DEFAULT_KNN_NEIGHBOURS
    isolation_forest_estimators: int = 200
    random_state: int = 42
    scaler: StandardScaler | None = field(default=None, init=False)
    detectors: dict[str, AnomalyDetector] = field(default_factory=dict, init=False)

    def fit(self, clean_features: np.ndarray) -> "DetectorSuite":
        self.scaler = StandardScaler().fit(clean_features)
        standardised = self.scaler.transform(clean_features)

        self.detectors = {}
        for k in self.knn_neighbours:
            if k <= len(standardised):
                detector = KNNDistance(n_neighbours=k)
                self.detectors[detector.name] = detector.fit(standardised)
        mahalanobis = MahalanobisDistance().fit(standardised)
        self.detectors[mahalanobis.name] = mahalanobis
        forest = IsolationForestDetector(
            n_estimators=self.isolation_forest_estimators,
            random_state=self.random_state,
        ).fit(standardised)
        self.detectors[forest.name] = forest
        return self

    def score_all(self, features: np.ndarray) -> dict[str, np.ndarray]:
        """Score with every fitted detector, keyed by detector name."""
        if self.scaler is None:
            raise RuntimeError("DetectorSuite must be fitted before scoring")
        standardised = self.scaler.transform(features)
        return {
            name: detector.score(standardised)
            for name, detector in self.detectors.items()
        }

    def describe(self) -> dict[str, Any]:
        return {name: detector.describe() for name, detector in self.detectors.items()}


def detector_names(knn_neighbours: tuple[int, ...] = DEFAULT_KNN_NEIGHBOURS) -> list[str]:
    """Names a fully-fitted suite would produce, without fitting one."""
    return [f"knn_k{k}" for k in knn_neighbours] + ["mahalanobis", "isolation_forest"]