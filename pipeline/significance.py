"""Bootstrap significance testing for AUROC comparisons.

Extracted from ad-hoc scratch code into a proper module because Stage 4's
fusion evaluation rests on paired comparisons between scorers, and getting the
pairing right matters more than the point estimates. The specific mistake this
prevents: **taking independent bootstrap CIs on each scorer and calling the
comparison significant when the CIs do not overlap** - that ignores the
correlation between the two scorers' errors and is much too conservative on
paired data. A paired bootstrap on the *difference*, using the same resampled
row indices for both scorers, is the correct test.

The concrete case that motivated this: the ChestX-ray14 occlusion result. The
embedding side's AUROC was 0.811 with an independent 95% CI of roughly
[0.766, 0.853] and the IQA side's point estimate was 0.779 - a naive reading
sees CI overlap and calls the difference insignificant, but does not actually
test the difference itself. Running the paired bootstrap gave a diff of +0.032
with a 95% CI of [-0.032, +0.099] and p(diff <= 0) = 0.170. Same conclusion
here (not significant), but produced by the right instrument.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_auc_score


N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42


@dataclass(frozen=True)
class BootstrapCI:
    """Point estimate plus a percentile bootstrap 95% CI."""

    point: float
    ci_low: float
    ci_high: float
    n_bootstrap: int


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Paired bootstrap on the AUROC difference (scorer_a - scorer_b).

    ``p_diff_le_zero`` is the bootstrap p-value for the one-sided null that
    scorer_a is no better than scorer_b. Reported alongside the CI because it
    quantifies "how close was this to a tie" independently of an arbitrary
    0.05 threshold.
    """

    auroc_a: float
    auroc_b: float
    diff: float
    ci_low: float
    ci_high: float
    p_diff_le_zero: float
    n_bootstrap: int

    @property
    def significant_at_95(self) -> bool:
        """True when the 95% CI on the difference excludes zero."""
        return self.ci_low > 0 or self.ci_high < 0


def _resample_indices(rng: np.random.Generator, n_rows: int) -> np.ndarray:
    return rng.integers(0, n_rows, n_rows)


def bootstrap_auroc_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapCI:
    """Point AUROC plus a percentile bootstrap 95% CI on the same rows.

    Resamples that end up containing only one class have no defined AUROC and
    are silently dropped - a warning is not informative here since it is an
    unavoidable consequence of resampling with replacement and does not
    reflect anything wrong with the data.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(labels) != len(scores):
        raise ValueError(
            f"labels and scores must have the same length, got "
            f"{len(labels)} and {len(scores)}"
        )
    if len(np.unique(labels)) < 2:
        return BootstrapCI(
            point=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_bootstrap=0,
        )
    point = float(roc_auc_score(labels, scores))
    rng = np.random.default_rng(seed)
    resampled: list[float] = []
    with warnings.catch_warnings():
        # A single-class resample is expected here (see the docstring), so the
        # UndefinedMetricWarning it raises inside sklearn is noise, not signal.
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        for _ in range(n_bootstrap):
            indices = _resample_indices(rng, len(labels))
            if len(np.unique(labels[indices])) < 2:
                continue
            resampled.append(roc_auc_score(labels[indices], scores[indices]))
    if not resampled:
        return BootstrapCI(
            point=point, ci_low=float("nan"), ci_high=float("nan"), n_bootstrap=0
        )
    low, high = np.percentile(resampled, [2.5, 97.5])
    return BootstrapCI(
        point=point, ci_low=float(low), ci_high=float(high), n_bootstrap=len(resampled)
    )


def paired_bootstrap_diff(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> PairedBootstrapResult:
    """Paired bootstrap on AUROC(scorer_a) - AUROC(scorer_b).

    Both scorers are evaluated on **the same resampled row indices**, so
    correlated errors cancel and the CI on the difference is narrower - and
    fairer - than what independent bootstraps would give.

    Both scorer arrays must be pre-oriented so that higher = more anomalous /
    more degraded. The caller owns polarity: this function does not know which
    direction a raw IQA metric is "worse". Passing an unoriented score is not
    detected here and will silently invert the sign of the difference.
    """
    labels = np.asarray(labels)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    if not (len(labels) == len(scores_a) == len(scores_b)):
        raise ValueError("labels and both scorer arrays must be the same length")

    auroc_a = float(roc_auc_score(labels, scores_a))
    auroc_b = float(roc_auc_score(labels, scores_b))

    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        for _ in range(n_bootstrap):
            indices = _resample_indices(rng, len(labels))
            if len(np.unique(labels[indices])) < 2:
                continue
            diff = (
                roc_auc_score(labels[indices], scores_a[indices])
                - roc_auc_score(labels[indices], scores_b[indices])
            )
            diffs.append(diff)

    if not diffs:
        return PairedBootstrapResult(
            auroc_a=auroc_a, auroc_b=auroc_b, diff=auroc_a - auroc_b,
            ci_low=float("nan"), ci_high=float("nan"),
            p_diff_le_zero=float("nan"), n_bootstrap=0,
        )
    diffs_array = np.asarray(diffs)
    low, high = np.percentile(diffs_array, [2.5, 97.5])
    return PairedBootstrapResult(
        auroc_a=auroc_a,
        auroc_b=auroc_b,
        diff=float(diffs_array.mean()),
        ci_low=float(low),
        ci_high=float(high),
        p_diff_le_zero=float((diffs_array <= 0).mean()),
        n_bootstrap=len(diffs_array),
    )


def orient_scores(raw_scores: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Flip sign so that, uniformly, higher = more degraded (anomalous).

    Kept next to the bootstrap functions because getting orientation wrong is
    the failure mode paired_bootstrap_diff cannot detect for you.
    """
    return -np.asarray(raw_scores) if higher_is_better else np.asarray(raw_scores)