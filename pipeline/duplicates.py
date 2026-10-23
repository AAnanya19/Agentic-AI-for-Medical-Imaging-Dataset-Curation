"""Duplicate and near-duplicate detection, by perceptual hash and by embedding.

Two distinct problems that are usually conflated, and which matter for different
reasons:

  * **Exact duplicates** - byte-different but pixel-identical images. They
    inflate the apparent size of a dataset and silently reweight whatever they
    duplicate. An identical perceptual hash is itself the evidence, so these
    need no external ground truth to confirm.
  * **Near duplicates** - different acquisitions of the same subject (the same
    patient's follow-up radiograph, the same lesion photographed twice). These
    are not errors in themselves, but if they land on opposite sides of a
    train/test split they leak, and the reported performance is inflated.

Two methods are compared:

  1. **Perceptual hashing** (pHash, dHash) - fast, exact-duplicate detection is
     essentially free, but degrades on geometric transformation.
  2. **Embedding cosine similarity** - reuses the Stage 3 feature caches at no
     extra compute cost, and is robust to transformations that defeat a hash,
     at the price of a threshold that has to be chosen rather than being
     definitionally zero.

**Ground truth availability differs by dataset, and this is a real limitation.**
ChestX-ray14's ``metadata.csv`` carries ``patient_id``, so near-duplicate
detection can be validated against genuine same-subject groups. ISIC's
downloader computes a lesion grouping for split assignment but never persists
it, so no equivalent column exists - on ISIC, exact duplicates are still
detectable (hash equality is self-evidencing) but near-duplicate candidates can
only be reported, not scored. Rather than fabricating a proxy, the functions
below take an optional group column and report precision/recall only when it is
supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import imagehash
    from PIL import Image
except ImportError as error:  # pragma: no cover - environment guard
    raise ImportError(
        "imagehash and Pillow are required for pipeline.duplicates"
    ) from error


# Hamming distance of 0: the images are perceptually identical at this hash
# resolution, which is the definition of an exact duplicate here.
EXACT_HASH_DISTANCE = 0
# Empirically the usual near-duplicate band for a 64-bit pHash on natural
# images. Reported rather than assumed correct - the precision/recall sweep is
# what justifies a choice.
DEFAULT_NEAR_HASH_DISTANCE = 10
DEFAULT_COSINE_THRESHOLD = 0.95

# **Hash resolution matters far more on medical images than on natural photos.**
# imagehash defaults to hash_size=8, i.e. a 64-bit signature derived from an 8x8
# DCT. Natural photographs differ enormously at that resolution, but chest
# radiographs are highly stereotyped - at 8x8 every one is the same grey ellipse
# with ribs. Measured on a stereotyped synthetic radiograph set, hash_size=8
# yielded only 17 distinct pHash values across 35 different subjects, collapsing
# precision against known patient groups to 0.007; hash_size=16 (a 256-bit
# signature) gave 34 distinct values and raised precision by ~30x. The default
# here is therefore 16, not imagehash's 8, and hash_saturation() below reports
# the diagnostic so a reader can see whether the hash had enough resolution to
# discriminate at all.
DEFAULT_HASH_SIZE = 16

# The four classical perceptual hashes used in the standard comparative
# evaluations of image deduplication (AHash, DHash, PHash, WHash). All four are
# computed because they fail differently: aHash thresholds against the mean and
# is the crudest, pHash uses a DCT and resists mild global change, dHash tracks
# horizontal gradients and is sensitive to structural edits, wHash uses a Haar
# wavelet decomposition. Agreement across them is stronger evidence than any
# one alone.
HASH_TYPES = ("ahash", "phash", "dhash", "whash")
# Below this ratio of distinct hashes to images, the hash is saturated and its
# duplicate verdicts are not trustworthy.
HASH_SATURATION_WARNING_RATIO = 0.9


@dataclass(frozen=True)
class DuplicatePair:
    """One candidate duplicate relationship, with the evidence for it."""

    image_id_a: str
    image_id_b: str
    method: str
    score: float
    is_exact: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id_a": self.image_id_a,
            "image_id_b": self.image_id_b,
            "method": self.method,
            "score": self.score,
            "is_exact": self.is_exact,
        }


# ---------------------------------------------------------------------------
# Perceptual hashing
# ---------------------------------------------------------------------------

def _power_of_two_at_most(value: int) -> int:
    """Largest power of two <= value. whash rejects non-power-of-two sizes."""
    power = 1
    while power * 2 <= value:
        power *= 2
    return power


def compute_hashes(
    paths: dict[str, Path], hash_size: int = DEFAULT_HASH_SIZE
) -> tuple[dict[str, Any], dict[str, str]]:
    """Compute pHash and dHash for each image, returning (hashes, failures).

    Both are computed because they fail differently: pHash uses a DCT and is
    robust to small global changes, dHash tracks horizontal gradients and is
    more sensitive to structural edits. Agreement between them is stronger
    evidence than either alone.
    """
    hashes: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for image_id, path in paths.items():
        try:
            with Image.open(path) as handle:
                image = handle.convert("L")
                hashes[image_id] = {
                    "ahash": imagehash.average_hash(image, hash_size=hash_size),
                    "phash": imagehash.phash(image, hash_size=hash_size),
                    "dhash": imagehash.dhash(image, hash_size=hash_size),
                    # whash requires a power-of-two hash_size; fall back to the
                    # nearest lower power of two rather than failing the row.
                    "whash": imagehash.whash(
                        image, hash_size=_power_of_two_at_most(hash_size)
                    ),
                }
        except Exception as error:  # noqa: BLE001 - one bad file must not abort
            failures[image_id] = str(error)
    return hashes, failures


def hash_saturation(hashes: dict[str, Any], hash_type: str = "phash") -> dict[str, Any]:
    """How many distinct hash values exist, versus how many images.

    The single most useful diagnostic for whether a perceptual hash has enough
    resolution for this dataset. A ratio well below 1.0 means many genuinely
    different subjects share a hash, so every "duplicate" verdict is suspect -
    which is exactly what happens to a 64-bit pHash on stereotyped radiographs.
    """
    values = {str(entry[hash_type]) for entry in hashes.values()}
    ratio = len(values) / len(hashes) if hashes else float("nan")
    return {
        "hash_type": hash_type,
        "n_images": len(hashes),
        "n_distinct_hashes": len(values),
        "distinct_ratio": ratio,
        "saturated": bool(ratio < HASH_SATURATION_WARNING_RATIO),
    }


def find_hash_duplicates(
    hashes: dict[str, Any],
    near_distance: int = DEFAULT_NEAR_HASH_DISTANCE,
    hash_type: str = "phash",
) -> list[DuplicatePair]:
    """All pairs within near_distance, marking distance-0 pairs as exact.

    Exact matches are found by bucketing on the hash value, which is O(n).
    Near matches require pairwise comparison, so this is O(n^2) in the number of
    images - acceptable at these dataset sizes (5,606 and 1,800) but the reason
    a production system would use an LSH index instead.
    """
    image_ids = sorted(hashes)
    pairs: list[DuplicatePair] = []

    # Exact duplicates: group by identical hash rather than comparing all pairs.
    buckets: dict[str, list[str]] = {}
    for image_id in image_ids:
        buckets.setdefault(str(hashes[image_id][hash_type]), []).append(image_id)
    exact_pairs: set[tuple[str, str]] = set()
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for index, first in enumerate(bucket):
            for second in bucket[index + 1 :]:
                exact_pairs.add((first, second))
                pairs.append(
                    DuplicatePair(first, second, f"{hash_type}_exact", 0.0, True)
                )

    if near_distance <= EXACT_HASH_DISTANCE:
        return pairs

    for index, first in enumerate(image_ids):
        for second in image_ids[index + 1 :]:
            if (first, second) in exact_pairs:
                continue  # already recorded as exact
            distance = hashes[first][hash_type] - hashes[second][hash_type]
            if distance <= near_distance:
                pairs.append(
                    DuplicatePair(
                        first, second, f"{hash_type}_near", float(distance), False
                    )
                )
    return pairs


# ---------------------------------------------------------------------------
# Embedding cosine similarity
# ---------------------------------------------------------------------------

def find_embedding_duplicates(
    image_ids: list[str],
    embeddings: np.ndarray,
    cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
) -> list[DuplicatePair]:
    """All pairs whose L2-normalised embeddings exceed cosine_threshold.

    Reuses the Stage 3 caches, so this costs no additional feature extraction.
    Normalising first turns the dot product into cosine similarity directly.
    """
    if len(image_ids) != len(embeddings):
        raise ValueError(
            f"got {len(image_ids)} image_ids for {len(embeddings)} embeddings"
        )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # A zero-norm row would divide by zero; guard rather than emit nan.
    norms[norms == 0] = 1.0
    normalised = embeddings / norms
    similarity = normalised @ normalised.T

    pairs: list[DuplicatePair] = []
    # Upper triangle only, so each unordered pair is reported once.
    rows, columns = np.triu_indices(len(image_ids), k=1)
    above = similarity[rows, columns] >= cosine_threshold
    for row, column, score in zip(
        rows[above], columns[above], similarity[rows, columns][above]
    ):
        pairs.append(
            DuplicatePair(
                image_ids[row], image_ids[column], "embedding_cosine",
                float(score), bool(score >= 0.999),
            )
        )
    return pairs


# ---------------------------------------------------------------------------
# Validation against known same-subject groups
# ---------------------------------------------------------------------------

def evaluate_against_groups(
    pairs: Iterable[DuplicatePair],
    group_by_image: dict[str, str],
    all_image_ids: list[str],
) -> dict[str, Any]:
    """Precision/recall of detected pairs against known same-subject groups.

    A pair counts as a true positive when both images belong to the same group
    (same patient, same lesion). Recall is measured against every same-group
    pair that *could* have been found, which is why the full image list is
    needed and not just the detected pairs.

    Only meaningful where a genuine grouping column exists - ChestX-ray14's
    ``patient_id``. See the module docstring on ISIC.
    """
    known_pairs: set[tuple[str, str]] = set()
    by_group: dict[str, list[str]] = {}
    for image_id in all_image_ids:
        group = group_by_image.get(image_id)
        if group is None:
            continue
        by_group.setdefault(group, []).append(image_id)
    for members in by_group.values():
        if len(members) < 2:
            continue
        ordered = sorted(members)
        for index, first in enumerate(ordered):
            for second in ordered[index + 1 :]:
                known_pairs.add((first, second))

    detected: set[tuple[str, str]] = set()
    for pair in pairs:
        detected.add(tuple(sorted((pair.image_id_a, pair.image_id_b))))

    true_positives = len(detected & known_pairs)
    precision = true_positives / len(detected) if detected else float("nan")
    recall = true_positives / len(known_pairs) if known_pairs else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if detected and known_pairs and (precision + recall) > 0
        else float("nan")
    )
    return {
        "n_detected_pairs": len(detected),
        "n_known_same_group_pairs": len(known_pairs),
        "n_true_positives": true_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_groups_with_multiple_images": sum(
            1 for members in by_group.values() if len(members) > 1
        ),
    }


def check_split_leakage(
    pairs: Iterable[DuplicatePair], split_by_image: dict[str, str]
) -> dict[str, Any]:
    """How many detected duplicate pairs straddle a train/val/test boundary.

    This is the reason near-duplicate detection matters at all: a duplicate
    within one split is harmless, whereas the same image effectively appearing
    in both train and test inflates every downstream result.
    """
    crossing: list[dict[str, str]] = []
    within = 0
    for pair in pairs:
        split_a = split_by_image.get(pair.image_id_a)
        split_b = split_by_image.get(pair.image_id_b)
        if split_a is None or split_b is None:
            continue
        if split_a == split_b:
            within += 1
        else:
            crossing.append({
                "image_id_a": pair.image_id_a, "split_a": split_a,
                "image_id_b": pair.image_id_b, "split_b": split_b,
                "method": pair.method, "score": pair.score,
            })
    return {
        "n_pairs_within_one_split": within,
        "n_pairs_crossing_splits": len(crossing),
        "crossing_examples": crossing[:20],
    }


def pairs_to_frame(pairs: Iterable[DuplicatePair]) -> pd.DataFrame:
    records = [pair.to_dict() for pair in pairs]
    columns = ["image_id_a", "image_id_b", "method", "score", "is_exact"]
    return pd.DataFrame(records, columns=columns)


def sweep_thresholds(
    hashes: dict[str, Any],
    group_by_image: dict[str, str],
    all_image_ids: list[str],
    distances: Iterable[int] = (0, 2, 4, 6, 8, 10, 12, 16),
    hash_type: str = "phash",
) -> pd.DataFrame:
    """Precision/recall across hash distances, so a threshold is chosen not assumed."""
    rows = []
    for distance in distances:
        pairs = find_hash_duplicates(hashes, near_distance=distance, hash_type=hash_type)
        metrics = evaluate_against_groups(pairs, group_by_image, all_image_ids)
        rows.append({"hash_distance": distance, **metrics})
    return pd.DataFrame(rows)


def sweep_cosine_thresholds(
    image_ids: list[str],
    embeddings: np.ndarray,
    group_by_image: dict[str, str],
    thresholds: Iterable[float] = (0.80, 0.85, 0.90, 0.95, 0.98, 0.99),
) -> pd.DataFrame:
    """Precision/recall across cosine thresholds, for the same reason."""
    rows = []
    for threshold in thresholds:
        pairs = find_embedding_duplicates(image_ids, embeddings, threshold)
        metrics = evaluate_against_groups(pairs, group_by_image, image_ids)
        rows.append({"cosine_threshold": threshold, **metrics})
    return pd.DataFrame(rows)