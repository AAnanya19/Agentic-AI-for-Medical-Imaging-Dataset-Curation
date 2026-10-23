# Medical image dataset storage

This directory is the local data root for the dissertation pipeline. Large raw
datasets and generated files should normally stay out of version control.

Each dataset has the same broad layout:

```text
data/
├── chestxray14/{raw,processed,metadata.csv}
├── chexpert/{raw,processed,metadata.csv}
├── isic/
│   ├── {raw,processed,metadata.csv}
│   └── legacy_unstratified/{raw,metadata.csv,README.md}
├── brats/{raw,processed,metadata.csv}
└── logs/
```

- `raw/` contains untouched downloaded or extracted source files.
- `processed/` is reserved for later resizing/normalization code and is empty initially.
- `metadata.csv` is the normalized index. Its `path` values are relative to the
  corresponding dataset directory, not to the repository root.
- `logs/` contains one log per downloader.

Downloader reruns merge into an existing `metadata.csv` instead of replacing it:
existing keyed rows retain their values, unseen rows append, and a verified
reprocessed file may update its row in place. The keys are `image_id` for
ChestX-ray14, CheXpert, and ISIC, and `path` for BraTS. Use `--fresh` only when
you intentionally want the current run to replace the manifest; it logs a
warning and does not delete anything under `raw/`.

## Setup

Install dependencies and create all directories plus empty metadata headers:

```bash
python -m pip install -r requirements.txt
python scripts/setup_all.py
```

Use `--dry-run` on any command to validate required local inputs (and Kaggle/API
access where applicable) without downloading, extracting, or writing data.
`--limit N` is useful for test runs. For ChestX-ray14, KaggleHub must still fetch
the complete archive; the limit applies while building metadata.

## Sources and rerunning

### NIH ChestX-ray14

Source: the Kaggle mirror `nih-chest-xrays/data` of the NIH clinical-center
release. Configure `~/.kaggle/kaggle.json`, then run:

```bash
python scripts/download_chestxray14.py --data-dir data
```

The original finding labels are retained. A seeded patient-level 70/15/15 split
prevents a patient's images from crossing train, validation, and test sets.

### CheXpert Small

Source: Stanford ML Group's CheXpert Small release. Registration is required;
download the archive manually and pass its local path:

```bash
python scripts/download_chexpert.py --data-dir data --source-path /path/to/CheXpert-v1.0-small.zip
```

The supplied `train.csv` and `valid.csv` assignments are preserved (`valid` is
normalized to `val`). Label columns are serialized as JSON in `source_labels`.

### ISIC Archive

Source: the public ISIC Archive API v2. Before downloading image bytes, the
downloader indexes the API's lightweight metadata listings and builds a seeded,
diagnosis-stratified cohort. Detailed diagnoses such as `Nevus, Dysplastic`
remain unchanged in `metadata.csv`; their parent diagnosis (for example,
`Nevus`) is used only for sampling and split assignment.

The active local cohort was rebuilt on 2026-07-31 as one balanced, seed-42
sample of 1,800 verified images. Its manifest is `isic/metadata.csv`, and only
the 1,800 files indexed there belong to the active pipeline. The earlier 2,000
image sequential/unstratified cohort is preserved, not deleted, under
`isic/legacy_unstratified/`; its local README records why upload/ID-order
sampling made it unsuitable as the active diagnosis-balanced cohort. ISIC
`--fresh` mode is scoped to the active manifest and `isic/raw/`; it does not
read, overwrite, or remove anything in the legacy archive.

By default, a limited cohort follows the archive's diagnosis proportions. Use
`--balanced` for an approximately equal number from each available class, or
`--stratify-by` to select another API metadata field:

```bash
python scripts/download_isic.py --data-dir data
python scripts/download_isic.py --data-dir data --limit 1000 --seed 42
python scripts/download_isic.py --data-dir data --limit 1000 --balanced --seed 42
python scripts/download_isic.py --data-dir data --limit 1000 --stratify-by image_type
python scripts/download_isic.py --data-dir data --repair
python scripts/download_isic.py --data-dir data --fresh --limit 1000
python scripts/download_isic.py --data-dir data --fresh --balanced --limit 1800 --seed 42
```

Candidate IDs are sorted and then shuffled independently within each stratum
using `--seed`, so the same archive state, arguments, and seed reproduce both the
selected cohort and its split assignments even if API pagination order changes.
The selected cohort receives a diagnosis-stratified 70/15/15 `train`/`val`/`test`
split. When ISIC supplies a lesion identifier, every image from that lesion is
kept in one split to prevent leakage; images without one are treated
independently.

For the current 1,800-image cohort, lesion grouping produced 1,277 train, 258
validation, and 265 test images (approximately 70.9/14.3/14.7%). The complete
diagnosis-parent-by-split table is in `logs/isic_split_crosstab.csv`.

The archive exposed 67 diagnosis-parent strata. Thirty-one have fewer than 15
usable public images in the entire archive and therefore cannot independently
support a well-powered defect-severity analysis. No minimum-count exclusion was
applied: these categories were retained at their full available capacity and
flagged in `logs/isic_rare_categories.json`. Capacity-sufficient categories
contribute 54 or 55 images each to this balanced cohort.

The final planned and achieved counts are logged and written to
`logs/isic_sample_distribution.json`. The downloader still retries transient
failures with exponential backoff and resumes by reusing matching image files
already present as regular files in `isic/raw/`. Resume decisions are based only
on actual file existence at the expected path, never on an image ID appearing in
metadata. Failed downloads therefore remain eligible on the next run, and a
metadata row is written only after the completed temporary download is non-empty
and passes Pillow image verification. The verified file is then moved atomically
to its final path. Raw files and metadata rows from older cohorts are retained,
so `isic/metadata.csv` is the cumulative manifest unless `--fresh` is supplied.

`--repair` is a local, report-only audit: it reads `isic/metadata.csv`, resolves
each relative `path` safely inside the ISIC dataset directory, and logs every row
whose target is missing, empty, or fails image verification. It performs no API
requests and does not remove metadata rows or raw files. It also reports total
metadata rows, raw image files, unindexed files, and the extension breakdown.
The current result is stored in `logs/isic_active_audit.json`: 1,800 rows/files,
zero orphaned rows, zero unindexed files, 1,799 `.jpg`, and one `.png`. Readers,
including `inject_defects.py`, use the exact `path` value from metadata rather
than reconstructing a filename or assuming `.jpg`.

A cached manifest can be safely reconciled with verified on-disk files using
`--rebuild-metadata-from /path/to/cached.csv`; the current manifest is backed up
before the atomic merge. Pass `-` as the source to read CSV data from standard
input.
Documented image type and quality-control/artifact fields are retained as JSON
in `artifact_flags`.

### BraTS

Source: an approved BraTS release obtained after accepting its Synapse/data-use
terms. Pass the downloaded `.zip` or tar archive:

```bash
python scripts/download_brats.py --data-dir data --source-path /path/to/brats.zip
```

The indexer discovers NIfTI/MHA T1, contrast-enhanced T1, T2, and FLAIR volumes
inside per-patient directories. It derives a split from folder names where one is
present and otherwise records `unassigned`.

## Running several downloaders

The setup entry point can invoke a comma-separated selection:

```bash
python scripts/setup_all.py --datasets chestxray14,isic --limit 100
python scripts/setup_all.py --datasets chestxray14,isic --limit 100 --fresh
python scripts/setup_all.py --datasets chexpert,brats \
  --chexpert-source /path/to/chexpert.zip --brats-source /path/to/brats.tar
```

Re-running extraction copies/extracts over matching files but does not delete
anything already under `raw/`. Raw files are never transformed by these scripts.

## Creating synthetic IQA defects

The defect injector reads a dataset's `metadata.csv`, preserves its split labels,
and writes labeled JPEG derivatives under `processed/` plus
`processed_metadata.csv`:

```bash
# Validate source paths and show the planned split counts without writing files.
python scripts/inject_defects.py --dataset isic --data-dir data --dry-run

# Default: all defects and severities, with 30% clean in each split.
python scripts/inject_defects.py --dataset isic --data-dir data --seed 42

# Stack two distinct defects and create one visual pair per defect/severity.
python scripts/inject_defects.py --dataset chestxray14 --data-dir data \
  --allow-combinations --visualize 1
```

Use `--defects gaussian_noise,motion_blur`, `--severities 1,2`, and
`--fraction-clean 0.3` to select a smaller experiment. Occlusion fills can be
fixed with `--occlusion-fill black|noise|edge`; the default samples among them.
Every source image is center-cropped to a square and resized to a fixed
canonical size (default 256×256) before any defect is applied; override with
`--canonical-size 384` or similar. See the methodology note below for why this
step exists and is applied uniformly regardless of source resolution.

Every processed metadata row records the ordered defect steps, exact sampled
parameters, and random seeds in `defect_params_json`. Severity ranges are defined
centrally in `pipeline/defects.py`. Visual checks are saved to
`logs/defect_examples_<dataset>.png`, and run logs to
`logs/<dataset>_defect_injection.log`.

### Resolution and spatial-severity methodology note

ISIC source images vary substantially in native resolution — observed examples
range from approximately 1,872×1,053 pixels up to 6,000×4,000 pixels and
larger, including a handful of extreme outliers above 270 million pixels.
Because several defects (`gaussian_blur`, `motion_blur`, and the fixed-pixel
context used for `occlusion_patch`/`rotation_crop`) are parameterized in
absolute pixels, a nominal severity level occupied a much smaller effective
fraction of a high-resolution image than of a lower-resolution one, making
severity levels visually and quantitatively inconsistent across the dataset.

**Resolution: every source image is now center-cropped to a square (using its
shorter dimension) and resized to a fixed canonical size — 256×256 by default,
configurable via `--canonical-size` — as the first step of the pipeline,
before any defect is applied.** This was chosen over the alternative of
scaling kernel sizes relative to each image's own dimensions because it is
simpler to justify methodologically, matches common practice in the IQA/
anomaly-detection literature this project follows, and gives the planned
embedding-based anomaly detection stage (Stage 3) a single consistent input
size for its backbone as well, rather than two separate resizing logics to
keep in sync. Absolute-pixel kernel sizes and offsets are unchanged; they are
now comparable across images because the input resolution is fixed first.

The applied crop box, original dimensions, and canonical size are recorded per
image in `processed_metadata.csv` (`canonical_size` column and the
`baseline_preprocessing.canonical_resize` block within `defect_params_json`),
so this preprocessing step is fully traceable and reproducible.

Pillow's decompression-bomb guard is deliberately relaxed (`Image.MAX_IMAGE_PIXELS
= None`) for the duration of loading and cropping each source image, since
these are curated, project-downloaded academic archives rather than arbitrary
untrusted uploads — not just at the initial `Image.open()` call, but also
around the square-crop step, since Pillow's `Image.crop()` enforces the same
guard independently of `Image.open()`. The guard is restored immediately
afterward, and every image is canonically resized down before any further
processing, so no oversized array persists in memory beyond that step. The
four ISIC images that previously failed as "oversized" now process
successfully as a result and no longer need special-case exclusion; whether
any dataset needs that guard relaxed depends on trusting its source, which
should be reassessed for each new dataset (e.g., BraTS, ChestX-ray14) rather
than assumed.

Raw files are never modified. Reusable defect functions preserve grayscale mode
and NumPy dtype; CLI artifacts are intentionally written as 8-bit JPEGs (quality
100 for the final container) to match the requested output naming. The selected
one-time input mapping and final encoding are recorded in `defect_params_json`.
Here, `clean` means “no synthetic defect after the common baseline conversion,”
not byte-identical to a PNG or differently encoded source file.
On rerun, matching filenames are overwritten but unrelated/stale JPEGs are not
deleted; treat `processed_metadata.csv` as the authoritative run manifest or use
a fresh `--output-dir` for each experiment.