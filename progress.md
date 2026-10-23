# Project Progress Log

**Project:** An Agentic AI System for Automated Medical Imaging Dataset Curation
and Quality Control — image quality assessment (IQA) component / "Data Inspector Agent"

**Purpose of this file:** A running record of what is *actually implemented and
verified* versus what is planned, plus the reasoning behind design decisions.
Written because "already built but not mentioned in the handoff" has bitten this
project more than once.

**Last updated:** 2026-08-02

---

## Scope

My dissertation covers only the **image quality / anomaly detection** slice of a
larger agentic system. Other agents (label auditing, dataset optimisation, LLM
reporting) are out of scope.

| Dataset | Role | Status |
|---|---|---|
| NIH ChestX-ray14 | **Primary** | **Complete** — 5,606 images |
| ISIC (dermoscopy) | Generalisation test | **Complete** — 1,800 images |
| CheXpert | Optional secondary | Scaffolded, never run |
| BraTS (3D MRI) | Stretch goal | Stub, never run |

---

## Status by component

### ✅ Dataset acquisition — both primary datasets complete

**ChestX-ray14** (`scripts/download_chestxray14.py`)
```
5,606 images   official Kaggle random sample (nih-chest-xrays/sample, 4.2 GB)
Splits         3,925 train / 844 test / 837 val  (70.0 / 15.1 / 14.9%)
Strata         15 finding strata, ~70/15/15 within each
Audit          0 orphaned rows, 0 unindexed files, all .png
Rare           Hernia (13 images) retained at capacity, documented
```
Used the *sample* rather than the full 45 GB release: it is a genuine random
sample (no ID-ordering bias), original-resolution PNGs, and 5,606 images exceeds
what this work needs. Pre-resized 224×224 mirrors were rejected — they are
re-encoded, so injecting defects onto them would contaminate the ground truth.

**ISIC** (`scripts/download_isic.py`)
```
1,800 images   balanced across 67 diagnosis-parent strata, seed 42
Splits         1,277 train / 258 val / 265 test, lesion-grouped
Extensions     1,799 .jpg + 1 .png
Audit          0 orphaned rows, 0 unindexed files
Rare           31 strata under 15 images, retained at capacity
```
Legacy 2,000-image sequential cohort archived under `isic/legacy_unstratified/`.

### ✅ Synthetic defect injection — complete for both datasets

9 defect types × 3 severities. Both datasets: **all images processed, 0 failures.**

### ✅ Perceptual severity calibration — implemented, run, mostly validated

`scripts/calibrate_severity.py`. Each severity is a target **median SSIM**;
bisection solves for the parameter that reaches it on each dataset. Targets
**0.963 / 0.846 / 0.729** chosen from the intersection of both datasets'
measured feasible bands (`--probe`), not guessed. All 21 entries per dataset
report `status: reached`.

### ✅ Tests — 69 passing

---

## Key decisions (and why)

**Canonical 256×256 resize before injection** — ISIC natives span 1,872×1,053 to
6,000×4,000+, so fixed-pixel kernels were not comparable. Chosen over
resolution-relative kernels: simpler to defend, and gives the planned embedding
backbone one consistent input size.

**Perceptual calibration over hand-tuned ranges** — the defaults, tuned by eye on
dermoscopy, destroyed radiographs with Poisson noise (severity 3 was 22× too
strong) while JPEG stayed nearly invisible. A defect that annihilates the image
is trivially detectable and contributes nothing to a benchmark.

**Targets chosen from the data** — the feasible band is bounded by physical
limits, not preference: JPEG stops at quality 1 (SSIM ≈ 0.63) and motion blur
cannot use a kernel below 3×3 (SSIM ≈ 0.97). Targets sit inside the intersection
of both datasets' bands so severity means the same thing in both.

**occlusion_patch and rotation_crop are not calibrated** — already parameterised
in modality-independent units (area fraction, degrees). SSIM is a bad target for
them: a few degrees of rotation collapses SSIM without comparable quality loss.

**ChestX-ray14 kept at natural class proportions; ISIC balanced** — ISIC needed
balancing to correct a sampling artifact. The ChestX-ray14 sample is already
random, so balancing would discard real data to manufacture a distribution that
does not exist clinically. Must be stated in the methodology.

---

## Cross-modality calibration result

Same SSIM target, different parameters required — this is a reportable finding:

| Defect (severity 3) | ChestX-ray14 | ISIC |
|---|---|---|
| Gaussian noise σ | 7.28 | 6.66 |
| Poisson scale | 0.0033 | 0.0026 |
| Brightness offset | 84 | 94 |
| Gaussian blur σ | 7.7 (k 19–29) | **30.1 (k 73–109)** |
| Motion blur kernel | 32 | **79** |
| Contrast factor | 0.29 | 0.14 |
| JPEG quality (sev 1) | 50 | **91** |

Noise and brightness transfer almost unchanged. Blur does not. Radiographs
tolerate compression far better (quality 50 vs 91 for the same mild target)
because they are smooth and low-texture.

---

## ⚠️ Open problem: ISIC blur at severity 3 is unusable

Visual inspection (2026-08-02) shows ISIC `gaussian_blur` severity 3 (109×109
kernel, 43% of the frame) and `motion_blur` severity 3 (65–95) **destroy the
lesion entirely**, despite hitting SSIM 0.723 exactly on target.

Cause: ISIC images are downsampled ~20× to canonical size, so they arrive
band-limited. Blur removes little *local* structure — which is what SSIM
measures — while perceptually erasing the lesion. **SSIM is not a valid proxy
for blur degradation on already-smoothed images.**

ChestX-ray14 is unaffected (only ~4× downsample, retains fine detail).

Options, not yet decided:
1. Cap blur kernels at a fraction of image width (~15–20%, i.e. ≤51px at 256)
   and accept blur cannot reach severity 3 on ISIC. Document as a physical limit
   like JPEG's quality-1 floor. **Currently favoured.**
2. Use a blur-appropriate target metric (e.g. gradient-energy or Laplacian
   variance ratio) instead of SSIM for the two blur defects.
3. Accept as-is and document that ISIC severity-3 blur is a total-loss case.

---

## Other open questions

- [ ] **Canonical size 256 vs 224** — depends on final backbone choice.
- [ ] **Upscaling small sources** fabricates detail; needs a warning log before
      this is pointed at any dataset with sub-256px images.
- [ ] **`setup_all.py` writes a stale ChestX-ray14 header** missing
      `finding_stratum`; re-running it recreates the schema mismatch.
- [ ] **Deduplicate** the allocation/split helpers now duplicated between
      `download_isic.py` and `common.py`.
- [ ] **BMAD alignment** — confirm whether its splits/metrics can be matched.
- [ ] **Legacy 2,000 ISIC images** — archived; decide if they ever re-enter.

---

## Bug history

Several of these are methodological findings, not plumbing, and belong in the
"Data Preparation and Validation" write-up.

| Bug | Impact | Fix |
|---|---|---|
| ISIC `/download/` endpoint 404 | Nothing downloaded | Use `files.full.url` |
| Sequential ID-order download | Severe diagnosis clustering | Stratified sampling |
| Resume checked metadata, not disk | Failed downloads marked complete | Check the filesystem |
| `metadata.csv` overwritten each run | Lost track of 2,000 images | Merge-by-key; affected all 4 downloaders |
| Row written before file verified | Row pointing at a missing file | Pillow-verify + atomic rename |
| `.jpg` extension assumed | Would crash on the 1 `.png` | Read `path` verbatim |
| Hard error on any failure | 0.22% failures aborted the run | Fail only above 5% |
| Fixed kernels vs 10× resolution range | Blur nearly invisible (MAD 1.7) | Canonical resize (MAD 6.4) |
| Decompression guard at **two** call sites | `crop()` re-raised after `open()` was relaxed | Relax around both |
| 23 ChestX-ray14 images are RGBA | Silently dropped | Flatten alpha; collapse identical RGB to L for a uniform dataset |
| Calibration bounds too narrow | Poisson/blur clamped at bounds | Widened after measuring against real data |
| Calibrated contrast crossed 1.0 | Became contrast *increase* | Clamp, narrowing symmetrically |
| "CLAMPED" conflated two causes | Bound-limited vs integer granularity | Three distinct statuses |

**False alarm worth remembering:** defect *rendering* looked broken, but the
diagnostic proved application and saving were correct — images were being
downsampled 20–30× for display. Check the saved artefact, not the figure.

---

## Next steps

1. **Resolve the ISIC blur problem** (options above).
2. **Write "Data Preparation and Validation"** — everything needed exists.
3. **Stage 2 — no-reference IQA baselines:** Laplacian-variance sharpness,
   noise, contrast; BRISQUE; CLIP-IQA. First real result: Spearman correlation
   of score vs injected severity.
4. **Stage 3 — embedding anomaly detection:** frozen ResNet-18 /
   EfficientNet-B0, then DINO; Isolation Forest vs Mahalanobis vs kNN.
5. **Stage 4 — fusion**, per-defect ablation table. The intended contribution.
6. **Stage 5 — generalisation** across the two modalities.

---

## Environment gotchas

- **Activate the venv first**, or macOS falls back to system Python 3.9:
  `source .venv/bin/activate` (prompt should show `(.venv)`).
- **Use `python -m pytest`**, not bare `pytest` — the latter omits the project
  root from `sys.path`. Permanent fix (not yet applied):
  ```toml
  [tool.pytest.ini_options]
  pythonpath = ["."]
  ```
- Kagglehub downloads public datasets **anonymously**; `~/.kaggle/kaggle.json`
  is optional, and requiring it was a bug.
- ISIC metadata indexing scans ~550k records (~23 min) with no caching.

---

## Commands

```bash
source .venv/bin/activate

# Acquisition
python scripts/download_chestxray14.py --data-dir data --source sample --seed 42
python scripts/download_isic.py --data-dir data --fresh --balanced --limit 1800 --seed 42

# Audits (report-only)
python scripts/download_chestxray14.py --data-dir data --repair
python scripts/download_isic.py --data-dir data --repair

# Severity calibration
python scripts/calibrate_severity.py --dataset <ds> --data-dir data --sample-size 60 --probe
python scripts/calibrate_severity.py --dataset <ds> --data-dir data \
    --sample-size 60 --targets 0.963,0.846,0.729

# Defect injection
python scripts/inject_defects.py --dataset <ds> --data-dir data \
    --canonical-size 256 --seed 42 --visualize 3 \
    --severity-profile data/<ds>/severity_profile.json

python -m pytest tests/ -v
```

**Outputs to check after any run:** `processed_metadata.csv`,
`logs/defect_examples_<ds>.png`, `logs/<ds>_defect_injection_failures.json`,
`logs/<ds>_severity_calibration.json`, `logs/<ds>_split_crosstab.csv`,
`logs/<ds>_severity_probe.json`