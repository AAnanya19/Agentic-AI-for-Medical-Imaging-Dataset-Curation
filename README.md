# An Agentic AI System for Automated Medical Imaging Dataset Curation and Quality Control

Automated quality control for medical imaging datasets. Three agents run in
sequence: a **Data Inspector** decides accept / flag / reject per image, a
**Dataset Summary Agent** analyses what acting on those decisions would do to the
dataset, and an **LLM Explainer** turns the result into a human-readable audit
report.

MSc dissertation, University of Surrey.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# The full audit, assuming datasets are already prepared (see Reproduction below)
python scripts/run_pipeline.py --dataset chestxray14 --data-dir data
```

The pipeline writes an audit report to
`data/logs/chestxray14_audit_report_template.md` and a run manifest recording
every stage's provenance to `data/logs/chestxray14_pipeline_manifest.json`.

---

## What the system checks

Ten checks, each with a validated implementation:

| Check | Method | Notes |
|---|---|---|
| Blur | Laplacian variance | Confounded upward by noise - see below |
| Noise | Immerkaer estimator | |
| Contrast | Michelson contrast | |
| Brightness | Deviation from a fitted clean reference | Injected offset is randomly signed, so absolute deviation is required |
| Compression | BRISQUE | |
| Motion blur | BRISQUE | Shares its scorer with compression |
| Occlusion | CLIP-IQA | The weakest check; every pixel-statistics metric was at or below chance |
| Rotation | Laplacian variance | Binary detection only - severity is not ordinal under SSIM |
| Resolution | Recorded source dimensions | Assessed *pre*-canonicalisation, since resizing erases resolution |
| Duplicates | 256-bit pHash + embedding cosine | Three distinct tasks; see below |

The accept / flag / reject decision is driven by a **logistic fusion** of all
metric scores, not by any single check. The per-check results supply attribution.

---

## Architecture

```
data/<dataset>/iqa_scores.csv            (Stage 2)
data/<dataset>/anomaly_scores_*.csv      (Stage 3)
data/<dataset>/processed_metadata.csv    (Stage 1)
                    |
                    v
        agents/data_inspector.py   ->  logs/<ds>_inspection.json
                    |
                    v
        agents/dataset_summary.py  ->  logs/<ds>_summary.json
                    |
                    v
        agents/llm_explainer.py    ->  logs/<ds>_audit_report_*.md
                    |
        agents/orchestrator.py     ->  logs/<ds>_pipeline_manifest.json
```

Each agent output carries a provenance block with a run id, timestamp, schema
version, full configuration and library versions, so any figure in a report can
be traced to the parameters that produced it. The handoff schema is versioned; a
consumer handed output from an incompatible major version fails loudly rather
than misreading it.

Orchestration is a plain sequential script. The dependency graph is a straight
line and the handoff is a JSON file, so an agent framework would add a dependency
without adding capability.

**Partial success is a real outcome.** The Explainer depends on an external model
server; if it is unreachable, the Inspector and Summary outputs are still written
and the manifest records which stage failed and why. The pipeline exits non-zero
only when a *required* stage fails.

---

## Configuration

All decision policy lives in a JSON file rather than in code - which checks run,
what false-positive rates they target, and how flagging is separated from
rejection:

```bash
python scripts/run_inspector.py --dataset chestxray14 --write-default-config policy.json
# edit policy.json
python scripts/run_pipeline.py --dataset chestxray14 --data-dir data --config policy.json
```

Each check carries a `rationale` string citing the empirical result that justifies
it, so the config doubles as documentation. The exact policy used is written to
`logs/<dataset>_inspector_config_used.json` on every run.

Two settings worth understanding before changing them:

- **`fpr_policy`** defaults to `composite_only`. With nine checks each targeting a
  5% false-positive rate independently, the image-level rate compounds to
  1 - 0.95^9, roughly 37%; this was measured at 28.6% before the policy existed.
  Under `composite_only` only the fused score decides, while the per-check results
  are still reported for attribution, so the configured rate is the rate
  delivered.
- **`allow_cross_modality_thresholds`** defaults to `False`. A threshold set at 5%
  false-positive rate on ChestX-ray14 produced 91.2% on ISIC, an eighteen-fold
  inflation, because the clean-score distributions barely overlap. Thresholds must
  be derived per modality.

---

## Reproduction from raw data

Datasets are downloaded from public sources; no data is committed.

```bash
# Stage 1 - acquisition, calibration, defect injection
python scripts/download_chestxray14.py --data-dir data --source sample --seed 42
python scripts/download_isic.py --data-dir data --fresh --balanced --limit 1800 --seed 42

python scripts/calibrate_severity.py --dataset chestxray14 --data-dir data \
    --sample-size 60 --targets 0.963,0.846,0.729
python scripts/inject_defects.py --dataset chestxray14 --data-dir data \
    --canonical-size 256 --seed 42 --visualize 3 \
    --severity-profile data/chestxray14/severity_profile.json
python scripts/validate_ground_truth.py --dataset chestxray14 --data-dir data

# Stage 2 - classical no-reference IQA
python scripts/compute_iqa.py  --dataset chestxray14 --data-dir data
python scripts/evaluate_iqa.py --dataset chestxray14 --data-dir data \
    --eval-splits val,test --output-suffix _valtest

# Stage 3 - frozen-embedding anomaly detection
python scripts/extract_embeddings.py --dataset chestxray14 --data-dir data
python scripts/evaluate_anomaly.py   --dataset chestxray14 --data-dir data

# Stage 4 / 4b - fusion and the feature-set ablation
python scripts/evaluate_fusion.py --dataset chestxray14 --data-dir data \
    --backbone resnet18

# Stage 5 - cross-modality transfer
python scripts/evaluate_transfer.py --datasets chestxray14,isic --data-dir data

# Stage 6 - resolution and duplicates
python scripts/build_duplicate_benchmark.py --dataset chestxray14 --data-dir data
python scripts/extract_embeddings.py --dataset chestxray14 --data-dir data \
    --subset benchmark
python scripts/detect_duplicates.py --dataset chestxray14 --data-dir data \
    --group-column patient_id

# Stages 7-10 - the agent pipeline
python scripts/run_pipeline.py --dataset chestxray14 --data-dir data
```

Repeat with `--dataset isic` throughout. ISIC has no persisted subject
identifier, so omit `--group-column` there; the re-identification task is skipped
rather than evaluated against a proxy.

### Supporting analyses

```bash
# The false-reject / recall trade-off across operating points
python scripts/sweep_operating_points.py --dataset chestxray14 --data-dir data

# Why particular clean images were flagged
python scripts/diagnose_false_positives.py --dataset chestxray14 --data-dir data

# Explainer backends compared on the same summary
python scripts/run_explainer.py --dataset chestxray14 --data-dir data --timeout 600 \
    --compare template,ollama:llama3.2:3b,ollama:qwen3:8b
```

---

## Findings that shaped the design

Several results contradicted the initial hypotheses and are reported as findings
rather than worked around. Each is encoded in a regression test so a future change
cannot silently alter a reported number.

- **Frozen deep embeddings add nothing measurable over classical IQA metrics.**
  Paired bootstrap over 1,681 matched ChestX-ray14 rows: classical metrics
  significantly better on 7 of 9 defects, no significant difference on 2,
  embeddings better on none. The marginal contribution of six embedding detectors
  to logistic fusion was +0.006 AUROC, not significant.
- **Fusion helps, but not through complementarity.** Logistic fusion beat the best
  single scorer on the aggregate task (+0.098, 95% CI [+0.072, +0.122]), yet
  classical metrics alone reached 0.909 of that 0.915. The mechanism is learned
  metric weighting with confound correction: the model assigns Laplacian variance
  a *negative* coefficient, independently recovering the noise confound.
- **Discriminative ability transfers across modalities; calibration does not.**
  Hence per-modality thresholds.
- **A 64-bit perceptual hash is unusable on stereotyped medical images.** 35 chest
  radiographs produced only 17 distinct hash values, collapsing precision to
  0.007. The default here is 256-bit, with a saturation diagnostic.
- **Perceptual hashing beats embedding similarity on mild transformation
  near-duplicates** (F1 1.000 vs 0.927 on ChestX-ray14) at a twentieth of the
  compute. Neither detects same-patient re-identification, which requires
  purpose-trained metric learning.
- **Quality rejection can destroy a rare class.** All three Hernia images in the
  ChestX-ray14 test split were rejected. The Summary Agent surfaces this tension
  rather than resolving it.

### Known limitations

- Severity-1 defects are detected at roughly 35% (ChestX-ray14) at a 5%
  false-positive operating point. Mild degradation largely passes inspection.
- ISIC has 77 clean validation images, so operating points below about 1.25%
  false-reject are not measurable there. This follows from balancing across 67
  diagnosis strata.
- Ground truth is synthetic. Injected defects may not represent the distribution
  of naturally occurring artefacts.
- `compression`/`motion_blur` and `blur`/`rotation` share scorers, so those
  attributions should be read as one finding each.
- SSIM is a poor calibration target for rotation (it measures misalignment) and
  for blur on heavily downsampled images.

---

## Tests

```bash
python -m pytest tests/ -v
```

Roughly 400 tests. They encode findings as well as behaviour: the noise confound,
the hash saturation threshold, the false-positive-rate compounding and the exact
occlusion statistics are all asserted, so a refactor cannot quietly change a
number that appears in the dissertation.

---

## Requirements

Python 3.10+. See `requirements.txt`. The Explainer's LLM backend needs
[Ollama](https://ollama.com) running locally with a model pulled:

```bash
ollama pull llama3.2:3b
```

The `template` backend requires no model and is the default; it is also the
zero-hallucination control against which any model's factual consistency is
measured.