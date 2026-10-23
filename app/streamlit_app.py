"""Streamlit interface for the Data Inspector: upload an image, get a decision.

Run with:

    streamlit run app/streamlit_app.py

**Scope, stated deliberately.** Thresholds are a property of a dataset, not of an
image - Stage 5 measured an eighteen-fold false-positive inflation when
ChestX-ray14 thresholds were applied to ISIC - so the interface asks which
dataset's thresholds to use and shows that choice on screen. It does not classify
the modality of an upload, because no such classifier exists in this system;
selecting the wrong one produces a confident and meaningless verdict, and the
interface says so rather than hiding it.

The composite fusion scorer is fitted per dataset during a batch run and is not
reconstructed here. Where it is unavailable the per-check attribution is still
shown, and the overall verdict is reported as unavailable rather than substituted
with a single metric dressed up as the composite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.inference import (  # noqa: E402
    InspectionContext,
    UnsupportedImageError,
    available_datasets,
    inspect_single_image,
    per_check_table,
)

DECISION_STYLE = {
    "accept": ("✅", "Accepted", "This image passed every enabled check."),
    "flag": (
        "⚠️", "Flagged for review",
        "A human should look at this image; it was not discarded.",
    ),
    "reject": (
        "❌", "Rejected",
        "The composite quality score exceeded the rejection threshold.",
    ),
}


st.set_page_config(page_title="Medical Image Quality Inspector", layout="wide")
st.title("Medical imaging dataset quality inspector")
st.caption(
    "Upload a single image to see the accept / flag / reject decision and the "
    "evidence behind it."
)

data_dir = Path(st.sidebar.text_input("Data directory", value="data"))
datasets = available_datasets(data_dir)

if not datasets:
    st.error(
        f"No dataset with completed thresholds was found under `{data_dir}`. "
        "Single-image inspection reuses thresholds derived from a batch run, so "
        "run the pipeline first:\n\n"
        "```\npython scripts/run_pipeline.py --dataset chestxray14 --data-dir data\n```"
    )
    st.stop()

dataset = st.sidebar.selectbox("Threshold source dataset", datasets)
st.sidebar.warning(
    "Thresholds do not transfer between modalities. A ChestX-ray14 threshold "
    "applied to dermoscopy produced an eighteen-fold false-positive inflation in "
    "testing, so select the dataset matching your image."
)

try:
    context = InspectionContext.load(dataset, data_dir)
except (FileNotFoundError, KeyError) as error:
    st.error(str(error))
    st.stop()

with st.sidebar.expander("Threshold provenance"):
    st.write(f"**Dataset:** {context.dataset}")
    st.write(f"**Batch run:** `{context.inspection_run_id}`")
    st.write(f"**Decision policy:** `{context.config.fpr_policy}`")
    st.write(
        f"**Brightness reference:** {context.brightness_reference.reference_mean:.1f} "
        f"(from {context.brightness_reference.n_images} clean images)"
    )

upload = st.file_uploader(
    "Image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    label_visibility="collapsed",
)

if upload is None:
    st.info("Upload an image to begin.")
    st.stop()

try:
    result, canonical = inspect_single_image(upload, context)
except UnsupportedImageError as error:
    st.error(f"Could not inspect this image: {error}")
    st.stop()

# ---------------------------------------------------------------- verdict
icon, label, explanation = DECISION_STYLE[result["decision"]]
left, right = st.columns([1, 2])

with left:
    st.image(canonical, caption=f"Canonicalised to {result['canonical_size']}px")
    source = result["source"]
    st.caption(
        f"Source: {source['original_width']}×{source['original_height']} "
        f"({source['megapixels']:.2f} MP), mode {source['original_mode']}"
    )
    if source["upscaled_to_canonical"]:
        st.warning(
            "This image was **upscaled** to the canonical size, so its apparent "
            "detail is interpolation rather than measurement. Sharpness-based "
            "checks should be read with that in mind."
        )

with right:
    st.header(f"{icon} {label}")
    st.write(explanation)

    confidence = result.get("confidence", {})
    band = confidence.get("band")
    if band == "borderline":
        st.warning(
            "This was a **borderline** call: the score sat close to the "
            "threshold, so the verdict could reasonably have gone either way. "
            "Borderline cases are where human review is most worthwhile."
        )
    elif band == "clear":
        distance = confidence.get("distance_from_boundary")
        st.success(
            "This was a **clear-cut** decision"
            + (f", {abs(distance):.1f} threshold-spans from the boundary."
               if distance is not None else ".")
        )
    elif band == "unscored":
        st.info(
            "No composite score was available, so no confidence can be stated. "
            "This is **not** the same as a confident acceptance - the per-check "
            "results below still apply."
        )

    if result.get("primary_issue"):
        st.metric("Primary issue", result["primary_issue"].replace("_", " "))
    flagged = result.get("flagged_checks", []) + result.get(
        "attribution_only_checks", []
    )
    if flagged:
        st.write("**Checks that fired:** " + ", ".join(flagged))

# ------------------------------------------------------------ per-check
st.subheader("Per-check detail")
st.caption(
    "Every check reports its score beside the threshold that produced the "
    "verdict, so a decision can be audited without re-running anything."
)
table = per_check_table(result)
if table.empty:
    st.info("No checks were assessable for this image.")
else:
    st.dataframe(
        table.style.format(
            {"score": "{:.4f}", "flag threshold": "{:.4f}"}, na_rep="—"
        ),
        use_container_width=True, hide_index=True,
    )

if result.get("attribution_only_checks"):
    st.caption(
        "Under the `composite_only` policy, individual checks are reported for "
        "attribution but do not themselves change the decision - nine checks each "
        "firing at a 5% false-positive rate would compound to roughly 37%."
    )

if result.get("structural_findings"):
    st.warning(
        "**Structural findings:** " + ", ".join(result["structural_findings"])
        + ". These are curation concerns rather than quality defects, so they do "
        "not affect the quality decision."
    )

# ----------------------------------------------------------- raw scores
with st.expander("Raw metric scores"):
    scores = pd.DataFrame(
        sorted(result["raw_scores"].items()), columns=["metric", "value"]
    )
    st.dataframe(
        scores.style.format({"value": "{:.4f}"}),
        use_container_width=True, hide_index=True,
    )

with st.expander("Full JSON output"):
    st.caption(
        "The same structure the Dataset Summary Agent consumes in the batch "
        "pipeline."
    )
    st.json(result)

st.divider()
st.caption(
    "Decisions use thresholds derived from clean validation images of "
    f"**{context.dataset}** at a stated false-positive rate. Ground truth in this "
    "system comes from synthetically injected defects, which may not represent "
    "the distribution of naturally occurring artefacts. This tool recommends; it "
    "does not modify any dataset."
)