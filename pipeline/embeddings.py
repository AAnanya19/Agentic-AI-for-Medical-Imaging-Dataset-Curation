"""Frozen-backbone feature extraction for embedding-based anomaly detection.

Two backbones, both frozen (no fine-tuning), both used as pure feature
extractors:

  * **resnet18** (512-d) - a conventional supervised ImageNet baseline.
  * **dinov2_vits14** (384-d) - self-supervised ViT-S/14 features. Included
    because the frozen-DINO-features line of work is what this stage is
    replicating in a medical-imaging setting.

Both are loaded through ``timm`` rather than ``torchvision``/``torch.hub`` so
that input size, interpolation, and channel normalisation all come from each
model's own published data config instead of being hand-copied and silently
drifting from what the weights expect.

Two input-size decisions worth knowing about:

  * DINOv2 ViT-S/14 defaults to 518x518, which costs roughly 1 s/image on CPU.
    It is instead run at **252 = 14 x 18**, the largest patch-aligned size at or
    below our canonical 256, which measured ~205 ms/image - the difference
    between ~25 minutes and ~2 hours across both datasets. Patch alignment
    matters for a ViT; 252 is exact rather than relying on timm's dynamic
    position-embedding interpolation.
  * Canonical images are **resized** (not centre-cropped) to each backbone's
    expected input. A centre crop would clip ``occlusion_patch`` defects that
    sit near the frame edge, silently changing their effective severity - and
    occlusion is precisely the defect this stage is expected to be good at.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import timm
    import torch
    from PIL import Image
except ImportError as error:  # pragma: no cover - environment guard
    raise ImportError(
        "timm, torch and Pillow are required for pipeline.embeddings"
    ) from error


@dataclass(frozen=True)
class BackboneSpec:
    """One frozen feature extractor: which timm model, at what input size."""

    timm_name: str
    input_size: int
    expected_dim: int
    note: str = ""


BACKBONES: dict[str, BackboneSpec] = {
    "resnet18": BackboneSpec(
        timm_name="resnet18",
        input_size=224,
        expected_dim=512,
        note="supervised ImageNet baseline; native config is 224",
    ),
    "dinov2_vits14": BackboneSpec(
        timm_name="vit_small_patch14_dinov2.lvd142m",
        # 252 = 14 * 18: patch-aligned, and ~5x faster than the 518 default.
        input_size=252,
        expected_dim=384,
        note="self-supervised ViT-S/14; run at 252 rather than its 518 default",
    ),
}


def available_backbones() -> tuple[str, ...]:
    return tuple(BACKBONES)


class FeatureExtractor:
    """A frozen timm backbone plus its own preprocessing, in eval mode.

    ``num_classes=0`` makes timm return pooled penultimate features rather than
    logits, which is what the anomaly detectors consume.
    """

    def __init__(self, backbone: str, device: str = "cpu", pretrained: bool = True) -> None:
        if backbone not in BACKBONES:
            raise ValueError(
                f"unknown backbone {backbone!r}; available: {sorted(BACKBONES)}"
            )
        self.backbone = backbone
        self.spec = BACKBONES[backbone]
        self.device = torch.device(device)

        kwargs: dict[str, Any] = {"pretrained": pretrained, "num_classes": 0}
        if "vit" in self.spec.timm_name:
            # A ViT bakes its input resolution into its position embeddings, so
            # the non-default size must be declared at construction time.
            kwargs["img_size"] = self.spec.input_size
        self.model = timm.create_model(self.spec.timm_name, **kwargs)
        self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)  # frozen: never fine-tuned

        config = timm.data.resolve_data_config({}, model=self.model)
        self.mean = np.asarray(config["mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(config["std"], dtype=np.float32).reshape(3, 1, 1)

    def describe(self) -> dict[str, Any]:
        """Provenance for the output manifest: exactly what produced the features."""
        return {
            "backbone": self.backbone,
            "timm_name": self.spec.timm_name,
            "input_size": self.spec.input_size,
            "expected_dim": self.spec.expected_dim,
            "normalisation_mean": self.mean.reshape(-1).tolist(),
            "normalisation_std": self.std.reshape(-1).tolist(),
            "resize_policy": "resize (not centre-crop), to preserve edge occlusions",
            "note": self.spec.note,
        }

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """PIL image -> normalised CHW float32 array at the backbone's size.

        Grayscale is replicated to 3 channels because both backbones were
        pretrained on RGB; ChestX-ray14 processed images are mode ``L``.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        size = self.spec.input_size
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return (array - self.mean) / self.std

    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Extract features for a batch, returning an (N, D) float32 array."""
        if not images:
            return np.zeros((0, self.spec.expected_dim), dtype=np.float32)
        batch = np.stack([self.preprocess(image) for image in images])
        tensor = torch.from_numpy(batch).to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
        array = features.detach().cpu().numpy().astype(np.float32)
        if array.ndim != 2:
            raise RuntimeError(
                f"expected pooled 2-D features from {self.backbone}, got shape {array.shape}"
            )
        if array.shape[1] != self.spec.expected_dim:
            raise RuntimeError(
                f"{self.backbone} produced {array.shape[1]}-d features, "
                f"expected {self.spec.expected_dim}-d"
            )
        return array


def save_embeddings(
    path: Path,
    image_ids: list[str],
    embeddings: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Cache embeddings to a compressed .npz alongside their provenance.

    Extraction is the expensive step in this stage, so it is done once and
    reused by every detector and by Stage 4's fusion.
    """
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image_ids=np.asarray(image_ids, dtype=object),
        embeddings=embeddings,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    """Load a cached .npz produced by save_embeddings."""
    import json

    with np.load(path, allow_pickle=True) as payload:
        image_ids = [str(value) for value in payload["image_ids"]]
        embeddings = payload["embeddings"]
        metadata = json.loads(str(payload["metadata_json"]))
    return image_ids, embeddings, metadata