"""ViT-based feature extraction for CBIR.

This module processes large image collections using ViT-B/16 with batch
inference and L2-normalized embeddings. Results are written to pickle files
for downstream retrieval and FAISS indexing.
"""

import os
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from torchvision.models import ViT_B_16_Weights, vit_b_16

# Default datasets; override via CLI or by instantiating the class directly.
DATASETS: Dict[str, str] = {
    "Corel-1K": "dataset/Corel-1K",
    "Corel-5K": "dataset/Corel-5K",
    "Corel-10K": "dataset/Corel-10K",
    "Caltech-101": "dataset/caltech-101",
}


@dataclass
class ExtractionResult:
    features_path: str
    paths_path: str
    num_images: int
    num_failures: int


class ViTFeatureExtractor:
    """Batch feature extractor built on ViT-B/16.

    L2-normalized embeddings guarantee that $\|x\|_2 = 1$, so the dot product
    between two vectors equals their cosine similarity. This keeps retrieval
    consistent whether we use cosine distance or FAISS' L2 metric.
    """

    def __init__(
        self,
        dataset_dir: str,
        output_dir: str = "features",
        batch_size: int = 32,
        num_workers: int = 0,
        device: Optional[str] = None,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        # Remove the classification head to expose the 768-dim embedding.
        if hasattr(model.heads, "head"):
            model.heads.head = torch.nn.Identity()
        else:
            model.heads = torch.nn.Identity()

        self.model = model.to(self.device).eval()
        self.embedding_dim = 768
        # Use weights-native transforms for consistent normalization.
        self.preprocess = weights.transforms()

        os.makedirs(self.output_dir, exist_ok=True)
        self.failed: List[Tuple[str, str]] = []

    def _list_images(self) -> List[str]:
        image_paths: List[str] = []
        for root, _, files in os.walk(self.dataset_dir):
            for name in files:
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(root, name))
        return image_paths

    def _load_batch(self, paths: Sequence[str]) -> Tuple[Optional[torch.Tensor], List[str]]:
        tensors: List[torch.Tensor] = []
        valid_paths: List[str] = []
        for path in paths:
            try:
                img = Image.open(path).convert("RGB")
                tensors.append(self.preprocess(img))
                valid_paths.append(path)
            except Exception as exc:  # Corrupt or unreadable image
                self.failed.append((path, str(exc)))
        if not tensors:
            return None, []
        batch = torch.stack(tensors, dim=0)
        return batch, valid_paths

    def extract(self) -> Tuple[np.ndarray, List[str]]:
        image_paths = self._list_images()
        if not image_paths:
            raise FileNotFoundError(f"No images found in {self.dataset_dir}")

        all_features: List[np.ndarray] = []
        kept_paths: List[str] = []

        for start in tqdm(range(0, len(image_paths), self.batch_size), desc="ViT-B/16 extraction"):
            batch_paths = image_paths[start : start + self.batch_size]
            batch, valid_paths = self._load_batch(batch_paths)
            if batch is None:
                continue

            batch = batch.to(self.device, non_blocking=True)
            with torch.no_grad():
                embeds = self.model(batch)
                if embeds.ndim > 2:
                    # Safety fallback if the backend returns spatial features.
                    embeds = embeds.mean(dim=(2, 3))
                embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)  # Unit-length vectors
            all_features.append(embeds.cpu().numpy())
            kept_paths.extend(valid_paths)

        if not all_features:
            raise RuntimeError("No features extracted; all images may have failed to load.")

        features = np.vstack(all_features).astype(np.float32)
        return features, kept_paths

    def save(self, features: np.ndarray, paths: List[str], prefix: str) -> ExtractionResult:
        feats_path = os.path.join(self.output_dir, f"{prefix}_vit_features.pkl")
        paths_path = os.path.join(self.output_dir, f"{prefix}_vit_paths.pkl")

        with open(feats_path, "wb") as f:
            pickle.dump(features, f)
        with open(paths_path, "wb") as f:
            pickle.dump(paths, f)

        return ExtractionResult(
            features_path=feats_path,
            paths_path=paths_path,
            num_images=len(paths),
            num_failures=len(self.failed),
        )

    def run(self, prefix: str) -> ExtractionResult:
        feats_path = os.path.join(self.output_dir, f"{prefix}_vit_features.pkl")
        paths_path = os.path.join(self.output_dir, f"{prefix}_vit_paths.pkl")

        # Skip extraction if pickles already exist (resume-friendly).
        if os.path.exists(feats_path) and os.path.exists(paths_path):
            with open(feats_path, "rb") as f:
                features = pickle.load(f)
            with open(paths_path, "rb") as f:
                paths = pickle.load(f)
            self.failed = []
            return ExtractionResult(
                features_path=feats_path,
                paths_path=paths_path,
                num_images=len(paths),
                num_failures=0,
            )

        features, paths = self.extract()
        return self.save(features, paths, prefix)


def combine_feature_pickles(prefixes: Iterable[str], output_dir: str = "features") -> None:
    """Concatenate per-dataset pickles into a combined ViT bank."""
    feature_blocks: List[np.ndarray] = []
    path_blocks: List[List[str]] = []

    for name in prefixes:
        feats_file = os.path.join(output_dir, f"{name}_vit_features.pkl")
        paths_file = os.path.join(output_dir, f"{name}_vit_paths.pkl")
        if not (os.path.exists(feats_file) and os.path.exists(paths_file)):
            continue
        with open(feats_file, "rb") as f:
            feature_blocks.append(pickle.load(f))
        with open(paths_file, "rb") as f:
            path_blocks.append(pickle.load(f))

    if not feature_blocks:
        raise FileNotFoundError("No per-dataset ViT pickles found to combine.")

    combined_features = np.vstack(feature_blocks).astype(np.float32)
    combined_paths: List[str] = [p for block in path_blocks for p in block]

    with open(os.path.join(output_dir, "combined_vit_features.pkl"), "wb") as f:
        pickle.dump(combined_features, f)
    with open(os.path.join(output_dir, "combined_vit_paths.pkl"), "wb") as f:
        pickle.dump(combined_paths, f)


def _run_cli() -> None:
    print("=" * 60)
    print("ViT-B/16 Feature Extraction Pipeline")
    print("=" * 60)

    completed: List[str] = []
    for dataset_name, dataset_dir in DATASETS.items():
        if not os.path.exists(dataset_dir):
            print(f"[SKIP] {dataset_name}: path not found -> {dataset_dir}")
            continue

        print(f"\n[RUN] {dataset_name} ({dataset_dir})")
        extractor = ViTFeatureExtractor(dataset_dir=dataset_dir, batch_size=8)
        result = extractor.run(prefix=dataset_name)
        completed.append(dataset_name)

        print(f"   Saved: {result.features_path}")
        print(f"   Images: {result.num_images} | Failed: {result.num_failures}")

    if completed:
        print("\nCombining feature banks...")
        combine_feature_pickles(completed)
        print("   Saved: features/combined_vit_features.pkl")
        print("   Saved: features/combined_vit_paths.pkl")

    print("\n✅ ViT extraction complete.")


if __name__ == "__main__":
    _run_cli()
