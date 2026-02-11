import os
import pickle
from typing import Dict, Tuple

import numpy as np
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, vit_b_16

from vector_engine import FaissVectorEngine

# Set working directory to project root (so relative paths always work)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

MODEL_CONFIG: Dict[str, Dict[str, str]] = {
    "ResNet50": {
        "features": "features/combined_features.pkl",
        "paths": "features/combined_paths.pkl",
        "faiss_flat": "features/faiss_resnet_flat.index",
        "faiss_ivf": "features/faiss_resnet_ivf.index",
    },
    "ViT-B/16": {
        "features": "features/combined_vit_features.pkl",
        "paths": "features/combined_vit_paths.pkl",
        "faiss_flat": "features/faiss_vit_flat.index",
        "faiss_ivf": "features/faiss_vit_ivf.index",
    },
}


# 1️⃣ Page Setup
st.set_page_config(page_title="CBIR Search Engine", layout="wide")
st.title("🔍 Content-Based Image Retrieval")
st.write("All Datasets (Corel-1K + Corel-5K + Corel-10K + Caltech-101) — 25,144 Images")


# 2️⃣ Loaders (cached)
@st.cache_resource
def load_backbone(model_name: str) -> Tuple[torch.nn.Module, transforms.Compose]:
    if model_name == "ResNet50":
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        preprocess = weights.transforms()
    else:
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        if hasattr(model.heads, "head"):
            model.heads.head = torch.nn.Identity()
        else:
            model.heads = torch.nn.Identity()
        preprocess = weights.transforms()
    model.eval()
    return model, preprocess


@st.cache_resource
def load_feature_bank(model_name: str) -> Tuple[np.ndarray, list[str]]:
    cfg = MODEL_CONFIG[model_name]
    with open(cfg["features"], "rb") as f:
        feats = pickle.load(f)
    with open(cfg["paths"], "rb") as f:
        paths = pickle.load(f)
    return feats, paths


def _estimate_nlist(num_vectors: int) -> int:
    return int(np.clip(np.sqrt(num_vectors), 32, 4096))


@st.cache_resource
def load_faiss_engine(model_name: str, index_type: str) -> FaissVectorEngine:
    cfg = MODEL_CONFIG[model_name]
    index_path = cfg["faiss_flat"] if index_type == "flat" else cfg["faiss_ivf"]

    if os.path.exists(index_path):
        return FaissVectorEngine.load_index(index_path)

    features, _ = load_feature_bank(model_name)
    nlist = _estimate_nlist(len(features)) if index_type == "ivf" else 256
    engine = FaissVectorEngine(dim=features.shape[1], index_type=index_type, nlist=nlist)
    engine.build_index(features)
    engine.save_index(index_path)
    return engine


# 3️⃣ UI Sidebar
st.sidebar.header("Upload Image")

available_models = [name for name, cfg in MODEL_CONFIG.items() if os.path.exists(cfg["features"])]
if not available_models:
    st.sidebar.error("No feature bank found. Run the extractors first.")
    st.stop()

model_choice = st.sidebar.selectbox("Backbone", options=available_models, index=0)

backend_choice = st.sidebar.radio(
    "Search backend",
    options=["FAISS (Flat L2)", "FAISS (IVF Flat)", "Cosine (sklearn)"]
)

uploaded_file = st.sidebar.file_uploader("Milti julti images dhoondne ke liye upload karein", type=["jpg", "png", "jpeg"])
top_k = st.sidebar.slider("Top Results", 1, 20, 5)


if uploaded_file and model_choice:
    query_img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Your Image")
        st.image(query_img, width="stretch")

    if st.button("Start Search"):
        with st.spinner("Dhoond raha hoon..."):
            model, preprocess = load_backbone(model_choice)
            feature_bank, image_paths = load_feature_bank(model_choice)

            input_tensor = preprocess(query_img).unsqueeze(0)
            with torch.no_grad():
                query_feat = model(input_tensor)
                if query_feat.ndim > 2:
                    query_feat = query_feat.mean(dim=(2, 3))
                query_feat = query_feat.squeeze().numpy().reshape(1, -1)
            query_feat = query_feat / np.linalg.norm(query_feat)

            if backend_choice.startswith("FAISS"):
                index_type = "flat" if "Flat L2" in backend_choice else "ivf"
                engine = load_faiss_engine(model_choice, index_type)
                result = engine.search(query_feat.astype(np.float32), k=top_k)
                similarities = 1.0 - (result.distances.flatten() / 2.0)
                indices = result.indices.flatten()
                latency_ms = result.latency_ms
            else:
                similarities = cosine_similarity(query_feat, feature_bank).flatten()
                indices = np.argsort(similarities)[::-1][:top_k]
                latency_ms = 0.0

            with col2:
                st.subheader("Search Results")
                cols = st.columns(3)
                for i, idx in enumerate(indices[:top_k]):
                    with cols[i % 3]:
                        full_path = image_paths[idx]
                        category = os.path.basename(os.path.dirname(full_path))
                        parts = full_path.replace("\\", "/").split("/")
                        dataset_name = parts[1] if len(parts) > 2 else "Unknown"

                        st.image(full_path, width="stretch")
                        st.caption(
                            f"📁 {dataset_name} / {category} | Score: {similarities[i]:.2f}"
                        )

                if backend_choice.startswith("FAISS"):
                    st.info(f"Average search latency: {latency_ms:.2f} ms ({backend_choice})")
