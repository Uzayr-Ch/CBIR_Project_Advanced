# 01 — Project Overview

## What is CBIR?

**Content-Based Image Retrieval (CBIR)** is an AI-powered image search engine that finds visually similar images based on their actual content (colors, textures, shapes) — not metadata or text tags.

You upload an image, and it returns the most visually similar images from the database.

---

## Key Features (Updated)

| Feature | Description |
|:--------|:------------|
| **Visual Search** | Upload any image, get similar ones back |
| **Deep Learning** | Supports both ResNet50 (2048-D) and ViT-B/16 (768-D) for feature extraction |
| **Model Comparison** | Benchmark and compare ResNet50 vs ViT-B/16 on accuracy and speed |
| **Multi-Dataset** | Searches across 25,144 images from 4 datasets |
| **Fast Retrieval** | FAISS-based similarity search for instant results, especially with ViT features |
| **Web Interface** | Streamlit-based UI with model selection |
| **Modern Visualizations** | Publication-ready graphs, dashboards, and radar charts |

---

## System Architecture (Updated)

```
┌──────────────────────────────────────────────────────────────┐
│                    CBIR SYSTEM WORKFLOW                     │
└──────────────────────────────────────────────────────────────┘

  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
  │   Upload    │      │  ResNet50   │      │   2048-D    │
  │   Query     │ ───► │  or ViT-B/16│ ───► │  or 768-D   │
  │   Image     │      │  Extractor  │      │  Feature    │
  └─────────────┘      └─────────────┘      └─────────────┘
                                            │
                                            ▼
  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
  │  Display    │      │   Sort by   │      │   Cosine    │
  │   Top-K     │ ◄─── │  Similarity │ ◄─── │  Similarity │
  │  Results    │      │   Score     │      │  or FAISS   │
  └─────────────┘      └─────────────┘      └─────────────┘
```

### How it works (step by step):

1. **User uploads** a query image via the Streamlit UI
2. **Model selection:** ResNet50 (2048-D) or ViT-B/16 (768-D)
3. **L2 normalization** applied to the feature vector
4. **Cosine similarity or FAISS** computed against all pre-computed feature vectors
5. **Top-K results** sorted by similarity score and displayed with category info

---

## Datasets Used

| Dataset | Images | Categories |
|:--------|-------:|-----------:|
| Corel-1K | 1,000 | 10 |
| Corel-5K | 5,000 | 50 |
| Corel-10K | 10,000 | 100 |
| Caltech-101 | 9,144 | 102 |
| **Combined** | **25,144** | **~262** |

All features are merged into a single combined index for unified search.

---

## Work Completed (Updated)

| # | Task | Status |
|:-:|:-----|:------:|
| 1 | Virtual Environment Setup (Python 3.10.2) | ✅ |
| 2 | Dataset Collection (4 datasets) | ✅ |
| 3 | Feature Extraction Script (`src/extractor.py`, `src/extractor_vit.py`) | ✅ |
| 4 | Per-Dataset Feature Extraction (all 4, both models) | ✅ |
| 5 | Combined Feature Index (25,144 images) | ✅ |
| 6 | Web UI (`src/app.py` — Streamlit, model selection) | ✅ |
| 7 | Single Dataset Evaluation (`src/evaluate_system.py`) | ✅ |
| 8 | Batch Evaluation — all datasets (`src/batch_evaluate.py`) | ✅ |
| 9 | Model Comparison & Benchmarking (`src/compare_benchmarks.py`) | ✅ |
| 10 | Graph Generation (`src/generate_graphs.py`, comparison graphs) | ✅ |
| 11 | Data Validation (`src/validate_data.py`) | ✅ |
| 12 | FAISS Fast Search Integration | ✅ |
| 13 | Documentation Split & Cleanup | ✅ |

---

## Project Summary (New)

- **Supports both ResNet50 and ViT-B/16 for feature extraction and retrieval**
- **Model comparison framework**: accuracy, speed, and feature size
- **Modern, publication-ready visualizations**
- **FAISS integration for fast search**
- **Ready for research, demo, or production use**

---

*Next: [02 - Setup & Environment](02_setup.md)*
