# 01 — Project Overview

## What is CBIR?

**Content-Based Image Retrieval (CBIR)** is an AI-powered image search engine that finds visually similar images based on their actual content (colors, textures, shapes) — not metadata or text tags.

You upload an image, and it returns the most visually similar images from the database.

---

## Key Features

| Feature | Description |
|:--------|:------------|
| **Visual Search** | Upload any image, get similar ones back |
| **Deep Learning** | ResNet50 extracts 2048-D semantic features |
| **Multi-Dataset** | Searches across 25,144 images from 4 datasets |
| **Fast Retrieval** | Pre-computed features = instant results |
| **Web Interface** | Streamlit-based UI |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        CBIR SYSTEM WORKFLOW                       │
└──────────────────────────────────────────────────────────────────┘

  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
  │   Upload    │      │  ResNet50   │      │   2048-D    │
  │   Query     │ ───► │  Extract    │ ───► │   Feature   │
  │   Image     │      │  Features   │      │   Vector    │
  └─────────────┘      └─────────────┘      └──────┬──────┘
                                                   │
                                                   ▼
  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
  │  Display    │      │   Sort by   │      │   Cosine    │
  │   Top-K     │ ◄─── │  Similarity │ ◄─── │  Similarity │
  │  Results    │      │   Score     │      │   Compute   │
  └─────────────┘      └─────────────┘      └─────────────┘
```

### How it works (step by step):

1. **User uploads** a query image via the Streamlit UI
2. **ResNet50** processes the image → produces a 2048-dimensional feature vector
3. **L2 normalization** applied to the feature vector
4. **Cosine similarity** computed against all 25,144 pre-computed feature vectors
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

## Work Completed

| # | Task | Status |
|:-:|:-----|:------:|
| 1 | Virtual Environment Setup (Python 3.10.2) | ✅ |
| 2 | Dataset Collection (4 datasets) | ✅ |
| 3 | Feature Extraction Script (`src/extractor.py`) | ✅ |
| 4 | Per-Dataset Feature Extraction (all 4) | ✅ |
| 5 | Combined Feature Index (25,144 images) | ✅ |
| 6 | Web UI (`src/app.py` — Streamlit) | ✅ |
| 7 | Single Dataset Evaluation (`src/evaluate_system.py`) | ✅ |
| 8 | Batch Evaluation — all datasets (`src/batch_evaluate.py`) | ✅ |
| 9 | Graph Generation (`src/generate_graphs.py`) | ✅ |
| 10 | Data Validation (`src/validate_data.py`) | ✅ |
| 11 | Documentation Split & Cleanup | ✅ |

---

*Next: [02 - Setup & Environment](02_setup.md)*
