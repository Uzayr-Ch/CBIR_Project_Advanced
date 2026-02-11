# CBIR Project Documentation

> Content-Based Image Retrieval System using Deep Learning (ResNet50)

---

## Quick Navigation

| Document | Description |
|:---------|:------------|
| [01 - Project Overview](01_overview.md) | Introduction, features, architecture |
| [02 - Setup & Environment](02_setup.md) | Installation, venv, packages |
| [03 - Datasets](03_datasets.md) | All 4 datasets described |
| [04 - Code Reference](04_code_reference.md) | Every script explained |
| [05 - Evaluation Results](05_results.md) | mAP, Precision@10, latency |
| [06 - Graphs & Visualizations](06_graphs.md) | Generated charts info |
| [07 - Model & Theory](07_model_theory.md) | ResNet50, cosine similarity, mAP |
| [08 - Usage Guide](08_usage_guide.md) | How to run everything |
| [09 - Troubleshooting](09_troubleshooting.md) | Common issues & fixes |
| [10 - References](10_references.md) | Academic sources |

---

## Project Summary

| Field | Value |
|:------|:------|
| **Project** | Content-Based Image Retrieval (CBIR) |
| **Domain** | Computer Vision / Deep Learning |
| **Language** | Python 3.10 |
| **Framework** | PyTorch + Streamlit |
| **Model** | ResNet50 (ImageNet pretrained) |
| **Total Images** | 25,144 (combined across 4 datasets) |
| **Best mAP** | 97.63% (Corel-1K) |
| **Date** | February 2026 |

---

## Directory Structure

```
CBIR_Project/
│
├── src/                            # Source Code
│   ├── app.py                      # Streamlit Web Application (all datasets)
│   ├── extractor.py                # Feature Extraction (ResNet50)
│   ├── evaluate_system.py          # Single Dataset Evaluation
│   ├── validate_data.py            # Image Validation Script
│   ├── batch_evaluate.py           # Batch Evaluation (All Datasets)
│   └── generate_graphs.py          # Research Visualization Generator
│
├── docs/                           # Documentation (you are here)
│   ├── README.md                   # This file — Table of Contents
│   ├── 01_overview.md              # Project overview
│   ├── 02_setup.md                 # Setup & environment
│   ├── 03_datasets.md              # Dataset descriptions
│   ├── 04_code_reference.md        # Code file explanations
│   ├── 05_results.md               # Evaluation results
│   ├── 06_graphs.md                # Graphs & visualizations
│   ├── 07_model_theory.md          # Model & theory concepts
│   ├── 08_usage_guide.md           # How to use the system
│   ├── 09_troubleshooting.md       # Common issues
│   └── 10_references.md            # Academic references
│
├── cbir_env/                       # Python Virtual Environment
│   ├── pyvenv.cfg                  # Python 3.10.2 config
│   ├── Scripts/                    # Executables (activate, pip)
│   └── Lib/site-packages/          # Installed packages
│
├── features/                       # Pre-computed Features
│   ├── combined_features.pkl       # All datasets merged (25,144 × 2048)
│   ├── combined_paths.pkl          # All image paths (25,144)
│   ├── features.pkl                # Legacy Corel-10K features
│   ├── image_paths.pkl             # Legacy Corel-10K paths
│   ├── Corel-1K_features.pkl       # Per-dataset features
│   ├── Corel-1K_paths.pkl          #   └── Corel-1K paths
│   ├── Corel-5K_features.pkl       # Per-dataset features
│   ├── Corel-5K_paths.pkl          #   └── Corel-5K paths
│   ├── Corel-10K_features.pkl      # Per-dataset features
│   ├── Corel-10K_paths.pkl         #   └── Corel-10K paths
│   ├── Caltech-101_features.pkl    # Per-dataset features
│   ├── Caltech-101_paths.pkl       #   └── Caltech-101 paths
│   └── evaluation_results.pkl      # All evaluation metrics
│
├── graphs/                         # Generated Visualizations
│   ├── scalability_graph.png/pdf   # mAP vs Dataset Size
│   ├── precision_comparison.png/pdf# Bar chart comparison
│   ├── latency_accuracy.png/pdf    # Speed vs accuracy
│   └── combined_metrics.png/pdf    # Summary dashboard
│
├── cbir_env/                       # Python Virtual Environment
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
├── README.md                       # Project README
│
└── dataset/                        # Image Datasets
    ├── Corel-1K/                   # 10 categories × 100 images
    ├── Corel-5K/                   # 50 categories × 100 images
    ├── Corel-10K/                  # 100 categories × 100 images
    └── caltech-101/                # 102 categories × ~40-800 images
```

---

*Last Updated: February 2026*
