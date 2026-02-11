# Content-Based Image Retrieval (CBIR)

> AI-powered image search engine using deep learning — finds visually similar images across 25,144 images from 4 benchmark datasets.

---

## Quick Start

```powershell
# 1. Activate environment
.\cbir_env\Scripts\Activate.ps1

# 2. Launch web app
streamlit run src/app.py
```

Open http://localhost:8501 → Upload an image → Get similar results.

---

## Project Structure

```
CBIR_Project/
│
├── src/                            # Source Code
│   ├── app.py                      # Streamlit Web Application
│   ├── extractor.py                # Feature Extraction (ResNet50)
│   ├── evaluate_system.py          # Single Dataset Evaluation
│   ├── batch_evaluate.py           # Batch Evaluation (All Datasets)
│   ├── generate_graphs.py          # Research Visualization Generator
│   └── validate_data.py            # Dataset Validation
│
├── docs/                           # Documentation (split into 10 files)
│   ├── README.md                   # Documentation index
│   ├── 01_overview.md              # Project overview
│   ├── 02_setup.md                 # Setup & environment
│   ├── 03_datasets.md              # Dataset descriptions
│   ├── 04_code_reference.md        # Code file explanations
│   ├── 05_results.md               # Evaluation results & analysis
│   ├── 06_graphs.md                # Graphs & visualizations
│   ├── 07_model_theory.md          # ResNet50, cosine similarity, mAP
│   ├── 08_usage_guide.md           # How to run everything
│   ├── 09_troubleshooting.md       # Common issues & fixes
│   └── 10_references.md            # Academic references
│
├── dataset/                        # Image Datasets
│   ├── Corel-1K/                   # 10 categories × 100 images
│   ├── Corel-5K/                   # 50 categories × 100 images
│   ├── Corel-10K/                  # 100 categories × 100 images
│   └── caltech-101/                # 102 categories × ~40-800 images
│
├── features/                       # Pre-computed Features (.pkl)
│   ├── combined_features.pkl       # All datasets merged (25,144 × 2048)
│   ├── combined_paths.pkl          # All image paths
│   ├── Corel-1K_features.pkl       # Per-dataset features
│   ├── Corel-5K_features.pkl
│   ├── Corel-10K_features.pkl
│   ├── Caltech-101_features.pkl
│   └── evaluation_results.pkl      # Evaluation metrics
│
├── graphs/                         # Generated Visualizations (PNG + PDF)
│
├── cbir_env/                       # Python Virtual Environment
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

---

## Results

| Dataset | Images | Categories | mAP (%) | Latency (ms) |
|:--------|-------:|-----------:|--------:|-------------:|
| Corel-1K | 1,000 | 10 | **97.63** | 9.70 |
| Corel-5K | 5,000 | 50 | **51.90** | 36.95 |
| Caltech-101 | 9,144 | 102 | **85.20** | 65.99 |
| Corel-10K | 10,000 | 100 | **81.92** | 77.47 |

---

## Important Note: Local Data

> **Dataset and Features Not Included**  
> The `dataset/` (25K+ images) and `features/` (.pkl files ~500MB+) directories are **kept local** for privacy and repository size reasons. They are excluded via `.gitignore`.

**To run this project locally:**
1. Download the datasets (Corel-1K, Corel-5K, Corel-10K, Caltech-101) and place them in `dataset/`
2. Run feature extraction: `python src/extractor.py` (ResNet50) or `python src/extractor_vit.py` (ViT-B/16)
3. The system will generate the required `.pkl` feature files automatically

The code is fully ready for classification and retrieval tasks once you provide your own image data.

---

## Commands

| Action | Command |
|:-------|:--------|
| Start Web App | `streamlit run src/app.py` |
| Extract Features | `python src/extractor.py` |
| Evaluate All Datasets | `python src/batch_evaluate.py` |
| Generate Graphs | `python src/generate_graphs.py` |
| Validate Images | `python src/validate_data.py` |

> Always activate venv first: `.\cbir_env\Scripts\Activate.ps1`

---

## Tech Stack

| Component | Technology |
|:----------|:-----------|
| Models | ResNet50, ViT-B/16 (ImageNet pretrained) |
| Features | 2048-D (ResNet), 768-D (ViT), L2 normalized |
| Similarity | Cosine Similarity |
| Indexing | FAISS (optional fast search) |
| Backend | PyTorch, scikit-learn, timm |
| Frontend | Streamlit |
| Language | Python 3.10 |

---

## Documentation

Full documentation is in the [`docs/`](docs/README.md) folder, split into 10 readable sections.

---

*February 2026*
