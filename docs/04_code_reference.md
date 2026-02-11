# 04 — Code Reference

## File Summary

| File | Purpose | Input | Output |
|:-----|:--------|:------|:-------|
| `src/app.py` | Streamlit web UI | User-uploaded image | Top-K similar results |
| `src/extractor.py` | Feature extraction | Dataset images | `.pkl` feature files |
| `src/evaluate_system.py` | Single dataset eval | `features.pkl` | mAP score printed |
| `src/batch_evaluate.py` | All datasets eval | All dataset folders | Per-dataset metrics + `.pkl` |
| `src/generate_graphs.py` | Visualization | `evaluation_results.pkl` | PNG/PDF graphs |
| `src/validate_data.py` | Image validation | Dataset folder | Corrupt file report |

---

## `src/app.py` — Web Application

**Purpose:** Streamlit-based UI for searching similar images across all datasets.

**Key Components:**

| Component | Function |
|:----------|:---------|
| `load_data()` | Loads ResNet50 model + combined features (cached) |
| `preprocess` | Resize to 224×224 + ImageNet normalize |
| Sidebar | Image upload widget + Top-K slider (1-20) |
| Search Logic | Cosine similarity → sort → display Top-K |

**Data Source:** `features/combined_features.pkl` + `features/combined_paths.pkl` (25,144 images from all 4 datasets)

**How to run:**
```powershell
.\cbir_env\Scripts\Activate.ps1
streamlit run src/app.py
```

---

## `src/extractor.py` — Feature Extraction

**Purpose:** Extract ResNet50 features from a dataset and save to pickle files.

**Process Flow:**
```
Load ResNet50 (pretrained, no classification layer)
         ↓
Walk dataset directory → find all .jpg/.jpeg/.png
         ↓
For each image:
  → Resize to 224×224
  → Normalize (ImageNet mean/std)
  → Forward pass through ResNet50
  → Get 2048-D feature vector
  → L2 normalize
         ↓
Save to features/features.pkl + features/image_paths.pkl
```

**Output Files:**
- `features/features.pkl` — NumPy array shape (N, 2048)
- `features/image_paths.pkl` — List of N file paths

---

## `src/evaluate_system.py` — Single Dataset Evaluation

**Purpose:** Calculate mAP and Precision@K for the default feature set.

**Method:** Leave-One-Out evaluation
- Each image acts as a query (one at a time)
- For each query, find Top-K most similar images (excluding itself)
- Count how many belong to the same category
- Average precision across all queries = mAP

**Formulas:**
```
Precision@K = (Relevant in top K) / K
mAP = mean(all Precision@K values)
```

---

## `src/batch_evaluate.py` — Batch Evaluation

**Purpose:** Evaluate all 4 datasets in one run and generate comprehensive metrics.

**What it does:**
1. Loads ResNet50 model
2. For each dataset (Corel-1K, 5K, 10K, Caltech-101):
   - Extracts features (or loads from cache)
   - Saves per-dataset `.pkl` files
   - Runs leave-one-out evaluation
   - Calculates mAP, latency, precision stats
3. Saves all results to `features/evaluation_results.pkl`
4. Prints markdown table for documentation

**Output:**
- `features/{dataset}_features.pkl` — Feature vectors per dataset
- `features/{dataset}_paths.pkl` — Image paths per dataset
- `features/evaluation_results.pkl` — All metrics combined

---

## `src/generate_graphs.py` — Research Visualizations

**Purpose:** Generate publication-quality charts from evaluation results.

**Graphs Generated:**

| # | Graph | File | Description |
|:-:|:------|:-----|:------------|
| 1 | Scalability Line Graph | `scalability_graph.png/pdf` | mAP vs Dataset Size |
| 2 | Precision Bar Chart | `precision_comparison.png/pdf` | Side-by-side comparison |
| 3 | Latency-Accuracy Plot | `latency_accuracy.png/pdf` | Speed vs quality trade-off |
| 4 | Combined Metrics | `combined_metrics.png/pdf` | Summary dashboard |

**Output location:** `graphs/` folder (PNG for viewing, PDF at 300 DPI for papers)

---

## `src/validate_data.py` — Data Validation

**Purpose:** Detect corrupt or unreadable images in the dataset.

**Checks performed:**
- File is readable by PIL
- Image can be verified (`img.verify()`)
- Valid extension (.jpg, .jpeg, .png)

---

*Previous: [03 - Datasets](03_datasets.md) | Next: [05 - Results](05_results.md)*
