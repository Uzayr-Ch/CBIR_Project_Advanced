# 08 — Usage Guide

## Quick Start

```powershell
# 1. Navigate to project
cd z:\Project\CBIR_Project

# 2. Activate virtual environment
.\cbir_env\Scripts\Activate.ps1

# 3. Run the web app
streamlit run src/app.py
```

Open `http://localhost:8501` in your browser.

---

## All Commands

| Action | Command |
|:-------|:--------|
| **Start Web App** | `streamlit run src/app.py` |
| Extract Features | `python src/extractor.py` |
| Evaluate Single Dataset | `python src/evaluate_system.py` |
| Batch Evaluate All | `python src/batch_evaluate.py` |
| Generate Graphs | `python src/generate_graphs.py` |
| Validate Images | `python src/validate_data.py` |

> Always activate the virtual environment first: `.\cbir_env\Scripts\Activate.ps1`

---

## Using the Web App

1. Open `http://localhost:8501` after running `streamlit run app.py`
2. Use the **sidebar** to upload a query image (JPG, PNG, JPEG)
3. Adjust the **Top Results** slider (1-20)
4. Click **Start Search**
5. Results show:
   - Similar images from all 4 datasets
   - Dataset name + category for each result
   - Similarity score (0 to 1)

---

## Running Evaluation

### Option A: Single dataset (Corel-10K only)
```powershell
python src/evaluate_system.py
```
Uses `features/features.pkl` — the legacy Corel-10K features.

### Option B: All datasets at once
```powershell
python src/batch_evaluate.py
```
Evaluates Corel-1K, 5K, 10K, and Caltech-101. Saves results to `features/evaluation_results.pkl`.

---

## Generating Graphs

```powershell
# Run batch evaluation first (if not already done)
python src/batch_evaluate.py

# Then generate graphs
python src/generate_graphs.py
```

Graphs are saved in `graphs/` folder (PNG + PDF).

---

## Re-extracting Features

If you add new images or modify the dataset:

```powershell
# Extract for a single dataset (edit src/extractor.py path first)
python src/extractor.py

# Or run batch_evaluate.py (extracts + evaluates all)
python src/batch_evaluate.py
```

To rebuild the combined index after re-extraction, the combine script needs to be run again (features from all 4 datasets are merged into `combined_features.pkl`).

---

*Previous: [07 - Model & Theory](07_model_theory.md) | Next: [09 - Troubleshooting](09_troubleshooting.md)*
