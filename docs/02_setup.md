# 02 — Setup & Environment

## Virtual Environment

_________________________________________
|    Setting   |         Value          |
|:-------------|:-----------------------|
| **Python**   |         3.10.2         |
| **Location** |       `cbir_env/`      |
| **Isolated** | Yes (project-specific) |
|______________|________________________|

---

## Activation Commands

```powershell
# Windows PowerShell
.\cbir_env\Scripts\Activate.ps1

# Windows CMD
.\cbir_env\Scripts\activate.bat

# Linux / macOS
source cbir_env/bin/activate
```

> If activation fails on PowerShell, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

## Installed Packages

| Package | Version | Purpose |
|:--------|:--------|:--------|
| `torch` | 2.10.0+cpu | Deep learning framework |
| `torchvision` | 0.25.0+cpu | Pre-trained models (ResNet50) |
| `streamlit` | 1.54.0 | Web UI framework |
| `numpy` | 2.2.6 | Numerical operations |
| `pillow` | 12.1.0 | Image loading & processing |
| `scikit-learn` | - | Cosine similarity computation |
| `tqdm` | 4.67.1 | Progress bars |
| `matplotlib` | 3.10.8 | Graph generation |

---

## System Requirements

| Component | Minimum | Recommended |
|:----------|:--------|:------------|
| **OS** | Windows 10, Linux, macOS | Windows 10+ |
| **Python** | 3.8 | 3.10+ |
| **RAM** | 4 GB | 8 GB |
| **Storage** | 2 GB | 5 GB |
| **GPU** | Not required | Optional (faster extraction) |

---

## Fresh Install (from scratch)

```powershell
# 1. Create virtual environment
python -m venv cbir_env

# 2. Activate it
.\cbir_env\Scripts\Activate.ps1

# 3. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install streamlit numpy pillow scikit-learn tqdm matplotlib
```

---

*Previous: [01 - Overview](01_overview.md) | Next: [03 - Datasets](03_datasets.md)*
