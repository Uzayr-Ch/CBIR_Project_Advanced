# 03 — Datasets

## Overview

| Dataset | Categories | Total Images | Images/Class | Type |
|:--------|:----------:|-------------:|:------------:|:-----|
| Corel-1K | 10 | 1,000 | 100 | Nature/Objects |
| Corel-5K | 50 | 5,000 | 100 | Nature/Objects |
| Corel-10K | 100 | 10,000 | 100 | Nature/Objects |
| Caltech-101 | 102 | 9,144 | ~40-800 | Object Recognition |
| **Combined** | **~262** | **25,144** | — | **All merged** |

All datasets are stored in `dataset/` folder.

---

## Corel-1K

- **Path:** `dataset/Corel-1K/`
- **Purpose:** Quick testing and benchmarking
- **Structure:** 10 folders, each with 100 images

| # | Category | Images |
|:-:|:---------|:------:|
| 1 | africans | 100 |
| 2 | beaches | 100 |
| 3 | buildings | 100 |
| 4 | buses | 100 |
| 5 | dinosaurs | 100 |
| 6 | elephants | 100 |
| 7 | flowers | 100 |
| 8 | food | 100 |
| 9 | horses | 100 |
| 10 | mountains | 100 |

---

## Corel-5K

- **Path:** `dataset/Corel-5K/`
- **Purpose:** Medium-scale evaluation
- **Structure:** 50 numbered folders (0-49), each with 100 images
- **Image Format:** JPEG

---

## Corel-10K

- **Path:** `dataset/Corel-10K/`
- **Purpose:** Large-scale evaluation — primary benchmark
- **Structure:** 100 numbered folders (0-99), each with 100 images
- **Image Format:** JPEG

---

## Caltech-101

- **Path:** `dataset/caltech-101/`
- **Purpose:** Object recognition benchmark with high category diversity
- **Structure:** 102 named folders (accordion, airplanes, anchor, ant, etc.)
- **Images per class:** Varies significantly (40 to 800)
- **Image Format:** JPG

Notable categories include: airplanes, butterfly, camera, elephant, Faces, helicopter, laptop, Motorbikes, pizza, sunflower, watch, etc.

---

## Combined Feature Index

All 4 datasets are merged into a single retrieval index:

| File | Contents | Shape |
|:-----|:---------|:------|
| `features/combined_features.pkl` | All feature vectors | (25,144 × 2048) |
| `features/combined_paths.pkl` | All image paths | 25,144 entries |

**Per-dataset feature files are preserved separately** for individual evaluation:

| File | Dataset |
|:-----|:--------|
| `features/Corel-1K_features.pkl` / `_paths.pkl` | Corel-1K |
| `features/Corel-5K_features.pkl` / `_paths.pkl` | Corel-5K |
| `features/Corel-10K_features.pkl` / `_paths.pkl` | Corel-10K |
| `features/Caltech-101_features.pkl` / `_paths.pkl` | Caltech-101 |

---

*Previous: [02 - Setup](02_setup.md) | Next: [04 - Code Reference](04_code_reference.md)*
