# 05 — Evaluation Results

## System Configuration

______________________________________________________
|     Parameter     |            Value               |
|:------------------|:-------------------------------|
| Model             | ResNet50 (ImageNet pretrained) |
| Feature Dimension | 2048                           |
| Similarity Metric | Cosine Similarity              |
| Evaluation Method | Leave-One-Out                  |
| Primary Metric    | Precision@10 (mAP)             |
| Hardware          | CPU                            |
|___________________|________________________________|

---

## Results Table

___________________________________________________________________________
|     Dataset      | Images | Categories |    mAP (%)  | Avg Latency (ms) |
|:-----------------|-------:|-----------:|------------:|-----------------:|
| **Corel-1K**     | 1,000  |    10      | **97.63%**  |     9.70         |
| **Corel-5K**     | 5,000  |    50      | **51.90%**  |    36.95         |
| **Caltech-101**  | 9,144  |   102      |  **85.20%** |    65.99         |
| **Corel-10K**    | 10,000 |   100      | **81.92%**  |    77.47         |
|__________________|________|____________|_____________|__________________|

---

## Multi-Model Results: ResNet50 vs ViT-B/16

| Dataset      | Model      | mAP (%) | Latency (ms) | Winner |
|:------------|:-----------|--------:|-------------:|:------:|
| Corel-1K    | ResNet50   | 95.74   | 6.47         |        |
| Corel-1K    | **ViT-B/16** | **96.16** | **2.89**   | ✓ ViT |
| Corel-5K    | **ResNet50** | **51.56** | 33.97      | ✓ ResNet |
| Corel-5K    | ViT-B/16   | 49.26   | **14.53**    |        |
| Corel-10K   | **ResNet50** | **85.40** | 69.95      | ✓ ResNet |
| Corel-10K   | ViT-B/16   | 77.72   | **27.54**    |        |
| Caltech-101 | ResNet50   | 99.02   | 64.53        |        |
| Caltech-101 | **ViT-B/16** | **99.24** | **25.05** | ✓ ViT |

---

## Updated Analysis: ViT-B/16 vs ResNet50

- **ViT-B/16 outperforms ResNet50 on Corel-1K and Caltech-101** in both accuracy and speed.
- **ResNet50 remains best for Corel-5K and Corel-10K** in accuracy, but ViT is much faster.
- **ViT features are 768-D (vs 2048-D for ResNet50)**, making them more compact and efficient for indexing.
- **Retrieval latency is 2-3x lower with ViT** on all datasets.
- **Speedup:** ViT enables real-time search even on large datasets.

### Key Takeaways

1. **ViT-B/16 is recommended for fast, interactive search** where speed is critical and accuracy is comparable.
2. **ResNet50 is still preferred for maximum accuracy** on large, diverse datasets.
3. **Both models are supported** and can be selected based on deployment needs.

---

## Key Findings

1. **mAP decreases as dataset complexity increases** — this is expected behavior for any retrieval system.
2. **Deep learning features maintain robust performance** — even at 10K scale, mAP stays above 80%.
3. **Caltech-101 outperforms Corel-10K** despite having more categories — because Caltech images are more object-focused, making features more discriminative.
4. **Corel-5K is the hardest** — many visually similar nature/landscape categories cause confusion.
5. **Latency scales linearly** with dataset size (cosine similarity is O(n)).

---

## Performance Trends

_________________________________________________
|    Metric    | Corel-1K → Corel-10K |  Change |
|:-------------|:--------------------:|:-------:|
| Dataset Size |       1K → 10K       | +900%   |
| Categories   |       10 → 100       | +900%   |
| mAP Score    |    97.63% → 81.92%   | -15.71% |
| Latency      |    9.7ms → 77.5ms    | +699%   |
|______________|______________________|_________|

---

## Research Paper Excerpt

> The proposed Content-Based Image Retrieval (CBIR) system was evaluated on four benchmark datasets using Mean Average Precision (mAP) at K=10. Our system achieved **97.63% mAP on Corel-1K**, **51.90% on Corel-5K**, **85.20% on Caltech-101**, and **81.92% on Corel-10K**.
>
> The results validate the effectiveness of transfer learning from ImageNet-pretrained models. The 2048-dimensional features extracted from ResNet50's penultimate layer capture rich semantic information that generalizes well across varying dataset scales, confirming the suitability of our framework for real-world image retrieval scenarios.

---

*Previous: [04 - Code Reference](04_code_reference.md) | Next: [06 - Graphs](06_graphs.md)*
