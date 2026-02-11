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

## Analysis

### Corel-1K — 97.63% mAP

- Only 10 categories → easier to discriminate
- 100 images per category → strong class representation
- High intra-class visual similarity
- Near-perfect retrieval performance

### Corel-5K — 51.90% mAP

- 50 categories → much more diverse
- Some categories visually overlap (nature scenes etc.)
- Lower per-class sample density compared to search space
- Hardest dataset for this system

### Caltech-101 — 85.20% mAP

- 102 object categories (named: accordion, butterfly, etc.)
- Unbalanced (40-800 images per class)
- Object-centric images → cleaner features
- Surprisingly strong despite high category count

### Corel-10K — 81.92% mAP

- 100 categories, 100 images each
- High diversity in visual content
- Some categories are visually ambiguous
- Still strong performance (>80%)

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
