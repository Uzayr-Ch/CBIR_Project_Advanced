# 10 — References

## Academic Sources

| # | Reference |
|:-:|:----------|
| 1 | He, K. et al. — *"Deep Residual Learning for Image Recognition"*, CVPR 2016 |
| 2 | Deng, J. et al. — *ImageNet: A Large-Scale Hierarchical Image Database*, CVPR 2009 |
| 3 | Wang, J.Z. et al. — *"SIMPLIcity: Semantics-sensitive Integrated Matching for Picture Libraries"*, IEEE TPAMI 2001 |
| 4 | Fei-Fei, L. et al. — *"Learning Generative Visual Models from Few Training Examples"*, CVPR 2004 (Caltech-101) |

---

## Tools & Libraries

| Tool | URL |
|:-----|:----|
| PyTorch | https://pytorch.org/docs/ |
| TorchVision | https://pytorch.org/vision/stable/ |
| Streamlit | https://docs.streamlit.io/ |
| scikit-learn | https://scikit-learn.org/stable/ |
| NumPy | https://numpy.org/doc/ |
| Pillow | https://pillow.readthedocs.io/ |
| Matplotlib | https://matplotlib.org/stable/ |

---

## Key Concepts

| Concept | Brief Description |
|:--------|:-----------------|
| CBIR | Image retrieval based on visual content, not text metadata |
| Transfer Learning | Using features from a model trained on one task (ImageNet) for another (retrieval) |
| Feature Extraction | Converting images into numerical vectors that capture semantic information |
| Cosine Similarity | Measuring angle between two vectors to determine similarity |
| mAP | Mean Average Precision — standard retrieval evaluation metric |
| Leave-One-Out | Evaluation method where each image is used as a query against the rest |

---

## Future Improvements

- [ ] GPU acceleration for faster feature extraction
- [ ] Multiple models (VGG16, EfficientNet, ViT)
- [ ] Image segmentation for local features
- [ ] User relevance feedback loop
- [ ] FAISS integration for billion-scale retrieval
- [ ] Fine-tuning on domain-specific data

---

*Previous: [09 - Troubleshooting](09_troubleshooting.md) | Back to: [README](README.md)*
