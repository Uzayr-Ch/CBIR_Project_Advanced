# 07 — Model & Theory

## ResNet50

### Architecture

| Property | Value |
|:---------|:------|
| **Type** | Residual Network (ResNet) |
| **Layers** | 50 |
| **Pretrained On** | ImageNet (1.2M images, 1000 classes) |
| **Input Size** | 224 × 224 × 3 (RGB) |
| **Output (our use)** | 2048-dimensional feature vector |
| **Classification Layer** | Removed (we only use features) |

### Why ResNet50?

| Advantage | Explanation |
|:----------|:------------|
| **Quality** | State-of-the-art feature extraction |
| **Transfer Learning** | ImageNet pre-training generalizes to any image domain |
| **Speed** | Good balance between accuracy and computation |
| **CPU Friendly** | Runs efficiently without GPU |
| **Proven** | Widely used in CBIR research literature |

### What we use from ResNet50

We remove the final classification layer (`fc`) and use the output of the **global average pooling layer** — this gives us a 2048-dimensional vector that encodes the semantic content of the image.

```python
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove fc layer
model.eval()
```

---

## Image Preprocessing

Every image (query or dataset) goes through the same pipeline:

| Step | Operation | Details |
|:----:|:----------|:--------|
| 1 | **Resize** | 224 × 224 pixels |
| 2 | **ToTensor** | Convert PIL Image to PyTorch tensor (0-1 range) |
| 3 | **Normalize** | ImageNet mean/std normalization |

```python
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet RGB means
        std=[0.229, 0.224, 0.225]     # ImageNet RGB stds
    )
])
```

---

## L2 Normalization

After extracting the 2048-D feature vector, we normalize it to unit length:

$$\vec{v}_{normalized} = \frac{\vec{v}}{||\vec{v}||_2}$$

This ensures:
- All vectors have magnitude = 1
- Cosine similarity becomes equivalent to dot product
- Fair comparison regardless of image brightness/contrast

---

## Cosine Similarity

Measures the angle between two feature vectors:

$$\cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{||\vec{A}|| \times ||\vec{B}||}$$

| Value | Meaning |
|:-----:|:--------|
| 1.0 | Identical |
| 0.0 | Completely different (orthogonal) |
| -1.0 | Opposite |

Since our vectors are L2-normalized, this simplifies to:

$$\text{similarity} = \vec{A} \cdot \vec{B}$$

We use `sklearn.metrics.pairwise.cosine_similarity` for efficient batch computation.

---

## Mean Average Precision (mAP)

### Precision@K

For each query image, look at the Top-K results and count how many are from the same category:

$$\text{Precision@K} = \frac{\text{Relevant images in top K}}{K}$$

### mAP

Average the Precision@K across all queries:

$$\text{mAP} = \frac{1}{N} \sum_{i=1}^{N} \text{Precision@K}_i$$

Where $N$ is the total number of images used as queries.

### Our Evaluation Method

- **Leave-One-Out:** Each image in the dataset serves as a query
- **K = 10:** We look at the Top-10 results
- **Ground truth:** Images from the same folder/category are considered relevant

---

## Feature Storage

| Detail | Value |
|:-------|:------|
| Dimensions per image | 2048 |
| Bytes per dimension | 4 (float32) |
| Bytes per image | 8,192 (8 KB) |
| Combined index size | 25,144 × 2048 × 4 ≈ **196 MB** |

---

*Previous: [06 - Graphs](06_graphs.md) | Next: [08 - Usage Guide](08_usage_guide.md)*
