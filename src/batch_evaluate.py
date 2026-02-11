"""
Batch Evaluation Script for CBIR System
========================================
This script evaluates all datasets and generates research-grade metrics.

Author: CBIR Research Project
Date: February 2026
"""

import os
import time
import torch
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASETS = {
    "Corel-1K": "dataset/Corel-1K",
    "Corel-5K": "dataset/Corel-5K", 
    "Corel-10K": "dataset/Corel-10K",
    "Caltech-101": "dataset/caltech-101"
}

FEATURES_DIR = "features"
RESULTS_FILE = "evaluation_results.pkl"
K_VALUE = 10  # Precision@K

# ============================================================================
# MODEL SETUP
# ============================================================================

print("=" * 60)
print("CBIR BATCH EVALUATION SYSTEM")
print("=" * 60)

print("\n[1/4] Loading ResNet50 model...")
device = torch.device("cpu")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()
model.to(device)

# Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

os.makedirs(FEATURES_DIR, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_images(dataset_path):
    """Get all image paths from a dataset directory."""
    all_files = []
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(root, f))
    return all_files

def get_category(path):
    """Extract category name from image path."""
    return os.path.basename(os.path.dirname(path))

def extract_features(dataset_name, dataset_path):
    """Extract features for a dataset and save to pickle files."""
    features_file = os.path.join(FEATURES_DIR, f"{dataset_name}_features.pkl")
    paths_file = os.path.join(FEATURES_DIR, f"{dataset_name}_paths.pkl")
    
    # Check if already extracted
    if os.path.exists(features_file) and os.path.exists(paths_file):
        print(f"   Loading cached features for {dataset_name}...")
        with open(features_file, 'rb') as f:
            features = pickle.load(f)
        with open(paths_file, 'rb') as f:
            image_paths = pickle.load(f)
        return features, image_paths
    
    print(f"   Extracting features for {dataset_name}...")
    all_files = get_all_images(dataset_path)
    
    if len(all_files) == 0:
        print(f"   WARNING: No images found in {dataset_path}")
        return None, None
    
    features = []
    image_paths = []
    
    for img_path in tqdm(all_files, desc=f"   {dataset_name}", leave=True):
        try:
            img = Image.open(img_path).convert('RGB')
            input_tensor = preprocess(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                feature = model(input_tensor).squeeze().numpy()
                feature = feature / np.linalg.norm(feature)  # L2 normalize
            
            features.append(feature)
            image_paths.append(img_path)
        except Exception as e:
            continue
    
    features = np.array(features)
    
    # Save to pickle
    with open(features_file, 'wb') as f:
        pickle.dump(features, f)
    with open(paths_file, 'wb') as f:
        pickle.dump(image_paths, f)
    
    return features, image_paths

def calculate_metrics(features, image_paths, k=10):
    """Calculate mAP and Precision@K using leave-one-out evaluation."""
    all_precisions = []
    total_latency = 0
    
    for i in range(len(features)):
        query_feat = features[i].reshape(1, -1)
        query_cat = get_category(image_paths[i])
        
        # Measure retrieval latency
        start_time = time.time()
        sims = cosine_similarity(query_feat, features).flatten()
        indices = np.argsort(sims)[::-1][1:k+1]  # Skip query itself
        end_time = time.time()
        
        total_latency += (end_time - start_time)
        
        # Calculate precision
        hits = sum(1 for idx in indices if get_category(image_paths[idx]) == query_cat)
        precision = hits / k
        all_precisions.append(precision)
    
    mAP = np.mean(all_precisions) * 100  # Convert to percentage
    avg_latency = (total_latency / len(features)) * 1000  # Convert to ms
    
    return mAP, avg_latency, all_precisions

def count_categories(image_paths):
    """Count unique categories in dataset."""
    categories = set(get_category(p) for p in image_paths)
    return len(categories)

# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================

print("\n[2/4] Extracting features for all datasets...")
all_results = {}

for dataset_name, dataset_path in DATASETS.items():
    print(f"\n{'─' * 50}")
    print(f"Processing: {dataset_name}")
    print(f"{'─' * 50}")
    
    if not os.path.exists(dataset_path):
        print(f"   SKIPPED: Path not found - {dataset_path}")
        continue
    
    features, image_paths = extract_features(dataset_name, dataset_path)
    
    if features is None:
        continue
    
    num_images = len(image_paths)
    num_categories = count_categories(image_paths)
    
    print(f"   Images: {num_images}, Categories: {num_categories}")
    
    all_results[dataset_name] = {
        "features": features,
        "image_paths": image_paths,
        "num_images": num_images,
        "num_categories": num_categories
    }

# ============================================================================
# CALCULATE METRICS
# ============================================================================

print("\n[3/4] Calculating evaluation metrics...")

for dataset_name, data in all_results.items():
    print(f"\n   Evaluating {dataset_name}...")
    
    mAP, latency, precisions = calculate_metrics(
        data["features"], 
        data["image_paths"], 
        k=K_VALUE
    )
    
    all_results[dataset_name]["mAP"] = mAP
    all_results[dataset_name]["latency_ms"] = latency
    all_results[dataset_name]["precisions"] = precisions
    all_results[dataset_name]["max_precision"] = max(precisions) * 100
    all_results[dataset_name]["min_precision"] = min(precisions) * 100

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n[4/4] Saving results...")

# Save comprehensive results
results_for_save = {
    name: {k: v for k, v in data.items() if k not in ['features', 'precisions']}
    for name, data in all_results.items()
}

with open(os.path.join(FEATURES_DIR, RESULTS_FILE), 'wb') as f:
    pickle.dump(results_for_save, f)

# ============================================================================
# PRINT RESULTS TABLE
# ============================================================================

print("\n")
print("=" * 80)
print("EVALUATION RESULTS SUMMARY")
print("=" * 80)

print(f"\n{'Dataset':<15} {'Images':>10} {'Categories':>12} {'mAP (%)':>12} {'Latency (ms)':>15}")
print("─" * 80)

for name, data in sorted(all_results.items(), key=lambda x: x[1]['num_images']):
    print(f"{name:<15} {data['num_images']:>10,} {data['num_categories']:>12} "
          f"{data['mAP']:>11.2f}% {data['latency_ms']:>14.3f}")

print("─" * 80)

# ============================================================================
# MARKDOWN TABLE OUTPUT
# ============================================================================

print("\n\n📋 MARKDOWN TABLE (Copy for documentation):")
print("─" * 80)

print("\n| Dataset | Total Images | Categories | mAP (%) | Precision@10 | Latency (ms) |")
print("|:--------|-------------:|-----------:|--------:|-------------:|-------------:|")

for name, data in sorted(all_results.items(), key=lambda x: x[1]['num_images']):
    print(f"| {name} | {data['num_images']:,} | {data['num_categories']} | "
          f"{data['mAP']:.2f}% | {data['mAP']:.2f}% | {data['latency_ms']:.3f} |")

print("\n")
print("=" * 80)
print("✅ Evaluation Complete!")
print(f"   Results saved to: {FEATURES_DIR}/{RESULTS_FILE}")
print("=" * 80)
