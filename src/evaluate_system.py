import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# 1. Load Data
with open('features/features.pkl', 'rb') as f:
    features = pickle.load(f)
with open('features/image_paths.pkl', 'rb') as f:
    image_paths = pickle.load(f)

def get_category(path):
    return os.path.basename(os.path.dirname(path))

def calculate_metrics(features, image_paths, k=10):
    all_precisions = []
    
    print(f"Calculating Precision@{k} and mAP for 10K images...")
    
    # Har image ko as a query use karenge (Leave-one-out evaluation)
    for i in tqdm(range(len(features))):
        query_feat = features[i].reshape(1, -1)
        query_cat = get_category(image_paths[i])
        
        # Calculate similarity with all images
        sims = cosine_similarity(query_feat, features).flatten()
        
        # Get top K+1 (to skip the query image itself)
        indices = np.argsort(sims)[::-1][1:k+1]
        
        # Check how many are from the same category
        hits = 0
        for idx in indices:
            if get_category(image_paths[idx]) == query_cat:
                hits += 1
        
        precision = hits / k
        all_precisions.append(precision)
    
    mAP = np.mean(all_precisions)
    return mAP, all_precisions

# Run Evaluation
k_value = 10
mean_precision, precisions = calculate_metrics(features, image_paths, k=k_value)

print(f"\n--- Research Findings Summary ---")
print(f"Dataset Size: {len(image_paths)} images")
print(f"Metric: Precision@{k_value}")
print(f"Mean Average Precision (mAP): {mean_precision:.4f}")
print(f"Max Precision achieved: {max(precisions):.2f}")
