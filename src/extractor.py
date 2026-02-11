import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm

# -------------------------------
# 1️: Configuration
# -------------------------------
# Tumhare folder structure ke mutabiq path set kiya hai
DATASET_DIR = "dataset/Corel-1K"  
FEATURES_DIR = "features"
FEATURES_FILE = os.path.join(FEATURES_DIR, "features.pkl")
PATHS_FILE = os.path.join(FEATURES_DIR, "image_paths.pkl")

# Features folder banana agar nahi bana hua
os.makedirs(FEATURES_DIR, exist_ok=True)

# -------------------------------
# 2️: Pretrained Model Setup (CPU)
# -------------------------------
print("Loading ResNet50 model...")
device = torch.device("cpu")
# Pehli baar run hone par ye weights download karega (approx 100MB)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# Classification layer ko hata kar sirf feature extractor rakha hai
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()
model.to(device)

# -------------------------------
# 3️: Image Preprocessing
# -------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# 4️: Feature Extraction Loop
# -------------------------------
features = []
image_paths = []

print(f"Starting extraction from {DATASET_DIR}...")

# Saari images ki list pehle hi bana lete hain taake tqdm sahi se chale
all_files = []
for root, _, files in os.walk(DATASET_DIR):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_files.append(os.path.join(root, f))

# Asal extraction loop
for img_path in tqdm(all_files, desc="Extracting Features"):
    try:
        # Image load aur RGB mein convert
        img = Image.open(img_path).convert('RGB')
        
        # Preprocess aur batch dimension add karna
        input_tensor = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Feature nikalna
            feature = model(input_tensor).squeeze().numpy()
            # L2 Normalization (Cosine similarity ke liye zaroori hai)
            feature = feature / np.linalg.norm(feature)
            
        features.append(feature)
        image_paths.append(img_path)
    except Exception as e:
        print(f"\nSkipping {img_path} due to error: {e}")

# Numpy array mein convert karna
features = np.array(features)
print(f"\n✅ Success: {len(features)} images ke features extract ho gaye.")

# -------------------------------
# 5️: Saving Results
# -------------------------------
print("Saving features to disk...")
with open(FEATURES_FILE, 'wb') as f:
    pickle.dump(features, f)

with open(PATHS_FILE, 'wb') as f:
    pickle.dump(image_paths, f)

print(f"Done! Pickles saved in '{FEATURES_DIR}/' folder.")
