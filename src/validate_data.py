import os
from PIL import Image
from tqdm import tqdm

def validate_dataset(root_path):
    print(f"--- Validating Dataset at: {root_path} ---")
    if not os.path.exists(root_path):
        print(f"Error: Path '{root_path}' nahi mila!")
        return

    total_images = 0
    corrupt_files = []
    
    # Categories check karein (africans, beaches, etc.)
    categories = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    
    for cat in categories:
        cat_path = os.path.join(root_path, cat)
        images = [f for f in os.listdir(cat_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Checking {cat}: {len(images)} images...")
        total_images += len(images)
        
        for img_name in tqdm(images, desc=f"Scanning {cat}", leave=False):
            img_path = os.path.join(cat_path, img_name)
            try:
                with Image.open(img_path) as img:
                    img.verify() 
            except Exception:
                corrupt_files.append(img_path)

    print(f"\n--- Validation Summary ---")
    print(f"Total Categories: {len(categories)}")
    print(f"Total Images Found: {total_images}")
    
    if corrupt_files:
        print(f"Alert: {len(corrupt_files)} corrupt images mili hain!")
        for f in corrupt_files: print(f" - {f}")
    else:
        print("Success: Saari images extraction ke liye fit hain!")

if __name__ == "__main__":
    # Tumhara set kiya hua path
    dataset_path = 'dataset/Corel-1K/'
    validate_dataset(dataset_path)
