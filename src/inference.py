import os
import sys
# Make sure the project root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch
import numpy as np
from src.model import get_unet_model

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_WEIGHTS_PATH = "best_unet_model.pt"
TEST_IMG_DIR = "test-new/Fluo-N2DL-HeLa/01"
OUTPUT_MASK_DIR = "test-new/Fluo-N2DL-HeLa/01_RES"
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Running inference on device: {DEVICE}")
    print("1. Loading Model and Weights...")
    
    model = get_unet_model(DEVICE)
    
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Could not find weights at '{MODEL_WEIGHTS_PATH}'")
        return
        
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
    model.eval() 
    print("Model loaded successfully!")

    if not os.path.exists(OUTPUT_MASK_DIR):
        os.makedirs(OUTPUT_MASK_DIR)
        print(f"Created output directory: {OUTPUT_MASK_DIR}")
    
    image_files = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.tif')]
    print(f"Found {len(image_files)} images in {TEST_IMG_DIR}")
    
    for img_name in image_files:
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None: continue

        # Normalization
        image_norm = image.astype(np.float32) / image.max()
        # Ensure native resolution is maintained for final testing unless specifically resizing
        # In predict.py earlier we did not resize so we keep it native or we resize back.
        # But we must match what we did. Let's keep native.
        
        image_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(image_norm, axis=0), axis=0)).to(DEVICE)
        
        with torch.no_grad():
            raw_logits = model(image_tensor)
            
        probabilities = torch.sigmoid(raw_logits)
        mask = (probabilities > THRESHOLD).float()
        
        mask_np = mask.squeeze().cpu().numpy()
        mask_save = (mask_np * 255).astype(np.uint8)
        
        save_name = "mask_" + img_name
        save_path = os.path.join(OUTPUT_MASK_DIR, save_name)
        cv2.imwrite(save_path, mask_save)
        
        print(f"Processed: {img_name} -> Saved as {save_name}")

    print("Prediction completed successfully!")

if __name__ == "__main__":
    main()
