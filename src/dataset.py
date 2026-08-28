import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class CellDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = [img for img in os.listdir(images_dir) if img.endswith('.tif')]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_name = "man_seg" + img_name[1:]
        mask_path = os.path.join(self.masks_dir, mask_name) 
        
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        # Resize for VRAM optimization
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        image = image.astype(np.float32) / image.max() 
        mask = (mask > 0).astype(np.float32)
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        else:
            image = torch.from_numpy(np.expand_dims(image, axis=0))
            mask = torch.from_numpy(np.expand_dims(mask, axis=0))
            
        return image, mask
