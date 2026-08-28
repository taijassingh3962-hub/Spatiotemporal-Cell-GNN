import os
import sys
# Make sure the project root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from src.dataset import CellDataset
from src.model import ComboLoss, get_unet_model

# ==========================================
# CONFIGURATION
# ==========================================
TRAIN_IMG_DIR = "train-new/Fluo-N2DL-HeLa/01"
TRAIN_MASK_DIR = "train-new/Fluo-N2DL-HeLa/01_ST/SEG"
VAL_IMG_DIR = "train-new/Fluo-N2DL-HeLa/02"
VAL_MASK_DIR = "train-new/Fluo-N2DL-HeLa/02_ST/SEG"

BATCH_SIZE = 8
NUM_EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Training on device: {DEVICE}")

    train_dataset = CellDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=None)
    val_dataset = CellDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=None)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = get_unet_model(DEVICE)

    weights_path = "best_unet_model.pt"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print("Loaded previous weights for fine-tuning!")

    criterion = ComboLoss()

    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 1e-5},
        {'params': model.decoder.parameters(), 'lr': 1e-3},
        {'params': model.segmentation_head.parameters(), 'lr': 1e-3},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )

    scaler = torch.amp.GradScaler('cuda')
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_unet_model.pt")
            print("--> Best model saved!")

if __name__ == "__main__":
    main()
