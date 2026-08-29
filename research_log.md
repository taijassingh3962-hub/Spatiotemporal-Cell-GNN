# 🔬 Research & Implementation Log: HeLa Cell Segmentation (U-Net)

This document is a step-by-step breakdown of our entire development process. It details the architecture we built, the bugs we encountered, how we overcame them, and the **Actual Logic** and **Mathematical Reasoning** behind every architectural decision.

---

## 1. The Core Architecture: U-Net
For biomedical segmentation, we utilized an FCN (Fully Convolutional Network) architecture, specifically the **U-Net**. Its Encoder-Decoder structure with skip connections is the state-of-the-art approach for extracting deep spatial features and reconstructing them into original-sized high-fidelity masks.

### Q: Why did you choose ResNet-34 as the encoder instead of ResNet-50?
> **Answer (The Logic):** ResNet-50 is a very deep network with a significantly higher parameter count. Biomedical datasets (like HeLa cells) typically have a smaller number of images compared to general datasets. If we had used ResNet-50, two major issues would have occurred:
> 1. **VRAM Bottlenecks:** The high parameter count would easily lead to Out Of Memory (OOM) errors during training.
> 2. **Overfitting:** The model would essentially memorize the limited training data and fail to generalize on unseen test data.
> ResNet-34 acted as the perfect **"Sweet Spot"**—it is lightweight, fits comfortably in VRAM, and possesses sufficient depth to extract complex features without severely overfitting.

---

## 2. Loss Function Design: ComboLoss
Instead of relying on a standard single loss function, we designed a custom `ComboLoss` which is a 50%-50% weighted mixture of **BCE (Binary Cross Entropy) Loss** and **Dice Loss**.

### Q: Why did you combine BCE and Dice Loss? Why not just use one?
> **Answer (The Logic):** Microscopy images suffer heavily from the **Class Imbalance** problem. Typically, 80% of the image is completely black (background) and only 20% contains actual cells (foreground).
> - **BCE Loss** evaluates on a strict pixel-by-pixel basis. If the model lazily predicts the entire image as black, BCE might still yield an 80% accuracy score, which is entirely misleading and incorrect for segmentation.
> - **Dice Loss** specifically focuses on the *Overlap* (Intersection over Union) and the overall *Shape*. If even a small cell is missed, the Dice Loss penalizes the model heavily, thus forcing it to focus on the foreground.
> By combining both, we get the best of both worlds: BCE ensures pixel-level perfection, while Dice Loss handles the class imbalance and maintains global structural integrity.

---

## 3. The "IoU 0.0" Bug & Domain Shift 
When we evaluated our model for the first time, our Mean IoU and Dice Score were exactly **0.0000**. The model predicted completely empty (black) masks.

### Q: How did you debug the IoU 0.0 issue?
> **Answer (The Logic):** Initial debugging revealed that the code was structurally flawless. However, we were victims of a classic **"Domain Shift"**.
> - Our model was initially trained for just 2 epochs on the `PhC-C2DH-U373` dataset (Phase Contrast microscopy—featuring dark cells on a light background).
> - However, we were evaluating it on the `Fluo-N2DL-HeLa` dataset (Fluorescence microscopy—featuring bright cells on a dark background).
> - Deep learning models can only generalize well when the test distribution matches the training distribution. The visual modalities were completely opposite!
> **The Fix:** We updated the data pipelines to point directly to the target dataset (`Fluo-N2DL-HeLa`) and trained the model fully for **30 epochs** to achieve convergence. The result? The Dice Score jumped from 0.0 straight to an outstanding **95.9%**!

---

## 4. Hardware Optimization & VRAM Management
During training, the RAM and VRAM were maxing out, causing system slowdowns. To tackle this, we engineered two major optimizations:

### Optimization A: Automatic Mixed Precision (AMP)
We utilized `torch.amp.autocast()` and a `GradScaler`.
> **Why?:** By default, Neural Networks store every floating-point number in **Float32** (32-bit precision). The `autocast` context manager automatically converts tensor operations to **Float16** (half-precision) wherever numerically safe. This instantly slashed our VRAM consumption by **50%** and accelerated training via GPU Tensor Cores, without sacrificing accuracy. The `GradScaler` prevents gradient underflow (gradients turning into absolute zeros) which is a common issue when using Float16.

### Optimization B: Dynamic Resizing (512x512)
The native resolution (700x1100) was crashing the VRAM due to the massive activation maps.
> We dynamically downscaled the inputs to `512x512` during dataset loading using OpenCV.
> **Important Technical Detail:** We used `cv2.INTER_LINEAR` (smooth blending) for the images. However, **for the masks, we intentionally used `cv2.INTER_NEAREST` (Nearest-Neighbor)**. This was critical to ensure the masks remained strictly binary (0 or 1) and didn't become fuzzy/grayscale along the edges due to interpolation.

---

## 5. Differential Learning Rates
Instead of applying a blanket Learning Rate (LR) across the entire model, we used fine-grained control within the AdamW optimizer:
> - **Encoder (ResNet-34) LR = 1e-5:** Since this backbone was pre-trained on ImageNet, we used a very small learning rate to preserve its already-learned edge and shape detection capabilities, avoiding "catastrophic forgetting".
> - **Decoder & Segmentation Head LR = 1e-3:** These layers were initialized from scratch specifically for our cell segmentation task. We assigned them a higher learning rate so they could learn patterns quickly.
> - **Scheduler:** We paired this with a `CosineAnnealingWarmRestarts` scheduler. If the model got stuck in a local minimum (a dead end) during training, the scheduler would force the learning rate to jump back up periodically (every 10 epochs), allowing it to escape and find a better global optimum.

---
**Final Metric Achieved on Fluo-N2DL-HeLa:**
- **Mean IoU:** 92.17%
- **Mean Dice:** 95.92%
