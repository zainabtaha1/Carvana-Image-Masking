# Carvana Image Masking Challenge

## Project Overview
This project tackles **image segmentation** for the Carvana dataset, aiming to **automatically identify the boundaries of cars** in images.  
It leverages **U-Net** and **U-Net with ResNet50 encoder** architectures for precise mask prediction, useful for background removal or object detection.

---

## Description
Carvana, an online used car startup, captures **16 standard images of each vehicle** using a custom rotating photo studio.  
High-quality images sometimes introduce errors due to reflections or cars blending with the background.  
This project automates **mask generation** to identify car boundaries, reducing manual editing effort.

---

## Dataset
- **Images:** `.jpg` and `.png` files  
- **Masks:** `.gif` files representing car masks  
- **Directory structure example:**
/train/train/.jpg
/train_masks/train_masks/.gif

**Source:** [Carvana Image Masking Challenge](https://www.kaggle.com/competitions/carvana-image-masking-challenge)
## Notebook
You can view and run the full notebook on Kaggle here: [Carvana Image Masking Notebook](<https://www.kaggle.com/code/zainabelsayedtaha/carvana-image-masking#2nd-model:-Unet-Model-+-Resnet-50-Model>)

---

## Features
- Custom **U-Net** and **U-Net + ResNet50** segmentation models
- Training pipeline with **Dice + BCE loss**
- Evaluation metrics: Loss, IoU (Intersection over Union), Confusion Matrix, Classification Report
- Visualization of **original images**, **ground truth masks**, and **predicted masks**
- Overlay masks on original images with custom colormap

---

## Installation
Clone the repository and install dependencies:
```bash
git clone <your-repo-link>
cd Carvana-Image-Masking
pip install -r requirements.txt

---

Dependencies: Python 3.x, PyTorch, Torchvision, OpenCV, NumPy, Matplotlib, PIL, Seaborn, scikit-learn

Usage
1. Prepare Dataset
DATA_DIR = '/kaggle/working/'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train/train')
TRAIN_MASK_DIR = os.path.join(DATA_DIR, 'train_masks/train_masks')

2. Training
model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)
# Train for 15 epochs


Or using U-Net + ResNet50 encoder:

model1 = UNetResNet50(num_classes=1, pretrained=True).to(DEVICE)
# Train for 5 epochs

3. Evaluation

Compute metrics:

Loss & Val IoU

Confusion Matrix

Classification Report

from sklearn.metrics import confusion_matrix, classification_report

4. Visualization

Visualize predictions with ground truth:

predict_and_visualize(model, SegmentationDataset(val_imgs, val_masks_p, IMG_SIZE), num_samples=3)


Overlay masks with colors:

Green: Ground truth

Red: Predicted mask

Model Architecture

U-Net: Encoder-decoder with skip connections

U-Net + ResNet50: Uses ResNet50 pretrained as encoder for better feature extraction

Loss: Combination of BCEWithLogitsLoss and Dice Loss

Optimizer: Adam

Results

Training and validation loss curves

Validation IoU curve

Confusion matrix heatmap

Sample prediction visualization

Saving the Model
torch.save(model.state_dict(), "model.pth")
