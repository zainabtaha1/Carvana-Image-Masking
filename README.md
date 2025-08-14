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

## Dependences

* Python 3.x

* PyTorch

* Torchvision

* OpenCV

* NumPy

* Matplotlib

* PIL

* Seaborn

* scikit-learn

