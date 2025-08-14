import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import torch.nn as nn

# ---------------- UNet Model ----------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32,64,128]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        rev_f = features[::-1]
        for f in rev_f:
            self.ups.append(nn.ConvTranspose2d(f*2 if f!=rev_f[0] else features[-1]*2, f, 2, stride=2))
            self.ups.append(DoubleConv(f*2, f))

        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i//2]
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i+1](x)

        return self.final_conv(x)

# ---------------- Settings ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (128, 128)

# Load model
model = UNet()
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------- Helper Functions ----------------
def predict_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img, IMG_SIZE)
    img_norm = img_resized / 255.0
    img_tensor = torch.tensor(img_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float().cpu().squeeze().numpy()

    # Convert mask to color
    mask_color = np.zeros_like(img_resized)
    mask_color[pred == 1] = [0, 255, 0]  # Green mask

    # Blend original and mask
    overlay = cv2.addWeighted(img_resized, 0.7, mask_color, 0.3, 0)

    return overlay, pred

# ---------------- Streamlit UI ----------------
st.title("🚗 Carvana Image Segmentation")
uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Original Image", use_column_width=True)
    
    st.write("⏳ Predicting...")
    overlay, mask = predict_image(image)

    st.image(overlay, caption="🎨 Overlay Result", use_column_width=True)
    st.image(mask, caption="🖌️ Predicted Mask", use_column_width=True)