# Code working -epoch = 20
import os
import timm
import torch
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "/home/aic_u3/aic_u3/ComputerVision/EfficientNet/efficientnet_b4_leaf.pth"
VAL_DIR = "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Benchmark_Dataset-CDDM_images/Benchmark_Dataset-CDDM_images/images"
NUM_CLASSES = 13
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

val_transforms = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# -------------------------
# MODEL
# -------------------------
model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)
model.eval()

# -------------------------
# EVALUATION
# -------------------------
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print(f"\n📊 TEST RESULTS")
print("="*50)
print(f"Total images        : {len(all_labels)}")
print(f"Correct predictions : {(acc*len(all_labels)):.0f}")
print(f"Wrong predictions   : {len(all_labels) - (acc*len(all_labels)):.0f}")
print(f"Overall Accuracy    : {acc*100:.2f}%")

# Optional: Plot confusion matrix
import seaborn as sns
import pandas as pd

class_names = val_dataset.classes
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(12,10))
sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.title("Normalized Confusion Matrix")
plt.show()

# 📊 TEST RESULTS
# ==================================================
# Total images        : 52146
# Correct predictions : 47734
# Wrong predictions   : 4412
# Overall Accuracy    : 91.54%

