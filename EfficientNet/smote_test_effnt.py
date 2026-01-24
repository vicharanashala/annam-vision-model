# ============================================================
# Rice Disease Dataset – UNSEEN TEST
# EfficientNet-B4 + SMOTE (Feature Space)
# ============================================================

import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -------------------------
# CONFIG
# -------------------------
DATASET_DIR = "/home/aic_u3/aic_u3/ComputerVision/Rice Disease Dataset Test"
MODEL_PATH = "/home/aic_u3/aic_u3/ComputerVision/EfficientNet/checkpoints_Rice/best_model_smote_b6.pth"

IMG_SIZE = 528  # b4=380 | b5=456 | b6=528
BATCH_SIZE = 8  # b4=16, b5=16, b6=8, b7=8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# TRANSFORMS (MUST MATCH TRAINING)
# -------------------------
test_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------
# DATASET
# -------------------------
test_dataset = datasets.ImageFolder(
    root=DATASET_DIR,
    transform=test_transforms
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

class_names = test_dataset.classes
NUM_CLASSES = len(class_names)

print(f"Detected {NUM_CLASSES} classes")
print("Classes:", class_names)

# -------------------------
# LOAD FEATURE EXTRACTOR
# -------------------------
feature_extractor = create_model(
    "tf_efficientnet_b6_ns",  # "efficientnet_b4" "efficientnet_b5" "tf_efficientnet_b6_ns"
    pretrained=False,
    num_classes=0   # IMPORTANT
).to(DEVICE)

# -------------------------
# LOAD CLASSIFIER HEAD
# -------------------------
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

feature_extractor.load_state_dict(checkpoint["feature_extractor"])

# infer feature dimension
with torch.no_grad():
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    feat_dim = feature_extractor(dummy).shape[1]

classifier = nn.Linear(feat_dim, NUM_CLASSES).to(DEVICE)
classifier.load_state_dict(checkpoint["classifier"])

feature_extractor.eval()
classifier.eval()

# -------------------------
# INFERENCE
# -------------------------
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        features = feature_extractor(imgs)
        outputs = classifier(features)
        preds = outputs.argmax(dim=1)

        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# -------------------------
# METRICS
# -------------------------
acc = accuracy_score(y_true, y_pred) * 100
cm = confusion_matrix(y_true, y_pred)

print("\n📊 BENCHMARK TEST RESULTS (SMOTE)")
print("=" * 60)
print(f"Total images        : {len(y_true)}")
print(f"Correct predictions : {(acc/100)*len(y_true):.0f}")
print(f"Wrong predictions   : {len(y_true) - (acc/100)*len(y_true):.0f}")
print(f"Overall Accuracy    : {acc:.2f}%")

# -------------------------
# PER-CLASS ACCURACY
# -------------------------
print("\n📌 Per-Class Accuracy:")
for i, cls in enumerate(class_names):
    class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
    print(f"{cls:30s}: {class_acc*100:.2f}%")

# -------------------------
# PER-CLASS RECALL & MACRO AVG
# -------------------------
print("\n📌 Classification Report (Recall, Macro Avg):")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
))
