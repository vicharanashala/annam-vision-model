# Efficientnet-b4
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
from sklearn.metrics import accuracy_score

# -------------------------
# CONFIG
# -------------------------
DATASET_DIR = "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Benchmark_Dataset-CDDM_images/Benchmark_Dataset-CDDM_images/images"
MODEL_PATH = "/home/aic_u3/aic_u3/ComputerVision/EfficientNet/checkpoints/best_model_50epochs.pth"
IMG_SIZE = 380
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# TRANSFORMS (must match training)
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

NUM_CLASSES = len(test_dataset.classes)
print(f"Detected {NUM_CLASSES} classes")

# -------------------------
# MODEL
# -------------------------
model = create_model(
    "efficientnet_b4",
    pretrained=False,
    num_classes=NUM_CLASSES
)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

# -------------------------
# EVALUATION
# -------------------------
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(imgs)
        preds = outputs.argmax(dim=1)

        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

acc = accuracy_score(y_true, y_pred) * 100

print("\n📊 BENCHMARK TEST RESULTS")
print("=" * 50)
print(f"Total images        : {len(y_true)}")
print(f"Correct predictions : {(acc/100)*len(y_true):.0f}")
print(f"Wrong predictions   : {len(y_true) - (acc/100)*len(y_true):.0f}")
print(f"Overall Accuracy    : {acc:.2f}%")

# ###############################################################
# Test Result using Bechmark - # both models are working 

# 1. Epochs= 20 
# 📊 BENCHMARK DATASET TEST RESULTS
# ==================================================
# Total images        : 52146
# Correct predictions : 48425
# Wrong predictions   : 3721
# Overall Accuracy    : 92.86%
# .......................................................
# 📊 FIELD IMAGE DATASET TEST RESULTS
# ==================================================
# Total images        : 996
# Correct predictions : 375
# Wrong predictions   : 621
# Overall Accuracy    : 37.65%
# ..............................................

# 2. Epochs=50

# 📊 BENCHMARK DATASET TEST RESULTS
# ==================================================
# Total images        : 52146
# Correct predictions : 48530
# Wrong predictions   : 3616
# Overall Accuracy    : 93.07%

# so for further process model with epochs = 50 is selecting

# "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Benchmark_Dataset-CDDM_images/images"


# "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Field_Images"

# 📊 FIELD IMAGE DATASET TEST RESULTS
# ==================================================
# Total images        : 996
# Correct predictions : 365
# Wrong predictions   : 631
# Overall Accuracy    : 36.65%

# =========================
# EfficientNet-B7 Training Script 
# =========================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os

# =========================
# CONFIG
# =========================
DATASET_DIR = "/home/aic_u3/aic_u3/ComputerVision/Rice Disease Dataset split"
BATCH_SIZE = 8  # Reduced for B7 to avoid CUDA OOM
EPOCHS = 50
LR = 3e-4
IMG_SIZE = 600  # B7 native input size
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# TRANSFORMS
# =========================
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# DATA
# =========================
train_ds = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=train_tfms)
val_ds   = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"), transform=val_tfms)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

num_classes = len(train_ds.classes)
print("Classes:", train_ds.classes)

# =========================
# MODEL
# =========================
model = models.efficientnet_b7(
    weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1
)

# Replace classifier head
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

model = model.to(DEVICE)

# =========================
# LOSS, OPTIMIZER, SCHEDULER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS
)

# =========================
# AMP SCALER
# =========================
scaler = torch.cuda.amp.GradScaler()

# =========================
# TRAINING LOOP
# =========================
best_acc = 0.0
os.makedirs("checkpoints_Rice", exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        # Automatic Mixed Precision (AMP)
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)

    scheduler.step()
    train_loss = running_loss / len(train_ds)

    # -------- Validation --------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100.0 * correct / total
    print(
        f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Acc={val_acc:.2f}%"
    )

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(
            model.state_dict(),
            "checkpoints_Rice/best_model_b7.pth"
        )
        print(f"✅ Saved best model ({best_acc:.2f}%)")

print(f"\n🎯 Best Validation Accuracy: {best_acc:.2f}%")

