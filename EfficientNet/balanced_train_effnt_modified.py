# using balanced dataset for training : balance the training signal -control how often classes are seen (will not change or create new images)
# =========================
# IMPORTS
# =========================
import torch
import torch.nn as nn

# 🔴 CHANGE 1: ADD WeightedRandomSampler + numpy
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

from torchvision import datasets, transforms
from timm import create_model
from tqdm import tqdm
import os

# =========================
# CONFIG
# =========================
DATASET_DIR = "/home/aic_u3/aic_u3/ComputerVision/Rice Disease Dataset split"
BATCH_SIZE = 16
EPOCHS = 50
LR = 3e-4
IMG_SIZE = 380
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# TRANSFORMS (UNCHANGED)
# =========================
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================
# DATASETS (UNCHANGED)
# =========================
train_ds = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "train"),
    transform=train_tfms
)

val_ds = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "val"),
    transform=val_tfms
)

num_classes = len(train_ds.classes)
print("Classes:", train_ds.classes)

# =========================================================
# 🔴 CHANGE 2: COMPUTE SAMPLE WEIGHTS (NEW BLOCK)
# =========================================================

# Extract class labels for each image
targets = [label for _, label in train_ds.samples]

# Count images per class
class_counts = np.bincount(targets)
print("Class counts:", class_counts)

# Inverse frequency for class weighting
class_weights = 1.0 / class_counts

# Assign weight to each sample
sample_weights = [class_weights[label] for label in targets]

# Create sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# =========================================================
# 🔴 CHANGE 3: TRAIN LOADER USES SAMPLER (shuffle REMOVED)
# =========================================================
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=sampler,          # ✅ BALANCED SAMPLING
    num_workers=NUM_WORKERS
)

# =========================
# VALIDATION LOADER (UNCHANGED)
# =========================
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# =========================
# MODEL (UNCHANGED)
# =========================
model = create_model(
    "efficientnet_b4",
    pretrained=True,
    num_classes=num_classes
)
model = model.to(DEVICE)

# =========================
# LOSS & OPTIMIZER (UNCHANGED)
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

# =========================
# TRAINING LOOP (UNCHANGED)
# =========================
best_acc = 0.0
os.makedirs("checkpoints_Rice", exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, labels in tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    ):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    scheduler.step()

    train_loss = running_loss / len(train_ds)

    # -------- VALIDATION --------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100.0 * correct / total

    print(
        f"Epoch {epoch+1}: "
        f"Train Loss={train_loss:.4f} | "
        f"Val Acc={val_acc:.2f}%"
    )

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(
            model.state_dict(),
            "checkpoints_Rice/best_model_balanced.pth"
        )
        print(f"✅ Saved best model ({best_acc:.2f}%)")

print(f"\n🎯 Best Validation Accuracy: {best_acc:.2f}%")
