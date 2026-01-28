# using balanced dataset for training
# =========================
# IMPORTS
# =========================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

from torchvision import datasets, transforms
from timm import create_model
from tqdm import tqdm
import os

# =========================
# CONFIG
# =========================
DATASET_DIR = "/home/aic_u3/aic_u3/ComputerVision/Rice Disease Dataset split_ORIGINAL"
# DATASET_DIR = "/home/aic_u3/aic_u3/ComputerVision/Perception_Models/Potato_Tomato_G-Models/Dataset_Tomato-Potato_split_T_V"
BATCH_SIZE = 8  # b4=16, b5=16, b6=8, b7=8
EPOCHS = 50
LR = 3e-4
IMG_SIZE = 528  # b4=380, b5=456, b6==528
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
# DATASETS
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
# COMPUTE CLASS COUNTS & WEIGHTS
# =========================================================

# labels for each training image
targets = [label for _, label in train_ds.samples]

# count per class
class_counts = np.bincount(targets)
print("Class counts:", class_counts)

# inverse-frequency class weights
class_weights = 1.0 / class_counts

# normalize (optional but recommended)
class_weights = class_weights / class_weights.sum() * num_classes

# sample-level weights for sampler
sample_weights = [class_weights[label] for label in targets]

# =========================================================
# WEIGHTED RANDOM SAMPLER (OVERSAMPLING)
# =========================================================
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# =========================
# MODEL
# =========================
model = create_model(
    "tf_efficientnet_b6_ns", # "efficientnet_b4" "efficientnet_b5" "tf_efficientnet_b6_ns"
    pretrained=True,
    num_classes=num_classes
)
model = model.to(DEVICE)

# =========================================================
# 🔴 CLASS-WEIGHTED LOSS (NEW & IMPORTANT)
# =========================================================
class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float
).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

# =========================
# TRAINING LOOP
# =========================
best_acc = 0.0
os.makedirs("checkpoints_Rice", exist_ok=True)  # checkpoints-P_T  checkpoints_Rice

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
            "checkpoints_Rice/best_model_weightedrandomsampler_weighted_loss_b6.pth"
        )
        print(f"✅ Saved best model ({best_acc:.2f}%)")

print(f"\n🎯 Best Validation Accuracy: {best_acc:.2f}%")


# WeightedRandomSampler balances the training data by showing minority-class images more frequently during each epoch without creating new images.
# Class-Weighted Loss increases the penalty for misclassifying rare classes, forcing the model to learn minority patterns more strongly.
# The sampler controls what the model sees, while the weighted loss controls what the model cares about.
# Together, they reduce class bias and improve performance on severely imbalanced datasets.
