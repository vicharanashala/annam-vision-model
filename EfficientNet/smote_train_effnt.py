# ============================================================
# Rice Disease Classification with SMOTE (Feature Space)
# EfficientNet-B4
# ============================================================

import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from timm import create_model
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DATASET_DIR = "/home/aic_u3/aic_u3/ComputerVision/Rice Disease Dataset split"
BATCH_SIZE = 32
EPOCHS = 50
LR = 3e-4
IMG_SIZE = 380
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# TRANSFORMS
# =========================
tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================
# DATASET
# =========================
train_ds = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "train"),
    transform=tfms
)
val_ds = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "val"),
    transform=tfms
)

num_classes = len(train_ds.classes)
class_names = train_ds.classes
print("Classes:", class_names)

# =========================
# FEATURE EXTRACTOR
# =========================
feature_extractor = create_model(
    "efficientnet_b4",
    pretrained=True,
    num_classes=0  # removes classifier
).to(DEVICE)

feature_extractor.eval()
for p in feature_extractor.parameters():
    p.requires_grad = False

# =========================
# EXTRACT FEATURES
# =========================
def extract_features(dataset):
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    features, labels = [], []

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(DEVICE)
            feats = feature_extractor(imgs)
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())

    return np.vstack(features), np.hstack(labels)

X_train, y_train = extract_features(train_ds)
X_val, y_val = extract_features(val_ds)

# =========================
# CLASS COUNTS (BEFORE)
# =========================
print("\nClass counts BEFORE SMOTE:")
for i, name in enumerate(class_names):
    print(f"{name:25s}: {(y_train == i).sum()}")

# =========================
# APPLY SMOTE
# =========================
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# =========================
# CLASS COUNTS (AFTER)
# =========================
print("\nClass counts AFTER SMOTE:")
for i, name in enumerate(class_names):
    print(f"{name:25s}: {(y_train_sm == i).sum()}")

# =========================
# DATALOADERS
# =========================
train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train_sm, dtype=torch.float32),
        torch.tensor(y_train_sm, dtype=torch.long)
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================
# CLASSIFIER HEAD
# =========================
classifier = nn.Linear(X_train.shape[1], num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(classifier.parameters(), lr=LR)

# =========================
# TRAINING LOOP (with epoch loss + validation accuracy)
# =========================
best_acc = 0.0
os.makedirs("checkpoints_Rice", exist_ok=True)

for epoch in range(EPOCHS):
    classifier.train()
    running_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = classifier(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # Validation after each epoch
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = classifier(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "feature_extractor": feature_extractor.state_dict(),
            "classifier": classifier.state_dict()
        }, "checkpoints_Rice/best_model_smote1.pth")
        print(f"✅ Saved best model ({best_acc:.2f}%)")

print(f"\n🎯 Best Validation Accuracy: {best_acc:.2f}%")

# =========================
# FINAL VALIDATION METRICS
# =========================
classifier.eval()
y_pred = []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(DEVICE)
        preds = classifier(x).argmax(dim=1)
        y_pred.extend(preds.cpu().numpy())

print("\n📊 Validation Report (SMOTE)")
print(classification_report(
    y_val,
    y_pred,
    target_names=class_names,
    digits=4
))

print("\n✅ SMOTE model saved")
