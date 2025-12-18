# code worked and epoch=20 
# ##################################################################
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from timm import create_model
# from tqdm import tqdm
# import os

# # =========================
# # CONFIG
# # =========================
# DATASET_DIR = "/home/aic_u3/aic_u3/ComputerVision/Perception_Models/Potato_Tomato_G-Models/Dataset_Tomato-Potato_split_T_V"
# BATCH_SIZE = 16
# EPOCHS = 50
# LR = 3e-4
# IMG_SIZE = 380
# NUM_WORKERS = 4
# DEVICE = "cuda"

# # =========================
# # TRANSFORMS
# # =========================
# train_tfms = transforms.Compose([
#     transforms.RandomResizedCrop(IMG_SIZE),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# val_tfms = transforms.Compose([
#     transforms.Resize(IMG_SIZE + 32),
#     transforms.CenterCrop(IMG_SIZE),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# # =========================
# # DATA
# # =========================
# train_ds = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=train_tfms)
# val_ds   = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"), transform=val_tfms)

# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
# val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# num_classes = len(train_ds.classes)
# print("Classes:", train_ds.classes)

# # =========================
# # MODEL
# # =========================
# model = create_model("efficientnet_b4", pretrained=True, num_classes=num_classes)
# model = model.to(DEVICE)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# # =========================
# # TRAINING LOOP
# # =========================
# best_acc = 0.0
# os.makedirs("checkpoints", exist_ok=True)

# for epoch in range(EPOCHS):
#     model.train()
#     train_loss = 0.0

#     for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
#         imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

#         optimizer.zero_grad()
#         outputs = model(imgs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()

#     scheduler.step()

#     # -------- Validation --------
#     model.eval()
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for imgs, labels in val_loader:
#             imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
#             outputs = model(imgs)
#             preds = outputs.argmax(dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#     val_acc = 100 * correct / total
#     print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Acc={val_acc:.2f}%")

#     if val_acc > best_acc:
#         best_acc = val_acc
#         torch.save(model.state_dict(), "checkpoints/best_model.pth")

# print(f"\n✅ Best Validation Accuracy: {best_acc:.2f}%")

# ####################################################################################3
# .............................epochs=50..........................................

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model
from tqdm import tqdm
import os

# =========================
# CONFIG
# =========================
DATASET_DIR = "/home/aic_u3/aic_u3/ComputerVision/Perception_Models/Potato_Tomato_G-Models/Dataset_Tomato-Potato_split_T_V"
BATCH_SIZE = 16
EPOCHS = 50
LR = 3e-4
IMG_SIZE = 380
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
model = create_model(
    "efficientnet_b4",
    pretrained=True,
    num_classes=num_classes
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS
)

# =========================
# TRAINING LOOP
# =========================
best_acc = 0.0
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
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

    # -------- Validation --------
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
            "checkpoints/best_model.pth"
        )
        print(f"✅ Saved best model ({best_acc:.2f}%)")

print(f"\n🎯 Best Validation Accuracy: {best_acc:.2f}%")
