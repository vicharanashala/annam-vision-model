# Code for bot Rice disease datset and Potato-Tomato Disease Detection
# Conclusion: EfficientNet-B4 is working better than EfficientNet-B7 for Rice Disease Dataset
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

####################################################################################
# .............................epochs=50.......b4, b5 & b6...................................

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
#DATASET_DIR = "/home/aic_u3/aic_u3/ComputerVision/Rice Disease Dataset split_ORIGINAL"
BATCH_SIZE = 8
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
    "tf_efficientnet_b6_ns", # "efficientnet_b4" "efficientnet_b5" "tf_efficientnet_b6_ns"
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
os.makedirs("checkpoints-P_T", exist_ok=True)

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
            "checkpoints-P_T/best_model_b6.pth"
        )
        print(f"✅ Saved best model ({best_acc:.2f}%)")

print(f"\n🎯 Best Validation Accuracy: {best_acc:.2f}%")



# =================================================
# DATASET SPLITTING (Train and Validation) - RICE DISEASE DATASET (ONE-TIME NEEDED)
# =================================================
# import os
# import shutil
# import random

# SRC_DATASET = "/home/aic_u3/aic_u3/ComputerVision/Rice Disease Dataset"
# SPLIT_DATASET = "/home/aic_u3/aic_u3/ComputerVision/Rice Disease Dataset split" # it is used for GAN generation 
# (so made a copy of it and Now changed its name as .../home/aic_u3/aic_u3/ComputerVision/Rice Disease Dataset split_ORIGINAL)
# TRAIN_RATIO = 0.8
# RANDOM_SEED = 42

# random.seed(RANDOM_SEED)

# # Create train/val directories
# for split in ["train", "val"]:
#     os.makedirs(os.path.join(SPLIT_DATASET, split), exist_ok=True)

# # Loop over each class folder
# for class_name in os.listdir(SRC_DATASET):
#     class_path = os.path.join(SRC_DATASET, class_name)
#     if not os.path.isdir(class_path):
#         continue

#     images = os.listdir(class_path)
#     random.shuffle(images)

#     split_idx = int(len(images) * TRAIN_RATIO)
#     train_imgs = images[:split_idx]
#     val_imgs = images[split_idx:]

#     # Create class directories
#     train_class_dir = os.path.join(SPLIT_DATASET, "train", class_name)
#     val_class_dir   = os.path.join(SPLIT_DATASET, "val", class_name)
#     os.makedirs(train_class_dir, exist_ok=True)
#     os.makedirs(val_class_dir, exist_ok=True)

#     # Copy images
#     for img in train_imgs:
#         shutil.copy(
#             os.path.join(class_path, img),
#             os.path.join(train_class_dir, img)
#         )

#     for img in val_imgs:
#         shutil.copy(
#             os.path.join(class_path, img),
#             os.path.join(val_class_dir, img)
#         )

# print("✅ Dataset split completed successfully!")
# print("📂 Saved at:", SPLIT_DATASET)

# ################################################################################
#  EfficientNet-B7
# .............................epochs=50, batchsize 16 - (EfficientNet B7).memory issue..not working.........................................

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# # from timm import create_model
# from torchvision import models
# from tqdm import tqdm
# import os

# # =========================
# # CONFIG
# # =========================
# # DATASET_DIR = "/home/aic_u3/aic_u3/ComputerVision/Perception_Models/Potato_Tomato_G-Models/Dataset_Tomato-Potato_split_T_V"
# DATASET_DIR = "/home/aic_u3/aic_u3/ComputerVision/Rice Disease Dataset split_ORIGINAL"
# BATCH_SIZE = 16
# EPOCHS = 50
# LR = 3e-4
# IMG_SIZE = 600
# NUM_WORKERS = 4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# train_loader = DataLoader(
#     train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
# )
# val_loader = DataLoader(
#     val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
# )

# num_classes = len(train_ds.classes)
# print("Classes:", train_ds.classes)

# # =========================
# # MODEL
# # =========================
# model = models.efficientnet_b7(
#     weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1
# )

# # Replace classifier head (VERY IMPORTANT)
# in_features = model.classifier[1].in_features
# model.classifier[1] = nn.Linear(in_features, num_classes)

# model = model.to(DEVICE)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=EPOCHS
# )

# # =========================
# # TRAINING LOOP
# # =========================
# best_acc = 0.0
# os.makedirs("checkpoints_Rice", exist_ok=True)

# for epoch in range(EPOCHS):
#     model.train()
#     running_loss = 0.0

#     for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
#         imgs = imgs.to(DEVICE)
#         labels = labels.to(DEVICE)

#         optimizer.zero_grad()
#         outputs = model(imgs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item() * imgs.size(0)

#     scheduler.step()

#     train_loss = running_loss / len(train_ds)

#     # -------- Validation --------
#     model.eval()
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for imgs, labels in val_loader:
#             imgs = imgs.to(DEVICE)
#             labels = labels.to(DEVICE)

#             outputs = model(imgs)
#             preds = outputs.argmax(dim=1)

#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#     val_acc = 100.0 * correct / total
#     print(
#         f"Epoch {epoch+1}: "
#         f"Train Loss={train_loss:.4f} | "
#         f"Val Acc={val_acc:.2f}%"
#     )

#     if val_acc > best_acc:
#         best_acc = val_acc
#         torch.save(
#             model.state_dict(),
#             "checkpoints_Rice/best_model_b7.pth"
#         )
#         print(f"✅ Saved best model ({best_acc:.2f}%)")

# print(f"\n🎯 Best Validation Accuracy: {best_acc:.2f}%")

# ################################################################################
# # =========================
# # EfficientNet-B7 Training Script (epoch 50,batchsize 8) - working
# # =========================

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms, models
# from tqdm import tqdm
# import os

# # =========================
# # CONFIG
# # =========================
# DATASET_DIR = "/home/aic_u3/aic_u3/ComputerVision/Rice Disease Dataset split_ORIGINAL"
# BATCH_SIZE = 8  # Reduced for B7 to avoid CUDA OOM
# EPOCHS = 50
# LR = 3e-4
# IMG_SIZE = 600  # B7 native input size
# NUM_WORKERS = 4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# train_loader = DataLoader(
#     train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
# )
# val_loader = DataLoader(
#     val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
# )

# num_classes = len(train_ds.classes)
# print("Classes:", train_ds.classes)

# # =========================
# # MODEL
# # =========================
# model = models.efficientnet_b7(
#     weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1
# )

# # Replace classifier head
# in_features = model.classifier[1].in_features
# model.classifier[1] = nn.Linear(in_features, num_classes)

# model = model.to(DEVICE)

# # =========================
# # LOSS, OPTIMIZER, SCHEDULER
# # =========================
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=EPOCHS
# )

# # =========================
# # AMP SCALER
# # =========================
# scaler = torch.cuda.amp.GradScaler()

# # =========================
# # TRAINING LOOP
# # =========================
# best_acc = 0.0
# os.makedirs("checkpoints_Rice", exist_ok=True)

# for epoch in range(EPOCHS):
#     model.train()
#     running_loss = 0.0

#     for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
#         imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

#         optimizer.zero_grad()

#         # Automatic Mixed Precision (AMP)
#         with torch.cuda.amp.autocast():
#             outputs = model(imgs)
#             loss = criterion(outputs, labels)

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         running_loss += loss.item() * imgs.size(0)

#     scheduler.step()
#     train_loss = running_loss / len(train_ds)

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

#     val_acc = 100.0 * correct / total
#     print(
#         f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Acc={val_acc:.2f}%"
#     )

#     if val_acc > best_acc:
#         best_acc = val_acc
#         torch.save(
#             model.state_dict(),
#             "checkpoints_Rice/best_model_b7.pth"
#         )
#         print(f"✅ Saved best model ({best_acc:.2f}%)")

# print(f"\n🎯 Best Validation Accuracy: {best_acc:.2f}%")



