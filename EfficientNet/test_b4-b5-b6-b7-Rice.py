# # b4-b5-b6 
# NB:  b'z of space issue i deleted trained model for b5 and b7 for rice datset, b'z its given low accuacy in testing. if needed retrain the model first 
# =====================================================
# EfficientNet-B4 / B5 / B6 TEST SCRIPT 
# =====================================================
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import numpy as np

# -------------------------
# CONFIG
# -------------------------
DATASET_DIR = "/home/aic_u3/aic_u3/ComputerVision/Rice Disease Dataset Test"

# MODEL_PATH = "/home/aic_u3/aic_u3/ComputerVision/EfficientNet/checkpoints_Rice/best_model_b4.pth"
# MODEL_PATH = "/home/aic_u3/aic_u3/ComputerVision/EfficientNet/checkpoints_Rice/best_model_b6.pth"
# ################################
# WeightedRandomSampler + class-weighted loss with the EfficientNet-B4 and EfficientNet-B6 variants
# MODEL_PATH = "/home/aic_u3/aic_u3/ComputerVision/EfficientNet/checkpoints_Rice/best_model_balanced_weighted_loss_b4.pth"
# MODEL_PATH = "/home/aic_u3/aic_u3/ComputerVision/EfficientNet/checkpoints_Rice/best_model_balanced_weighted_loss_b6.pth"
MODEL_PATH = "/home/aic_u3/aic_u3/ComputerVision/EfficientNet/checkpoints_Rice/best_model_balanced_weighted_loss_b6.pth"



IMG_SIZE = 528  # b4=380 | b5=456 | b6=528
BATCH_SIZE = 8  # b4=16, b5=16, b6=8, b7=8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# TRANSFORMS (UNCHANGED)
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
class_names = test_dataset.classes
print(f"Detected {NUM_CLASSES} classes")
print("Classes:", class_names)

# -------------------------
# MODEL
# -------------------------
model = create_model(
    "tf_efficientnet_b6_ns",  # "efficientnet_b4" "efficientnet_b5" "tf_efficientnet_b6_ns"
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

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# -------------------------
# METRICS
# -------------------------
overall_acc = accuracy_score(y_true, y_pred) * 100
overall_precision = precision_score(
    y_true, y_pred, average="weighted", zero_division=0
) * 100

cm = confusion_matrix(y_true, y_pred)

per_class_acc = []
per_class_prec = []

print("\n📊 BENCHMARK TEST RESULTS")
print("=" * 70)
print(f"Total images        : {len(y_true)}")
print(f"Correct predictions : {(overall_acc/100)*len(y_true):.0f}")
print(f"Wrong predictions   : {len(y_true) - (overall_acc/100)*len(y_true):.0f}")
print(f"Overall Accuracy    : {overall_acc:.2f}%")
print(f"Overall Precision   : {overall_precision:.2f}%\n")

print("Per-Class Metrics")
print("-" * 70)
print(f"{'Class':25s} | {'Accuracy (%)':>12s} | {'Precision (%)':>14s}")
print("-" * 70)

for i, cls in enumerate(class_names):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)

    class_acc = TP / (TP + FN) * 100
    class_prec = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0.0

    per_class_acc.append(class_acc)
    per_class_prec.append(class_prec)

    print(f"{cls:25s} | {class_acc:12.2f} | {class_prec:14.2f}")

print("-" * 70)
print(f"Macro Avg Accuracy  : {np.mean(per_class_acc):.2f}%")
print(f"Macro Avg Precision : {np.mean(per_class_prec):.2f}%")
print("=" * 70)


# # =================================================================================
# # EfficientNet-B7 TEST SCRIPT (# B& model is different)
# # =================================================================================
# import os
# import torch
# import torch.nn as nn
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
# import numpy as np

# # -------------------------
# # CONFIG
# # -------------------------
# # Dataset - Rice dsease

# DATASET_DIR = "/home/aic_u3/aic_u3/ComputerVision/Rice Disease Dataset Test"
# MODEL_PATH  = "/home/aic_u3/aic_u3/ComputerVision/EfficientNet/checkpoints_Rice/best_model_b7.pth"

# IMG_SIZE = 600
# BATCH_SIZE = 8
# NUM_WORKERS = 4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # -------------------------
# # TRANSFORMS (UNCHANGED)
# # -------------------------
# test_transforms = transforms.Compose([
#     transforms.Resize(IMG_SIZE + 32),
#     transforms.CenterCrop(IMG_SIZE),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# # -------------------------
# # DATASET
# # -------------------------
# test_dataset = datasets.ImageFolder(
#     root=DATASET_DIR,
#     transform=test_transforms
# )

# test_loader = DataLoader(
#     test_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=NUM_WORKERS,
#     pin_memory=True
# )

# NUM_CLASSES = len(test_dataset.classes)
# class_names = test_dataset.classes
# print(f"Detected {NUM_CLASSES} classes")
# print("Classes:", class_names)

# # -------------------------
# # MODEL (EfficientNet-B7)
# # -------------------------
# model = models.efficientnet_b7(weights=None)

# in_features = model.classifier[1].in_features
# model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

# state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
# model.load_state_dict(state_dict)

# model = model.to(DEVICE)
# model.eval()

# # -------------------------
# # EVALUATION
# # -------------------------
# y_true, y_pred = [], []

# with torch.no_grad():
#     for imgs, labels in test_loader:
#         imgs = imgs.to(DEVICE)
#         labels = labels.to(DEVICE)

#         outputs = model(imgs)
#         preds = outputs.argmax(dim=1)

#         y_pred.extend(preds.cpu().numpy())
#         y_true.extend(labels.cpu().numpy())

# y_true = np.array(y_true)
# y_pred = np.array(y_pred)

# # -------------------------
# # METRICS (SAME AS B4–B6)
# # -------------------------
# overall_acc = accuracy_score(y_true, y_pred) * 100
# overall_precision = precision_score(
#     y_true, y_pred, average="weighted", zero_division=0
# ) * 100

# cm = confusion_matrix(y_true, y_pred)

# per_class_acc = []
# per_class_prec = []

# print("\n📊 BENCHMARK TEST RESULTS")
# print("=" * 70)
# print(f"Total images        : {len(y_true)}")
# print(f"Correct predictions : {(overall_acc/100)*len(y_true):.0f}")
# print(f"Wrong predictions   : {len(y_true) - (overall_acc/100)*len(y_true):.0f}")
# print(f"Overall Accuracy    : {overall_acc:.2f}%")
# print(f"Overall Precision   : {overall_precision:.2f}%\n")

# print("Per-Class Metrics")
# print("-" * 70)
# print(f"{'Class':25s} | {'Accuracy (%)':>12s} | {'Precision (%)':>14s}")
# print("-" * 70)

# for i, cls in enumerate(class_names):
#     TP = cm[i, i]
#     FP = cm[:, i].sum() - TP
#     FN = cm[i, :].sum() - TP
#     TN = cm.sum() - (TP + FP + FN)

#     class_acc = (TP + TN) / cm.sum() * 100
#     class_prec = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0.0

#     per_class_acc.append(class_acc)
#     per_class_prec.append(class_prec)

#     print(f"{cls:25s} | {class_acc:12.2f} | {class_prec:14.2f}")

# print("-" * 70)
# print(f"Macro Avg Accuracy  : {np.mean(per_class_acc):.2f}%")
# print(f"Macro Avg Precision : {np.mean(per_class_prec):.2f}%")
# print("=" * 70)

