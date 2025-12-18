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


