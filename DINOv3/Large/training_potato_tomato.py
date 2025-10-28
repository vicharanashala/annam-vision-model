import warnings
warnings.filterwarnings("ignore")

import os
import time
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoModel, AutoConfig

from knowledge_graph import load_knowledge_graph  

# -----------------------------
# CONFIG
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device) 

data_dir = "/home/aic_u3/aic_u3/ComputerVision/Perception_Models/Potato_Tomato_G-Models/Dataset_Tomato-Potato_split_T_V"
save_path = "/home/aic_u3/aic_u3/ComputerVision/DINO_large/saved_models/best_Potato_Tomato.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m" # large
# model_name = "facebook/dinov3-vits16-pretrain-lvd1689m" # small
model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m" # base
hf_token = os.getenv("HF_TOKEN", "") # it works only after exporting token in terminal
os.environ["HF_TOKEN"] = hf_token
#
batch_size = 16
epochs = 100
patience = 5
tol = 0.001
use_sampler = True
resize_schedule = [224, 384]

# -----------------------------
# MODEL WRAPPER
# -----------------------------
class DinoClassifier(nn.Module):
    def __init__(self, base_model, hidden_size, num_classes):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        outputs = self.base(x)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)

# -----------------------------
# DATA TRANSFORMS
# -----------------------------
def get_transforms(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)
    ])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# DATASET + DATALOADERS
# -----------------------------
def get_dataloaders(image_size):
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"),
                                    transform=get_transforms(image_size))
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"),
                                  transform=val_tfms)

    if use_sampler:
        class_counts = np.bincount(train_ds.targets)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = [class_weights[t] for t in train_ds.targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, train_ds.classes

# -----------------------------
# TRAIN FUNCTION
# -----------------------------
def train_model():
    config = AutoConfig.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name, config=config)

    train_loader, val_loader, class_names = get_dataloaders(resize_schedule[0])
    num_classes = len(class_names)

    model = DinoClassifier(base_model, config.hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0
    patience_counter = 0
    start_time = time.time()

    for size in resize_schedule:
        print(f"\n🔄 Progressive Resize: {size}x{size}")
        train_loader, val_loader, _ = get_dataloaders(size)

        for epoch in range(epochs):
            model.train()
            train_loss, train_correct, total = 0.0, 0, 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (size {size})")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                total += labels.size(0)
                pbar.set_postfix({"loss": loss.item()})

            train_acc = train_correct / total

            # Validation
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = val_correct / val_total
            print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, "
                  f"Train Loss={train_loss/total:.4f}, Val Loss={val_loss/val_total:.4f}")

            # Early stopping & save best
            if val_acc > best_val_acc + tol:
                best_val_acc = val_acc
                torch.save({'model_state_dict': model.state_dict(),
                            'classes': class_names}, save_path)
                print(f"✅ New best model saved: Val Acc={val_acc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("⏹️ Early stopping triggered")
                    total_time = time.time() - start_time
                    print(f"Total Training Time: {total_time/60:.2f} minutes")
                    break  # exit epoch loop

    total_time = time.time() - start_time
    print(f"\n🎉 Training complete. Best Val Acc={best_val_acc:.4f}")
    print(f"Total Training Time: {total_time/60:.2f} minutes")

    # -----------------------------
    # RETURN FOR MAIN.PY
    # -----------------------------
    # Load best model weights
    checkpoint = torch.load(save_path, map_location=device)
    class_names = checkpoint['classes']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load Knowledge Graph
    G, symptom_map = load_knowledge_graph("/home/aic_u3/aic_u3/ComputerVision/DINO_large/knowledge_graph_potato_tomato.json")

    return model, G, symptom_map, class_names, val_tfms  # val_tfms used for testing


# RUN TRAINING
# -----------------------------
if __name__ == "__main__":
    train_model()    
