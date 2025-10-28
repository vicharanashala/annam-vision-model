# ---------------------------------------------------
# Single Image, Single folder, Multiclass folders.
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import AutoModel, AutoConfig
from sklearn.metrics import accuracy_score

# -----------------------------
# CONFIG
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# MODEL WRAPPER (same as training)
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
# IMAGE TRANSFORM
# -----------------------------
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

val_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model(checkpoint_path, model_name):
    config = AutoConfig.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name, config=config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint['classes']
    num_classes = len(class_names)

    model = DinoClassifier(base_model, config.hidden_size, num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, class_names

# -----------------------------
# PREDICT SINGLE IMAGE
# -----------------------------
def predict_image(img_path, model, class_names, transform=val_tfms):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_idx = model(x).argmax(1).item()
    return class_names[pred_idx]

# -----------------------------
# PREDICT ALL IMAGES IN A FOLDER
# -----------------------------
def predict_folder(folder_path, model, class_names, true_class=None):
    results = {}
    count = 0
    correct = 0

    for img_file in os.listdir(folder_path):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, img_file)
            pred = predict_image(img_path, model, class_names)
            results[img_file] = pred
            count += 1
            if true_class is not None and pred == true_class:
                correct += 1

    if true_class is not None:
        acc = correct / count if count > 0 else 0
        print(f"\n✅ Folder: {folder_path}")
        print(f"   True class: {true_class}")
        print(f"   Number of predictions: {count}")
        print(f"   Correct predictions: {correct}")
        print(f"   Accuracy: {acc:.4f}")

    return results

# -----------------------------
# PREDICT MULTI-FOLDER STRUCTURE
# -----------------------------
def predict_multi_class(base_folder, model, class_names):
    results = {}
    overall_y_true, overall_y_pred = [], []

    for class_folder in os.listdir(base_folder):
        class_dir = os.path.join(base_folder, class_folder)
        if os.path.isdir(class_dir):
            results[class_folder] = {}
            y_true, y_pred = [], []
            count, correct = 0, 0

            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_file)
                    pred = predict_image(img_path, model, class_names)
                    results[class_folder][img_file] = pred

                    y_true.append(class_folder)
                    y_pred.append(pred)
                    count += 1
                    if pred == class_folder:
                        correct += 1

            # Individual folder accuracy
            acc = correct / count if count > 0 else 0
            print(f"\n[{class_folder}]")
            print(f"  Number of predictions: {count}")
            print(f"  Correct predictions: {correct}")
            print(f"  Accuracy: {acc:.4f}")

            overall_y_true.extend(y_true)
            overall_y_pred.extend(y_pred)

    # Overall multi-folder accuracy
    if len(overall_y_true) > 0:
        overall_acc = accuracy_score(overall_y_true, overall_y_pred)
        print(f"\n✅ Overall Multi-folder Accuracy: {overall_acc:.4f} ({sum([t==p for t,p in zip(overall_y_true, overall_y_pred)])}/{len(overall_y_true)})")

    return results

# -----------------------------
# MAIN USAGE
# -----------------------------
if __name__ == "__main__":
    model_choice = "base"  # choose from: "large", "small", or "base"

    if model_choice == "large":
        ckpt = "/home/aic_u3/aic_u3/ComputerVision/DINO_large/saved_models/best_Potato_Tomato_L.pt"
        model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    elif model_choice == "small":
        ckpt = "/home/aic_u3/aic_u3/ComputerVision/DINO_large/saved_models/best_Potato_Tomato_S.pt"
        model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"  # will change after fine-tuning with new dataset

    elif model_choice == "base":
        ckpt = "/home/aic_u3/aic_u3/ComputerVision/DINO_large/saved_models/best_Potato_Tomato.pt"
        model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # will change after fine-tuning with new dataset

    model, class_names = load_model(ckpt, model_name)

    # -----------------------------
    # OPTIONS
    # -----------------------------
    mode = "multi"   # "single_image", "single_folder", or "multi"

    if mode == "single_image":
        img_path = "/home/aic_u3/ComputerVision/DINO_large/Sugarcane_Leaf_Image_Dataset_split_T_V/train/Dried_Leaves/image002.jpg"
        pred_class = predict_image(img_path, model, class_names)
        print(f"\n✅ Prediction for {img_path}: {pred_class}")

    elif mode == "single_folder":
        folder_path = "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Field_Images/Potato_Early_blight"
        true_class = "Potato_Early_blight"   # optional, for accuracy calc
        folder_results = predict_folder(folder_path, model, class_names, true_class=true_class)

        print(f"\nSample predictions from folder {folder_path}:")
        for img, pred in list(folder_results.items())[:5]:
            print(f"  {img} -> {pred}")

    elif mode == "multi":
        # base_folder = "/home/aic_u3/aic_u3/ComputerVision/Perception_Models/Potato_Tomato_G-Models/Dataset_Tomato-Potato_split_T_V/val"
        base_folder = "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Field_Images"
        multi_results = predict_multi_class(base_folder, model, class_names)

        print("\nMulti-folder predictions (sample):")
        for folder, res in list(multi_results.items())[:3]:  # show first 3 folders only
            print(f"[{folder}] -> {list(res.items())[:3]}")


