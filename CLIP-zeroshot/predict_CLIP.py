# Not unsloth family
# CLIP zero shot
"""
CLIP Zero-shot Plant Disease Classification
Prompt Ensembling + ViT-L/14
"""

import os
import torch
import clip
from PIL import Image
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = (
    "/home/aic_u3/aic_u3/ComputerVision/"
    "DINO_large/Benchmark_Dataset-CDDM_images/"
    "Benchmark_Dataset-CDDM_images/images"
)

MAX_IMAGES_PER_CLASS = 50
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# ----------------------------
# PROMPT ENSEMBLE
# ----------------------------
PROMPT_TEMPLATES = [
    "a photo of a {} leaf",
    "a close-up photo of a {} leaf",
    "a plant leaf showing {}",
    "a diseased leaf with {}",
    "a leaf affected by {}",
]

CLASS_NAMES = [
    "potato early blight",
    "potato late blight",
    "healthy potato leaf",
    "tomato bacterial spot",
    "tomato early blight",
    "healthy tomato leaf",
    "tomato late blight",
    "tomato leaf mold",
    "tomato mosaic virus",
    "tomato septoria leaf spot",
    "tomato spider mites",
    "tomato target spot",
    "tomato yellow leaf curl virus",
]

DISPLAY_NAMES = [
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Healthy",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Mosaic Virus",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
]

# ----------------------------
# LOAD CLIP (BIGGER MODEL)
# ----------------------------
print("Loading CLIP ViT-L/14...")
model, preprocess = clip.load("ViT-L/14", device=DEVICE)
model.eval()

# ----------------------------
# BUILD TEXT FEATURES (ENSEMBLE)
# ----------------------------
text_features = []

with torch.no_grad():
    for cls in CLASS_NAMES:
        prompts = [t.format(cls) for t in PROMPT_TEMPLATES]
        tokens = clip.tokenize(prompts).to(DEVICE)
        feats = model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
        text_features.append(feats.mean(dim=0))

text_features = torch.stack(text_features).to(DEVICE)

# ----------------------------
# EVALUATION
# ----------------------------
y_true, y_pred = [], []

print("\nRunning zero-shot evaluation...\n")

with torch.no_grad():
    for true_cls, display_name in zip(CLASS_NAMES, DISPLAY_NAMES):
        folder = os.path.join(DATA_ROOT, display_name)

        if not os.path.isdir(folder):
            print(f"⚠️ Missing folder: {display_name}")
            continue

        images = [
            f for f in os.listdir(folder)
            if f.lower().endswith(IMAGE_EXTS)
        ]

        if MAX_IMAGES_PER_CLASS:
            images = images[:MAX_IMAGES_PER_CLASS]

        for img_name in tqdm(images, desc=display_name, leave=False):
            img_path = os.path.join(folder, img_name)

            image = preprocess(
                Image.open(img_path).convert("RGB")
            ).unsqueeze(0).to(DEVICE)

            image_feat = model.encode_image(image)
            image_feat /= image_feat.norm(dim=-1, keepdim=True)

            similarity = image_feat @ text_features.T
            pred_idx = similarity.argmax(dim=-1).item()

            y_true.append(display_name)
            y_pred.append(DISPLAY_NAMES[pred_idx])

# ----------------------------
# RESULTS
# ----------------------------
acc = accuracy_score(y_true, y_pred)

print("\n" + "=" * 60)
print("📊 CLIP ZERO-SHOT (PROMPT ENSEMBLE)")
print("=" * 60)
print(f"Total images        : {len(y_true)}")
print(f"Correct predictions : {sum(t == p for t, p in zip(y_true, y_pred))}")
print(f"Wrong predictions   : {sum(t != p for t, p in zip(y_true, y_pred))}")
print(f"Overall Accuracy    : {acc * 100:.2f}%")
print("=" * 60)


# #######################################################
# Running zero-shot evaluation...

# ⚠️ Missing folder: Tomato Spider Mites                                                           
                                                                                                
# ============================================================
# 📊 CLIP ZERO-SHOT (PROMPT ENSEMBLE)
# ============================================================
# Total images        : 600
# Correct predictions : 149
# Wrong predictions   : 451
# Overall Accuracy    : 24.83%
# ============================================================