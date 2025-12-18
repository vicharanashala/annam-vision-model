# """
# Inference script for Qwen3-VL-8B base model (without fine-tuning)
# Tests plant leaf disease classification on images from Benchmark_Dataset-CDDM_images
# """

# import os
# import re
# from PIL import Image
# from unsloth import FastVisionModel
# from sklearn.metrics import accuracy_score

# # -------------------------
# # CONFIGURATION
# # -------------------------
# # MODE: 0 = Single image, 1 = Single folder, 2 = Multi-folder evaluation
# MODE = 2

# # Set to True to show individual predictions (False = only summary)
# VERBOSE = False  # Set to False for faster execution

# # Limit images per folder for faster testing (None = all images)
# MAX_IMAGES_PER_FOLDER = 10  # 10 images per class for testing

# # Paths
# MODEL_NAME = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
# TEST_ROOT = "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Benchmark_Dataset-CDDM_images/Benchmark_Dataset-CDDM_images/images"
# SINGLE_IMAGE_PATH = os.path.join(TEST_ROOT, "Tomato Healthy/plant_22126.jpg")
# SINGLE_FOLDER_PATH = os.path.join(TEST_ROOT, "Tomato Yellow Leaf Curl Virus")

# # -------------------------
# # CLASS LABELS (expected classes)
# # -------------------------
# CLASS_LABELS = [
#     "Potato Early Blight",
#     "Potato Late Blight",
#     "Potato Healthy",
#     "Tomato Bacterial Spot",
#     "Tomato Early Blight",
#     "Tomato Healthy",
#     "Tomato Late Blight",
#     "Tomato Leaf Mold",
#     "Tomato Mosaic Virus",
#     "Tomato Septoria Leaf Spot",
#     "Tomato Spider Mites",
#     "Tomato Target Spot",
#     "Tomato Yellow Leaf Curl Virus",
# ]

# # -------------------------
# # NORMALIZATION FUNCTIONS
# # -------------------------
# def normalize_label(text: str) -> str:
#     """Normalize text for comparison"""
#     text = text.lower()
#     text = re.sub(r"[^\w\s]", "", text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()

# NORMALIZED_LABEL_MAP = {normalize_label(lbl): lbl for lbl in CLASS_LABELS}

# def folder_to_label(folder_name: str) -> str:
#     """Convert test folder name to closest matching training label"""
#     norm_folder = normalize_label(folder_name)
    
#     # Try exact match first
#     if norm_folder in NORMALIZED_LABEL_MAP:
#         return NORMALIZED_LABEL_MAP[norm_folder]
    
#     # Try partial match
#     for norm_label, true_label in NORMALIZED_LABEL_MAP.items():
#         if norm_label in norm_folder or norm_folder in norm_label:
#             return true_label
    
#     return folder_name  # Return original if no match found

# # -------------------------
# # LOAD MODEL (Base model, no fine-tuning)
# # -------------------------
# print("Loading base model (no fine-tuning)...")
# model, processor = FastVisionModel.from_pretrained(
#     model_name=MODEL_NAME,
#     max_seq_length=16384,  # Use larger context for VLMs as per Unsloth docs
#     load_in_4bit=True,
#     fast_inference=False,
# )
# model.eval()
# print("Model loaded successfully!")

# # -------------------------
# # PROMPT (Simple classification prompt)
# # -------------------------
# # Try simpler prompt that matches training format better
# PROMPT = (
#     "<|vision_start|><|image_pad|><|vision_end|>\n"
#     "Identify the plant and the disease from the image and provide:\n"
#     "- Plant name\n"
#     "- Disease name\n"
#     "- Symptoms\n"
#     "- Treatment steps\n\n"
#     "Disease name:"
# )

# # -------------------------
# # PREDICTION FUNCTION
# # -------------------------
# def predict_single_image(image_path):
#     """Predict disease label for a single image"""
#     try:
#         image = Image.open(image_path).convert("RGB")
#     except Exception as e:
#         print(f"Error loading image {image_path}: {e}")
#         return "Unknown"
    
#     inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(model.device)
    
#     # Generate prediction - reduced tokens for faster inference
#     output_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False, temperature=0.1)
    
#     # Extract generated text
#     full_output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    
#     # Extract generated text - find where prompt ends
#     prompt_end_marker = "Disease name:"
#     if prompt_end_marker in full_output:
#         idx = full_output.find(prompt_end_marker)
#         if idx != -1:
#             start_idx = idx + len(prompt_end_marker)
#             # Skip whitespace
#             while start_idx < len(full_output) and full_output[start_idx] in ['\n', '\r', ' ']:
#                 start_idx += 1
#             generated_text = full_output[start_idx:].strip()
#         else:
#             generated_text = ""
#     else:
#         # Fallback: extract by token IDs (more reliable)
#         input_length = inputs["input_ids"].shape[1]
#         generated_ids = output_ids[0][input_length:]
#         if len(generated_ids) > 0:
#             generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#         else:
#             generated_text = ""
    
#     # If still empty, try extracting from full output by removing input prompt
#     if not generated_text or len(generated_text) < 3:
#         input_text = processor.batch_decode(inputs["input_ids"], skip_special_tokens=True)[0]
#         if full_output.startswith(input_text):
#             generated_text = full_output[len(input_text):].strip()
    
#     # Debug output
#     first_line = generated_text.split('\n')[0].strip() if '\n' in generated_text else generated_text.strip()
#     if VERBOSE:
#         print(f"[DEBUG] Generated text (first 200 chars): {repr(generated_text[:200])}")
#         print(f"[DEBUG] First line: {repr(first_line)}")
    
#     # Extract disease name from generated text
#     generated_text_norm = normalize_label(generated_text)
    
#     # Strategy 1: Check first line (model usually outputs label first)
#     first_line = generated_text.split('\n')[0].strip() if '\n' in generated_text else generated_text.strip()
#     first_line_norm = normalize_label(first_line)
#     if first_line_norm in NORMALIZED_LABEL_MAP:
#         return NORMALIZED_LABEL_MAP[first_line_norm]
    
#     # Strategy 2: Check if first line starts with a label
#     for norm_label, label in sorted(NORMALIZED_LABEL_MAP.items(), key=lambda x: len(x[0]), reverse=True):
#         if first_line_norm.startswith(norm_label):
#             return label
    
#     # Strategy 3: Search for label in entire generated text (prioritize longer matches)
#     for norm_label, label in sorted(NORMALIZED_LABEL_MAP.items(), key=lambda x: len(x[0]), reverse=True):
#         if norm_label in generated_text_norm:
#             return label
    
#     # Strategy 4: Check each line
#     lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
#     for line in lines[:5]:  # Check first 5 lines
#         line_norm = normalize_label(line)
#         if line_norm in NORMALIZED_LABEL_MAP:
#             return NORMALIZED_LABEL_MAP[line_norm]
#         # Also check if line contains a label
#         for norm_label, label in sorted(NORMALIZED_LABEL_MAP.items(), key=lambda x: len(x[0]), reverse=True):
#             if norm_label in line_norm:
#                 return label
    
#     # Strategy 5: Look for disease name pattern
#     disease_patterns = [
#         r"disease name[:\-\s]+([^\n]+)",
#         r"- Disease name[:\-\s]+([^\n]+)",
#         r"disease[:\-\s]+([^\n]+)",
#     ]
    
#     for pattern in disease_patterns:
#         match = re.search(pattern, generated_text, re.IGNORECASE)
#         if match:
#             disease_text = match.group(1).strip()
#             disease_text_norm = normalize_label(disease_text)
#             if disease_text_norm in NORMALIZED_LABEL_MAP:
#                 return NORMALIZED_LABEL_MAP[disease_text_norm]
#             # Try partial match
#             for norm_label, label in NORMALIZED_LABEL_MAP.items():
#                 if norm_label in disease_text_norm or disease_text_norm in norm_label:
#                     return label
    
#     if VERBOSE:
#         print(f"[DEBUG] Could not extract label. Generated text: {repr(generated_text[:300])}")
    
#     return "Unknown"

# # -------------------------
# # SINGLE FOLDER EVALUATION
# # -------------------------
# def evaluate_single_folder(folder_path):
#     """Evaluate all images in a single folder"""
#     folder_name = os.path.basename(folder_path)
#     true_label = folder_to_label(folder_name)
    
#     y_true, y_pred = [], []
#     images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
#     if MAX_IMAGES_PER_FOLDER:
#         images = sorted(images)[:MAX_IMAGES_PER_FOLDER]
    
#     print(f"\nEvaluating folder: {folder_name} ({len(images)} images)")
    
#     for img in sorted(images):
#         img_path = os.path.join(folder_path, img)
#         pred = predict_single_image(img_path)
#         y_true.append(true_label)
#         y_pred.append(pred)
        
#         if VERBOSE:
#             status = "✓" if pred == true_label else "✗"
#             print(f"  {status} {img} -> {pred}")
    
#     total = len(y_true)
#     correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
#     acc = accuracy_score(y_true, y_pred)
    
#     print("\n" + "="*60)
#     print("📊 SINGLE FOLDER ACCURACY")
#     print("="*60)
#     print(f"Folder               : {folder_name}")
#     print(f"True label           : {true_label}")
#     print(f"Total images         : {total}")
#     print(f"Correct predictions  : {correct}")
#     print(f"Wrong predictions    : {total - correct}")
#     print(f"Accuracy             : {acc * 100:.2f}%")
    
#     return acc

# # -------------------------
# # MULTI-FOLDER EVALUATION
# # -------------------------
# def evaluate_multifolder_dataset(root_folder):
#     """Evaluate all folders in the dataset"""
#     y_true, y_pred = [], []
#     total_images = 0
#     per_class_stats = {}
    
#     for folder in sorted(os.listdir(root_folder)):
#         folder_path = os.path.join(root_folder, folder)
#         if not os.path.isdir(folder_path):
#             continue
        
#         true_label = folder_to_label(folder)
#         images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
#         if len(images) == 0:
#             print(f"⚠️ No images in folder: {folder}")
#             continue
        
#         if MAX_IMAGES_PER_FOLDER:
#             images = sorted(images)[:MAX_IMAGES_PER_FOLDER]
        
#         print(f"\nProcessing folder: {folder} ({len(images)} images)")
#         folder_correct = 0
        
#         for i, img in enumerate(images, 1):
#             img_path = os.path.join(folder_path, img)
#             pred = predict_single_image(img_path)
#             y_true.append(true_label)
#             y_pred.append(pred)
#             total_images += 1
            
#             if pred == true_label:
#                 folder_correct += 1
            
#             if VERBOSE:
#                 status = "✓" if pred == true_label else "✗"
#                 print(f"  [{i}/{len(images)}] {status} {img} -> {pred}")
#             elif i % 5 == 0:  # Show progress every 5 images
#                 print(f"  Processed {i}/{len(images)} images...", end='\r')
        
#         folder_acc = folder_correct / len(images) if len(images) > 0 else 0.0
#         per_class_stats[true_label] = {
#             'total': len(images),
#             'correct': folder_correct,
#             'accuracy': folder_acc
#         }
        
#         if not VERBOSE:
#             print(f"  Folder accuracy: {folder_correct}/{len(images)} ({folder_acc*100:.1f}%)")
    
#     if total_images == 0:
#         print("\n❌ No images processed. Check TEST_ROOT path.")
#         return None
    
#     correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
#     acc = accuracy_score(y_true, y_pred)
    
#     print("\n" + "="*60)
#     print("📊 MULTI-FOLDER ACCURACY REPORT")
#     print("="*60)
#     print(f"\nOverall Statistics:")
#     print(f"  Total images         : {total_images}")
#     print(f"  Correct predictions  : {correct}")
#     print(f"  Wrong predictions    : {total_images - correct}")
#     print(f"  Overall Accuracy     : {acc * 100:.2f}%")
    
#     print(f"\nPer-Class Accuracy:")
#     for class_name, stats in sorted(per_class_stats.items()):
#         print(f"  {class_name:35s} : {stats['correct']:3d}/{stats['total']:3d} ({stats['accuracy']*100:5.1f}%)")
    
#     return acc

# # -------------------------
# # MAIN
# # -------------------------
# if __name__ == "__main__":
#     print("\n" + "="*60)
#     print("🔬 BASE MODEL INFERENCE (No Fine-tuning)")
#     print("="*60)
#     print(f"Model: {MODEL_NAME}")
#     print(f"Mode: {MODE} ({'Single Image' if MODE == 0 else 'Single Folder' if MODE == 1 else 'Multi-Folder'})")
#     print("="*60 + "\n")
    
#     if MODE == 0:
#         print(f"Testing single image: {SINGLE_IMAGE_PATH}")
#         result = predict_single_image(SINGLE_IMAGE_PATH)
#         print(f"\n✅ Prediction: {result}")
    
#     elif MODE == 1:
#         evaluate_single_folder(SINGLE_FOLDER_PATH)
    
#     elif MODE == 2:
#         evaluate_multifolder_dataset(TEST_ROOT)
    
#     else:
#         print("❌ Invalid MODE. Use 0, 1, or 2.")


# 2 ############################################################
"""
Inference script for Qwen3-VL-8B base model (without fine-tuning)
Tests plant leaf disease classification on images from Benchmark_Dataset-CDDM_images
"""

import os
import re
from PIL import Image
from unsloth import FastVisionModel
from sklearn.metrics import accuracy_score

# -------------------------
# CONFIGURATION
# -------------------------
# MODE: 0 = Single image, 1 = Single folder, 2 = Multi-folder evaluation
MODE = 2

# Set to True to show individual predictions (False = only summary)
VERBOSE = False  # Set to False for faster execution

# Limit images per folder for faster testing (None = all images)
MAX_IMAGES_PER_FOLDER = 5  # 10 images per class for testing

# Paths
MODEL_NAME = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
TEST_ROOT = "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Benchmark_Dataset-CDDM_images/Benchmark_Dataset-CDDM_images/images"
SINGLE_IMAGE_PATH = os.path.join(TEST_ROOT, "Tomato Healthy/plant_22126.jpg")
SINGLE_FOLDER_PATH = os.path.join(TEST_ROOT, "Tomato Yellow Leaf Curl Virus")

# -------------------------
# CLASS LABELS (expected classes)
# -------------------------
CLASS_LABELS = [
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

# -------------------------
# NORMALIZATION FUNCTIONS
# -------------------------
def normalize_label(text: str) -> str:
    """Normalize text for comparison"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

NORMALIZED_LABEL_MAP = {normalize_label(lbl): lbl for lbl in CLASS_LABELS}

def folder_to_label(folder_name: str) -> str:
    """Convert test folder name to closest matching training label"""
    norm_folder = normalize_label(folder_name)
    
    # Try exact match first
    if norm_folder in NORMALIZED_LABEL_MAP:
        return NORMALIZED_LABEL_MAP[norm_folder]
    
    # Try partial match
    for norm_label, true_label in NORMALIZED_LABEL_MAP.items():
        if norm_label in norm_folder or norm_folder in norm_label:
            return true_label
    
    return folder_name  # Return original if no match found

# -------------------------
# LOAD MODEL (Base model, no fine-tuning)
# -------------------------
print("Loading base model (no fine-tuning)...")
model, processor = FastVisionModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=16384,  # Use larger context for VLMs as per Unsloth docs
    load_in_4bit=True,
    fast_inference=False,
)
model.eval()
print("Model loaded successfully!")

# -------------------------
# PROMPT (Simple classification prompt)
# -------------------------
# Try simpler prompt that matches training format better
PROMPT = (
    "<|vision_start|><|image_pad|><|vision_end|>\n"
    "Identify the plant and the disease from the image and provide:\n"
    "- Plant name\n"
    "- Disease name\n"
    "- Symptoms\n"
    "- Treatment steps\n\n"
    "Disease name:"
)

# -------------------------
# PREDICTION FUNCTION
# -------------------------
def predict_single_image(image_path):
    """Predict disease label for a single image"""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return "Unknown"
    
    inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(model.device)
    
    # Generate prediction - reduced tokens for faster inference
    output_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False, temperature=0.1)
    
    # Extract generated text
    full_output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    # Extract generated text - find where prompt ends
    prompt_end_marker = "Disease name:"
    if prompt_end_marker in full_output:
        idx = full_output.find(prompt_end_marker)
        if idx != -1:
            start_idx = idx + len(prompt_end_marker)
            # Skip whitespace
            while start_idx < len(full_output) and full_output[start_idx] in ['\n', '\r', ' ']:
                start_idx += 1
            generated_text = full_output[start_idx:].strip()
        else:
            generated_text = ""
    else:
        # Fallback: extract by token IDs (more reliable)
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_length:]
        if len(generated_ids) > 0:
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        else:
            generated_text = ""
    
    # If still empty, try extracting from full output by removing input prompt
    if not generated_text or len(generated_text) < 3:
        input_text = processor.batch_decode(inputs["input_ids"], skip_special_tokens=True)[0]
        if full_output.startswith(input_text):
            generated_text = full_output[len(input_text):].strip()
    
    # Debug output
    first_line = generated_text.split('\n')[0].strip() if '\n' in generated_text else generated_text.strip()
    if VERBOSE:
        print(f"[DEBUG] Generated text (first 200 chars): {repr(generated_text[:200])}")
        print(f"[DEBUG] First line: {repr(first_line)}")
    
    # Extract disease name from generated text
    generated_text_norm = normalize_label(generated_text)
    
    # Strategy 1: Check first line (model usually outputs label first)
    first_line = generated_text.split('\n')[0].strip() if '\n' in generated_text else generated_text.strip()
    first_line_norm = normalize_label(first_line)
    if first_line_norm in NORMALIZED_LABEL_MAP:
        return NORMALIZED_LABEL_MAP[first_line_norm]
    
    # Strategy 2: Check if first line starts with a label
    for norm_label, label in sorted(NORMALIZED_LABEL_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if first_line_norm.startswith(norm_label):
            return label
    
    # Strategy 3: Search for label in entire generated text (prioritize longer matches)
    for norm_label, label in sorted(NORMALIZED_LABEL_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if norm_label in generated_text_norm:
            return label
    
    # Strategy 4: Check each line
    lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
    for line in lines[:5]:  # Check first 5 lines
        line_norm = normalize_label(line)
        if line_norm in NORMALIZED_LABEL_MAP:
            return NORMALIZED_LABEL_MAP[line_norm]
        # Also check if line contains a label
        for norm_label, label in sorted(NORMALIZED_LABEL_MAP.items(), key=lambda x: len(x[0]), reverse=True):
            if norm_label in line_norm:
                return label
    
    # Strategy 5: Look for disease name pattern
    disease_patterns = [
        r"disease name[:\-\s]+([^\n]+)",
        r"- Disease name[:\-\s]+([^\n]+)",
        r"disease[:\-\s]+([^\n]+)",
    ]
    
    for pattern in disease_patterns:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match:
            disease_text = match.group(1).strip()
            disease_text_norm = normalize_label(disease_text)
            if disease_text_norm in NORMALIZED_LABEL_MAP:
                return NORMALIZED_LABEL_MAP[disease_text_norm]
            # Try partial match
            for norm_label, label in NORMALIZED_LABEL_MAP.items():
                if norm_label in disease_text_norm or disease_text_norm in norm_label:
                    return label
    
    if VERBOSE:
        print(f"[DEBUG] Could not extract label. Generated text: {repr(generated_text[:300])}")
    
    return "Unknown"

# -------------------------
# SINGLE FOLDER EVALUATION
# -------------------------
def evaluate_single_folder(folder_path):
    """Evaluate all images in a single folder"""
    folder_name = os.path.basename(folder_path)
    true_label = folder_to_label(folder_name)
    
    y_true, y_pred = [], []
    images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    if MAX_IMAGES_PER_FOLDER:
        images = sorted(images)[:MAX_IMAGES_PER_FOLDER]
    
    print(f"\nEvaluating folder: {folder_name} ({len(images)} images)")
    
    for img in sorted(images):
        img_path = os.path.join(folder_path, img)
        pred = predict_single_image(img_path)
        y_true.append(true_label)
        y_pred.append(pred)
        
        if VERBOSE:
            status = "✓" if pred == true_label else "✗"
            print(f"  {status} {img} -> {pred}")
    
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*60)
    print("📊 SINGLE FOLDER ACCURACY")
    print("="*60)
    print(f"Folder               : {folder_name}")
    print(f"True label           : {true_label}")
    print(f"Total images         : {total}")
    print(f"Correct predictions  : {correct}")
    print(f"Wrong predictions    : {total - correct}")
    print(f"Accuracy             : {acc * 100:.2f}%")
    
    return acc

# -------------------------
# MULTI-FOLDER EVALUATION
# -------------------------
def evaluate_multifolder_dataset(root_folder):
    """Evaluate all folders in the dataset"""
    y_true, y_pred = [], []
    total_images = 0
    per_class_stats = {}
    
    for folder in sorted(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        
        true_label = folder_to_label(folder)
        images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        if len(images) == 0:
            print(f"⚠️ No images in folder: {folder}")
            continue
        
        if MAX_IMAGES_PER_FOLDER:
            images = sorted(images)[:MAX_IMAGES_PER_FOLDER]
        
        print(f"\nProcessing folder: {folder} ({len(images)} images)")
        folder_correct = 0
        
        for i, img in enumerate(images, 1):
            img_path = os.path.join(folder_path, img)
            pred = predict_single_image(img_path)
            y_true.append(true_label)
            y_pred.append(pred)
            total_images += 1
            
            if pred == true_label:
                folder_correct += 1
            
            if VERBOSE:
                status = "✓" if pred == true_label else "✗"
                print(f"  [{i}/{len(images)}] {status} {img} -> {pred}")
            elif i % 5 == 0:  # Show progress every 5 images
                print(f"  Processed {i}/{len(images)} images...", end='\r')
        
        folder_acc = folder_correct / len(images) if len(images) > 0 else 0.0
        per_class_stats[true_label] = {
            'total': len(images),
            'correct': folder_correct,
            'accuracy': folder_acc
        }
        
        if not VERBOSE:
            print(f"  Folder accuracy: {folder_correct}/{len(images)} ({folder_acc*100:.1f}%)")
    
    if total_images == 0:
        print("\n❌ No images processed. Check TEST_ROOT path.")
        return None
    
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*60)
    print("📊 MULTI-FOLDER ACCURACY REPORT")
    print("="*60)
    print(f"\nOverall Statistics:")
    print(f"  Total images         : {total_images}")
    print(f"  Correct predictions  : {correct}")
    print(f"  Wrong predictions    : {total_images - correct}")
    print(f"  Overall Accuracy     : {acc * 100:.2f}%")
    
    print(f"\nPer-Class Accuracy:")
    for class_name, stats in sorted(per_class_stats.items()):
        print(f"  {class_name:35s} : {stats['correct']:3d}/{stats['total']:3d} ({stats['accuracy']*100:5.1f}%)")
    
    return acc

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔬 BASE MODEL INFERENCE (No Fine-tuning)")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Mode: {MODE} ({'Single Image' if MODE == 0 else 'Single Folder' if MODE == 1 else 'Multi-Folder'})")
    print("="*60 + "\n")
    
    if MODE == 0:
        print(f"Testing single image: {SINGLE_IMAGE_PATH}")
        result = predict_single_image(SINGLE_IMAGE_PATH)
        print(f"\n✅ Prediction: {result}")
    
    elif MODE == 1:
        evaluate_single_folder(SINGLE_FOLDER_PATH)
    
    elif MODE == 2:
        evaluate_multifolder_dataset(TEST_ROOT)
    
    else:
        print("❌ Invalid MODE. Use 0, 1, or 2.")