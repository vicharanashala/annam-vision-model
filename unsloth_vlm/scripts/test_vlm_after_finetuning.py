# # import os
# # from PIL import Image
# # from unsloth import FastVisionModel
# # import re
# # from sklearn.metrics import accuracy_score


# # # ----------------------------------------------------
# # # 1. CHANGE THIS VALUE TO SELECT MODE
# # # ----------------------------------------------------
# # # 0 = Single image prediction
# # # 1 = Single folder prediction
# # # 2 = Multi-folder evaluation (accuracy)
# # MODE = 1


# # # ----------------------------------------------------
# # # 2. PATHS — MODIFY FOR YOUR SYSTEM
# # # ----------------------------------------------------
# # MODEL_DIR = "/home/aic_u3/aic_u3/ComputerVision/unsloth_vlm/outputs/leaf_vlm_lora"
# # TEST_ROOT = "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Benchmark_Dataset-CDDM_images"

# # SINGLE_IMAGE_PATH = "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Benchmark_Dataset-CDDM_images/Benchmark_Dataset-CDDM_images/images/Tomato_Leaf_Mold/plant_33562.jpg"
# # SINGLE_FOLDER_PATH = "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Benchmark_Dataset-CDDM_images/Benchmark_Dataset-CDDM_images/images/Tomato_Yellow_Leaf_Curl_Virus"


# # # ----------------------------------------------------
# # # 3. LOAD MODEL
# # # ----------------------------------------------------
# # model, processor = FastVisionModel.from_pretrained(
# #     model_name=MODEL_DIR,
# #     load_in_4bit=True,
# #     fast_inference=False,
# # )


# # # ----------------------------------------------------
# # # 4. PROMPT
# # # ----------------------------------------------------
# # PROMPT = (
# #     "<|vision_start|><|image_pad|><|vision_end|>\n"
# #     "Identify the plant and the disease from the image and provide:\n"
# #     "- Plant name\n"
# #     "- Disease name\n"
# #     "- Symptoms\n"
# #     "- Treatment steps\n"
# # )


# # # ----------------------------------------------------
# # # 5. Predict a single image
# # # ----------------------------------------------------
# # def predict_single_image(image_path):
# #     image = Image.open(image_path).convert("RGB")
# #     inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(model.device)
# #     out_ids = model.generate(**inputs, max_new_tokens=300)
# #     out_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
# #     return out_text


# # # ----------------------------------------------------
# # # 6. Extract predicted disease name from text
# # # ----------------------------------------------------
# # def extract_disease_name(model_output):
# #     match = re.search(r"Disease name[:\-]\s*(.*)", model_output, re.IGNORECASE)
# #     if match:
# #         return match.group(1).strip().split("\n")[0]
# #     return None


# # # ----------------------------------------------------
# # # 7. Single folder prediction
# # # ----------------------------------------------------
# # def predict_single_folder(folder_path):
# #     for img in sorted(os.listdir(folder_path)):
# #         if not img.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
# #             continue

# #         img_path = os.path.join(folder_path, img)
# #         print(f"\n🖼️ Image: {img_path}")

# #         out_text = predict_single_image(img_path)
# #         print(f"📌 Prediction:\n{out_text}\n")


# # # ----------------------------------------------------
# # # 8. Multi-folder evaluation (accuracy)
# # # ----------------------------------------------------
# # def evaluate_multifolder_dataset(root_folder):
# #     y_true = []
# #     y_pred = []

# #     for class_name in sorted(os.listdir(root_folder)):
# #         class_folder = os.path.join(root_folder, class_name)
# #         if not os.path.isdir(class_folder):
# #             continue

# #         print(f"\n============================")
# #         print(f"🔍 Class Folder: {class_name}")
# #         print("============================")

# #         for img in sorted(os.listdir(class_folder)):
# #             if not img.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
# #                 continue

# #             img_path = os.path.join(class_folder, img)
# #             print(f"\n🖼️ {img_path}")

# #             text = predict_single_image(img_path)
# #             print(f"📌 Model Output:\n{text}")

# #             disease_pred = extract_disease_name(text)

# #             y_true.append(class_name.lower())
# #             y_pred.append("" if disease_pred is None else disease_pred.lower())

# #     # clean labels
# #     y_true = [t.replace("_", " ").strip() for t in y_true]
# #     y_pred = [p.replace("_", " ").strip() for p in y_pred]

# #     acc = accuracy_score(y_true, y_pred)

# #     print("\n============================")
# #     print("📊 FINAL ACCURACY REPORT")
# #     print("============================")
# #     print(f"🎯 Overall Accuracy: {acc*100:.2f}%")

# #     return acc


# # # ----------------------------------------------------
# # # 9. MAIN (Runs based on MODE)
# # # ----------------------------------------------------
# # if __name__ == "__main__":

# #     if MODE == 0:
# #         print("\n=== MODE 0: Single Image Prediction ===")
# #         result = predict_single_image(SINGLE_IMAGE_PATH)
# #         print("\n📌 Final Prediction:\n", result)

# #     elif MODE == 1:
# #         print("\n=== MODE 1: Single Folder Prediction ===")
# #         predict_single_folder(SINGLE_FOLDER_PATH)
# #         print("\n✔ Completed folder prediction.")

# #     elif MODE == 2:
# #         print("\n=== MODE 2: Multi-folder Evaluation ===")
# #         evaluate_multifolder_dataset(TEST_ROOT)

# #     else:
# #         print("❌ ERROR: Invalid MODE value")


# # 2. ##################################################

# import os
# import re
# from PIL import Image
# from unsloth import FastVisionModel
# from sklearn.metrics import accuracy_score


# # ----------------------------------------------------
# # MODE
# # ----------------------------------------------------
# # 0 = Single image prediction
# # 1 = Single folder accuracy
# # 2 = Multi-folder accuracy
# MODE = 2


# # ----------------------------------------------------
# # PATHS
# # ----------------------------------------------------
# MODEL_DIR = "/home/aic_u3/aic_u3/ComputerVision/unsloth_vlm/outputs/leaf_vlm_lora"

# TEST_ROOT = "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Benchmark_Dataset-CDDM_images/Benchmark_Dataset-CDDM_images/images"

# SINGLE_IMAGE_PATH = (
#     "/home/aic_u3/aic_u3/ComputerVision/DINO_large/"
#     "Benchmark_Dataset-CDDM_images/Benchmark_Dataset-CDDM_images/"
#     "images/Tomato_Leaf_Mold/plant_33562.jpg"
# )

# SINGLE_FOLDER_PATH = (
#     "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Benchmark_Dataset-CDDM_images/Benchmark_Dataset-CDDM_images/images/Potato Early Blight"
# )


# # ----------------------------------------------------
# # CANONICAL CLASS LABELS (TRAINING LABELS)
# # ----------------------------------------------------
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


# # ----------------------------------------------------
# # LABEL NORMALIZATION
# # ----------------------------------------------------
# def normalize_label(text: str) -> str:
#     """
#     Makes labels comparable:
#     - lowercase
#     - remove underscores, hyphens
#     - collapse spaces
#     """
#     text = text.lower()
#     text = re.sub(r"[_\-]", " ", text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()


# NORMALIZED_LABEL_MAP = {
#     normalize_label(lbl): lbl for lbl in CLASS_LABELS
# }


# def folder_to_label(folder_name: str) -> str:
#     """
#     Convert test folder name → closest training label
#     """
#     norm = normalize_label(folder_name)

#     for key in NORMALIZED_LABEL_MAP:
#         if key in norm or norm in key:
#             return NORMALIZED_LABEL_MAP[key]

#     return "Unknown"


# # ----------------------------------------------------
# # LOAD MODEL
# # ----------------------------------------------------
# model, processor = FastVisionModel.from_pretrained(
#     model_name=MODEL_DIR,
#     load_in_4bit=True,
#     fast_inference=False,
# )

# model.eval()


# # ----------------------------------------------------
# # PROMPT (CLASSIFICATION ONLY)
# # ----------------------------------------------------
# PROMPT = (
#     "<|vision_start|><|image_pad|><|vision_end|>\n"
#     "Identify the plant leaf disease.\n"
#     "Choose ONLY ONE label from the list below.\n\n"
#     + "\n".join(CLASS_LABELS)
#     + "\n\nAnswer with only the label name."
# )


# # ----------------------------------------------------
# # SINGLE IMAGE PREDICTION
# # ----------------------------------------------------
# def predict_single_image(image_path):
#     image = Image.open(image_path).convert("RGB")

#     inputs = processor(
#         images=image,
#         text=PROMPT,
#         return_tensors="pt"
#     ).to(model.device)

#     output_ids = model.generate(
#         **inputs,
#         max_new_tokens=8,
#         do_sample=False,
#     )

#     output = processor.batch_decode(
#         output_ids, skip_special_tokens=True
#     )[0]

#     output_norm = normalize_label(output)

#     for norm_lbl, true_lbl in NORMALIZED_LABEL_MAP.items():
#         if norm_lbl in output_norm:
#             return true_lbl

#     return "Unknown"


# # ----------------------------------------------------
# # SINGLE FOLDER ACCURACY
# # ----------------------------------------------------
# def evaluate_single_folder(folder_path):
#     folder_name = os.path.basename(folder_path)
#     true_label = folder_to_label(folder_name)

#     y_true, y_pred = [], []

#     for img in sorted(os.listdir(folder_path)):
#         if img.lower().endswith((".jpg", ".jpeg", ".png")):
#             img_path = os.path.join(folder_path, img)
#             pred = predict_single_image(img_path)

#             y_true.append(true_label)
#             y_pred.append(pred)

#             print(f"{img} -> {pred}")

#     acc = accuracy_score(y_true, y_pred)

#     print("\n============================")
#     print("📊 SINGLE FOLDER ACCURACY")
#     print("============================")
#     print(f"Folder : {folder_name}")
#     print(f"Label  : {true_label}")
#     print(f"Accuracy: {acc * 100:.2f}%")

#     return acc


# # ----------------------------------------------------
# # MULTI-FOLDER ACCURACY
# # ----------------------------------------------------
# def evaluate_multifolder_dataset(root_folder):
#     y_true, y_pred = [], []

#     for folder in sorted(os.listdir(root_folder)):
#         folder_path = os.path.join(root_folder, folder)
#         if not os.path.isdir(folder_path):
#             continue

#         true_label = folder_to_label(folder)

#         for img in os.listdir(folder_path):
#             if img.lower().endswith((".jpg", ".jpeg", ".png")):
#                 img_path = os.path.join(folder_path, img)
#                 pred = predict_single_image(img_path)

#                 y_true.append(true_label)
#                 y_pred.append(pred)

#     acc = accuracy_score(y_true, y_pred)

#     print("\n============================")
#     print("📊 MULTI-FOLDER ACCURACY")
#     print("============================")
#     print(f"Accuracy: {acc * 100:.2f}%")

#     return acc


# # ----------------------------------------------------
# # MAIN
# # ----------------------------------------------------
# if __name__ == "__main__":

#     if MODE == 0:
#         print(predict_single_image(SINGLE_IMAGE_PATH))

#     elif MODE == 1:
#         evaluate_single_folder(SINGLE_FOLDER_PATH)

#     elif MODE == 2:
#         evaluate_multifolder_dataset(TEST_ROOT)

#     else:
#         print("Invalid MODE")


# 3.###########################################################
# import os
# import re
# from PIL import Image
# from unsloth import FastVisionModel
# from sklearn.metrics import accuracy_score


# # ----------------------------------------------------
# # MODE
# # ----------------------------------------------------
# # 0 = Single image prediction
# # 1 = Single folder accuracy
# # 2 = Multi-folder accuracy
# MODE = 0


# # ----------------------------------------------------
# # PATHS
# # ----------------------------------------------------
# MODEL_DIR = "/home/aic_u3/aic_u3/ComputerVision/unsloth_vlm/outputs/leaf_vlm_lora"

# TEST_ROOT = "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Benchmark_Dataset-CDDM_images/Benchmark_Dataset-CDDM_images/images"

# SINGLE_IMAGE_PATH = (
#     "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Benchmark_Dataset-CDDM_images/Benchmark_Dataset-CDDM_images/images/Tomato Target Spot/plant_50957.jpg"
# )

# SINGLE_FOLDER_PATH = (
#     "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Benchmark_Dataset-CDDM_images/Benchmark_Dataset-CDDM_images/images/Tomato YellowLeaf Curl Virus"
# )


# # ----------------------------------------------------
# # CANONICAL CLASS LABELS (TRAINING LABELS)
# # ----------------------------------------------------
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


# # ----------------------------------------------------
# # LABEL NORMALIZATION
# # ----------------------------------------------------
# def normalize_label(text: str) -> str:
#     text = text.lower()
#     text = re.sub(r"[_\-]", " ", text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()


# NORMALIZED_LABEL_MAP = {
#     normalize_label(lbl): lbl for lbl in CLASS_LABELS
# }


# def folder_to_label(folder_name: str) -> str:
#     norm = normalize_label(folder_name)
#     for key in NORMALIZED_LABEL_MAP:
#         if key in norm or norm in key:
#             return NORMALIZED_LABEL_MAP[key]
#     return "Unknown"


# # ----------------------------------------------------
# # LOAD MODEL
# # ----------------------------------------------------
# model, processor = FastVisionModel.from_pretrained(
#     model_name=MODEL_DIR,
#     load_in_4bit=True,
#     fast_inference=False,
# )

# model.eval()


# # ----------------------------------------------------
# # PROMPT
# # ----------------------------------------------------
# PROMPT = (
#     "<|vision_start|><|image_pad|><|vision_end|>\n"
#     "Identify the plant leaf disease.\n"
#     "Choose ONLY ONE label from the list below.\n\n"
#     + "\n".join(CLASS_LABELS)
#     + "\n\nAnswer with only the label name."
# )


# # ----------------------------------------------------
# # SINGLE IMAGE PREDICTION
# # ----------------------------------------------------
# def predict_single_image(image_path):
#     image = Image.open(image_path).convert("RGB")

#     inputs = processor(
#         images=image,
#         text=PROMPT,
#         return_tensors="pt"
#     ).to(model.device)

#     output_ids = model.generate(
#         **inputs,
#         max_new_tokens=8,
#         do_sample=False,
#     )

#     output = processor.batch_decode(
#         output_ids, skip_special_tokens=True
#     )[0]

#     output_norm = normalize_label(output)

#     for norm_lbl, true_lbl in NORMALIZED_LABEL_MAP.items():
#         if norm_lbl in output_norm:
#             return true_lbl

#     return "Unknown"


# # ----------------------------------------------------
# # SINGLE FOLDER ACCURACY
# # ----------------------------------------------------
# def evaluate_single_folder(folder_path):
#     folder_name = os.path.basename(folder_path)
#     true_label = folder_to_label(folder_name)

#     y_true, y_pred = [], []

#     for img in sorted(os.listdir(folder_path)):
#         if img.lower().endswith((".jpg", ".jpeg", ".png")):
#             img_path = os.path.join(folder_path, img)
#             pred = predict_single_image(img_path)

#             y_true.append(true_label)
#             y_pred.append(pred)

#             print(f"{img} -> {pred}")

#     total = len(y_true)
#     correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
#     wrong = total - correct
#     acc = accuracy_score(y_true, y_pred)

#     print("\n============================")
#     print("📊 SINGLE FOLDER ACCURACY")
#     print("============================")
#     print(f"Folder               : {folder_name}")
#     print(f"True label           : {true_label}")
#     print(f"Total images         : {total}")
#     print(f"Correct predictions  : {correct}")
#     print(f"Wrong predictions    : {wrong}")
#     print(f"Accuracy             : {acc * 100:.2f}%")

#     return acc


# # ----------------------------------------------------
# # MULTI-FOLDER ACCURACY
# # ----------------------------------------------------
# # def evaluate_multifolder_dataset(root_folder):
# #     y_true, y_pred = [], []

# #     for folder in sorted(os.listdir(root_folder)):
# #         folder_path = os.path.join(root_folder, folder)
# #         if not os.path.isdir(folder_path):
# #             continue

# #         true_label = folder_to_label(folder)

# #         for img in os.listdir(folder_path):
# #             if img.lower().endswith((".jpg", ".jpeg", ".png")):
# #                 img_path = os.path.join(folder_path, img)
# #                 pred = predict_single_image(img_path)

# #                 y_true.append(true_label)
# #                 y_pred.append(pred)

# #     total = len(y_true)
# #     correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
# #     wrong = total - correct
# #     acc = accuracy_score(y_true, y_pred)

# #     print("\n============================")
# #     print("📊 MULTI-FOLDER ACCURACY")
# #     print("============================")
# #     print(f"Total images         : {total}")
# #     print(f"Correct predictions  : {correct}")
# #     print(f"Wrong predictions    : {wrong}")
# #     print(f"Accuracy             : {acc * 100:.2f}%")

# #     return acc


# def evaluate_multifolder_dataset(root_folder):
#     y_true, y_pred = [], []
#     total_images = 0

#     for folder in sorted(os.listdir(root_folder)):
#         folder_path = os.path.join(root_folder, folder)

#         if not os.path.isdir(folder_path):
#             continue

#         true_label = folder_to_label(folder)

#         images = [
#             f for f in os.listdir(folder_path)
#             if f.lower().endswith((".jpg", ".jpeg", ".png"))
#         ]

#         if len(images) == 0:
#             print(f"⚠️ No images found in folder: {folder}")
#             continue

#         print(f"\nProcessing folder: {folder} ({len(images)} images)")

#         for img in images:
#             img_path = os.path.join(folder_path, img)
#             pred = predict_single_image(img_path)

#             y_true.append(true_label)
#             y_pred.append(pred)
#             total_images += 1

#     if total_images == 0:
#         print("\n❌ No images processed. Check TEST_ROOT path.")
#         return None

#     correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
#     wrong = total_images - correct
#     acc = accuracy_score(y_true, y_pred)

#     print("\n============================")
#     print("📊 MULTI-FOLDER ACCURACY")
#     print("============================")
#     print(f"Total images         : {total_images}")
#     print(f"Correct predictions  : {correct}")
#     print(f"Wrong predictions    : {wrong}")
#     print(f"Accuracy             : {acc * 100:.2f}%")

#     return acc


# # ----------------------------------------------------
# # MAIN
# # ----------------------------------------------------
# if __name__ == "__main__":

#     if MODE == 0:
#         print(predict_single_image(SINGLE_IMAGE_PATH))

#     elif MODE == 1:
#         evaluate_single_folder(SINGLE_FOLDER_PATH)

#     elif MODE == 2:
#         evaluate_multifolder_dataset(TEST_ROOT)

#     else:
#         print("Invalid MODE")


# # 4 #######################
# import os
# import re
# from PIL import Image
# from unsloth import FastVisionModel
# from sklearn.metrics import accuracy_score

# # ----------------------------------------------------
# # MODE
# # 0 = Single image prediction
# # 1 = Single folder accuracy
# # 2 = Multi-folder accuracy
# MODE = 0

# # ----------------------------------------------------
# # PATHS
# MODEL_DIR = "/home/aic_u3/aic_u3/ComputerVision/unsloth_vlm/outputs/leaf_vlm_lora"
# TEST_ROOT = "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Benchmark_Dataset-CDDM_images/Benchmark_Dataset-CDDM_images/images"
# SINGLE_IMAGE_PATH = os.path.join(TEST_ROOT, "Potato Early Blight/plant_64488.jpg")
# SINGLE_FOLDER_PATH = os.path.join(TEST_ROOT, "Tomato Yellow Leaf Curl Virus")

# # ----------------------------------------------------
# # CLASS LABELS
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

# # ----------------------------------------------------
# # NORMALIZE LABELS
# def normalize_label(text: str) -> str:
#     text = text.lower()
#     text = re.sub(r"[_\-]", " ", text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()

# NORMALIZED_LABEL_MAP = {normalize_label(lbl): lbl for lbl in CLASS_LABELS}

# def folder_to_label(folder_name: str) -> str:
#     norm = normalize_label(folder_name)
#     for key, true_lbl in NORMALIZED_LABEL_MAP.items():
#         if key == norm:
#             return true_lbl
#     return "Unknown"

# # ----------------------------------------------------
# # LOAD MODEL
# model, processor = FastVisionModel.from_pretrained(
#     model_name=MODEL_DIR,
#     load_in_4bit=True,
#     fast_inference=False,
# )
# model.eval()

# # ----------------------------------------------------
# # PROMPT
# PROMPT = (
#     "<|vision_start|><|image_pad|><|vision_end|>\n"
#     "Identify the plant leaf disease.\n"
#     "Choose ONLY ONE label from the list below.\n\n"
#     + "\n".join(CLASS_LABELS)
#     + "\n\nAnswer with only the label name."
# )

# # ----------------------------------------------------
# # SINGLE IMAGE PREDICTION
# def predict_single_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(model.device)

#     output_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
#     output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

#     output_norm = normalize_label(output)
#     # strict matching
#     return NORMALIZED_LABEL_MAP.get(output_norm, "Unknown")

# # ----------------------------------------------------
# # SINGLE FOLDER ACCURACY
# def evaluate_single_folder(folder_path):
#     folder_name = os.path.basename(folder_path)
#     true_label = folder_to_label(folder_name)

#     y_true, y_pred = [], []

#     for img in sorted(os.listdir(folder_path)):
#         if img.lower().endswith((".jpg", ".jpeg", ".png")):
#             img_path = os.path.join(folder_path, img)
#             pred = predict_single_image(img_path)
#             y_true.append(true_label)
#             y_pred.append(pred)
#             print(f"{img} -> {pred}")

#     total = len(y_true)
#     correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
#     acc = accuracy_score(y_true, y_pred)

#     print("\n============================")
#     print("📊 SINGLE FOLDER ACCURACY")
#     print("============================")
#     print(f"Folder               : {folder_name}")
#     print(f"True label           : {true_label}")
#     print(f"Total images         : {total}")
#     print(f"Correct predictions  : {correct}")
#     print(f"Wrong predictions    : {total - correct}")
#     print(f"Accuracy             : {acc * 100:.2f}%")

#     return acc

# # ----------------------------------------------------
# # MULTI-FOLDER ACCURACY
# def evaluate_multifolder_dataset(root_folder):
#     y_true, y_pred = [], []
#     total_images = 0

#     for folder in sorted(os.listdir(root_folder)):
#         folder_path = os.path.join(root_folder, folder)
#         if not os.path.isdir(folder_path):
#             continue

#         true_label = folder_to_label(folder)
#         images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

#         if len(images) == 0:
#             print(f"⚠️ No images in folder: {folder}")
#             continue

#         print(f"\nProcessing folder: {folder} ({len(images)} images)")
#         for img in images:
#             img_path = os.path.join(folder_path, img)
#             pred = predict_single_image(img_path)
#             y_true.append(true_label)
#             y_pred.append(pred)
#             total_images += 1

#     if total_images == 0:
#         print("\n❌ No images processed. Check TEST_ROOT path.")
#         return None

#     correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
#     acc = accuracy_score(y_true, y_pred)

#     print("\n============================")
#     print("📊 MULTI-FOLDER ACCURACY")
#     print("============================")
#     print(f"Total images         : {total_images}")
#     print(f"Correct predictions  : {correct}")
#     print(f"Wrong predictions    : {total_images - correct}")
#     print(f"Accuracy             : {acc * 100:.2f}%")

#     return acc

# # ----------------------------------------------------
# # MAIN
# if __name__ == "__main__":
#     if MODE == 0:
#         print(predict_single_image(SINGLE_IMAGE_PATH))
#     elif MODE == 1:
#         evaluate_single_folder(SINGLE_FOLDER_PATH)
#     elif MODE == 2:
#         evaluate_multifolder_dataset(TEST_ROOT)
#     else:
#         print("Invalid MODE")

# # 5.#########################################################
import os
import re
from PIL import Image
from unsloth import FastVisionModel
from sklearn.metrics import accuracy_score

# -------------------------
# MODE
# 0 = Single image prediction
# 1 = Single folder accuracy
# 2 = Multi-folder accuracy
MODE = 2

# Set to True to show individual predictions (False = only summary)
VERBOSE = False

# Limit images per folder for faster evaluation (set to None to disable)
MAX_IMAGES_PER_FOLDER = 5

# -------------------------
# PATHS
MODEL_DIR = "/home/aic_u3/aic_u3/ComputerVision/unsloth_vlm/outputs/leaf_vlm_lora"
TEST_ROOT = "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Benchmark_Dataset-CDDM_images/Benchmark_Dataset-CDDM_images/images"
SINGLE_IMAGE_PATH = os.path.join(TEST_ROOT, "Tomato Leaf Mold/plant_33557.jpg")
SINGLE_FOLDER_PATH = os.path.join(TEST_ROOT, "Tomato Yellow Leaf Curl Virus")

# -------------------------
# CLASS LABELS
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
# Normalize labels
def normalize_label(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

NORMALIZED_LABEL_MAP = {normalize_label(lbl): lbl for lbl in CLASS_LABELS}

# -------------------------
# FOLDER TO LABEL MAPPING
def folder_to_label(folder_name: str) -> str:
    """Convert test folder name to closest matching training label."""
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
# LOAD MODEL
model, processor = FastVisionModel.from_pretrained(
    model_name=MODEL_DIR,
    load_in_4bit=True,
    fast_inference=False,
)
model.eval()

# -------------------------
# PROMPT - MUST MATCH TRAINING FORMAT
# Training format from dataset_to_jsonl.py:
# "<|vision_start|><|image_pad|><|vision_end|>\nIdentify the plant and the disease from the image and provide:\n- Plant name\n- Disease name\n- Symptoms\n- Treatment steps\n\n<LABEL>"
USER_INSTRUCTION = (
    "Identify the plant and the disease from the image and provide:\n"
    "- Plant name\n"
    "- Disease name\n"
    "- Symptoms\n"
    "- Treatment steps"
)

PROMPT = (
    "<|vision_start|><|image_pad|><|vision_end|>\n"
    + USER_INSTRUCTION
)

# -------------------------
# PREDICT SINGLE IMAGE
def predict_single_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(model.device)

    # generate text - reduced tokens since we only need the label (model outputs label first)
    output_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    
    # Decode full output
    full_output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    # Extract generated text by finding where the prompt ends
    # The prompt ends with "Treatment steps\n\n", after which the model should generate the label
    prompt_end_marker = "Treatment steps"
    if prompt_end_marker in full_output:
        # Find the position after the prompt end marker and newlines
        idx = full_output.find(prompt_end_marker)
        if idx != -1:
            # Find the start of generated text (after marker and newlines)
            start_idx = full_output.find('\n', idx + len(prompt_end_marker))
            if start_idx != -1:
                # Skip newlines to get to actual generated text
                while start_idx < len(full_output) and full_output[start_idx] in ['\n', '\r', ' ']:
                    start_idx += 1
                generated_text = full_output[start_idx:].strip()
            else:
                generated_text = ""
        else:
            generated_text = ""
    else:
        # Fallback: try to extract by token IDs
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_length:]
        if len(generated_ids) > 0:
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        else:
            generated_text = ""
    
    # Debug output to diagnose misclassification
    first_line = generated_text.split('\n')[0].strip() if '\n' in generated_text else generated_text.strip()
    print(f"\n[DEBUG] First line from model: {repr(first_line)}")
    print(f"[DEBUG] First 200 chars of generated text: {repr(generated_text[:200])}")
    
    # The model was trained to output the label directly after the prompt (e.g., "Potato Early Blight")
    # Try multiple extraction strategies
    
    # Strategy 1: Check if generated text IS a label (exact match after normalization)
    generated_text_norm = normalize_label(generated_text)
    if generated_text_norm in NORMALIZED_LABEL_MAP:
        return NORMALIZED_LABEL_MAP[generated_text_norm]
    
    # Strategy 1b: Check first line only (model usually outputs label on first line)
    first_line = generated_text.split('\n')[0].strip() if '\n' in generated_text else generated_text.strip()
    first_line_norm = normalize_label(first_line)
    if first_line_norm in NORMALIZED_LABEL_MAP:
        return NORMALIZED_LABEL_MAP[first_line_norm]
    # Also check if first line starts with a label
    for norm_label, label in sorted(NORMALIZED_LABEL_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if first_line_norm.startswith(norm_label):
            return label
    
    # Strategy 2: Check each line separately (model might output label on first line)
    # Prioritize first line as it's most likely to contain the label
    lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
    for line in lines[:3]:  # Only check first 3 lines to avoid false matches
        line_norm = normalize_label(line)
        # Exact match first
        if line_norm in NORMALIZED_LABEL_MAP:
            return NORMALIZED_LABEL_MAP[line_norm]
        # Check if line starts with a label (most reliable)
        for norm_label, label in NORMALIZED_LABEL_MAP.items():
            if line_norm.startswith(norm_label):
                return label
        # Fallback: check if label is in line (but prefer longer matches)
        best_match = None
        best_length = 0
        for norm_label, label in NORMALIZED_LABEL_MAP.items():
            if norm_label in line_norm and len(norm_label) > best_length:
                best_match = label
                best_length = len(norm_label)
        if best_match:
            return best_match
    
    # Strategy 3: Look for disease name in structured format
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
    
    # Strategy 4: Search for labels in the beginning of generated text (first 200 chars)
    # Models typically output the label at the start, so prioritize matches there
    text_start = generated_text_norm[:200]  # Check first 200 chars
    best_match = None
    best_pos = len(text_start)  # Prefer matches closer to start
    best_length = 0  # Prefer longer/more specific matches
    
    for norm_label, label in NORMALIZED_LABEL_MAP.items():
        pos = text_start.find(norm_label)
        if pos != -1:
            # Found match - prefer earlier and longer matches
            if pos < best_pos or (pos == best_pos and len(norm_label) > best_length):
                best_match = label
                best_pos = pos
                best_length = len(norm_label)
    
    if best_match:
        return best_match
    
    # Fallback: search entire text but prefer longer matches
    for norm_label, label in sorted(NORMALIZED_LABEL_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if norm_label in generated_text_norm:
            return label
    
    # Strategy 5: Check individual words/phrases
    words = generated_text_norm.split()
    for i in range(len(words)):
        for j in range(i+1, min(i+5, len(words)+1)):  # Check up to 4-word phrases
            phrase = ' '.join(words[i:j])
            if phrase in NORMALIZED_LABEL_MAP:
                return NORMALIZED_LABEL_MAP[phrase]

    # fallback: if nothing matches, return Unknown
    print(f"[DEBUG] No match found. Generated: {repr(generated_text)}")
    print(f"[DEBUG] Normalized: {repr(generated_text_norm)}")
    return "Unknown"

# -------------------------
# SINGLE FOLDER ACCURACY
def evaluate_single_folder(folder_path):
    folder_name = os.path.basename(folder_path)
    true_label = folder_to_label(folder_name)

    y_true, y_pred = [], []

    for img in sorted(os.listdir(folder_path)):
        if img.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, img)
            pred = predict_single_image(img_path)
            y_true.append(true_label)
            y_pred.append(pred)
            if VERBOSE:
                print(f"{img} -> {pred}")

    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = accuracy_score(y_true, y_pred)

    print("\n============================")
    print("📊 SINGLE FOLDER ACCURACY")
    print("============================")
    print(f"Folder               : {folder_name}")
    print(f"True label           : {true_label}")
    print(f"Total images         : {total}")
    print(f"Correct predictions  : {correct}")
    print(f"Wrong predictions    : {total - correct}")
    print(f"Accuracy             : {acc * 100:.2f}%")

    return acc

# -------------------------
# MULTI-FOLDER ACCURACY
def evaluate_multifolder_dataset(root_folder):
    y_true, y_pred = [], []
    total_images = 0
    per_class_stats = {}  # Track accuracy per class

    for folder in sorted(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, folder)
        if not os.path.isdir(folder_path):
            continue

        true_label = folder_to_label(folder)
        images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        images = sorted(images)
        if MAX_IMAGES_PER_FOLDER is not None:
            images = images[:MAX_IMAGES_PER_FOLDER]

        if len(images) == 0:
            print(f"⚠️ No images in folder: {folder}")
            continue

        print(f"\nProcessing folder: {folder} ({len(images)} images)")
        folder_correct = 0
        folder_predictions = []
        for img in images:
            img_path = os.path.join(folder_path, img)
            pred = predict_single_image(img_path)
            y_true.append(true_label)
            y_pred.append(pred)
            folder_predictions.append(pred)
            total_images += 1
            if pred == true_label:
                folder_correct += 1
            if VERBOSE:
                status = "✓" if pred == true_label else "✗"
                print(f"  {status} {img} -> {pred}")
        
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
if __name__ == "__main__":
    if MODE == 0:
        print(predict_single_image(SINGLE_IMAGE_PATH))
    elif MODE == 1:
        evaluate_single_folder(SINGLE_FOLDER_PATH)
    elif MODE == 2:
        evaluate_multifolder_dataset(TEST_ROOT)
    else:
        print("Invalid MODE")
