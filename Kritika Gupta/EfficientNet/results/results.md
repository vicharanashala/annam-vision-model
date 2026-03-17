# Wheat Disease Classification — Results Report


This document summarizes the dataset preparation, baseline model performance, advanced model experiments, and architecture comparison conducted across the following notebooks:

* 01_baseline.ipynb
* 02_dataset_builder.ipynb
* 03_efficientnetv2s.ipynb
* 04_model_comparison.ipynb

---

# 01_baseline.ipynb

## Objective

Train a **baseline deep learning model** for wheat leaf disease classification using EfficientNet.

## Dataset Used

* Wheat Leaf Disease Dataset-1
* Wheat Leaf Disease Dataset-2

These datasets contain images of plant leaves belonging to different wheat disease classes.

---

## Model Architecture

**EfficientNet-B0 (Base Model)**

Transfer learning was applied using pretrained ImageNet weights.

Training strategy:

1. Freeze backbone initially
2. Train classifier layer
3. Unfreeze backbone for fine-tuning

Trainable parameters after unfreezing:

**4,336,769 parameters**

---

## Training Configuration

| Parameter     | Value             |
| ------------- | ----------------- |
| Framework     | PyTorch           |
| Model         | EfficientNet-B0   |
| Image Size    | 224 × 224         |
| Optimizer     | Adam              |
| Loss Function | CrossEntropyLoss  |
| LR Scheduler  | CosineAnnealingLR |
| Epochs        | 30                |

---

## Training Progress

| Epoch | Train Accuracy | Validation Accuracy |
| ----- | -------------- | ------------------- |
| 1     | 71.56%         | 81.16%              |
| 10    | 86.08%         | 89.37%              |
| 20    | 91.38%         | 95.17%              |
| 29    | 92.42%         | **96.62%**          |

---

## Final Baseline Results

| Metric                   | Score      |
| ------------------------ | ---------- |
| Best Validation Accuracy | **96.62%** |
| Final Training Accuracy  | **92.11%** |
| Final Validation Loss    | **0.1108** |

---

## Observations

* EfficientNet-B0 achieved strong performance on wheat disease classification.
* Fine-tuning the backbone improved validation accuracy.
* Transfer learning significantly reduced training time.

---

# 02_dataset_builder.ipynb

## Objective

Create a **clean unified dataset** by merging multiple crop disease datasets and filtering wheat classes.

---

## Source Datasets

* 20k Multi-Class Crop Disease Images
* Wheat Leaf Disease Dataset
* CGIAR Computer Vision Crop Disease Dataset
* Wheat Plant Disease Dataset

Non-wheat disease folders were automatically filtered.

---

## Wheat Classes

Final dataset contains **8 wheat classes**:

* brown_rust
* healthy
* loose_smut
* powdery_mildew
* scab
* septoria
* stem_rust
* yellow_rust

---

## Raw Dataset Statistics

| Class          | Images |
| -------------- | ------ |
| brown_rust     | 2632   |
| healthy        | 2915   |
| loose_smut     | 1189   |
| powdery_mildew | 211    |
| scab           | 118    |
| septoria       | 1671   |
| stem_rust      | 528    |
| yellow_rust    | 2807   |

Total raw images collected:

**12,071 images**

---

## Image Validation

Invalid or corrupt images were removed.

Examples:

* yellow_rust → 2807 → **2671 valid**
* brown_rust → 2632 → **2410 valid**
* loose_smut → 1189 → **1186 valid**

Final cleaned dataset size:

**11,710 images**

---

## Data Augmentation

Augmentation was applied to underrepresented classes:

* loose_smut
* scab
* stem_rust
* powdery_mildew

---

## Final Dataset Split

| Class          | Train | Validation | Test | Total |
| -------------- | ----- | ---------- | ---- | ----- |
| brown_rust     | 1628  | 348        | 350  | 2326  |
| healthy        | 1948  | 417        | 419  | 2784  |
| loose_smut     | 840   | 180        | 150  | 1170  |
| powdery_mildew | 840   | 180        | 33   | 1053  |
| scab           | 840   | 180        | 19   | 1039  |
| septoria       | 1169  | 250        | 252  | 1671  |
| stem_rust      | 840   | 180        | 69   | 1089  |
| yellow_rust    | 1803  | 386        | 387  | 2576  |

---

# 03_efficientnetv2s.ipynb

## Objective

Train a **larger EfficientNet architecture** to evaluate potential performance improvements.

---

## Model Architecture

Model: **tf_efficientnetv2_s**

| Property             | Value      |
| -------------------- | ---------- |
| Total Parameters     | 20,187,736 |
| Trainable Parameters | 20,187,736 |

---

## Class Weights

| Class          | Weight |
| -------------- | ------ |
| brown_rust     | 0.6732 |
| healthy        | 0.5626 |
| loose_smut     | 1.3047 |
| powdery_mildew | 1.3047 |
| scab           | 1.3047 |
| septoria       | 0.9375 |
| stem_rust      | 1.3047 |
| yellow_rust    | 0.6079 |

These weights were applied to address dataset imbalance.

---

## Training Strategy

Two-phase training pipeline.

### Phase 1 — Train Classifier Head (5 epochs)

| Epoch | Train Acc | Val Acc |
| ----- | --------- | ------- |
| 1     | 0.1847    | 0.3187  |
| 2     | 0.3202    | 0.4729  |
| 3     | 0.4101    | 0.5639  |
| 4     | 0.4657    | 0.6167  |
| 5     | 0.5102    | 0.6506  |

---

### Phase 2 — Full Fine-Tuning

| Epoch | Train Acc | Val Acc    |
| ----- | --------- | ---------- |
| 1     | 0.6173    | 0.7822     |
| 2     | 0.6975    | 0.8152     |
| 3     | 0.7469    | 0.8326     |
| 4     | 0.7799    | 0.8472     |
| 5     | 0.8070    | **0.8637** |

Current best validation accuracy:

**86.37%**

Training planned for **100 epochs**.

---

# 04_model_comparison.ipynb

## Objective

Compare different EfficientNet architectures for crop disease classification.

---

## Models Evaluated

* EfficientNet-B0
* EfficientNetV2-S
* EfficientNet-B7

---

## Architecture Comparison

| Model            | Parameters | Characteristics                                   |
| ---------------- | ---------- | ------------------------------------------------- |
| EfficientNet-B0  | ~5.3M      | Lightweight baseline                              |
| EfficientNetV2-S | ~20M       | Faster training and improved scaling              |
| EfficientNet-B7  | ~66M       | Large architecture with higher computational cost |

---

## Key Insights

* EfficientNet-B0 already achieves **96.62% validation accuracy**.
* EfficientNetV2-S provides improved architecture but requires more training.
* EfficientNet-B7 may provide further improvements but requires significantly more compute resources.

---

# Overall Summary

| Model            | Validation Accuracy           |
| ---------------- | ----------------------------- |
| EfficientNet-B0  | **96.62%**                    |
| EfficientNetV2-S | **86.37%** (training ongoing) |

EfficientNet-B0 currently serves as the **strong baseline model**, while EfficientNetV2-S and EfficientNet-B7 are being explored for further improvements.
