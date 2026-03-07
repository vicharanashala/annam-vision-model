# Weekly Progress Report

**Project:** Wheat Leaf Disease Classification
**Week:** 02 March 2026 – 07 March 2026

---

# 1. Objective

The goal of this work is to build a **deep learning model for wheat disease classification** using multiple crop disease datasets.
The model aims to automatically identify wheat leaf diseases from images using transfer learning with EfficientNet architectures.

---

# 2. Datasets Used

The following datasets were used and combined for training:

* 20k+ Multi-Class Crop Disease Images Dataset
* Wheat Leaf Disease Dataset (Dataset-1)
* Wheat Leaf Disease Dataset (Dataset-2)
* CGIAR Crop Disease Dataset
* Wheat Plant Disease Dataset

These datasets contain images of plant leaves representing different crop diseases.

---

# 3. Data Preparation

Several preprocessing steps were performed to prepare the dataset for model training:

* Filtered out **non-wheat disease classes**
* Validated images and removed **corrupt or invalid images**
* Combined datasets into a **single unified dataset**
* Applied **data augmentation** to improve class balance
* Created **train / validation / test splits**

### Wheat Classes Used

The final dataset contains **8 wheat disease classes**:

* brown_rust
* healthy
* loose_smut
* powdery_mildew
* scab
* septoria
* stem_rust
* yellow_rust

---

# 4. Model Development

## Initial Model

**EfficientNet-B0 (Base Model)**

Transfer learning was applied using pretrained ImageNet weights.

### Training Configuration

| Parameter     | Value             |
| ------------- | ----------------- |
| Framework     | PyTorch           |
| Model         | EfficientNet-B0   |
| Image Size    | 224 × 224         |
| Optimizer     | Adam              |
| Loss Function | CrossEntropyLoss  |
| LR Scheduler  | CosineAnnealingLR |
| Epochs        | 30                |

### Trainable Parameters

4,336,769 parameters

---

# 5. Model Training Results

The EfficientNet-B0 model was trained for **30 epochs**.

### Training Progress

Epoch 1
Train Accuracy: **71.56%**
Validation Accuracy: **81.16%**

Epoch 10
Train Accuracy: **86.08%**
Validation Accuracy: **89.37%**

Epoch 20
Train Accuracy: **91.38%**
Validation Accuracy: **95.17%**

Epoch 29
Train Accuracy: **92.42%**
Validation Accuracy: **96.62% (Best)**

---

# 6. Final Model Performance

| Metric                   | Score      |
| ------------------------ | ---------- |
| Best Validation Accuracy | **96.62%** |
| Final Training Accuracy  | **92.11%** |
| Final Validation Loss    | **0.1108** |

### Observations

* EfficientNet-B0 performed very well for crop disease classification.
* Transfer learning significantly improved model performance.
* Fine-tuning the backbone helped achieve higher accuracy.
* The model achieved **over 96% validation accuracy**, showing strong ability to distinguish between disease classes.

---

# 7. Additional Model Experimentation

To explore further improvements, work began on training:

### EfficientNetV2-S

Model parameters:

* Total Parameters: **20,187,736**
* Trainable Parameters: **20,187,736**

A **two-stage training strategy** was implemented:

**Phase 1:** Train classifier head only
**Phase 2:** Full fine-tuning of all layers

Training is currently ongoing.

---

# 8. Model Research

Research was conducted on **EfficientNet-B7** to evaluate whether a larger architecture could improve performance.

### Model Comparison

| Model            | Parameters | Characteristics                           |
| ---------------- | ---------- | ----------------------------------------- |
| EfficientNetV2-S | ~20M       | Faster training, optimized architecture   |
| EfficientNet-B7  | ~66M       | Larger network, higher computational cost |

### Key Observations

**EfficientNetV2-S**

* Faster training
* Efficient architecture
* Suitable for medium-sized datasets

**EfficientNet-B7**

* Larger model
* Higher GPU memory requirement
* Potentially higher accuracy with very large datasets

---

# 9. Current Status

* Dataset preprocessing completed
* Multiple datasets successfully combined
* EfficientNet-B0 baseline model trained
* Achieved **96.62% validation accuracy**
* EfficientNetV2-S training currently running
* Research on larger EfficientNet architectures completed

---

# 10. Next Steps

* Complete EfficientNetV2-S training
* Compare performance with EfficientNet-B0
* Experiment with EfficientNet-B7
* Perform further hyperparameter tuning
* Evaluate models on unseen test images
