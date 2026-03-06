# Daily Update

## Date: 05/06/2026

### Work Done
- Worked on the **20k+ Multi-Class Crop Disease Images Dataset**.
- Worked on the **Wheat Leaf Disease Dataset**.
- Preprocessed and explored the datasets.
- Trained the **EfficientNet-B0 (base model)** for crop disease classification.

### Notes
- Dataset includes multiple crop disease classes.
- Model training was performed using transfer learning with EfficientNet.
- Further evaluation and fine-tuning will be done in the next steps.


## 📅 Daily Update — 06 March 2026

### 🧠 Model Development
- Started training **tf_efficientnetv2_s** for wheat disease classification.
- Model Parameters:
  - Total Parameters: **20,187,736**
  - Trainable Parameters: **20,187,736**

### 🌾 Classes Selected
Defined **8 wheat disease classes** for the model:

- brown_rust
- healthy
- loose_smut
- powdery_mildew
- scab
- septoria
- stem_rust
- yellow_rust

### 📂 Dataset Work
Combined multiple datasets for training:

- 20k Multi-Class Crop Disease Images
- Wheat Leaf Disease Dataset
- CGIAR Crop Disease Dataset
- Wheat Plant Diseases Dataset

Filtered out **non-wheat disease folders** from the datasets.

### 🧹 Data Cleaning
- Validated images and removed corrupt/tiny images.
- Final valid dataset prepared for training.

### ⚖️ Handling Class Imbalance
Applied **class weights** to balance the dataset during training.

### 🔁 Data Augmentation
Performed augmentation for minority classes including:
- loose_smut
- scab
- stem_rust
- powdery_mildew

### 📊 Dataset Preparation
Created **train / validation / test splits** for the final dataset.

### ⚙️ Training Strategy
Implemented **two-stage training pipeline**:

**Phase 1**
- Train classifier head only.

**Phase 2**
- Full fine-tuning of the model.

### 🚀 Current Status
- Dataset preprocessing completed
- Model architecture finalized
- Training pipeline implemented
- Training currently running.
- Results and evaluation will be added after training completion.
