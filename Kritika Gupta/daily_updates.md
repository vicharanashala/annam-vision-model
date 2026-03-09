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


## 06 March 2026

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

## 07 March 2026

### 🧠 Model Training
Continued training of the **tf_efficientnetv2_s** model for wheat disease classification.

- Architecture: **EfficientNetV2-S**
- Total Parameters: **20,187,736**
- Training pipeline from previous day continued.
- Monitoring training loss and validation accuracy during fine-tuning phase.

### 🔬 Model Research
Conducted research on **EfficientNet-B7** architecture to evaluate whether a larger model could improve performance on the wheat disease dataset.

Studied:

- Model architecture
- Parameter size
- Computational requirements
- Performance reported in research papers
- Suitability for plant disease classification tasks

### ⚖️ Model Comparison

Comparison between **EfficientNetV2-S** and **EfficientNet-B7**:

| Model | Parameters | Characteristics |
|------|------|------|
| EfficientNetV2-S | ~20M | Faster training, optimized architecture, good balance between speed and accuracy |
| EfficientNet-B7 | ~66M | Much deeper network, higher computational cost, often higher accuracy on large datasets |

### 📊 Observations from Research

- **EfficientNetV2-S**
  - Faster training
  - More optimized for modern training pipelines
  - Better for medium-sized datasets

- **EfficientNet-B7**
  - Larger and more computationally expensive
  - Potentially higher accuracy with very large datasets
  - Requires more GPU memory and longer training time

### 📂 Dataset Work
Continued using the combined wheat disease dataset prepared earlier.

Dataset contains **8 wheat disease classes**:
- brown_rust
- healthy
- loose_smut
- powdery_mildew
- scab
- septoria
- stem_rust
- yellow_rust

### ⚙️ Current Status
- Model training still running
- Monitoring validation performance
- Researching larger architectures for potential improvement
- Results and final evaluation will be added after training completion


## 📅 09 March 2026

### 📂 Dataset Preparation
- Worked on building a clean wheat disease dataset using multiple sources.
- Combined the following datasets:
  - 20k+ Multi-Class Crop Disease Images
  - Wheat Leaf Disease Dataset
  - CGIAR Crop Disease Dataset
  - Wheat Plant Disease Dataset
- Filtered out **non-wheat disease classes** from the datasets.
- Validated images and removed corrupted or invalid files.
- Final dataset prepared with **8 wheat disease classes**:
  - brown_rust
  - healthy
  - loose_smut
  - powdery_mildew
  - scab
  - septoria
  - stem_rust
  - yellow_rust

### 🔁 Data Processing
- Applied **data augmentation** for minority classes to handle class imbalance.
- Created **train, validation, and test splits** for the final dataset.

### 🧠 Baseline Model Training
- Implemented a baseline model using **EfficientNet-B0** with transfer learning.
- Training strategy:
  - Initially froze the backbone layers.
  - Trained the classifier head.
  - Later unfroze the backbone for fine-tuning.

### ⚙️ Training Configuration
- Framework: PyTorch  
- Model: EfficientNet-B0  
- Image Size: 224 × 224  
- Optimizer: Adam  
- Loss Function: CrossEntropyLoss  
- LR Scheduler: CosineAnnealingLR  
- Epochs: 30  

### 📊 Model Results
- Best Validation Accuracy: **96.62%**
- Final Training Accuracy: **92.11%**
- Final Validation Loss: **0.1108**

### 📌 Summary
- Successfully built a cleaned wheat disease dataset.
- Trained the **EfficientNet-B0 baseline model**.
- Achieved **96.62% validation accuracy** on wheat disease classification.
