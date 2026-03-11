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


## 09 March 2026

### Dataset Preparation
Created a clean wheat disease dataset using the **wheat-leaf-disease dataset**.

Dataset statistics:

Classes:
- Brown rust
- Healthy
- Loose Smut
- Septoria
- Yellow rust

Total Images: 5521

Dataset split:
- Train: 3864
- Validation: 828
- Test: 829

Data loaders were created and verified. Sample images were visualized to confirm dataset integrity.

---

### Model Experiments

Implemented EfficientNet based models for wheat disease classification.

Models experimented with:
- EfficientNet-B0
- EfficientNet-B1
- EfficientNet-B2
- EfficientNet-B3

Training setup:
- Transfer learning using pretrained EfficientNet weights
- Backbone initially frozen
- Backbone unfrozen after epoch 10
- Optimizer and scheduler configured
- Batch size: 16

Training executed on:
GPU: Tesla T4

---

### Additional Model Experiment

Created a separate experiment using:

Model: EfficientNetV2-S

Model parameters:
- Total parameters: ~20M
- Trainable head parameters: 6405 (initially)
- Backbone unfreezes during training

Class weights were applied to handle dataset imbalance.

---

### Current Status

- Dataset pipeline finalized
- EfficientNet baseline models trained
- Multiple EfficientNet variants evaluated
- EfficientNetV2-S training experiment created
- Model checkpoints saved during training


## 10 March 2026

### 🚀 Model Training Experiments

Continued large-scale experiments with **EfficientNet family models** for wheat disease classification.

Models trained today:
- EfficientNet-B5
- EfficientNet-V2-M
- EfficientNet-B6

All experiments used **transfer learning with pretrained ImageNet weights**.

Training setup:
- Backbone frozen initially
- Backbone unfrozen at **epoch 10**
- Automatic checkpoint saving enabled
- Best models pushed to repository after training

---

# EfficientNet-B5 Training

Configuration:
- Image size: **456**
- Batch size: **8**
- Parameters: **28.9M**

Training progress highlights:

- Epoch 1: Val Acc **0.8575**
- Epoch 5: Val Acc **0.8973**
- Epoch 10 (after unfreezing): **0.9191**
- Epoch 20: **0.9722**
- Epoch 26: **0.9746** (best)

Final results:
- **Test Accuracy:** 0.9807
- **F1 Score:** 0.9808
- Training time: **134 minutes**

---

# EfficientNet-V2-M Training

Configuration:
- Image size: **480**
- Batch size: **8**
- Parameters: **53.2M**

Training progress highlights:

- Epoch 1: Val Acc **0.8684**
- Epoch 5: **0.9215**
- Epoch 10 (after unfreezing): **0.9444**
- Epoch 15: **0.9795**
- Epoch 22: **0.9807** (best)

Final results:
- **Test Accuracy:** 0.9964
- **F1 Score:** 0.9964
- Training time: **152 minutes**

---

# EfficientNet-B6 Training

Configuration:
- Image size: **528**
- Batch size: **4**
- Parameters: **41.3M**

Training progress:

- Epoch 1: Val Acc **0.8237**
- Epoch 5: **0.8961**
- Epoch 10 (after unfreezing): **0.9251**
- Epoch 13: **0.9541**
- Epoch 20: **0.9626**

Training is **currently ongoing**.

---

# Model Checkpoints Generated

Saved checkpoints:

- best_EffNet-B0.pth
- best_EffNet-B1.pth
- best_EffNet-B2.pth
- best_EffNet-B3.pth
- best_EffNet-B4.pth
- best_EffNet-B5.pth
- best_EffNet-V2-M.pth

All checkpoints were pushed to the repository for version tracking.

---

# Current Status

- Multiple EfficientNet models successfully trained.
- EfficientNet-V2-M currently shows the **highest performance (Test Acc: 0.9964)**.
- EfficientNet-B6 training is still running for further evaluation.
- 

## 11 March 2026

### 🗣 Team Stand-up
Attended the morning stand-up meeting with the team to discuss the current progress of the wheat disease classification project and ongoing model experiments.

Discussed:
- Current EfficientNet training experiments
- Model evaluation pipeline
- GPU resource usage on the shared Jupyter environment
- Next steps for testing larger EfficientNet architectures

---

### ⚙ Environment Setup

Yesterday I received access to the **shared Jupyter notebook environment** provided by the team for GPU computation.

Today I completed the following setup tasks:

- Imported all project notebooks into the shared environment
- Mounted and verified all required datasets
- Restored previously saved model checkpoints
- Verified GPU availability and environment configuration
- Recreated the training and evaluation pipeline inside the shared system

This ensures all experiments can now run using the **team GPU infrastructure instead of local systems**.

---

### 📦 Checkpoint Integration

Integrated previously trained model checkpoints into the new environment.

Loaded checkpoints for the following models:

- EfficientNet-B0
- EfficientNet-B1
- EfficientNet-B2
- EfficientNet-B3
- EfficientNet-B4
- EfficientNet-B5
- EfficientNet-B6

These checkpoints were used to **re-evaluate models without retraining**, saving computation time.

Evaluation results were successfully generated and stored.

Progress so far:
- 7 out of 15 model evaluations completed.

---

### 🚀 EfficientNet-B7 Training

Started training **EfficientNet-B7**, the largest model in the EfficientNet family used in this project.

Configuration:
- Image size: 600
- Batch size: 16
- Total parameters: 64.4M
- Trainable parameters (initial): 0.66M
- Backbone unfrozen at epoch 10

Training progress highlights:

Early training phase:
- Epoch 1: Validation Accuracy 0.8671
- Epoch 4: Validation Accuracy 0.9118
- Epoch 8: Validation Accuracy 0.9336

After backbone unfreezing:
- Epoch 11: Validation Accuracy 0.9626
- Epoch 15: Validation Accuracy 0.9734
- Epoch 18: Validation Accuracy 0.9807
- Epoch 20: Validation Accuracy 0.9819 (best so far)

Training is **currently ongoing**.

---

### 📊 Current Progress Summary

Completed:
- Environment migration to shared GPU Jupyter system
- Dataset and checkpoint integration
- Evaluation of multiple EfficientNet models
- Initiated EfficientNet-B7 training

Ongoing:
- EfficientNet-B7 training and monitoring
- Remaining model evaluations

---


