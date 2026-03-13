# Experiment Results: Swin-HViT 5-Class Subset & SAM Regularization
**Date:** March 10, 2026  
**Model Architecture:** Swin-HViT (initialized with pretrained weights)   
**Datasets:** Rice disease dataset (5-class difficult subset)

## 1. Training Dynamics & Regularization Setup
**Setup:** Trained two variants of a pretrained Swin-HViT model on a highly difficult 5-class subset. The goal was to test if Sharpness-Aware Minimization (SAM) provides the same massive regularization benefits seen in the MLP-Mixer experiments.

* **General Observation:** Unlike the MLP-Mixer, SAM provided **no benefit** to the Swin-HViT architecture and actually accelerated performance degradation. This is likely because the hierarchical, window-based local attention in Swin Transformers already provides strong structural inductive biases, rendering the smoothing effects of SAM redundant or even harmful to the learning dynamics.

| Model Variant | Peak Val Accuracy | Epoch Reached | Notes |
| :--- | :--- | :--- | :--- |
| **Base Swin-HViT** | 88.8% | 15 | Showed strong early performance but stagnated in validation and eventually overfit the training data by epoch 30. |
| **SAM Swin-HViT** | 88.6% | 8 | Matched the base model early on but quickly degraded. Training was halted after 15 epochs due to poor test performance. |

## 2. Final Test Evaluation & The "Sensitive" Class Problem
**Setup:** Evaluation on the unseen test set to measure true generalization. The base model was tested mid-training (Epoch 15) and at the end of training (Epoch 30) to track regression.

* **Peak vs. Regression:** The base model achieved an impressive **95.0% test accuracy at Epoch 15**. However, continuing the training to Epoch 30 caused severe overfitting, dropping the final test accuracy to 82.5%. The SAM model performed exceptionally poorly, hitting only 72.5% by epoch 15.
* **The Core Failure Point:** Analysis of the confusion matrix revealed a massive, specific bottleneck: **Leaf Smut misclassified as Leaf Blast**. 
* **SAM's Negative Impact:** SAM actively worsened this specific confusion, increasing the misclassifications from 16 (in the base model) to 25. This single point of failure was responsible for the massive 10% drop in overall accuracy between the two models at the end of their respective runs.

| Model Variant | Peak Test Accuracy (Epoch) | Final Test Accuracy (Epoch) |
| :--- | :--- | :--- |
| **Base Swin-HViT** | **95.0%** (Epoch 15) | 82.5% (Epoch 30) |
| **SAM Swin-HViT** | N/A | 72.5% (Epoch 15) |

## 3. Key Takeaways & Next Steps
* **Stop Early:** The base Swin-HViT model is highly capable (hitting 95% on a difficult subset) but prone to severe overfitting if trained too long.
* **SAM is Architecture-Dependent:** While essential for isotropic models like MLP-Mixers, SAM is actively detrimental to models with strong local inductive biases like Swin-HViT in this specific context.
* **The "Two-Stage" Classification Hypothesis:** Leaf Smut and Leaf Blast are highly "sensitive" to confusion. 
* **Proposed Experiment:**
  1. Train a general Swin-HViT model on the remaining 17 distinct classes.
  2. Train a specialized, secondary binary-classification model dedicated exclusively to differentiating between Leaf Smut and Leaf Blast.