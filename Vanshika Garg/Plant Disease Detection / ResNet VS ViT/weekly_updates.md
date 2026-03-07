# Weekly Progress Report

**Project:** Plant Disease Detection using Deep Learning
**Author:** Vanshika Garg
**Week:** Model Evaluation, Domain Shift Analysis, and Dataset Expansion

---

# 1. Project Objective

The goal of this project is to build a robust deep learning system for **plant disease detection from leaf images** and evaluate how well models trained on controlled datasets generalize to **real-world agricultural environments**.

The project focuses on comparing convolutional and transformer-based architectures and studying **domain shift between laboratory datasets and field images**.

Primary models used:

* ResNet-50 (CNN architecture)
* Vision Transformer (ViT Base Patch16)

---

# 2. Datasets Used

## 2.1 PlantVillage Dataset

The primary dataset used for model training was **PlantVillage**, which contains laboratory-controlled images of plant leaves.

Characteristics:

* Clean background
* Centered leaves
* Uniform lighting
* Clear disease patterns
* 38 plant disease classes

This dataset provides an excellent **baseline for training deep learning models** but does not represent real-world agricultural conditions.

---

## 2.2 PlantDoc Dataset

To evaluate real-world performance, the **PlantDoc dataset** was introduced.

Characteristics:

* Images captured in real agricultural environments
* Complex backgrounds
* Variable lighting conditions
* Multiple leaves in a single image
* Noise and occlusions

This dataset is useful for evaluating **domain shift**, where a model trained on one dataset struggles when tested on another dataset with different characteristics.

---

# 3. Models Implemented

Two deep learning architectures were implemented and compared:

## 3.1 ResNet-50

ResNet-50 is a convolutional neural network that uses residual connections to enable deeper architectures.

Advantages:

* Strong performance on image classification
* Efficient inference speed
* Lower parameter count compared to transformers

## 3.2 Vision Transformer (ViT)

The Vision Transformer processes images as sequences of patches using self-attention mechanisms.

Advantages:

* Captures global image context
* Strong performance on large datasets
* Modern transformer-based architecture

---

# 4. Model Training

Both models were trained on the **PlantVillage dataset** with 38 disease classes.

Training pipeline included:

* Image resizing to 224×224
* Cross-entropy loss
* GPU training
* Model checkpoint saving
* Standard preprocessing and normalization

The trained models were then evaluated on the test split of PlantVillage.

---

# 5. Baseline Model Results (PlantVillage)

## ResNet-50 Performance

| Metric                | Value           |
| --------------------- | --------------- |
| Test Accuracy         | 0.9986          |
| F1 Score              | 0.9986          |
| Trainable Parameters  | 23,585,894      |
| Parameters (Millions) | 23.58 M         |
| Total Inference Time  | 28.80 seconds   |
| Time per Image        | 0.00353 seconds |

## Vision Transformer Performance

| Metric                | Value           |
| --------------------- | --------------- |
| Test Accuracy         | 0.9925          |
| F1 Score              | 0.9925          |
| Trainable Parameters  | 85,827,878      |
| Parameters (Millions) | 85.83 M         |
| Total Inference Time  | 100.89 seconds  |
| Time per Image        | 0.01238 seconds |

### Observations

* ResNet-50 achieved **higher accuracy** than Vision Transformer.
* ResNet-50 was **approximately 3.5× faster during inference**.
* Vision Transformer required **3.6× more parameters**.

This indicates that **CNN architectures remain highly competitive for structured agricultural datasets**.

---

# 6. Documentation and Result Tracking

A structured results file (`results.md`) was created to document:

* Model performance
* Parameter complexity
* Inference time
* Comparative analysis between architectures

This documentation helps maintain **reproducibility and experiment tracking**.

---

# 7. Motivation for Domain Shift Testing

The PlantVillage dataset represents **ideal laboratory conditions**, which are not representative of real-world agricultural environments.

Real-world plant disease images often contain:

* Complex backgrounds
* Varying illumination
* Partial leaves
* Noise and blur
* Occlusions

Therefore, evaluating models on a real-world dataset is essential to measure **true practical performance**.

---

# 8. Domain Shift Experiment

To evaluate real-world generalization, models trained on PlantVillage were tested on the PlantDoc dataset.

### Experimental Setup

Training dataset:
PlantVillage

Testing dataset:
PlantDoc

Objective:
Evaluate how well models generalize to **unseen environmental conditions**.

---

# 9. Domain Shift Results

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 0.0476 |
| F1 Score | 0.026  |

These results initially suggested extremely poor performance.

However, further investigation revealed that the primary issue was **label mismatch between datasets**.

---

# 10. Dataset Label Mismatch

The PlantVillage dataset uses labels like:

```
Apple___Apple_scab
Tomato___Early_blight
Potato___Late_blight
```

Whereas PlantDoc uses labels such as:

```
Apple_Scab_Leaf
Apple_leaf
Tomato_leaf_bacterial_spot
Tomato_leaf_late_blight
```

Differences include:

* Naming conventions
* Class definitions
* Additional or missing classes

Because of this mismatch, model predictions were incorrectly counted as wrong during evaluation.

---

# 11. Proposed Solution: Label Mapping

To correct the evaluation process, a **label mapping strategy** was proposed.

Example mapping:

| PlantDoc Label             | PlantVillage Equivalent  |
| -------------------------- | ------------------------ |
| Apple_Scab_Leaf            | Apple___Apple_scab       |
| Apple_leaf                 | Apple___healthy          |
| Apple_rust_leaf            | Apple___Cedar_apple_rust |
| Tomato_Early_blight_leaf   | Tomato___Early_blight    |
| Tomato_leaf_bacterial_spot | Tomato___Bacterial_spot  |

This mapping will allow proper cross-dataset evaluation.

---

# 12. Future Work

The next stages of the project will focus on improving real-world robustness.

## Stage 3 – Domain Adaptation

Combine datasets for training:

PlantVillage + PlantDoc

This will allow the model to learn from both **controlled and real-world conditions**.

---

## Stage 4 – Advanced Training Strategies

Possible improvements include:

* Data augmentation (lighting variation, blur, noise)
* Multi-dataset training
* Hybrid CNN-transformer architectures
* Domain generalization techniques

---

## Stage 5 – Advanced Models

Future experiments may involve modern architectures such as:

* ConvNeXt
* Swin Transformer
* Hybrid CNN–Transformer networks

---

# 13. Key Learnings

This week provided several important insights:

1. Very high accuracy on laboratory datasets does not guarantee real-world performance.
2. Dataset bias and domain shift significantly affect deep learning models.
3. Proper dataset alignment and label mapping are essential when combining multiple datasets.
4. Real-world evaluation is critical for practical AI deployment in agriculture.

---

# 14. Next Week Goals

Planned tasks for the upcoming week include:

* Creating a unified label mapping between datasets
* Re-running domain shift experiments with corrected labels
* Performing domain adaptation training using combined datasets
* Evaluating improved model robustness

---

# Conclusion

This week focused on **model evaluation, dataset analysis, and domain shift investigation**. Initial results show that while models perform extremely well on controlled datasets, real-world deployment requires careful dataset alignment and more robust training strategies. These findings provide a strong foundation for improving the model’s performance in practical agricultural scenarios.
