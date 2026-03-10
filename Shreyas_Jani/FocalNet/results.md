# Experiment Results: FocalNet Training Comparison
**Date:** March 9, 2026  
**Model Architecture:** FocalNet (`focalnet_tiny_srf`)   
**Datasets:** Rice disease dataset

## 1. Training Dynamics & Pretraining Effectiveness
**Setup:** Trained two variants of a small FocalNet model (`focalnet_tiny_srf`) to evaluate the impact of pretraining. One model was trained completely from scratch, while the other was initialized with pretrained weights.

* **General Observation:** The from-scratch model demonstrated a very consistent and steady learning curve, eventually surpassing previous baselines like Deit-tiny during mid-training. However, the pretrained model showed the immense benefit of prior weight initialization, starting at a massive 91% validation accuracy by just epoch 4, though it eventually stagnated rather than continuously improving.

| Model Variant | Peak Val Accuracy | Epoch Reached | Notes |
| :--- | :--- | :--- | :--- |
| **From Scratch** | 80.0% | 23 | Consistent, steady improvement throughout. Surpassed Deit-tiny performance during mid-epochs. |
| **Pretrained** | **93.0%** | 17 | Started very strong (91% at epoch 4) but stagnated at 93% by epoch 17. Train accuracy stalled at 96%, indicating an upper performance limit rather than severe overfitting. |

## 2. Final Test Evaluation
**Setup:** Final evaluation on the unseen test set to measure true generalization.

* **Per-Class Performance:** The pretrained model performed exceptionally well, making it the second-best model tested to date (behind Swin-HViT). Most class F1 scores were above 0.90, and all but one were above 0.70. 
* **Leaf Smut Struggle:** The only major failure point for the pretrained model was the "leaf smut" class, which suffered a highly problematic F1 score of 0.23. The from-scratch model suffered from poor per-class F1 across the board, performing worse overall than the SAM-regularized MLP-Mixer.

| Model Variant | Final Test Accuracy |
| :--- | :--- |
| **From Scratch** | 72.0% |
| **Pretrained** | **88.0%** |

## 3. Next Steps
* **Targeted Swin-HViT Training:** Create a difficult 5-class subset consisting of the most challenging and commonly misclassified diseases: *Sheath rot, leaf smut, leaf blast, ragged stunt virus,* and *brown spot*.
* **SAM Regularization on Swin-HViT:** Train Swin-HViT on this 5-class subset both normally and with SAM (Sharpness-Aware Minimization).
* **Objectives:** (1) Evaluate the model's capability on a concentrated, high-difficulty dataset, and (2) Test the regularization effectiveness of SAM on Swin-HViT before committing to the massive training times required for the full dataset.