# Experiment Results: MLP-Mixer Regularization Comparison
**Date:** March 7, 2026  
**Model Architecture:** MLP-Mixer small (16x16 patches)   
**Datasets:** Rice disease dataset

## 1. Training Dynamics & Regularization
**Setup:** Trained a baseline MLP-Mixer alongside two variations using Sharpness-Aware Minimization (SAM) and AdamP (as a Python alternative to Gradient Centralization) to combat the architecture's tendency to fall into sharp local minima . Models trained for up to 30 epochs.

* **General Observation:** The baseline model trained relatively fast (10-12 minutes per epoch) but quickly stagnated, confirming the local minima issue. Both SAM and AdamP successfully pushed past this barrier, though at the cost of significantly increased training time (2.5x slower) due to the double-pass nature of SAM and the Python-level implementation of AdamP.

| Optimizer / Regularizer | Peak Val Accuracy | Epoch Reached | Notes |
| :--- | :--- | :--- | :--- |
| **Baseline** | 72.0% | 29 | Stagnated at 72% early on (epoch 21). Fast training but got stuck in a local minimum. |
| **AdamP** | 73.5% | 17 | Passed baseline early (73% at epoch 8) but stagnated at 73.5% by epoch 10 and refused to budge further. |
| **SAM** | **79.5%** | 25 | Consistently climbed, reaching 79% by epoch 17 and maxing around 79.5% at epoch 19. |

## 2. Final Test Evaluation
**Setup:** Final evaluation on the unseen test set to measure true generalization. *(Note: AdamP test accuracy was not recorded due to early stagnation; peak validation is referenced instead).*

* **Class Imbalance:** Unlike previous models, the SAM-regularized MLP-Mixer was almost entirely unaffected by class imbalances. The confusion matrix was exceptionally clean.
* **F1 Scores:** The SAM model maintained highly stable F1 scores, with the lowest around 0.60 and the highest hitting a perfect 1.00. 

| Optimizer / Regularizer | Final Test Accuracy |
| :--- | :--- |
| **Baseline** | 65.5% |
| **AdamP** | *(Peak Val: 73.5%)* |
| **SAM** | **80.0%** *(Highest)* |

## 3. Key Takeaways
* **Regularization is Mandatory:** The standard MLP-Mixer easily overfits or gets stuck in sharp local minima. Regularizers like SAM are strictly necessary to unlock the model's performance, successfully boosting test accuracy from 65.5% to 80.0%.
* **Training vs. Inference Trade-off:** While SAM makes training 2.5x slower, the final inference time remains identical to the baseline. Given the massive performance bump and robustness to class imbalance, the training cost is highly justified.
* **AdamP Limitations:** The Python implementation of AdamP created the same overhead as SAM but failed to yield the same continuous improvements, stagnating early.

## 4. Next Steps
* **Performance Comparison:** Compare the per-class performance and F1 scores of the SAM MLP-Mixer directly against the fine-tuned Swin-HViT model.
* **Alternative Architectures:** Begin reading and evaluating the "Focal-HAIN" paper (March 2026) to see if Focal Modulation Networks (FocalNet) can offer better or more efficient results for crop disease classification.