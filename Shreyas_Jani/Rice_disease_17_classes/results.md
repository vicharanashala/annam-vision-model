# Experiment Results: 17-Class Rice Disease Dataset
**Date:** March 11, 2026  
**Models Evaluated:** Swin-HViT (Pretrained), MLP-Mixer (From Scratch)  
**Datasets:** Rice Disease (17 classes - excluding leaf smut & leaf blast)

---

## 1. Rice Disease Evaluation (17 Classes)
**Setup:** Removed "leaf smut" and "leaf blast" classes. SAM was excluded for Swin-HViT as prior tests showed no benefit. MLP-Mixer was trained from scratch due to a lack of pretrained weights in `timm`.

* **Swin-HViT (Base):** * Showed strong, rapid learning. Hit 96.3% validation accuracy by epoch 6.
    * **Test Accuracy:** Peaked at **92.84%** using the epoch 5 checkpoint (highest achieved so far).
    * **Note:** At epoch 10, test accuracy dropped to 89.3% due to misclassifying "narrow brown spot" as "brown spot" (needs further class variance analysis).
* **MLP-Mixer:** * Showed steady, consistent improvement during early epochs (hit 81.5% val at epoch 8).
    * Converged at epoch 17 with 85.7% validation accuracy.
    * **Test Accuracy:** Dropped to **75.0%**. 
    * **Note:** Unexpected performance drop compared to previous 19-class training where it handled leaf smut/blast better than others. 

### Summary of 17-Class Experiments

| Dataset | Model Architecture | Max Val Accuracy | Best Test Accuracy | Peak Epoch | Key Observations & F1 Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Rice Disease** *(17 classes)* | **Swin-HViT** *(Pretrained Base)* | 96.3% | **92.84%** | 5 | Very strong peak at epoch 5. Accuracy dropped to 89.3% at epoch 10 due to confusion between "narrow brown spot" and "brown spot." |
| **Rice Disease** *(17 classes)* | **MLP-Mixer** *(From Scratch)* | 85.7% | **75.0%** | 17 | Converged at epoch 17. Unexpectedly poor test generalization compared to previous 19-class runs. |

## 2. Next Steps
* **MLP-Mixer on Fruits:** Set up and run the fruit dataset on MLP-Mixer. While the lack of pretrained weights is a disadvantage, its consistent epoch-over-epoch improvement makes it worth testing. 
* **Model Analysis:** Investigate gMLP as a potential alternative, noting previous literature citing marginal improvements over standard MLP-Mixer for plant disease 