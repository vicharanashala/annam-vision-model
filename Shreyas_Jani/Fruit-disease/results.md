# Experiment Results: Fruit Disease
**Date:** 12th march 2026  
**Models Evaluated:** MLP-Mixer (with SAM, from scratch), Swin-HViT (Pretrained)  
**Datasets:** Fruit Disease (17 classes)

---

## 1. Fruit Disease Evaluation
**Setup:** MLP-Mixer trained from scratch (with SAM) to compare against the previous Swin-HViT fine-tuning results. 

* **MLP-Mixer (From Scratch):**
    * **Progression:** Showed steady, consistent improvement, requiring significantly more epochs than the fine-tuned models (as expected when learning low-level geometry from scratch).
        * Epoch 21: 87.2% val accuracy.
        * Epoch 30: 89.5% val accuracy.
        * Epoch 45: **92.0%** val accuracy (Peak).
        * Epoch 60: 92.0% val accuracy (Converged/Stagnated).
    * **Class-Specific Issues:** At epoch 30, `Alternaria_mango` suffered total mode collapse (F1 = 0, misclassified completely as Black rot). By epoch 60, its F1 improved to 0.3, but at the cost of slight regressions in other classes, keeping the overall accuracy flat.
    * **Conclusion:** Swin-HViT remains the top performer for this dataset.

---

### Summary of Latest Experiments

| Dataset | Model Architecture | Peak Val / Test Accuracy | Peak Epoch | Key Observations & F1 Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Fruits** *(17 classes)* | **Swin-HViT** *(Pretrained Base)* | 97.3% | 6 | Val set used as test set. Excellent F1 scores (most > 0.9). Lowest F1 was 0.55 (`Alternaria_mango`). |
| **Fruits** *(17 classes)* | **MLP-Mixer + SAM** *(From Scratch)* | 92.0% | 45 | Steady climb but hard ceiling at 92%. `Alternaria_mango` struggled heavily (mode collapse early on, maxing at 0.3 F1). |