# DINOv2 Large Results
**Model Description:** Scaling up the architecture, I evaluated the **DINOv2 Large** to test if drastically increasing the parameter count would naturally solve the domain adaptation issues. To prevent the massive network from aggressively overfitting on the training data, heavy regularization techniques had to be applied, including Dropout layers, Weighted Random Samplers, and intensified image augmentations. 

---

## Quick Summary of All Experiments

| Exp | Dataset Used | Backbone Status | Best Val Acc | Test Acc | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Normal PV + Normal PD | Unfroze Last 2 Blocks | 89.84% | 83.16% | Tested on 95 images. Massive parameter count overfits without deep fine-tuning. |
| **2** | Normal PV + Normal PD | Unfroze Last 3 Blocks | 90.91% | 84.21% | Tested on 95 images. Slight improvement over 2 blocks, but still overfitting on minority classes. |
| **3** | Normal PV + Normal PD | Unfroze Last 3 Blocks | 90.91% | 89.27% | Tested on 177 Expanded Images. Added regularization (WRS + Dropout) significantly improved generalization. |
| **4** | Normal PV + Normal PD | Unfroze Last 4 Blocks | 91.98% | 89.27% | Tested on 177 Expanded Images. Mathematically tied with 3 blocks, but improved minority Corn class recall. |
| **5** | Normal PV + Normal PD | Unfroze Last 6 Blocks | ~92.00% | 89.83% | Tested on 177 Expanded Images. Used 384x384 resolution and TTA. Peak performance for the Large model. |
| **6** | Segmented PV + Seg PD | Unfroze Last 3 Blocks | 88.24% | 88.70% | Tested on 177 Expanded Images. Good stability with fully segmented data. |
| **7** | Grayscale PV + Seg PD | Unfroze Last 3 Blocks | 88.24% | 88.70% | Tested on 177 Expanded Images. Performance tied with Exp 6, though class level confidence shifted. |

---

## Detailed Breakdown of Each Experiment

### Experiment 1: Initial DINOv2 Large Fine-Tuning (2 Blocks Unfrozen)
* **Dataset:** Normal PlantVillage + Normal PlantDoc (**No Segmentation Applied**)
* **Test Set:** 95 images (PlantDoc only)
* **Architecture Setup:** **Unfroze the last 2 blocks** of the DINOv2 Large backbone. Applied 0.4 Dropout and Weighted Random Sampling.
* **Validation Accuracy:** 89.84% (Peaked at Epoch 13, Early Stopping triggered at Epoch 23)
* **Testing Accuracy:** **83.16%**
* **Notes:** As an initial test on the Large architecture, maintaining the previous "sweet spot" of unfreezing 2 blocks led to significant overfitting. The model struggled heavily with the minority classes like `Corn_Gray_leaf_spot` (25%).

### Experiment 2: Moderate Fine-Tuning (3 Blocks Unfrozen)
* **Dataset:** Normal PlantVillage + Normal PlantDoc (**No Segmentation Applied**)
* **Test Set:** 95 images (PlantDoc only)
* **Architecture Setup:** **Unfroze the last 3 blocks** of the DINOv2 Large backbone.
* **Validation Accuracy:** 90.91% (Peaked at Epoch 11, Early Stopping at Epoch 21)
* **Testing Accuracy:** **84.21%**
* **Notes:** Testing the theory that the massive parameter count requires deeper fine-tuning, 3 blocks were unfrozen. While validation accuracy broke 90%, testing accuracy only improved slightly to 84.21%.

### Experiment 3: Optimized Moderate Fine-Tuning (3 Blocks Unfrozen + Regularization)
* **Dataset:** Normal PlantVillage + Normal PlantDoc (**No Segmentation Applied**)
* **Test Set:** **177 images** (95 PlantDoc + 82 manually collected real-world web images)
* **Architecture Setup:** **Unfroze the last 3 blocks** of the DINOv2 Large backbone. Applied 0.5 Dropout and Weighted Random Sampling.
* **Validation Accuracy:** 90.91% (Peaked at Epoch 11, Early Stopping triggered at Epoch 21)
* **Testing Accuracy:** **89.27%**
* **Notes:** Applying heavy regularization and testing against the expanded dataset resulted in a massive leap in accuracy (89.27%). The model successfully stabilized, achieving 100% precision and recall on multiple classes, though it still showed some weakness on the harder `Corn_leaf_blight` class (57%).


**Confusion Matrix (Exp 3):**
```text
[[11  4  0  0  0  0  0  0  0  0] 
 [ 9 12  0  0  0  0  0  0  0  0] 
 [ 1  1 19  0  0  0  0  0  0  0] 
 [ 0  0  0 22  0  0  0  0  0  0] 
 [ 0  0  0  1 13  0  0  0  0  0] 
 [ 0  0  0  0  0 12  0  3  0  0] 
 [ 0  0  0  0  0  0 14  0  0  0] 
 [ 0  0  0  0  0  0  0 18  0  0] 
 [ 0  0  0  0  0  0  0  0 18  0] 
 [ 0  0  0  0  0  0  0  0  0 19]]
```
| Corn_Gray_leaf_spot | 0.52 | 0.73 | 0.61 | 15 |
| Corn_leaf_blight | 0.71 | 0.57 | 0.63 | 21 |
| Corn_rust_leaf | 1.00 | 0.90 | 0.95 | 21 |
| Tomato_Septoria_leaf_spot | 0.96 | 1.00 | 0.98 | 22 |
| Tomato_leaf | 1.00 | 0.93 | 0.96 | 14 |
| Apple_Scab_Leaf | 1.00 | 0.80 | 0.89 | 15 |
| Apple_leaf | 1.00 | 1.00 | 1.00 | 14 |
| Apple_rust_leaf | 0.86 | 1.00 | 0.92 | 18 |
| grape_leaf | 1.00 | 1.00 | 1.00 | 18 |
| grape_leaf_black_rot | 1.00 | 1.00 | 1.00 | 19 |
| **Macro Avg** | **0.90** | **0.89** | **0.89** | **177** |
| **Weighted Avg** | **0.90** | **0.89** | **0.89** | **177** |

### Experiment 3: Optimized Moderate Fine-Tuning (3 Blocks Unfrozen + Regularization)
* **Dataset:** Normal PlantVillage + Normal PlantDoc (**No Segmentation Applied**)
* **Test Set:** **177 images** (95 PlantDoc + 82 manually collected real-world web images)
* **Architecture Setup:** **Unfroze the last 3 blocks** of the DINOv2 Large backbone. Applied 0.5 Dropout and Weighted Random Sampling.
* **Validation Accuracy:** 90.91% (Peaked at Epoch 11, Early Stopping triggered at Epoch 21)
* **Testing Accuracy:** **89.27%**
