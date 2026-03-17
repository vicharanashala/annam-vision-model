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

| Class Name | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
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

### Experiment 4: Deep Fine-Tuning with Heavy Regularization (4 Blocks Unfrozen)
* **Dataset:** Normal PlantVillage + Normal PlantDoc (No Segmentation Applied)
* **Test Set:** 177 images (95 PlantDoc + 82 manually collected real-world web images)
* **Architecture Setup:** Unfroze the last 4 blocks of the DINOv2 Large backbone. Applied 0.5 Dropout and Weighted Random Sampling.
* **Validation Accuracy:** 91.98% (Peaked at Epoch 13, Early Stopping triggered at Epoch 21)
* **Testing Accuracy:** 89.27%
* **Notes:** Pushing the fine-tuning 1 block deeper (4 blocks total) yielded the exact same overall testing accuracy (89.27%) as Exp 3. However, there was a clear trade-off: it improved recall on the minority Corn_Gray_leaf_spot class (from 73% to 80%) but sacrificed some accuracy on Corn_leaf_blight.

|Class|Precision|Recall (Acc)|F1-Score|Support|
|:----|:----|:----|:----|:----|
|Corn_Gray_leaf_spot|0.52|0.80|0.63|15|
|Corn_leaf_blight|0.73|0.52|0.61|21|
|Corn_rust_leaf|1.00|0.90|0.95|21|
|Tomato_Septoria_leaf_spot|1.00|1.00|1.00|22|
|Tomato_leaf|1.00|1.00|1.00|14|
|Apple_Scab_Leaf|0.87|0.87|0.87|15|
|Apple_leaf|1.00|0.93|0.96|14|
|Apple_rust_leaf|0.89|0.94|0.92|18|
|grape_leaf|1.00|1.00|1.00|18|
|grape_leaf_black_rot|1.00|1.00|1.00|19|
|Macro Avg|0.90|0.90|0.89|177|
|Weighted Avg|0.91|0.89|0.89|177|

### Experiment 5: Maximum Resolution & Deep Fine-Tuning (6 Blocks Unfrozen)
* **Dataset:** Normal PlantVillage + Normal PlantDoc (No Segmentation Applied)
* **Test Set:** 177 images (95 PlantDoc + 82 manually collected real-world web images)
* **Architecture Setup:** Unfroze the last 6 blocks. Image resolution increased to 384x384. Applied Test-Time Augmentation (TTA), 0.5 Dropout, and Weighted Random Sampling.
* **Validation Accuracy:** ~92.00%
* **Testing Accuracy:** 89.83%
* **Notes:** Pushing the Large model to its absolute limit, the image input size was increased to 384x384 to provide more fine-grained details, 6 blocks were unfrozen, and Test-Time Augmentation was used during evaluation. While the overall testing accuracy of 89.83% still didn't eclipse the DINOv2 Base model's record, this specific setup achieved the highest recall we have ever seen on the minority Corn_Gray_leaf_spot class (86.67%).

|Class|Precision|Recall (Acc)|F1-Score|Support|
|:----|:----|:----|:----|:----|
|Corn_Gray_leaf_spot|0.52|0.87|0.65|15|
|Corn_leaf_blight|0.85|0.52|0.65|21|
|Corn_rust_leaf|1.00|0.90|0.95|21|
|Tomato_Septoria_leaf_spot|1.00|1.00|1.00|22|
|Tomato_leaf|1.00|1.00|1.00|14|
|Apple_Scab_Leaf|0.87|0.87|0.87|15|
|Apple_leaf|1.00|0.86|0.92|14|
|Apple_rust_leaf|0.90|1.00|0.95|18|
|grape_leaf|1.00|1.00|1.00|18|
|grape_leaf_black_rot|1.00|1.00|1.00|19|
|Macro Avg|0.91|0.90|0.90|177|
|Weighted Avg|0.92|0.90|0.90|177|

### Experiment 6: Fully Segmented Datasets with Heavy Regularization
* **Dataset:** Segmented PlantVillage + Segmented PlantDoc
* **Test Set:** 177 images (95 PlantDoc + 82 manually collected real-world web images)
* **Architecture Setup:** Unfroze the last 3 blocks of the DINOv2 Large backbone. Applied 0.5 Dropout and Weighted Random Sampling.
* **Validation Accuracy:** 88.24% (Peaked at Epoch 15, Early Stopping triggered at Epoch 23)
* **Testing Accuracy:** 88.70%
* **Notes:** This experiment evaluated the Large architecture specifically on the pre-processed, fully segmented image datasets. While the testing accuracy proved robust and stable at 88.70%, it underperformed compared to training on the raw, unsegmented images (Exp 4 & 5).

|Class|Precision|Recall (Acc)|F1-Score|Support|
|:----|:----|:----|:----|:----|
|Corn_Gray_leaf_spot|0.44|0.53|0.48|15|
|Corn_leaf_blight|0.63|0.57|0.60|21|
|Corn_rust_leaf|1.00|0.95|0.98|21|
|Tomato_Septoria_leaf_spot|1.00|1.00|1.00|22|
|Tomato_leaf|1.00|1.00|1.00|14|
|Apple_Scab_Leaf|0.88|0.93|0.90|15|
|Apple_leaf|1.00|0.86|0.92|14|
|Apple_rust_leaf|0.95|1.00|0.97|18|
|grape_leaf|1.00|1.00|1.00|18|
|grape_leaf_black_rot|1.00|1.00|1.00|19|
|Macro Avg|0.89|0.88|0.89|177|
|Weighted Avg|0.89|0.89|0.89|177|

### Experiment 7: Grayscale Lab Data + Segmented Real-World Data
* **Dataset:** Grayscale PlantVillage + Segmented PlantDoc
* **Test Set:** 177 images (95 PlantDoc + 82 manually collected real-world web images)
* **Architecture Setup:** Unfroze the last 3 blocks of the DINOv2 Large backbone. Applied 0.5 Dropout and Weighted Random Sampling.
* **Validation Accuracy:** 88.24% (Peaked at Epoch 12, Early Stopping triggered at Epoch 20)
* **Testing Accuracy:** 88.70%
* **Notes:** Mixing the datasets (using grayscale for the lab data and segmentation for the real-world data) resulted in the exact same overall testing accuracy (88.70%) as the fully segmented dataset from Exp 6. However, there was a shift in class-level confidence: performance dropped on Corn_Gray_leaf_spot (40% vs 53%) but significantly improved on Corn_leaf_blight (76% vs 57%).

|Class|Precision|Recall (Acc)|F1-Score|Support|
|:----|:----|:----|:----|:----|
|Corn_Gray_leaf_spot|0.55|0.40|0.46|15|
|Corn_leaf_blight|0.59|0.76|0.67|21|
|Corn_rust_leaf|1.00|0.90|0.95|21|
|Tomato_Septoria_leaf_spot|1.00|1.00|1.00|22|
|Tomato_leaf|1.00|1.00|1.00|14|
|Apple_Scab_Leaf|0.92|0.80|0.86|15|
|Apple_leaf|1.00|0.93|0.96|14|
|Apple_rust_leaf|0.86|1.00|0.92|18|
|grape_leaf|1.00|1.00|1.00|18|
|grape_leaf_black_rot|1.00|1.00|1.00|19|
|Macro Avg|0.89|0.88|0.88|177|
|Weighted Avg|0.89|0.89|0.89|177|
