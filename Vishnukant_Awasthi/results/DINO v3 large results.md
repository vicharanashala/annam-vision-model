# DINOv3 Large Results

**Model Description:** Evaluating the massive **DINOv3 Large** architecture. The goal is to determine if the advancements in DINOv3's self-supervised pre-training combined with its massive parameter count can establish new project records while effectively handling complex domain gaps. 

---

## Quick Summary of All Experiments

| Exp | Dataset Used | Backbone Status | Best Val Acc | Test Acc | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Segmented PV + Seg PD | Unfroze Last 4 Blocks | 86.00% | 89.00% | Tested on 177 images. Solid performance on fully segmented data, but underperforms compared to unsegmented lab images. |
| **2** | **Normal PV + Seg PD** | **Unfroze Last 4 Blocks** | **89.84%** | **90.96%** | **Tested on 177 images. Mixing normal lab data with segmented test data yields an exceptional 90.96% accuracy, locking in 100% recall on 5 classes.** |
| **3** | Grayscale PV + Seg PD | Unfroze Last 4 Blocks | 90.91% | 90.40% | Tested on 177 images. Massive discovery: Unlike the Base and Small models, the Large architecture is nearly immune to the loss of color data. |
| **4** | Normal PV + Normal PD | Unfroze Last 4 Blocks | 91.44% | 89.27% | Tested on 177 images. Evaluated entirely on unsegmented images. Achieved highest validation score and amazing recall on difficult classes, though slight noise interference in testing. |

---

## Detailed Breakdown of Each Experiment

### Experiment 1: Fully Segmented Datasets (4 Blocks Unfrozen)
* **Dataset:** Segmented PlantVillage + Segmented PlantDoc (SAM-3 Enhanced)
* **Test Set:** **177 images** (95 PlantDoc + 82 manually collected real-world web images)
* **Architecture Setup:** **Unfroze the last 4 blocks** of the DINOv3 Large backbone. Image resolution set to **224x224**.
* **Validation Accuracy:** 86.00%
* **Testing Accuracy:** **89.00%**
* **Notes:** This experiment evaluated the DINOv3 Large architecture on fully segmented datasets for both training and testing. The model achieved a very solid 89.00% testing accuracy. Interestingly, this is lower than the 90.96% achieved when training on *unsegmented* normal lab images (Exp 2). This strongly suggests that the massive DINOv3 Large model actively utilizes the background context in raw images to build richer, more robust feature representations.

### Experiment 2: Normal Lab Data + Segmented Real-World Data (4 Blocks Unfrozen)
* **Dataset:** Normal PlantVillage (Unsegmented) + Segmented PlantDoc (SAM-3 Enhanced)
* **Test Set:** **177 images** (95 PlantDoc + 82 manually collected real-world web images)
* **Architecture Setup:** **Unfroze the last 4 blocks** of the DINOv3 Large backbone. Image resolution set to **224x224**. 
* **Validation Accuracy:** 89.84% (Peaked at Epoch 10)
* **Testing Accuracy:** **90.96%**
* **Notes:** Training the massive DINOv3 Large model on unsegmented lab images while keeping the real-world test images segmented resulted in a fantastic leap in generalizability. The testing accuracy hit an outstanding **90.96%**, locking in perfect 100% recall on 5 different classes. This reinforces that massive models like DINOv3 Large benefit from the background context of normal lab images during training to build stronger, more robust attention maps.

**Classification Report (Exp 2):**

| Class | Precision | Recall (Acc) | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Corn_Gray_leaf_spot | 0.56 | 0.67 | 0.61 | 15 |
| Corn_leaf_blight | 0.78 | 0.67 | 0.72 | 21 |
| Corn_rust_leaf | 0.95 | 0.95 | 0.95 | 21 |
| Tomato_Septoria_leaf_spot | 1.00 | 1.00 | 1.00 | 22 |
| Tomato_leaf | 0.93 | 1.00 | 0.97 | 14 |
| Apple_Scab_Leaf | 0.93 | 0.93 | 0.93 | 15 |
| Apple_leaf | 1.00 | 0.86 | 0.92 | 14 |
| Apple_rust_leaf | 0.95 | 1.00 | 0.97 | 18 |
| grape_leaf | 1.00 | 1.00 | 1.00 | 18 |
| grape_leaf_black_rot | 1.00 | 1.00 | 1.00 | 19 |
| **Macro Avg** | **0.91** | **0.91** | **0.91** | **177** |
| **Weighted Avg** | **0.91** | **0.91** | **0.91** | **177** |


### Experiment 3: Grayscale Lab Data + Segmented Real-World Data (4 Blocks Unfrozen)
* **Dataset:** Grayscale PlantVillage + Segmented PlantDoc (SAM-3 Enhanced)
* **Test Set:** **177 images** (95 PlantDoc + 82 manually collected real-world web images)
* **Architecture Setup:** **Unfroze the last 4 blocks** of the DINOv3 Large backbone. Image resolution set to **224x224**. 
* **Validation Accuracy:** 90.91% (Peaked at Epoch 12)
* **Testing Accuracy:** **90.40%**
* **Notes:** To test the model's reliance on color embeddings, the training data was converted to Grayscale. In previous experiments, doing this to the Base and Small models caused their accuracy to plummet to ~84% and ~83%. However, the DINOv3 Large architecture barely flinched, maintaining a highly robust **90.40% testing accuracy**. In fact, it actually *improved* recall on the hardest `Corn_Gray_leaf_spot` class compared to Exp 2 (jumping from 67% to 73%). This proves that the massive capacity of the Large model allows it to rely almost entirely on complex structural and textural features rather than taking the "shortcut" of color identification.

**Classification Report (Exp 3):**

| Class | Precision | Recall (Acc) | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Corn_Gray_leaf_spot | 0.65 | 0.73 | 0.69 | 15 |
| Corn_leaf_blight | 0.75 | 0.71 | 0.73 | 21 |
| Corn_rust_leaf | 1.00 | 0.95 | 0.98 | 21 |
| Tomato_Septoria_leaf_spot | 1.00 | 1.00 | 1.00 | 22 |
| Tomato_leaf | 1.00 | 1.00 | 1.00 | 14 |
| Apple_Scab_Leaf | 0.86 | 0.80 | 0.83 | 15 |
| Apple_leaf | 1.00 | 0.86 | 0.92 | 14 |
| Apple_rust_leaf | 0.86 | 1.00 | 0.92 | 18 |
| grape_leaf | 0.95 | 1.00 | 0.97 | 18 |
| grape_leaf_black_rot | 1.00 | 0.95 | 0.97 | 19 |
| **Macro Avg** | **0.91** | **0.90** | **0.90** | **177** |
| **Weighted Avg** | **0.91** | **0.90** | **0.90** | **177** |


### Experiment 4: Fully Unsegmented Data Baseline (4 Blocks Unfrozen)
* **Dataset:** Normal PlantVillage + Normal PlantDoc (**No Segmentation Applied**)
* **Test Set:** **177 images** (95 PlantDoc + 82 manually collected real-world web images)
* **Architecture Setup:** **Unfroze the last 4 blocks** of the DINOv3 Large backbone. Image resolution set to **224x224**.
* **Validation Accuracy:** 91.44% (Peaked at Epoch 7)
* **Testing Accuracy:** **89.27%**
* **Per-Class Testing Accuracy (Recall):**
  * Tomato_Septoria_leaf_spot: 100.00%
  * Apple_rust_leaf: 100.00%
  * grape_leaf: 100.00%
  * grape_leaf_black_rot: 100.00%
  * Tomato_leaf: 93.00%
  * Corn_rust_leaf: 90.00%
  * Apple_Scab_Leaf: 87.00%
  * Apple_leaf: 86.00%
  * Corn_Gray_leaf_spot: 80.00%
  * Corn_leaf_blight: 57.00%
* **Notes:** Removing all segmentation pre-processing yielded the highest validation accuracy achieved by this architecture (91.44%). While the overall testing accuracy dipped slightly to 89.27% compared to the mixed-dataset setup, the model achieved a spectacular 80% recall on the notoriously difficult `Corn_Gray_leaf_spot` class. This suggests that while chaotic real-world backgrounds can cause slight interference, the deep fine-tuning of the Large architecture allows it to successfully isolate the disease features organically. 
**Confusion Matrix (Exp 4):**
```text
[[12  3  0  0  0  0  0  0  0  0] 
 [ 9 12  0  0  0  0  0  0  0  0] 
 [ 0  2 19  0  0  0  0  0  0  0] 
 [ 0  0  0 22  0  0  0  0  0  0] 
 [ 0  0  0  1 13  0  0  0  0  0] 
 [ 0  0  0  0  0 13  0  2  0  0] 
 [ 0  0  0  0  0  2 12  0  0  0] 
 [ 0  0  0  0  0  0  0 18  0  0] 
 [ 0  0  0  0  0  0  0  0 18  0] 
 [ 0  0  0  0  0  0  0  0  0 19]]
```
|Class|Precision|Recall (Acc)|F1-Score|Support|
|:----|:----|:----|:----|:----|
|Corn_Gray_leaf_spot|0.57|0.80|0.67|15|
|Corn_leaf_blight|0.71|0.57|0.63|21|
|Corn_rust_leaf|1.00|0.90|0.95|21|
|Tomato_Septoria_leaf_spot|0.96|1.00|0.98|22|
|Tomato_leaf|1.00|0.93|0.96|14|
|Apple_Scab_Leaf|0.87|0.87|0.87|15|
|Apple_leaf|1.00|0.86|0.92|14|
|Apple_rust_leaf|0.90|1.00|0.95|18|
|grape_leaf|1.00|1.00|1.00|18|
|grape_leaf_black_rot|1.00|1.00|1.00|19|
|Macro Avg|0.90|0.89|0.89|177|
|Weighted Avg|0.90|0.89|0.89|177|

### Experiment 5: Fully Unsegmented Data Baseline (4 Blocks Unfrozen)
* **Dataset:** Normal PlantVillage + Normal PlantDoc (No Segmentation Applied)
* **Test Set:** 177 images (95 PlantDoc + 82 manually collected real-world web images)
* **Architecture Setup:** Unfroze the last 4 blocks of the DINOv3 Large backbone. Image resolution set to 224x224.
* **Validation Accuracy:** 91.44% (Peaked at Epoch 7)
* **Testing Accuracy:** 89.27%
* **Notes:** Removing all segmentation pre-processing yielded the highest validation accuracy achieved by this architecture (91.44%). While the overall testing accuracy dipped slightly to 89.27% compared to the mixed-dataset setup, the model achieved a spectacular 80% recall on the notoriously difficult Corn_Gray_leaf_spot class. This suggests that while chaotic real-world backgrounds can cause slight interference, the deep fine-tuning of the Large architecture allows it to successfully isolate the disease features organically.


|Class|Precision|Recall (Acc)|F1-Score|Support|
|:----|:----|:----|:----|:----|
|Corn_Gray_leaf_spot|0.57|0.80|0.67|15|
|Corn_leaf_blight|0.71|0.57|0.63|21|
|Corn_rust_leaf|1.00|0.90|0.95|21|
|Tomato_Septoria_leaf_spot|0.96|1.00|0.98|22|
|Tomato_leaf|1.00|0.93|0.96|14|
|Apple_Scab_Leaf|0.87|0.87|0.87|15|
|Apple_leaf|1.00|0.86|0.92|14|
|Apple_rust_leaf|0.90|1.00|0.95|18|
|grape_leaf|1.00|1.00|1.00|18|
|grape_leaf_black_rot|1.00|1.00|1.00|19|
|Macro Avg|0.90|0.89|0.89|177|
|Weighted Avg|0.90|0.89|0.89|177|
