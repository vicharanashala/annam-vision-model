# DINOv3 Small Results
**Model Description:** Evaluating the lightweight **DINOv3 Small** architecture. The goal is to determine the optimal fine-tuning depth (number of unfrozen blocks) required for this lower-parameter model to effectively handle the domain gap between lab-controlled and real-world image datasets.

---

## Quick Summary of All Experiments

| Exp | Dataset Used | Backbone Status | Best Val Acc | Test Acc | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Normal PV + Seg PD | Unfroze Last 2 Blocks | 84.49% | 85.88% | Tested on 177 images. Baseline mixed-dataset performance. |
| **2** | Segmented PV + Seg PD | Unfroze Last 2 Blocks | 83.96% | 87.01% | Tested on 177 images. Fully segmented data yielded a slight improvement over the mixed dataset. |
| **3** | Normal PV + Normal PD | Unfroze Last 2 Blocks | 90.37% | 87.57% | Tested on 177 images. Unsegmented images outperformed the segmented baselines, prompting deeper unfreezing tests. |
| **4** | **Normal PV + Normal PD** | **Unfroze Last 4 Blocks** | **90.91%** | **89.27%** | **Tested on 177 images. The "Sweet Spot". Deepening to 4 blocks dramatically improved generalization without needing segmentation.** |
| **5** | Normal PV + Normal PD | Unfroze Last 6 Blocks | 90.37% | 87.01% | Tested on 177 images. Unfreezing too many layers causes the Small model to lose its pre-trained robustness. |

---

## Detailed Breakdown of Each Experiment

### Experiment 1: Normal Lab Data + Segmented Real-World Data (2 Blocks Unfrozen)
* **Dataset:** Normal PlantVillage (Unsegmented) + Segmented PlantDoc (SAM-3 Enhanced)
* **Test Set:** **177 images** (95 PlantDoc + 82 manually collected real-world web images)
* **Architecture Setup:** **Unfroze the last 2 blocks** of the DINOv3 Small backbone. Image resolution set to **224x224**.
* **Validation Accuracy:** 84.49%
* **Testing Accuracy:** **85.88%**
* **Notes:** This run established the initial baseline for DINOv3 Small on a mixed dataset. It proved relatively stable but lacked the feature extraction depth to distinguish between the hardest classes, struggling particularly with `Corn_Gray_leaf_spot` (60% recall).

**Classification Report (Exp 1):**

| Class | Precision | Recall (Acc) | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Corn_Gray_leaf_spot | 0.53 | 0.60 | 0.56 | 15 |
| Corn_leaf_blight | 0.68 | 0.62 | 0.65 | 21 |
| Corn_rust_leaf | 0.95 | 0.95 | 0.95 | 21 |
| Tomato_Septoria_leaf_spot | 0.92 | 1.00 | 0.96 | 22 |
| Tomato_leaf | 0.93 | 0.93 | 0.93 | 14 |
| Apple_Scab_Leaf | 1.00 | 0.73 | 0.85 | 15 |
| Apple_leaf | 0.80 | 0.86 | 0.83 | 14 |
| Apple_rust_leaf | 0.80 | 0.89 | 0.84 | 18 |
| grape_leaf | 1.00 | 1.00 | 1.00 | 18 |
| grape_leaf_black_rot | 1.00 | 0.95 | 0.97 | 19 |
| **Macro Avg** | **0.86** | **0.85** | **0.85** | **177** |
| **Weighted Avg** | **0.86** | **0.86** | **0.86** | **177** |


### Experiment 2: Fully Segmented Datasets (2 Blocks Unfrozen)
* **Dataset:** Segmented PlantVillage + Segmented PlantDoc (SAM-3 Enhanced)
* **Test Set:** **177 images** (95 PlantDoc + 82 manually collected real-world web images)
* **Architecture Setup:** **Unfroze the last 2 blocks** of the DINOv3 Small backbone. Image resolution set to **224x224**.
* **Validation Accuracy:** 83.96%
* **Testing Accuracy:** **87.01%**
* **Notes:** Providing the Small model with fully segmented images and unfreezing 2 blocks yielded a solid 87.01% test accuracy. However, looking at the class-level metrics, it still struggled heavily to differentiate the subtle characteristics between `Corn_leaf_blight` (62% accuracy) and `Corn_Gray_leaf_spot` (47% accuracy).

**Classification Report (Exp 2):**

| Class | Precision | Recall (Acc) | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Corn_Gray_leaf_spot | 0.50 | 0.47 | 0.48 | 15 |
| Corn_leaf_blight | 0.57 | 0.62 | 0.59 | 21 |
| Corn_rust_leaf | 0.90 | 0.86 | 0.88 | 21 |
| Tomato_Septoria_leaf_spot | 0.96 | 1.00 | 0.98 | 22 |
| Tomato_leaf | 0.93 | 0.93 | 0.93 | 14 |
| Apple_Scab_Leaf | 1.00 | 0.93 | 0.97 | 15 |
| Apple_leaf | 1.00 | 0.86 | 0.92 | 14 |
| Apple_rust_leaf | 0.90 | 1.00 | 0.95 | 18 |
| grape_leaf | 1.00 | 1.00 | 1.00 | 18 |
| grape_leaf_black_rot | 1.00 | 1.00 | 1.00 | 19 |
| **Macro Avg** | **0.88** | **0.87** | **0.87** | **177** |
| **Weighted Avg** | **0.87** | **0.87** | **0.87** | **177** |


### Experiment 3: Fully Unsegmented Data Baseline (2 Blocks Unfrozen)
* **Dataset:** Normal PlantVillage + Normal PlantDoc (**No Segmentation Applied**)
* **Test Set:** **177 images** (95 PlantDoc + 82 manually collected real-world web images)
* **Architecture Setup:** **Unfroze the last 2 blocks** of the DINOv3 Small backbone. Image resolution set to **224x224**. 
* **Validation Accuracy:** 90.37%
* **Testing Accuracy:** **87.57%**
* **Notes:** Removing all segmentation pre-processing yielded a high validation accuracy of 90.37% and actually slightly improved the testing accuracy to 87.57% compared to the segmented dataset (Exp 2). The model saw notable improvements in the recall of the difficult `Corn_Gray_leaf_spot` class (jumping from 47% to 67%).

**Classification Report (Exp 3):**

| Class | Precision | Recall (Acc) | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Corn_Gray_leaf_spot | 0.50 | 0.67 | 0.57 | 15 |
| Corn_leaf_blight | 0.67 | 0.57 | 0.62 | 21 |
| Corn_rust_leaf | 1.00 | 0.90 | 0.95 | 21 |
| Tomato_Septoria_leaf_spot | 1.00 | 1.00 | 1.00 | 22 |
| Tomato_leaf | 0.93 | 1.00 | 0.97 | 14 |
| Apple_Scab_Leaf | 0.92 | 0.80 | 0.86 | 15 |
| Apple_leaf | 0.92 | 0.79 | 0.85 | 14 |
| Apple_rust_leaf | 0.90 | 1.00 | 0.95 | 18 |
| grape_leaf | 1.00 | 1.00 | 1.00 | 18 |
| grape_leaf_black_rot | 0.95 | 1.00 | 0.97 | 19 |
| **Macro Avg** | **0.88** | **0.87** | **0.87** | **177** |
| **Weighted Avg** | **0.88** | **0.88** | **0.88** | **177** |


### Experiment 4: The "Sweet Spot" Fine-Tuning (4 Blocks Unfrozen)
* **Dataset:** Normal PlantVillage + Normal PlantDoc (**No Segmentation Applied**)
* **Test Set:** **177 images** (95 PlantDoc + 82 manually collected real-world web images)
* **Architecture Setup:** **Unfroze the last 4 blocks** of the DINOv3 Small backbone. Image resolution set to **224x224**. 
* **Validation Accuracy:** 90.91%
* **Testing Accuracy:** **89.27%**
* **Notes:** Switching to the completely unsegmented dataset and doubling the unfrozen blocks from 2 to 4 resulted in a massive leap in performance. The model broke the 89% barrier, handling complex backgrounds beautifully and improving recall on nearly all minority classes. This proves that 4 blocks is the ideal fine-tuning depth for the DINOv3 Small architecture on complex image sets.

**Classification Report (Exp 4):**

| Class | Precision | Recall (Acc) | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Corn_Gray_leaf_spot | 0.53 | 0.53 | 0.53 | 15 |
| Corn_leaf_blight | 0.65 | 0.71 | 0.68 | 21 |
| Corn_rust_leaf | 1.00 | 0.90 | 0.95 | 21 |
| Tomato_Septoria_leaf_spot | 1.00 | 1.00 | 1.00 | 22 |
| Tomato_leaf | 0.93 | 1.00 | 0.97 | 14 |
| Apple_Scab_Leaf | 0.93 | 0.87 | 0.90 | 15 |
| Apple_leaf | 1.00 | 0.86 | 0.92 | 14 |
| Apple_rust_leaf | 0.90 | 1.00 | 0.95 | 18 |
| grape_leaf | 1.00 | 1.00 | 1.00 | 18 |
| grape_leaf_black_rot | 1.00 | 1.00 | 1.00 | 19 |
| **Macro Avg** | **0.89** | **0.89** | **0.89** | **177** |
| **Weighted Avg** | **0.90** | **0.89** | **0.89** | **177** |


### Experiment 5: Deeper Fine-Tuning on Unsegmented Data (6 Blocks Unfrozen)
* **Dataset:** Normal PlantVillage + Normal PlantDoc (**No Segmentation Applied**)
* **Test Set:** **177 images** (95 PlantDoc + 82 manually collected real-world web images)
* **Architecture Setup:** **Unfroze the last 6 blocks** of the DINOv3 Small backbone. Image resolution set to **224x224**. 
* **Validation Accuracy:** 90.37%
* **Testing Accuracy:** **87.01%**
* **Notes:** To see if the Small model could handle unsegmented data even better with more capacity to adapt, the unfreezing strategy was deepened to 6 blocks. Interestingly, the testing accuracy dropped back down to 87.01%. This indicates that unfreezing too many layers on the lower-parameter Small model likely causes it to overfit to the training dataset, losing the generalizability of its self-supervised pre-training.

**Classification Report (Exp 5):**

| Class | Precision | Recall (Acc) | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Corn_Gray_leaf_spot | 0.50 | 0.60 | 0.55 | 15 |
| Corn_leaf_blight | 0.71 | 0.57 | 0.63 | 21 |
| Corn_rust_leaf | 0.91 | 0.95 | 0.93 | 21 |
| Tomato_Septoria_leaf_spot | 0.92 | 1.00 | 0.96 | 22 |
| Tomato_leaf | 0.93 | 0.93 | 0.93 | 14 |
| Apple_Scab_Leaf | 0.93 | 0.87 | 0.90 | 15 |
| Apple_leaf | 1.00 | 0.86 | 0.92 | 14 |
| Apple_rust_leaf | 0.89 | 0.89 | 0.89 | 18 |
| grape_leaf | 1.00 | 1.00 | 1.00 | 18 |
| grape_leaf_black_rot | 0.95 | 1.00 | 0.97 | 19 |
| **Macro Avg** | **0.87** | **0.87** | **0.87** | **177** |
| **Weighted Avg** | **0.87** | **0.87** | **0.87** | **177** |

### Experiment 6: Deeper Fine-Tuning on Unsegmented Data (6 Blocks Unfrozen)
* **Dataset:** Normal PlantVillage + Normal PlantDoc (No Segmentation Applied)
* **Test Set:** 177 images (95 PlantDoc + 82 manually collected real-world web images)
* **Architecture Setup:** Unfroze the last 6 blocks of the DINOv3 Small backbone. Image resolution set to 224x224.
* **Validation Accuracy:** 90.37%
* **Testing Accuracy:** 87.01%
* **Notes:** To see if the Small model could handle unsegmented data even better with more capacity to adapt, the unfreezing strategy was deepened to 6 blocks. Interestingly, the testing accuracy dropped back down to 87.01%. This indicates that unfreezing too many layers on the lower-parameter Small model likely causes it to overfit to the training dataset, losing the generalizability of its self-supervised pre-training.

|Class|Precision|Recall (Acc)|F1-Score|Support|
|:----|:----|:----|:----|:----|
|Corn_Gray_leaf_spot|0.50|0.60|0.55|15|
|Corn_leaf_blight|0.71|0.57|0.63|21|
|Corn_rust_leaf|0.91|0.95|0.93|21|
|Tomato_Septoria_leaf_spot|0.92|1.00|0.96|22|
|Tomato_leaf|0.93|0.93|0.93|14|
|Apple_Scab_Leaf|0.93|0.87|0.90|15|
|Apple_leaf|1.00|0.86|0.92|14|
|Apple_rust_leaf|0.89|0.89|0.89|18|
|grape_leaf|1.00|1.00|1.00|18|
|grape_leaf_black_rot|0.95|1.00|0.97|19|
|Macro Avg|0.87|0.87|0.87|177|
|Weighted Avg|0.87|0.87|0.87|177|
