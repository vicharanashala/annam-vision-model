# DINOv2 Base Results

**Model Description:** After DINOv2 small upgraded the architecture to the larger and more powerful **DINOv2 Base**. The goal is to determine if a deeper Vision Transformer with more parameters can extract even richer features and further stabilize performance on the hardest minority classes. Across these experiments, the optimal fine tuning strategy discovered previously (unfreezing exactly the last 2 blocks) is maintained.

---

## Quick Summary of All Experiments

| Exp | Dataset Used | Backbone Status | Best Val Acc | Test Acc | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Segmented PV + Seg PD | Unfroze Last 2 Blocks | 88.24% | 89.47% | Tested on 95 images. Achieved 100% on 5 classes, though Corn classes still lag. |
| **2** | Grayscale PV + Seg PD | Unfroze Last 2 Blocks | 87.70% | 87.37% | Tested on 95 images. Minor performance drop compared to fully segmented data. |
| **3** | Segmented PV + Seg PD | Unfroze Last 2 Blocks | 88.24% | 91.53% | Tested on 177 Expanded Images. Massive improvement on minority classes. |
| **4** | Normal PV + Seg PD | Unfroze Last 2 Blocks | 86.10% | 90.40% | Tested on 177 Expanded Images. Excellent generalization from normal lab data. |
| **5** | **Normal PV + Normal PD** | **Unfroze Last 2 Blocks** | **91.44%** | **92.09%** | **No Segmentation Used.** Highest overall accuracy. |

---

## Detailed Breakdown of Each Experiment

### Experiment 1: Segmented PlantVillage + Segmented PlantDoc (Standard Test Set)
* **Dataset:** Segmented PlantVillage + Segmented PlantDoc
* **Test Set:** 95 images (PlantDoc only)
* **Architecture Setup:** **Unfroze the last 2 blocks** of the DINOv2 Base backbone.
* **Validation Accuracy:** 88.24%
* **Testing Accuracy:** 89.47%
* **Notes:** The Base model establishes a very strong initial baseline, perfectly categorizing 5 out of the 10 classes. However, it still exhibits similar struggles with `Corn_leaf_blight` and `Corn_Gray_leaf_spot`.

**Classification Report (Exp 1):**

| Class | Precision | Recall (Acc) | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Corn_Gray_leaf_spot | 0.25 | 0.50 | 0.33 | 4 |
| Corn_leaf_blight | 0.78 | 0.58 | 0.67 | 12 |
| Corn_rust_leaf | 1.00 | 0.90 | 0.95 | 10 |
| Tomato_Septoria_leaf_spot | 1.00 | 1.00 | 1.00 | 12 |
| Tomato_leaf | 1.00 | 1.00 | 1.00 | 8 |
| Apple_Scab_Leaf | 0.90 | 0.90 | 0.90 | 10 |
| Apple_leaf | 1.00 | 0.89 | 0.94 | 9 |
| Apple_rust_leaf | 0.91 | 1.00 | 0.95 | 10 |
| grape_leaf | 1.00 | 1.00 | 1.00 | 12 |
| grape_leaf_black_rot | 1.00 | 1.00 | 1.00 | 8 |
| **Macro Avg** | **0.88** | **0.88** | **0.87** | **95** |
| **Weighted Avg** | **0.92** | **0.89** | **0.90** | **95** |


### Experiment 2: Grayscale PlantVillage + Segmented PlantDoc
* **Dataset:** Grayscale PlantVillage + Segmented PlantDoc
* **Test Set:** 95 images (PlantDoc only)
* **Architecture Setup:** **Unfroze the last 2 blocks** of the DINOv2 Base backbone.
* **Validation Accuracy:** 87.70%
* **Testing Accuracy:** 87.37%
* **Notes:** Testing the mixed dataset strategy on the Base architecture resulted in a slight dip in overall testing accuracy (down to 87.37%) compared to the fully segmented approach.

**Classification Report (Exp 2):**

| Class | Precision | Recall (Acc) | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Corn_Gray_leaf_spot | 0.25 | 0.25 | 0.25 | 4 |
| Corn_leaf_blight | 0.75 | 0.75 | 0.75 | 12 |
| Corn_rust_leaf | 1.00 | 1.00 | 1.00 | 10 |
| Tomato_Septoria_leaf_spot | 0.86 | 1.00 | 0.92 | 12 |
| Tomato_leaf | 1.00 | 0.75 | 0.86 | 8 |
| Apple_Scab_Leaf | 1.00 | 0.90 | 0.95 | 10 |
| Apple_leaf | 1.00 | 0.89 | 0.94 | 9 |
| Apple_rust_leaf | 0.83 | 1.00 | 0.91 | 10 |
| grape_leaf | 0.86 | 1.00 | 0.92 | 12 |
| grape_leaf_black_rot | 1.00 | 0.75 | 0.86 | 8 |
| **Macro Avg** | **0.85** | **0.83** | **0.84** | **95** |
| **Weighted Avg** | **0.88** | **0.87** | **0.87** | **95** |


### Experiment 3: Segmented Datasets with Expanded Real-World Test Set
* **Dataset:** Segmented PlantVillage + Segmented PlantDoc
* **Test Set:** **177 images** (95 PlantDoc + 82 manually collected real-world web images)
* **Architecture Setup:** **Unfroze the last 2 blocks** of the DINOv2 Base backbone.
* **Validation Accuracy:** 88.24% (Peaked at Epoch 10, Early Stopping at Epoch 20)
* **Testing Accuracy:** **91.53%**
* **Notes:** To rigorously evaluate the model, the test set was nearly doubled using manually scraped real-world images. Surprisingly, the model performed *better* on the larger, more diverse dataset, breaking the 91.5% barrier. The expanded dataset also allowed the model to demonstrate a much stronger grasp of the minority `Corn_Gray_leaf_spot` class (jumping from 50% to over 73%).

**Classification Report (Exp 3):**

| Class | Precision | Recall (Acc) | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Corn_Gray_leaf_spot | 0.58 | 0.73 | 0.65 | 15 |
| Corn_leaf_blight | 0.79 | 0.71 | 0.75 | 21 |
| Corn_rust_leaf | 1.00 | 0.90 | 0.95 | 21 |
| Tomato_Septoria_leaf_spot | 1.00 | 1.00 | 1.00 | 22 |
| Tomato_leaf | 1.00 | 1.00 | 1.00 | 14 |
| Apple_Scab_Leaf | 0.93 | 0.87 | 0.90 | 15 |
| Apple_leaf | 1.00 | 0.93 | 0.96 | 14 |
| Apple_rust_leaf | 0.90 | 1.00 | 0.95 | 18 |
| grape_leaf | 1.00 | 1.00 | 1.00 | 18 |
| grape_leaf_black_rot | 1.00 | 1.00 | 1.00 | 19 |
| **Macro Avg** | **0.92** | **0.91** | **0.92** | **177** |
| **Weighted Avg** | **0.92** | **0.92** | **0.92** | **177** |


### Experiment 4: Normal Lab Data + Segmented Real-World Data (Expanded Test Set)
* **Dataset:** Normal PlantVillage (Color, Unsegmented) + Segmented PlantDoc
* **Test Set:** **177 images** (95 PlantDoc + 82 manually collected real-world web images)
* **Architecture Setup:** **Unfroze the last 2 blocks** of the DINOv2 Base backbone.
* **Validation Accuracy:** 86.10% (Peaked at Epoch 5, Early Stopping at Epoch 15)
* **Testing Accuracy:** 90.40%
* **Notes:** This experiment yielded a highly reliable 90.4% accuracy. It demonstrates that the DINOv2 Base model is robust enough to learn from unsegmented lab data (PlantVillage) as long as the complex real-world target data (PlantDoc) is properly segmented.

**Classification Report (Exp 4):**

| Class | Precision | Recall (Acc) | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Corn_Gray_leaf_spot | 0.55 | 0.73 | 0.63 | 15 |
| Corn_leaf_blight | 0.72 | 0.62 | 0.67 | 21 |
| Corn_rust_leaf | 1.00 | 0.90 | 0.95 | 21 |
| Tomato_Septoria_leaf_spot | 1.00 | 1.00 | 1.00 | 22 |
| Tomato_leaf | 1.00 | 1.00 | 1.00 | 14 |
| Apple_Scab_Leaf | 0.93 | 0.93 | 0.93 | 15 |
| Apple_leaf | 0.93 | 1.00 | 0.97 | 14 |
| Apple_rust_leaf | 0.94 | 0.89 | 0.91 | 18 |
| grape_leaf | 1.00 | 1.00 | 1.00 | 18 |
| grape_leaf_black_rot | 1.00 | 1.00 | 1.00 | 19 |
| **Macro Avg** | **0.91** | **0.91** | **0.91** | **177** |
| **Weighted Avg** | **0.91** | **0.90** | **0.91** | **177** |


### Experiment 5: The Unsegmented Breakthrough (Normal PV + Normal PD)
* **Dataset:** Normal PlantVillage + Normal PlantDoc (**No Segmentation Applied**)
* **Test Set:** **177 images** (95 PlantDoc + 82 manually collected real-world web images)
* **Architecture Setup:** **Unfroze the last 2 blocks** of the DINOv2 Base backbone.
* **Validation Accuracy:** 91.44% (Peaked at Epoch 12, Early Stopping at Epoch 22)
* **Testing Accuracy:** **92.09%**
* **Notes:** This is a project-defining result. By running the purely raw, unsegmented images through the DINOv2 Base model, it achieved the absolute highest testing accuracy of the entire project (92.09%). This proves that DINOv2's inherent self-attention mechanisms are sophisticated enough to separate the leaf/disease from complex real-world backgrounds without requiring a separate, computationally expensive segmentation preprocessing step.

**Classification Report (Exp 5):**

| Class | Precision | Recall (Acc) | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Corn_Gray_leaf_spot | 0.65 | 0.73 | 0.69 | 15 |
| Corn_leaf_blight | 0.75 | 0.71 | 0.73 | 21 |
| Corn_rust_leaf | 1.00 | 0.95 | 0.98 | 21 |
| Tomato_Septoria_leaf_spot | 1.00 | 1.00 | 1.00 | 22 |
| Tomato_leaf | 1.00 | 1.00 | 1.00 | 14 |
| Apple_Scab_Leaf | 0.93 | 0.87 | 0.90 | 15 |
| Apple_leaf | 1.00 | 0.93 | 0.96 | 14 |
| Apple_rust_leaf | 0.90 | 1.00 | 0.95 | 18 |
| grape_leaf | 1.00 | 1.00 | 1.00 | 18 |
| grape_leaf_black_rot | 1.00 | 1.00 | 1.00 | 19 |
| **Macro Avg** | **0.92** | **0.92** | **0.92** | **177** |
| **Weighted Avg** | **0.92** | **0.92** | **0.92** | **177** |
