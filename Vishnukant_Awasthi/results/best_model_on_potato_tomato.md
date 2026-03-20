# DINOv2 Base Results - Potato & Tomato Dataset

**Model Description:** Evaluating the **DINOv2 Base** architecture specifically on a multi-crop disease classification task. The goal of this experiment is to determine how well the self-supervised pre-trained features of DINOv2 generalize to a combined dataset containing 10 distinct classes of Potato and Tomato leaf diseases.

---

## Quick Summary of All Experiments

| Exp | Dataset Used | Backbone Status | Best Val Acc | Test Acc | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Potato & Tomato Disease Data | Unfroze Last 2 Blocks | 69.52% | 69.24% | Tested on 842 images. Established an initial baseline. Struggles with early blight cross-class confusion but handles healthy leaves exceptionally well. |
| **2** | **Potato & Tomato Disease Data** | **Unfroze Last 4 Blocks** | **69.37%** | **70.67%** | **Tested on 842 images. Deeper fine-tuning pushed the model past 70% accuracy and improved healthy/viral detection, but fungal similarity remains a bottleneck.** |

---

## Detailed Breakdown of Each Experiment

### Experiment 1: Initial Baseline Fine-Tuning (2 Blocks Unfrozen)
* **Dataset:** Combined Potato and Tomato Disease Dataset (10 Classes)
* **Test Set:** **842 images**
* **Architecture Setup:** **Unfroze the last 2 blocks** of the DINOv2 Base backbone. Image resolution set to **224x224**. 
* **Validation Accuracy:** 69.52%
* **Testing Accuracy:** **69.24%**
* **Notes:** This run established the initial baseline for applying DINOv2 Base to the mixed Potato and Tomato dataset. By unfreezing just 2 blocks, the model achieved a 69.24% testing accuracy. It performs exceptionally well on healthy leaves (95% and 84%) and distinct viral structures (`Yellow_Leaf_Curl_Virus` at 85%). However, it struggles heavily with closely related fungal profiles across the two species, specifically getting confused between `Tomato_Early_blight` (53%), `Potato_Early_blight` (58%), and `Tomato_Septoria_leaf_spot` (49%). 

|Class|Precision|Recall (Acc)|F1-Score|Support|
|:----|:----|:----|:----|:----|
|Potato_Early_blight|0.57|0.59|0.58|87|
|Potato_Lateblight|0.74|0.62|0.67|104|
|Potato_healthy|0.90|0.95|0.92|63|
|Tomato_Bacterial_spot|0.51|0.74|0.61|100|
|Tomato_Early_blight|0.43|0.53|0.47|62|
|Tomato_Late_blight|0.70|0.68|0.69|90|
|Tomato_Leaf_mold|0.83|0.75|0.79|73|
|Tomato_Septoria_leaf_spot|0.68|0.49|0.57|108|
|Tomato_Tomato_Yellow_Leaf_Curl_Virus|0.89|0.86|0.87|84|
|Tomato_healthy|0.90|0.85|0.87|71|
|Macro Avg|0.72|0.71|0.71|842|
|Weighted Avg|0.71|0.69|0.69|842|


### Experiment 2: Deeper Fine-Tuning (4 Blocks Unfrozen)
* **Dataset:** Combined Potato and Tomato Disease Dataset (10 Classes)
* **Test Set:** **842 images**
* **Architecture Setup:** **Unfroze the last 4 blocks** of the DINOv2 Base backbone. Image resolution set to **224x224**. 
* **Validation Accuracy:** 69.37%
* **Testing Accuracy:** **70.67%**
* **Per-Class Testing Accuracy (Recall):**
  * Potato_healthy: 98.41%
  * Tomato_Tomato_Yellow_Leaf_Curl_Virus: 92.86%
  * Tomato_Bacterial_spot: 82.00%
  * Tomato_healthy: 81.69%
  * Tomato_Leaf_mold: 76.71%
  * Tomato_Late_blight: 74.44%
  * Potato_Early_blight: 57.47%
  * Potato_Lateblight: 52.88%
  * Tomato_Early_blight: 51.61%
  * Tomato_Septoria_leaf_spot: 50.93%
* **Notes:** To combat the cross-class fungal confusion observed in the baseline, the fine-tuning depth was doubled to 4 blocks. This successfully raised the overall testing accuracy to 70.67%. The model's ability to identify highly distinct classes skyrocketed (e.g., `Potato_healthy` reached 98.41% and `Tomato_Tomato_Yellow_Leaf_Curl_Virus` jumped to 92.86%). However, unfreezing 4 blocks still did not give the model the nuanced feature extraction needed to reliably separate `Potato_Early_blight` and `Tomato_Early_blight`. This heavily implies that advanced regularization, focal loss, or higher image resolutions will be necessary to resolve the domain overlap between the two crops.


**Confusion Matrix (Exp 2):**
```text
[[50 17  0  7  6  3  1  3  0  0] 
 [33 55  2  0  3  7  1  2  1  0] 
 [ 1  0 62  0  0  0  0  0  0  0] 
 [ 0  0  2 82  1  0  1 13  0  1] 
 [ 7  0  0  5 32  9  2  5  0  2] 
 [ 0  0  0  2 11 67  1  1  4  4] 
 [ 0  0  0  5  3  6 56  2  0  1] 
 [ 4  0  0 27 17  1  1 55  2  1] 
 [ 0  0  0  2  0  0  1  1 78  2] 
 [ 0  0  3  0  0  1  1  0  8 58]]
```
|Class|Precision|Recall (Acc)|F1-Score|Support|
|:----|:----|:----|:----|:----|
|Potato_Early_blight|0.53|0.57|0.55|87|
|Potato_Lateblight|0.76|0.53|0.62|104|
|Potato_healthy|0.90|0.98|0.94|63|
|Tomato_Bacterial_spot|0.63|0.82|0.71|100|
|Tomato_Early_blight|0.44|0.52|0.47|62|
|Tomato_Late_blight|0.71|0.74|0.73|90|
|Tomato_Leaf_mold|0.86|0.77|0.81|73|
|Tomato_Septoria_leaf_spot|0.67|0.51|0.58|108|
|Tomato_Tomato_Yellow_Leaf_Curl_Virus|0.84|0.93|0.88|84|
|Tomato_healthy|0.84|0.82|0.83|71|
|Macro Avg|0.72|0.72|0.71|842|
|Weighted Avg|0.71|0.71|0.70|842|
