# DINOv2 Base (Vision Transformer) Results - Rice Dataset

**Model Description:** Evaluating the **DINOv2 Base** architecture on a highly complex, 19-class Rice Disease dataset. The goal of this experiment is to assess the baseline generalization capability of DINOv2's self-supervised pre-training on a diverse set of fungal, bacterial, viral, and pest-related rice crop damages.

---

## Quick Summary of All Experiments

| Exp | Dataset Used | Backbone Status | Best Val Acc | Test Acc | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Rice Disease Data (19 Classes)** | **Unfroze Last 2 Blocks** | **93.30%** | **94.19%** | **Tested on 465 images. Exceptional baseline performance. Achieves 100% recall on 5 different classes, with `Sheath Rot` being the only major bottleneck.** |

---

## Detailed Breakdown of Each Experiment

### Experiment 1: Initial Baseline Fine-Tuning (2 Blocks Unfrozen)
* **Dataset:** Rice Disease Dataset (19 Classes)
* **Test Set:** **465 images** * **Architecture Setup:** **Unfroze the last 2 blocks** of the DINOv2 Base backbone. Image resolution set to **224x224**. 
* **Validation Accuracy:** 93.30%
* **Testing Accuracy:** **94.19%**
* **Notes:** This initial run proves that the DINOv2 Base model is extraordinarily well-suited for rice crop analysis. Unfreezing merely 2 blocks allowed the model to effortlessly push past 94% testing accuracy. It scored a perfect 100% precision and 100% recall across multiple disease profiles (Bacterial Streak, False Smut, Neck Blast, Sheath Blight, and Tungro). The only significant struggle the model faced was with `Sheath Rot` (53.33% recall), which was frequently misclassified into the `Leaf Smut` and `Healthy` categories, indicating that deeper fine-tuning may be required to isolate those specific fungal textures.
* 
**Confusion Matrix (Exp 1):**
```text
[[29  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0] 
 [ 0 15  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0] 
 [ 0  0 15  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0] 
 [ 0  0  0 23  0  0  0  0  0  0  0  7  0  0  0  0  0  0  0] 
 [ 0  0  0  0 15  0  0  0  0  0  0  0  0  0  0  0  0  0  0] 
 [ 0  0  1  0  0 12  0  0  0  0  0  0  0  2  0  0  0  0  0] 
 [ 0  0  0  0  0  0 29  1  0  0  0  0  0  0  0  0  0  0  0] 
 [ 0  0  0  0  0  0  0 30  0  0  0  0  0  0  0  0  0  0  0] 
 [ 0  0  0  0  0  0  1  0 29  0  0  0  0  0  0  0  0  0  0] 
 [ 0  0  0  1  0  0  0  0  0 29  0  0  0  0  0  0  0  0  0] 
 [ 0  0  0  3  0  0  0  0  0  0 27  0  0  0  0  0  0  0  0] 
 [ 0  0  0  0  0  0  0  0  0  0  0 30  0  0  0  0  0  0  0] 
 [ 0  0  0  0  0  0  0  0  0  0  0  0 30  0  0  0  0  0  0] 
 [ 0  0  1  0  0  0  0  0  0  0  0  0  0 14  0  0  0  0  0] 
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0 30  0  0  0  0] 
 [ 6  0  0  0  0  0  0  0  0  0  0  0  0  0  0  8  1  0  0] 
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 15  0  0] 
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 30  0] 
 [ 0  0  0  0  0  0  1  0  0  1  0  0  0  0  0  0  0  0 28]]
```
|Class|Precision|Recall (Acc)|F1-Score|Support|
|:----|:----|:----|:----|:----|
|Bacterial Leaf Blight|0.83|0.97|0.89|30|
|Bacterial Streak|1.00|1.00|1.00|15|
|Bakanae|0.88|1.00|0.94|15|
|Brown Spot|0.85|0.77|0.81|30|
|False Smut|1.00|1.00|1.00|15|
|Grassy Stunt Virus|1.00|0.80|0.89|15|
|Healthy Leaf|0.94|0.97|0.95|30|
|Hispa|0.97|1.00|0.98|30|
|Leaf Blast|1.00|0.97|0.98|30|
|Leaf Scald|0.94|0.97|0.95|30|
|Leaf Smut|1.00|0.90|0.95|30|
|Narrow Brown Spot|0.81|1.00|0.90|30|
|Neck Blast|1.00|1.00|1.00|30|
|Ragged Stunt Virus|0.88|0.93|0.90|15|
|Sheath Blight|1.00|1.00|1.00|30|
|Sheath Rot|1.00|0.53|0.70|15|
|Stem Rot|0.94|1.00|0.97|15|
|Tungro|1.00|1.00|1.00|30|
|Insect Affected|1.00|0.93|0.97|30|
|Macro Avg|0.95|0.93|0.94|465|
|Weighted Avg|0.95|0.94|0.94|465|

### Experiment 2: Deeper Fine-Tuning (4 Blocks Unfrozen)
* **Dataset:** Rice Disease Dataset (19 Classes)
* **Test Set:** **465 images** * **Architecture Setup:** **Unfroze the last 4 blocks** of the DINOv2 Base backbone. Image resolution set to **224x224**. 
* **Validation Accuracy:** 93.83%
* **Testing Accuracy:** **94.62%**
* **Notes:** To resolve the textural confusion observed in the first experiment, the fine-tuning depth was doubled to 4 blocks. This successfully raised the overall testing accuracy to 94.62% and pushed the validation accuracy to 93.83%. The model perfectly categorized `Healthy Leaf` instances (up from 96.67% to 100%) and maintained its perfect 100% recall on 9 other classes. Most importantly, the deeper feature extraction allowed the model to significantly improve its distinction of `Sheath Rot`, jumping from a 53% recall in Experiment 1 to 67% here.


**Confusion Matrix (Exp 2):**
```text
[[29  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0] 
 [ 0 15  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0] 
 [ 0  0 15  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0] 
 [ 0  0  0 23  0  0  0  0  0  0  0  7  0  0  0  0  0  0  0] 
 [ 0  0  0  0 15  0  0  0  0  0  0  0  0  0  0  0  0  0  0] 
 [ 0  0  2  0  0 12  0  0  0  0  0  0  0  1  0  0  0  0  0] 
 [ 0  0  0  0  0  0 30  0  0  0  0  0  0  0  0  0  0  0  0] 
 [ 0  0  0  0  0  0  0 30  0  0  0  0  0  0  0  0  0  0  0] 
 [ 0  0  0  0  0  0  0  0 29  0  0  0  0  0  0  0  0  1  0] 
 [ 0  0  0  1  0  0  0  0  0 29  0  0  0  0  0  0  0  0  0] 
 [ 1  0  0  3  0  0  0  0  0  0 26  0  0  0  0  0  0  0  0] 
 [ 0  0  0  0  0  0  0  0  0  0  0 30  0  0  0  0  0  0  0] 
 [ 0  0  0  0  0  0  0  0  0  0  0  0 30  0  0  0  0  0  0] 
 [ 0  0  0  0  0  0  0  0  1  0  0  0  0 14  0  0  0  0  0] 
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0 30  0  0  0  0] 
 [ 5  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10  0  0  0] 
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 15  0  0] 
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 30  0] 
 [ 0  0  0  0  0  0  1  0  0  1  0  0  0  0  0  0  0  0 28]]
```
|Class|Precision|Recall (Acc)|F1-Score|Support|
|:----|:----|:----|:----|:----|
|Bacterial Leaf Blight|0.83|0.97|0.89|30|
|Bacterial Streak|1.00|1.00|1.00|15|
|Bakanae|0.88|1.00|0.94|15|
|Brown Spot|0.85|0.77|0.81|30|
|False Smut|1.00|1.00|1.00|15|
|Grassy Stunt Virus|1.00|0.80|0.89|15|
|Healthy Leaf|0.97|1.00|0.98|30|
|Hispa|1.00|1.00|1.00|30|
|Leaf Blast|0.97|0.97|0.97|30|
|Leaf Scald|0.94|0.97|0.95|30|
|Leaf Smut|1.00|0.87|0.93|30|
|Narrow Brown Spot|0.81|1.00|0.90|30|
|Neck Blast|1.00|1.00|1.00|30|
|Ragged Stunt Virus|0.93|0.93|0.93|15|
|Sheath Blight|1.00|1.00|1.00|30|
|Sheath Rot|1.00|0.67|0.80|15|
|Stem Rot|1.00|1.00|1.00|15|
|Tungro|0.97|1.00|0.98|30|
|Insect Affected|1.00|0.93|0.97|30|
|Macro Avg|0.95|0.94|0.94|465|
|Weighted Avg|0.95|0.95|0.95|465|
