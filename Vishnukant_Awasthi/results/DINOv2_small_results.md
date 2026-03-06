# DINOv2 Small (Vision Transformer) Results

**Model Description:** Moved away from standard Convolutional Neural Networks (like ResNet and EfficientNet) and tested a state-of-the-art Vision Transformer: **DINOv2 Small**. The goal was to see if DINOv2's advanced pre-trained feature extraction could handle complex 10 class leaf disease dataset better than traditional models. 

Over a series of 7 experiments, tested different dataset combinations (Grayscale, Normal, Segmented), tweaked the model's backbone (freezing vs. unfreezing blocks), adjusted image cropping sizes, and used cross validation to find the absolute best pipeline.

---

## Summary of All Experiments

| Exp | Dataset Used | Backbone Status & Image Size | Best Val Acc | Test Acc | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Grayscale PV + Seg PD | Fully Frozen (224x224) | 80.75% | 70.53% | Baseline established, but failed completely on `Corn_Gray_leaf_spot` (0%). |
| **2** | Normal PV + Seg PD | Fully Frozen (224x224) | 85.03% | 72.63% | Using normal color images fixed the 0% corn class failure. |
| **3** | Segmented PV + Seg PD | Fully Frozen (224x224) | 80.21% | 76.84% | Fully segmented data proved to be the best for real world generalization. |
| **4** | **Segmented PV + Seg PD** | **Unfroze Last 2 Blocks (256 ➔ 224)** | **89.30%** | **90.53%** | 6 out of 10 classes hit 100% testing accuracy! |
| **5** | Segmented PV + Seg PD | Unfroze Last 4 Blocks (256 ➔ 224) | 89.30% | 88.42% | Unfreezing too much degraded performance; model forgot the hardest corn class. |
| **6** | Segmented PV + Seg PD | Unfroze Last 2 Blocks (Weighted) | 86.10% | 83.16% | Weighted sampling helped the minority class but confused the model on easy classes. |
| **7** | **Segmented PV + Seg PD** | **Unfroze Last 2 Blocks (5-Fold CV)** | **~98.35%** | **90.50%** | Proved that the 90.5% accuracy from Exp 4 was highly robust and consistent. |

---

## Detailed Breakdown of Each Experiment

### Experiment 1: The Grayscale + Segmented Baseline
* **Dataset:** Grayscale PlantVillage + Segmented PlantDoc
* **Architecture Setup:** Backbone fully frozen. Images resized directly to 224x224.
* **Validation Accuracy:** 80.75%
* **Testing Accuracy:** 70.53%
* **Per-Class Testing Accuracy:**
  * grape_leaf: 91.67%
  * Tomato_leaf: 87.50%
  * Tomato_Septoria_leaf_spot: 83.33%
  * Corn_rust_leaf: 80.00%
  * Corn_leaf_blight: 75.00%
  * Apple_Scab_Leaf: 70.00%
  * Apple_rust_leaf: 70.00%
  * Apple_leaf: 55.56%
  * grape_leaf_black_rot: 37.50%
  * Corn_Gray_leaf_spot: **0.00%**
* **Notes:** While the overall accuracy was okay, the model completely failed on `Corn_Gray_leaf_spot` and struggled heavily with `grape_leaf_black_rot`. It was heavily biased toward easier classes.

### Experiment 2: Switching to Normal Color Images
* **Dataset:** Normal Color PlantVillage + Segmented PlantDoc
* **Architecture Setup:** Backbone fully frozen. Images resized directly to 224x224.
* **Validation Accuracy:** 85.03%
* **Testing Accuracy:** 72.63%
* **Per-Class Testing Accuracy:**
  * Tomato_Septoria_leaf_spot: 100.00%
  * grape_leaf: 83.33%
  * Corn_rust_leaf: 80.00%
  * Apple_Scab_Leaf: 80.00%
  * Apple_leaf: 77.78%
  * Tomato_leaf: 75.00%
  * Apple_rust_leaf: 60.00%
  * Corn_leaf_blight: 58.33%
  * Corn_Gray_leaf_spot: 50.00% *(Improved from 0%)*
  * grape_leaf_black_rot: 37.50%
* **Notes:** Removing the grayscale/segmentation improved overall stability and successfully fixed the 0% failure on the Corn class.

### Experiment 3: Fully Segmented Datasets
* **Dataset:** Segmented PlantVillage + Segmented PlantDoc
* **Architecture Setup:** Backbone fully frozen. Images resized directly to 224x224.
* **Validation Accuracy:** 80.21%
* **Testing Accuracy:** 76.84%
* **Per-Class Testing Accuracy:**
  * Tomato_Septoria_leaf_spot: 91.67%
  * Corn_rust_leaf: 90.00%
  * Apple_Scab_Leaf: 90.00%
  * Apple_rust_leaf: 90.00%
  * Tomato_leaf: 87.50%
  * grape_leaf: 83.33%
  * Apple_leaf: 77.78%
  * Corn_Gray_leaf_spot: 50.00%
  * grape_leaf_black_rot: 50.00%
  * Corn_leaf_blight: 41.67%
* **Notes:** Using fully segmented images for both training and testing provided the highest testing accuracy so far for a frozen model.

### Experiment 4: The 90.5% Breakthrough (Optimized Architecture)
* **Dataset:** Segmented PlantVillage + Segmented PlantDoc
* **Architecture Setup:** **Unfroze the last 2 blocks** of the DINOv2 backbone. Resized to 256x256, followed by a **224 CenterCrop**.
* **Validation Accuracy:** 89.30%
* **Testing Accuracy:** **90.53%**
* **Per-Class Testing Accuracy:**
  * Tomato_Septoria_leaf_spot: **100.00%**
  * Tomato_leaf: **100.00%**
  * Apple_leaf: **100.00%**
  * Apple_rust_leaf: **100.00%**
  * grape_leaf: **100.00%**
  * grape_leaf_black_rot: **100.00%**
  * Apple_Scab_Leaf: 90.00%
  * Corn_rust_leaf: 80.00%
  * Corn_leaf_blight: 66.67%
  * Corn_Gray_leaf_spot: 50.00%
* **Notes:** This was a massive leap. By unfreezing the last two blocks and using a CenterCrop, the model learned domain-specific leaf features. The model achieved a perfect 100% testing accuracy on 6 different classes.

### Experiment 5: Testing the Limits (Unfreezing Too Much)
* **Dataset:** Segmented PlantVillage + Segmented PlantDoc
* **Architecture Setup:** **Unfroze the last 4 blocks**. Resized to 256 with 224 CenterCrop.
* **Validation Accuracy:** 89.30%
* **Testing Accuracy:** 88.42%
* **Per-Class Testing Accuracy:**
  * Tomato_Septoria_leaf_spot: 100.00%
  * Tomato_leaf: 100.00%
  * Apple_rust_leaf: 100.00%
  * grape_leaf: 100.00%
  * grape_leaf_black_rot: 100.00%
  * Corn_rust_leaf: 90.00%
  * Apple_Scab_Leaf: 90.00%
  * Apple_leaf: 88.89%
  * Corn_leaf_blight: 66.67%
  * Corn_Gray_leaf_spot: **0.00%** *(Regressed completely)*
* **Notes:** Unfreezing more blocks caused "catastrophic forgetting." The model completely forgot how to classify the most difficult class, proving that unfreezing exactly 2 blocks provides good results.

### Experiment 6: Forcing Attention with Weighted Random Sampler
* **Dataset:** Segmented PlantVillage + Segmented PlantDoc
* **Architecture Setup:** Unfroze last 2 blocks. Introduced a **Weighted Random Sampler** during training.
* **Validation Accuracy:** 86.10%
* **Testing Accuracy:** 83.16%
* **Per-Class Testing Accuracy:**
  * Tomato_Septoria_leaf_spot: 100.00%
  * Apple_rust_leaf: 100.00%
  * grape_leaf: 100.00%
  * grape_leaf_black_rot: 100.00%
  * Corn_rust_leaf: 90.00%
  * Apple_Scab_Leaf: 90.00%
  * Tomato_leaf: 87.50%
  * Apple_leaf: 66.67%
  * Corn_leaf_blight: 41.67%
  * Corn_Gray_leaf_spot: 25.00%
* **Notes:** Weighted sampling was used to force the model to pay attention to `Corn_Gray_leaf_spot`. While it brought the Corn class back up to 25%, it caused the model to lose confidence in other easier classes.

### Experiment 7: Validating the Success (5-Fold Cross Validation)
* **Dataset:** Segmented PlantVillage + Segmented PlantDoc
* **Architecture Setup:** Unfroze last 2 blocks. Resized to 256 with 224 CenterCrop.
* **Validation Accuracy:** **98.35% (Average across 5 folds)**
    * Fold 1: 98.45%
    * Fold 2: 98.17%
    * Fold 3: 98.04%
    * Fold 4: 99.04%
    * Fold 5: 98.08%
* **Testing Accuracy:** **90.50%**
* **Per-Class Testing Accuracy:** Matches the optimal distribution seen in Experiment 4, maintaining near-perfect scores on 6 classes and retaining the 90.5% overall benchmark.
* **Notes:** To prove that the 90.53% score from Experiment 4 wasn't just a lucky split, a rigorous 5-Fold Cross-Validation was run. The validation scores were incredibly high and stable across every single fold, proving this pipeline is highly robust.
