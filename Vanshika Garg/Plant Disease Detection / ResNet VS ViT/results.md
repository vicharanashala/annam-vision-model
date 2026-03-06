# Plant Disease Detection: CNN vs Vision Transformer

## Dataset

**Dataset:** PlantVillage
**Number of Classes:** 38
**Task:** Plant disease classification from leaf images

---

# Model Comparison Results

| Model                  | Test Accuracy | F1 Score    | Parameters | Parameters (Millions) | Total Inference Time | Time per Image |
| ---------------------- | ------------- | ----------- | ---------- | --------------------- | -------------------- | -------------- |
| ResNet-50              | **0.99865**   | **0.99865** | 23,585,894 | 23.59M                | 28.80 s              | **0.00353 s**  |
| ViT-Base (Patch16-224) | 0.99251       | 0.99249     | 85,827,878 | 85.83M                | 100.89 s             | 0.01238 s      |

---

# Key Observations

### 1️⃣ Accuracy

**ResNet-50 achieved higher accuracy** than Vision Transformer on PlantVillage.

* ResNet-50: **99.86%**
* ViT-Base: **99.25%**

CNNs perform extremely well on structured visual tasks like leaf disease detection because they capture **local texture patterns** effectively.

---

### 2️⃣ Model Size

| Model     | Parameters |
| --------- | ---------- |
| ResNet-50 | 23.6M      |
| ViT-Base  | 85.8M      |

ViT has **~3.6× more parameters** than ResNet-50.

---

### 3️⃣ Inference Speed

| Model     | Time per Image |
| --------- | -------------- |
| ResNet-50 | **0.00353 s**  |
| ViT-Base  | 0.01238 s      |

ResNet-50 is **~3.5× faster** during inference.

---

# Conclusion

For **plant disease detection with limited compute resources**, **ResNet-50 outperforms Vision Transformer** in:

* Accuracy
* Model efficiency
* Inference speed

Therefore, CNN-based architectures remain a **strong baseline for agricultural computer vision tasks**.

---

# Future Work

Possible improvements for Vision Transformers:

* Hybrid CNN-ViT architecture
* Stronger data augmentation
* Self-supervised pretraining
* Real-world field image evaluation

---
