# Experiment Results: Custom Bottleneck DeiT-Tiny
**Date:** March 3, 2026  
**Model Architecture:** DeiT-tiny (trained from scratch) with a custom patch embedding that passes through a bottleneck first.  
**Datasets:** PlantVillage (PV) and PlantDoc (PD)

## 1. Zero-Shot Evaluation on PlantDoc
**Setup:** Models trained on PV and validated exclusively on PD to test zero-shot generalization.

* **General Observation:** Zero-shot classification on PlantDoc proved exceptionally difficult. Per-class accuracy and F1 scores were very low, often barely performing above a random guess, with severe prediction bias (e.g., heavy bias toward pepper bell bacterial spot and tomato late blight). 

| Bottleneck Size | Max PD Val Accuracy | Epoch Reached | Notes |
| :--- | :--- | :--- | :--- |
| **16** | 11.4% | 30 | Reached 81% on PV early on but failed to generalize. Horrible F1 scores (some classes at 0). |
| **32** | 10.5% | 30 | Lowest accuracy of the three, but yielded slightly better (though still poor) F1 scores. |
| **64** | 11.0% | 30 | Fastest initial improvement (10.5% at epoch 16) but stagnated shortly after. |

## 2. Mixed Dataset Evaluation
**Setup:** Created a combined dataset of PV and PD. Models were trained on the mix, and validation was later isolated to only the PD subset to measure direct improvement.

* **Mixed Validation (PV+PD):** All bottleneck sizes reached ~80% validation accuracy by epoch 10.
* **Isolated Validation (PD Only):** | Bottleneck Size | PD-Only Val Accuracy |
| **16** | 20.3% |
| **32** | 13.7% |
| **64** | **23.5%** *(Highest)* |

## 3. Key Takeaways
* **Underperformance:** Even with a mixed dataset, the highest accuracy achieved on PlantDoc was 23.5% (Bottleneck 64). 
* **Baseline Comparison:** This setup is significantly outperformed by the previously tested fine-tuned **Swin-HViT**, which achieved 60%+ accuracy under similar mixed-dataset conditions. 
* **Conclusion:** For this specific classification task, Swin-HViT remains the superior model. The custom bottleneck DeiT-tiny approach is not viable for this use case.

## 4. Next Steps
* **Dataset Exploration:** Investigate alternative datasets, such as PlantWild, to see if data diversity impacts learning.
* **Alternative Architectures:** Shift focus away from standard attention mechanisms. Next research targets are **MLP-mixer** and **FocalNet**, which provide unique structural alternatives to traditional Transformers.