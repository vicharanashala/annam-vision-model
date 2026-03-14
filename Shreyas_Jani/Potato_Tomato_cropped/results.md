# Experiment Results: Potato-Tomato Disease Dataset
**Date:** March 13, 2026  
**Model Architecture:** Swin-HViT (Pretrained Base) with Hybrid Classifier  
**Dataset:** Potato-Tomato Disease (Cropped Data, includes challenging PlantDoc images)

## 1. Experimental Setup
* **Data Preprocessing:** Dataset uploaded to Kaggle. Required manual cleaning to remove forbidden characters and truncate excessively long filenames due to Kaggle environment constraints.
* **Pipeline:** Custom `Dataset` class developed. Migrated the existing Swin-HViT Hybrid classifier class and training loop, updating them to accommodate the new dataset structure.

## 2. Training Progression & Results
* **Convergence Behavior:** The model displayed its typical fast-convergence pattern, reaching a strong initial peak early in the run.
* **Epoch 6:** Achieved **63.0%** validation accuracy. This is a solid baseline given the known difficulty of the PlantDoc images included in the dataset.
* **Epoch 7 to 15:** Validation accuracy largely stagnated, confirming that the model found its limit early (consistently peaking around epochs 5-7 during fine-tuning).
* **Final Test Accuracy (Epoch 15):** **64.51%** on the cropped data.

## 3. Per-Class Performance
* **F1 Scores:** Performance was decently distributed across the classes without any signs of severe mode collapse.
* **Lowest Performer:** `Tomato Septoria leaf spot` had the lowest F1 score at **0.49**. 

## 4. Key Takeaways
* Fine-tuning Swin-HViT continues to show highly consistent convergence timelines (epochs 5-7). 
* The ~64.5% accuracy ceiling makes sense given the realistic/wild nature of the PlantDoc data subset.