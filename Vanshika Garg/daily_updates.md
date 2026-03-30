# DAILY PROGRESS LOG  
## Plant Disease Detection Project (March 2026)

---

## 30/03/26

Today I focused on documenting the complete progress of the plant disease detection system. I prepared detailed weekly and monthly reports explaining the work done so far, including model training, performance issues, experiments with multiple disease classes, and the shift toward a more structured pipeline. I also reviewed the project workflow from dataset collection to model evaluation to ensure that the work is clearly understandable for anyone viewing the project on GitHub. The main goal today was to properly conclude the current phase of the project and present the research and findings in a clear and professional way.

---

## 28/03/26

Today I worked on finalizing the structured pipeline for solving the plant disease detection problem. Instead of directly jumping into model training, I created a clear step-by-step approach: dataset collection → dataset cleaning → binary classification (healthy vs diseased) → multi-class disease classification → infected region detection. I also reviewed the research paper I had studied earlier and aligned my approach with real-world research methods. This helped in creating a more systematic and research-driven workflow for the project.

---

## 27/03/26

Today I focused on improving the understanding of how to build a real-world AI system for agriculture instead of only working on dataset-based accuracy. I studied the limitations of existing datasets such as PlantVillage and realized that models trained only on clean datasets struggle when tested on real-world images. Based on this, I started planning a better dataset collection strategy that focuses on real agricultural crops, common diseases, and different environmental conditions. This helped in defining a more practical direction for the project.

---

## 26/03/26

Today I analysed the results obtained from the binary classification experiment (healthy vs diseased). The results were more stable compared to the earlier multi-class model, which confirmed that simplifying the problem first is a better approach. I carefully reviewed prediction outputs and verified that the model was now focusing more on disease patterns rather than background features. Based on this analysis, I decided to continue building the system in stages instead of directly solving the multi-class problem.

---

## 25/03/26

Today I worked on further improving the dataset after observing issues in model performance. I reviewed the dataset again and focused on making it more balanced and structured. I also studied different types of plant diseases and identified that many diseases look visually similar, which makes classification more difficult. This helped in understanding the importance of building a strong dataset before moving toward more complex models.

---

## 24/03/26

Today I focused on analysing the behaviour of the trained models and understanding their strengths and weaknesses. Instead of training new models immediately, I reviewed prediction outputs and compared correct and incorrect predictions. I observed that some diseases were getting predicted correctly while others were being confused due to similar visual patterns. These observations helped in improving the overall strategy for the project and avoiding unnecessary retraining.

---

## 23/03/26

Today I continued working on improving the overall quality of the plant disease detection system. I reviewed the dataset structure, checked the training pipeline again, and ensured that the data flow from dataset to model training to prediction was working correctly. I also reviewed the earlier experiments to understand what improvements were required next. The main focus today was on making the system more structured and reliable instead of only focusing on model accuracy.

---

## 21/03/26

Today I focused on analysing the performance of the plant disease detection model trained using multiple disease classes. After reviewing the prediction results carefully, I observed that the model accuracy was not stable across different classes. While some diseases were being predicted correctly, many classes with similar visual symptoms were getting confused. Based on this observation, I decided to simplify the problem and test whether the model could first learn the fundamental difference between healthy and diseased plants before moving toward multi-class classification. This helped in defining a clearer strategy for improving model performance.

---

## 20/03/26

Today I experimented with simplifying the dataset into two main categories: **Healthy** and **Diseased**. The main goal was to check whether the model can first understand the basic visual difference between a healthy leaf and an infected leaf. I reorganized the dataset accordingly and trained the model again using this simplified approach. After training, the model performance became more stable compared to the earlier multi-class setup. This experiment confirmed that learning the basic disease vs healthy concept first can help improve overall system performance in later stages.

---

## 19/03/26

Today I worked on improving the dataset quality to make the model more reliable. I carefully reviewed the dataset and removed duplicate images, very low-quality images, and images that could confuse the model during training. I also checked class balance and made sure that the dataset was more structured and consistent. This process helped in improving the overall data quality, which is very important for building a real-world plant disease detection system. After cleaning the dataset, I prepared it again for training and validation.

---

## 18/03/26

Today I focused on improving the training strategy of the model. I implemented stronger data augmentation techniques to make the model more robust to real-world conditions. These techniques included rotation, flipping, random cropping, and brightness/contrast adjustments. The goal was to ensure that the model does not overfit to a limited dataset and instead learns more generalized disease features. After implementing these changes, I prepared the model for retraining using the improved augmentation pipeline.

---

## 17/03/26

Today I spent time analysing the results of the previously trained models and trying to understand why the model was not performing well on unseen images. I compared training accuracy and validation accuracy and observed signs of overfitting. I also noticed that the model might be learning background patterns instead of learning actual disease features. Based on this observation, I planned to improve both the dataset quality and the training strategy in the coming days to make the model more reliable.

---

## 16/03/26

Today I reviewed the overall progress of the plant disease detection project and focused on understanding the main challenges in building a real-world AI solution for agriculture. Instead of directly training more models, I spent time analysing the problem more deeply, especially the challenges of multi-class disease classification, dataset limitations, and model generalization. Based on this analysis, I decided to improve the project in a more structured way by focusing on better datasets, better training strategies, and a step-by-step approach starting from simpler tasks toward more complex disease classification.

---

## 14/03/26

Today I focused on improving the robustness of the plant disease detection models. I trained the model using the combined **PlantVillage dataset (color, grayscale, and segmented images)** to reduce background bias and encourage the model to learn disease-specific features.

A stronger data augmentation pipeline was implemented, including:
- Random rotations  
- Color jitter  
- Random cropping  
- Random erasing  

These augmentations were designed to simulate real-world conditions and prevent shortcut learning.

I also verified the data pipeline to ensure grayscale images were properly converted to RGB format for compatibility with the **ResNet architecture**. After completing training, I began analyzing model behavior and planned to re-evaluate attention patterns using **Grad-CAM** to verify whether the model focuses more accurately on disease lesions.

---

## 13/03/26

Today I applied **Grad-CAM visualization** to analyze the decision-making behavior of the trained CNN model. I generated attention heatmaps for multiple input images to understand where the model focuses when predicting plant diseases.

The analysis revealed that while the model often attends to lesion areas, in some cases it still focuses on background regions or irrelevant parts of the leaf, which can lead to incorrect predictions.

I documented these observations and started comparing attention patterns between correct and incorrect predictions to better understand the model’s weaknesses.

---

## 12/03/26

Today I conducted a detailed **error analysis** of model predictions. I examined several incorrect predictions and compared them with the corresponding Grad-CAM heatmaps to determine why misclassifications occur.

I observed that certain diseases share visually similar lesion patterns across different plant species, which can cause confusion for the model. Additionally, some images contain complex backgrounds or multiple leaves, which may influence model predictions.

These findings highlight the importance of improving model robustness and reducing dataset bias.

---

## 11/03/26

Today I evaluated the trained models on a subset of images to validate the inference pipeline and verify prediction outputs.

I confirmed that the model can correctly classify some plant disease samples while still struggling with certain classes. I also verified the mapping between predicted class indices and dataset labels to ensure predictions are interpreted correctly.

After confirming that the prediction pipeline is working as expected, I prepared the setup for interpretability experiments using **Grad-CAM**.

---

## 10/03/26

Continued domain shift analysis between **PlantVillage** and **PlantDoc** datasets. Evaluated the **Vision Transformer (ViT)** model specifically on the *Apple_Scab_Leaf* class from the PlantDoc dataset to verify whether the model could correctly recognize real-world samples.

Compared the results with the previously tested **ResNet-50** model and observed that both models struggled to generalize, with only a very small number of correct predictions.

I also implemented **Grad-CAM visualization for both CNN and ViT models** to understand which image regions the models were focusing on during prediction. I debugged Grad-CAM implementation issues related to transformer token reshaping and successfully generated heatmaps to interpret model attention on leaf disease regions.

---

## 09/03/26

Today I focused on detailed **class-wise evaluation** to better understand model failure cases under domain shift.

I selected the *Apple_Scab_Leaf* class from the PlantDoc dataset (83 images) and created a testing pipeline to evaluate the trained **ResNet-50** model on this single class.

I implemented custom label mapping between PlantVillage and PlantDoc classes to resolve label mismatch issues across datasets. I then analyzed predictions and printed sample outputs to observe which incorrect classes the model predicted, helping identify patterns in model confusion.

---

## 07/03/26

Worked on improving the **cross-dataset evaluation pipeline** for testing models trained on PlantVillage against the PlantDoc dataset.

I investigated dataset structure differences and fixed issues related to class folder expectations and image loading errors. I also implemented a label mapping strategy to align class names between datasets and ensure consistent evaluation.

Finally, I verified the dataset pipeline and prepared the environment for running domain shift experiments on both **ResNet-50** and **Vision Transformer (ViT)** models.

---

## 06/03/26

Today I worked on strengthening my **plant disease detection project** to make it more relevant for real-world applications.

I first reviewed important evaluation concepts such as:
- True positives / False positives  
- Cross-entropy loss  
- Label smoothing  
- F1 score  

I then evaluated both **Vision Transformer (ViT)** and **ResNet-50** on the **PlantVillage Dataset**, documenting results such as accuracy, F1 score, parameter count, and inference speed, and created a `results.md` comparison table.

After realizing that PlantVillage is a lab-controlled dataset, I planned to test model robustness using real-world images from the **PlantDoc Dataset** to analyze domain shift.

Finally, I started implementing **Stage-2 domain shift testing** by setting up the dataset pipeline and evaluation code to test how my trained models perform on more realistic plant disease images.

---

## 05/03/26

### ViT vs CNN Model for Plant Disease Detection

- Trained a CNN model (e.g., **ResNet-50**) on the PlantVillage dataset using Kaggle GPU and achieved high validation accuracy.  
- Saved the trained model weights so retraining is not required in every session.  
- Evaluated the model by generating a confusion matrix, checking test accuracy/F1 score, counting parameters, and benchmarking inference time.  
- Started setting up the **Vision Transformer (ViT)** for transfer learning and fixed the classifier size mismatch error for the 38 plant disease classes.

---
