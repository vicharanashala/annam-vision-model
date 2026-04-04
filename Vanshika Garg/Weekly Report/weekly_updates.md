# WEEKLY REPORT  
**Project:** Literature Review on Synthetic Data Generation  
**Week:** 30/03 - 04/04 
**Status:** In Progress  

---

## 📌 Overview  
This week focused on initiating and conducting a structured literature review on synthetic data generation.  
The work included collecting research papers, analyzing various techniques, and identifying key insights.  
Major approaches studied include generative models, simulation techniques, and data augmentation.  
The progress lays a strong foundation for further research and development.

---

## ✅ Tasks Completed  

### 1. Research Planning  
- Defined scope and objectives of the literature review  
- Identified key focus areas: GANs, VAEs, simulations, augmentation  
- Created a structured research roadmap  
- Set milestones for completion  

### 2. Data Collection  
- Gathered research papers from IEEE, Springer, and arXiv  
- Filtered relevant and recent publications  
- Categorized papers based on techniques  
- Organized references for easy tracking  

### 3. Generative Models Analysis  
- Studied GANs, VAEs, and diffusion models  
- Understood architecture and working mechanisms  
- Analyzed real-world applications  
- Documented key differences and observations  

### 4. Simulation Techniques Study  
- Reviewed simulation-based data generation  
- Explored use cases in healthcare and finance  
- Identified strengths in controlled environments  
- Noted limitations in realism and scalability  

### 5. Data Augmentation Review  
- Examined augmentation techniques like transformations and noise injection  
- Compared effectiveness across domains  
- Studied impact on model performance  
- Documented suitable use cases  

---

## 📊 Key Findings  
- Synthetic data helps in privacy preservation and scalability  
- Generative models are powerful but computationally expensive  
- Simulation methods are domain-specific and less flexible  
- Data augmentation is effective but limited in diversity  

---

## ⚠️ Challenges Faced  
- Difficulty in comparing results due to lack of standard metrics  
- Variability in research quality across sources  
- Understanding complex architectures of advanced models  
- Limited real-world validation in some studies  

---

## 🔍 Research Gaps Identified  
- Lack of standardized evaluation frameworks  
- Challenges in high-dimensional data generation  
- Bias and fairness issues in synthetic datasets  
- Limited domain-specific optimization techniques  

---

## 🚀 Next Week Plan  
- Perform deeper comparative analysis of techniques  
- Explore evaluation metrics for synthetic data quality  
- Study advanced models and hybrid approaches  
- Begin drafting final literature review document  

---

## 📈 Overall Progress  
- Research Initiated: ✅  
- Data Collection: ✅  
- Initial Analysis: ✅  
- Comparative Study: 🔄 In Progress  
- Final Report: ⏳ Pending  

---

# WEEKLY UPDATE 

**Project Title:** AI-Based Plant Disease Detection System Using Computer Vision  
**Week Duration:** 23/03/26 – 29/03/26  

---

## 1. Objective of the Week

The main objective of this week was to begin practical implementation of the plant disease detection system. The focus was on training an object detection model, analysing its performance, identifying the limitations of the current dataset, and redesigning the problem in a more effective way to improve real-world accuracy.

---

## 2. Work Completed This Week

### 2.1 Initial Problem Understanding and Research

At the beginning of the week, I analysed the overall problem of plant disease detection and studied how existing systems work. I observed that most research work focuses only on classification using clean datasets, while real agricultural conditions are much more complex.

Instead of directly building a complex system, I decided to start from scratch and design the solution in stages so that the final system can work in real-world conditions.

---

### 2.2 Dataset Preparation and Initial Model Setup

An initial dataset was prepared consisting of plant leaf images with multiple disease classes. The dataset contained both healthy and diseased leaves across different plant categories.

After preparing the dataset, the YOLO model was selected for training because the final goal of the project is not only classification but also detection of infected regions on leaves.

The following steps were completed:

- Dataset preparation for object detection  
- Annotation of infected regions  
- Training configuration for YOLO  
- Running the first training experiment  

---

### 2.3 Training YOLO Model with Multiple Classes

The first model was trained using multiple disease classes. The aim was to train a model that could directly identify different plant diseases along with detecting the infected regions.

However, after training and testing, the following issues were observed:

- Accuracy was very low when multiple disease classes were used  
- The model was getting confused between different diseases  
- Some diseases were very similar visually, which reduced performance  
- The dataset was not large enough to support many classes  
- The model was detecting leaves but not classifying the disease correctly  

This step helped in understanding the real challenges involved in plant disease detection.

---

### 2.4 Problem Redesign Based on Results

Instead of continuing with a low-accuracy model, the problem was redesigned in a more effective way.

The new approach was:

**Step 1: Train the model only for two classes**
- Healthy leaf  
- Diseased leaf  

This was done so that the model first learns the basic difference between a healthy plant and a diseased plant before moving to detailed disease classification.

After simplifying the problem to two classes, the model performance improved and the training became more stable.

---

### 2.5 Designing a Better Pipeline

Based on the results obtained this week, a better pipeline was designed for the project:

- **Stage 1:** Healthy vs Diseased classification  
- **Stage 2:** Disease classification (after improving dataset)  
- **Stage 3:** Infected region detection using YOLO  
- **Stage 4:** Real-world testing with field images  

This staged approach will help in achieving higher accuracy and better real-world performance.

---

### 2.6 Short Work on Dataset Strategy

While analysing the model performance, it was clear that the main issue was not the model but the dataset. The dataset did not contain enough variation in terms of lighting conditions, infection stages, and leaf types.

Therefore, a new dataset collection strategy was planned, where the focus will be on:

- Common crops instead of too many plants  
- Healthy vs diseased images first  
- Real-world images instead of only clean datasets  
- Different disease stages (early, medium, severe)  

This planning will help in building a more accurate and practical plant disease detection system.

---

## 3. Learning and Skills Gained

During this week, I gained practical experience in:

- Training YOLO models for plant disease detection  
- Understanding the impact of dataset quality on model accuracy  
- Handling multi-class vs binary classification problems  
- Designing a step-by-step AI development pipeline  
- Analysing model performance and improving strategy based on results  

---

## 4. Challenges Faced

The following challenges were faced during the implementation:

- Low accuracy when training the model with multiple disease classes  
- Similar visual patterns between diseases caused classification confusion  
- Limited dataset size affected model performance  
- Difficulty in detecting early-stage diseases  
- Need for a more structured dataset for better training  

---

## 5. Conclusion of the Week

This week focused mainly on practical implementation and experimentation. The initial YOLO model was trained, and its performance was analysed. Based on the results, the problem was redesigned in a more effective way by first focusing on healthy vs diseased classification.

A clearer and more structured pipeline has now been created, which will help in improving model accuracy and building a real-world usable plant disease detection system.
---
# WEEKLY UPDATE  
**Week:** 16/03/26 – 22/03/26  

---

## Overview

This week was focused on improving the core model performance and understanding why the model was not generalizing well on different plant disease classes. Instead of directly training new models repeatedly, the focus was shifted toward analysing model behaviour, improving dataset structure, and experimenting with different training strategies.

---

## Work Completed

### 1. Model Performance Analysis

The first part of the week was spent analysing the performance of the previously trained CNN-based plant disease detection model. Although the model was performing well on the training dataset, the accuracy was not consistent when tested on new images.

After evaluating multiple predictions, it was observed that the model was relying heavily on background features instead of focusing on the infected regions of the leaf. This indicated that the model was learning shortcuts instead of learning real disease patterns.

---

### 2. Dataset Structure Improvement

To solve the generalization issue, the dataset structure was reviewed and improved. Instead of using only the original dataset format, the data was reorganized in a more structured way to make training more effective.

The following improvements were made:

- Removal of duplicate images  
- Removal of low-quality and blurry images  
- Better class balancing across disease categories  
- Separation of training and validation datasets in a more structured manner  

These changes helped improve the reliability of the training process.

---

### 3. Experimenting with Binary Classification

Another important improvement made this week was simplifying the problem. Instead of training the model directly on multiple disease classes, the focus was shifted to a binary classification problem:

- Healthy leaf  
- Diseased leaf  

This approach helped the model first learn the fundamental difference between a healthy plant and an infected plant. After training with this approach, the model performance became more stable and the accuracy improved compared to the earlier multi-class model.

---

### 4. Training Strategy Improvements

Several training improvements were also introduced this week:

- Improved data augmentation strategy  
- Better train-validation split  
- More balanced dataset usage  
- Hyperparameter tuning (learning rate and number of epochs)

These improvements helped in making the model training more stable and reducing overfitting.

---

## Observations

The following key observations were made during this week:

- Dataset quality has a bigger impact than model complexity  
- Multi-class disease classification is much harder with a small dataset  
- Binary classification is a better starting point for this problem  
- Models trained only on clean datasets struggle with real-world images  
- Proper dataset cleaning improves accuracy more than increasing model size  

---

## Challenges Faced

The following challenges were faced during this week:

- Difficulty in improving accuracy with multiple disease classes  
- Overfitting on the training dataset  
- Background features affecting model predictions  
- Limited number of high-quality images  
- Similar visual patterns between different plant diseases  

---

## Conclusion

This week focused mainly on analysing model behaviour and improving the dataset and training strategy. Instead of repeatedly training models without understanding the problem, more focus was given to improving data quality and simplifying the learning task.

These improvements helped in stabilizing the model performance and creating a stronger foundation for further development of the plant disease detection system.
---


# WEEKLY UPDATE  
**Week:** 09/03/26 – 15/03/26  

---

This week I focused on strengthening the plant disease detection project by analyzing model behavior, improving training strategies, and investigating model interpretability.

The week began with implementing **Grad-CAM visualization** to understand how the CNN model makes predictions. By generating heatmaps for different input images, I was able to analyze whether the model focuses on disease lesions or other irrelevant regions. This analysis helped identify cases where the model attends to background areas, indicating potential dataset bias.

I then performed **error analysis** on incorrect predictions to understand common failure cases. The analysis revealed that visually similar disease patterns across different plant species can lead to misclassification. Additionally, complex backgrounds and multiple leaves in images sometimes affect model predictions.

To improve model robustness, I introduced a **stronger data augmentation strategy** including:
- Random rotations  
- Color jitter  
- Random cropping  
- Random erasing  

These augmentations were designed to simulate real-world conditions and reduce shortcut learning based on leaf shape or background features.

Another major improvement involved training the model on multiple versions of the **PlantVillage dataset** (color, grayscale, and segmented images). This approach helps reduce background dependency and encourages the model to focus more on disease-related patterns.

Overall, this week’s work significantly improved the interpretability, robustness, and experimental depth of the project.
