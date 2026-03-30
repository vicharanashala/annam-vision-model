# Monthly Report  
## Project: AI-Based Plant Disease Detection System (YOLO + Deep Learning)

**Duration:** March 2026  
**Author:** Vanshika Garg  

---

# 1. Project Overview

The objective of this project is to build a real-world plant disease detection system using computer vision and deep learning. The goal is not only to classify plant diseases but also to detect infected regions on plant leaves so that the system can be useful for farmers in practical agricultural conditions.

This month mainly focused on understanding the real challenges of plant disease detection, experimenting with object detection models, analysing results, and redesigning the project in a more effective way.

---

# 2. Work Completed This Month

## 2.1 Understanding the Problem

Initially, the project was started with the assumption that plant disease detection is a standard multi-class image classification problem. However, after studying existing research papers and real datasets, it became clear that the problem is much more complex than expected.

The following challenges were identified:

- Large number of plant species and diseases  
- Diseases look different at different stages  
- Real-world images contain background noise, shadows, and lighting variations  
- Some diseases look visually very similar  
- Early-stage diseases are difficult to detect  

This analysis helped in redefining the approach to the project.

---

## 2.2 Dataset Preparation and Initial Experiments

A dataset of plant leaf images was collected and prepared for training. The dataset contained both healthy and diseased plant leaves with multiple disease classes.

The following steps were completed:

- Data collection from available datasets and public sources  
- Dataset cleaning and organisation  
- Annotation of infected regions for object detection  
- Preparation of training data for the YOLO model  

This stage helped in understanding the importance of data quality in deep learning projects.

---

## 2.3 Training the YOLO Model (Multi-Class Experiment)

The first major experiment performed this month was training a YOLO-based object detection model using multiple disease classes. The goal was to detect the infected region and classify the disease at the same time.

After training and testing the model, several issues were observed:

- The model accuracy was low when multiple disease classes were used  
- The model was getting confused between visually similar diseases  
- The dataset size was not sufficient for training a large multi-class model  
- The model was detecting leaves but not correctly identifying the disease  
- Early-stage diseases were not being detected properly  

These results showed that directly solving the problem as a large multi-class detection task was not effective.

---

## 2.4 Redesigning the Problem

Instead of continuing with a low-performing model, the problem was redesigned in a more practical way. The focus was shifted from a complex multi-class problem to a simpler and more structured pipeline.

The new approach was:

**Stage 1:** Healthy vs Diseased detection  
**Stage 2:** Disease classification (after improving the dataset)  
**Stage 3:** Infected region detection using object detection  
**Stage 4:** Real-world testing with field images  

This structured approach will help in improving accuracy step by step instead of solving everything at once.

---

## 2.5 Observations from Experiments

During the training and testing process, several important observations were made:

- Dataset quality affects performance more than model complexity  
- A large number of classes reduces accuracy when the dataset is small  
- Early-stage diseases are harder to detect than severe-stage diseases  
- Real-world images are much more difficult than clean research datasets  
- Training the model in stages gives better results than training everything together  

These observations played a major role in redesigning the project pipeline.

---

## 2.6 Dataset Strategy Improvement

One of the most important findings this month was that the main limitation of the project was not the model but the dataset.

Therefore, instead of focusing only on model improvement, more attention was given to creating a better dataset strategy. The improved strategy includes:

- Focusing on a small number of important crops instead of many plants  
- Collecting both healthy and diseased images  
- Including early-stage, medium-stage, and severe-stage infections  
- Using real-world images instead of only clean datasets  
- Creating a structured dataset instead of random data collection  

This will help in building a more accurate and real-world usable system.

---

# 3. Challenges Faced

The following challenges were faced during this month:

- Low accuracy when training the model with multiple disease classes  
- Limited dataset size for multi-class detection  
- Similar visual patterns between different diseases  
- Difficulty in detecting early-stage diseases  
- Data annotation required significant time and effort  
- Real-world agricultural images were difficult to find  

However, these challenges helped in understanding the real complexity of plant disease detection.

---

# 4. Key Learnings

This month helped in gaining practical knowledge in the following areas:

- Real-world computer vision problems  
- Dataset preparation and cleaning  
- Object detection using YOLO  
- Multi-class vs binary classification strategies  
- Importance of dataset quality in deep learning  
- Problem-solving approach in AI projects  
- Designing a structured AI pipeline  

---

# 5. Proposed Solution (Final Direction of the Project)

Based on the experiments and observations made this month, the following structured solution has been proposed:

### Step 1: Train the model to detect Healthy vs Diseased plants  
This helps the model learn the basic difference between a healthy leaf and an infected leaf.

### Step 2: Improve the dataset with real-world images  
Better dataset quality will directly improve model accuracy.

### Step 3: Add disease classification after improving binary accuracy  
This will reduce confusion between diseases and improve overall performance.

### Step 4: Use YOLO for detecting infected regions  
Instead of only classification, the system will also show where the disease is present.

This approach will help in building a more accurate and practical plant disease detection system.

---

# 6. Conclusion

This month focused mainly on experimentation, analysis, and improving the overall project strategy. Instead of continuing with a low-performing multi-class model, the project was redesigned in a more effective and structured way.

The experiments clearly showed that dataset quality plays a more important role than model complexity. By simplifying the problem and improving the dataset strategy, the project is now moving towards a more realistic and useful AI-based plant disease detection system.

---
