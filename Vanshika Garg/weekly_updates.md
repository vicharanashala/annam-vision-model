WEEKLY REPORT
Project Title: AI-Based Plant Disease Detection System Using Computer Vision
Week Duration: 23/03 - 29/03

1. Objective of This Week

The main objective of this week was to understand the plant disease detection problem deeply and design a strong foundation before starting model training. The focus was mainly on:

Studying existing plant disease detection systems
Understanding datasets used in research papers
Deciding the project scope
Planning a proper dataset collection strategy
Creating a step-by-step development roadmap
2. Work Completed This Week
2.1 Research on Existing Work

During this week, I studied several research papers and existing solutions related to plant disease detection using machine learning and deep learning. The main goal was to understand:

How current systems detect plant diseases
What type of datasets are used
What problems still exist in current research
How real-world agricultural problems are different from research datasets

I observed that most existing systems are trained on clean datasets where leaves are captured in controlled environments. However, real-world agricultural conditions include variations such as lighting changes, background noise, partial leaf visibility, and different weather conditions. This understanding helped in defining a more practical approach for the project.

2.2 Understanding the Problem in Depth

Instead of directly building a model, I first analyzed the challenges involved in plant disease detection. The following key issues were identified:

Large number of plant species and diseases
Variation in disease appearance at different stages
Different environmental conditions such as sunlight, humidity, and shadows
Differences in leaf texture, shape, and color across plants
Need for real-world usable AI instead of only academic accuracy

This step helped in narrowing down the project scope and focusing on a more realistic solution.

2.3 Dataset Planning and Strategy

A major part of this week was spent on designing a dataset collection strategy. Instead of collecting random images, a structured plan was created.

The following decisions were made:

Start with only a few important crops instead of collecting data for many plants
Focus on crops that cause major agricultural losses
Select only the most common diseases per crop
Include both healthy and diseased leaves in the dataset
Collect images in different lighting and environmental conditions

This approach will help in improving model accuracy and making the system useful for real farmers.

2.4 Selection of Priority Crops

Based on agricultural importance and availability of datasets, the following crops were selected as priority crops for the first phase of the project:

Tomato
Potato
Rice
Wheat
Cotton

Among these, tomato and potato were selected as the starting crops because they are easier to work with and have clearly visible disease patterns.

2.5 Designing the Dataset Structure

A proper dataset structure was planned to make the training process easier and more efficient. Instead of using a large number of classes, a simplified and effective structure was created.

The plan includes:

Healthy leaves
Early-stage disease
Medium-stage disease
Severe-stage disease

This structure will help the model learn the difference between healthy and diseased leaves before moving to detailed disease classification.

2.6 Planning the Model Development Approach

Instead of building a complex model directly, a step-by-step development strategy was created:

Stage 1: Binary classification (Healthy vs Diseased)
Stage 2: Multi-class disease classification
Stage 3: Infected region detection (segmentation)
Stage 4: Real-world dataset testing
Stage 5: Building a complete AI-based plant disease detection system

This approach will help in achieving higher accuracy and better results.

3. Skills and Knowledge Gained This Week

During this week, I improved my understanding of:

Computer vision in agriculture
Image dataset collection strategies
Real-world challenges in plant disease detection
Differences between academic datasets and real-world data
Model development planning in AI projects
Problem-solving approach in research-based projects
4. Challenges Faced

Some challenges were faced while understanding the project:

Very large number of plant diseases made it difficult to decide the scope
Different datasets had different formats and class labels
Understanding which crops to focus on required additional research
Designing a dataset that works in real-world conditions was challenging

However, these challenges helped in creating a more structured and practical project plan.

5. Conclusion of This Week

This week was mainly focused on research, planning, and understanding the problem in depth. Instead of directly starting model training, a strong foundation was created by analyzing existing systems and designing a proper dataset strategy.

The project is now ready to move to the next phase, which will focus on actual dataset collection and initial model training.

6. Plan for Next Week

The following tasks are planned for next week:

Start collecting dataset images
Organize images based on crop and disease type
Clean and verify collected data
Create the initial dataset structure
Start training the first classification model (Healthy vs Diseased)

-------------------------------------------------------------------------------------------------

Weekly Update
Week: 09/03/26 – 14/03/26

This week I focused on strengthening the plant disease detection project by analyzing model behavior, improving training strategies, and investigating model interpretability.

The week began with implementing Grad-CAM visualization to understand how the CNN model makes predictions. By generating heatmaps for different input images, I was able to analyze whether the model focuses on disease lesions or other irrelevant regions. This analysis helped identify cases where the model attends to background areas, indicating potential dataset bias.

I then performed error analysis on incorrect predictions to understand common failure cases. The analysis revealed that visually similar disease patterns across different plant species can lead to misclassification. Additionally, complex backgrounds and multiple leaves in images sometimes affect model predictions.

To improve model robustness, I introduced a stronger data augmentation strategy including random rotations, color jitter, random cropping, and random erasing. These augmentations were designed to simulate real-world conditions and reduce shortcut learning based on leaf shape or background features.

Another major improvement involved training the model on multiple versions of the PlantVillage dataset (color, grayscale, and segmented images). This approach helps reduce background dependency and encourages the model to focus more on disease-related patterns.

Overall, this week’s work significantly improved the interpretability, robustness, and experimental depth of the project. The next steps include evaluating the improved model on real-world plant disease images from the PlantDoc dataset to study domain shift and measure how well the model generalizes beyond controlled datasets.
