Weekly Update
Week: 09/03/26 – 14/03/26

This week I focused on strengthening the plant disease detection project by analyzing model behavior, improving training strategies, and investigating model interpretability.

The week began with implementing Grad-CAM visualization to understand how the CNN model makes predictions. By generating heatmaps for different input images, I was able to analyze whether the model focuses on disease lesions or other irrelevant regions. This analysis helped identify cases where the model attends to background areas, indicating potential dataset bias.

I then performed error analysis on incorrect predictions to understand common failure cases. The analysis revealed that visually similar disease patterns across different plant species can lead to misclassification. Additionally, complex backgrounds and multiple leaves in images sometimes affect model predictions.

To improve model robustness, I introduced a stronger data augmentation strategy including random rotations, color jitter, random cropping, and random erasing. These augmentations were designed to simulate real-world conditions and reduce shortcut learning based on leaf shape or background features.

Another major improvement involved training the model on multiple versions of the PlantVillage dataset (color, grayscale, and segmented images). This approach helps reduce background dependency and encourages the model to focus more on disease-related patterns.

Overall, this week’s work significantly improved the interpretability, robustness, and experimental depth of the project. The next steps include evaluating the improved model on real-world plant disease images from the PlantDoc dataset to study domain shift and measure how well the model generalizes beyond controlled datasets.
