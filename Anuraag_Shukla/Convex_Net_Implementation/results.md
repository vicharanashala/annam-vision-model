## Datasets

| Dataset Name | Size (Records) | Classes
| :---: | :---: | :---: |
| New Plant Disease Dataset (Augmented) | 87,000 | 38 |
| Plant-Doc | 2,572 | 28 |
| Plant-Wild v1 | 18,542 | 89 |
| Rice Disease Dataset | 45,400 | 19 |

## Results

| Dataset | Transformations Applied | Model Used | Optimizer | Epochs | Testing Accuracy |
| :---: | :---: | :--- | :---: | :---: | :---: |
| **Plant_Doc + Plant_Wild** | Resized to 224, YOLO v11 pipeline cropping, RandomHorizontalFlip | Convex-Net Large | AdamW (lr = 0.0001) | 10 | **56.77%** |
| **Plant_Doc + Plant_Wild** | Resized to 299, Rotation (0-180°), Histogram Equalization, YOLO v11 cropping, RandomHorizontalFlip | Convex-Net Base | AdamW (lr = 0.0001) | 20 | **59.86%** |
| **Plant_Doc + Plant_Wild** | Resized to 224, Rotation (0-180°), Histogram Equalization, YOLO v11 cropping, RandomHorizontalFlip | Convex-Net Small | AdamW (lr = 0.0001) | 20 | **55.34%** |
| **Plant_Doc + Plant_Wild [Potato Only Subclasses]** | Resized to 224, Rotation (0-180°), YOLO v11 pipeline cropping, RandomHorizontalFlip | Convex-Net Base | AdamW (lr = 0.0001) | 20 | **73.62%** |
| **Plant_Doc + Plant_Wild [Tomato Only Subclasses]** | Resized to 224, YOLO v11 pipeline cropping, RandomHorizontalFlip | Convex-Net Base | AdamW (lr = 0.0001) | 20 | **66.50%** |
| **Rice Disease Dataset** | Resized to 224, Rotation (0-180°), WeightedRandomSampling, RandomHorizontalFlip | Convex-Net Base | AdamW (lr = 0.0001) | 10 | **93.98%** |
| **Rice Disease Dataset** | Resized to 224, Rotation (0-180°), Histogram Equalization, RandomHorizontalFlip | Convex-Net Base | AdamW (lr = 0.0001) | 20 | **91.40%** |
| **Rice Disease Dataset** | Resized to 224, Rotation (0-180°), WeightedRandomSampling, RandomHorizontalFlip | Convex-Net Base | AdamW (lr = 0.0001) | 20 | **91.40%** |
| **Rice Disease Dataset** | Resized to 224, Rotation (0-180°), Center Cropping, RandomHorizontalFlip | Convex-Net Base | AdamW (lr = 0.0001) | 20 | **88.39%** |
| **Rice Disease Dataset** | Resized to 224, Rotation (0-180°), RandomHorizontalFlip | Convex-Net Large | AdamW (lr = 0.0001) | 12/15 | **87.31%/87.74%** |
| **Rice Disease Dataset** | Resized to 224, Rotation (0-180°), RandomHorizontalFlip, WeightedRandomSampling | Convex-Net Base | AdamW (lr = 0.0001) | 20 | **91.40%** |
| **Rice Disease Dataset** | Resized to 224, Rotation (0-180°), RandomHorizontalFlip, WeightedRandomSampling | Convex-Net Base | AdamW (lr = 0.0001) | 10 | **93.98%** |
| **Rice Disease Dataset** | Resized to 224, Rotation (0-180°), RandomHorizontalFlip, Replaced all images of the Brown Spot class with images from an alternative source | Convex-Net-Base | AdamW (lr = 0.0001) | 20 | **86.67%** |
| **Plant_Doc+Plant_Wild[27 Classes]** | Resized to 224, Rotation (0-180°), RandomHorizontalFlip, Histogram Equalization | Convex-Net-Base | AdamW (lr = 0.0001) | 20 | **73.73%** |

## Results [Best Models]

| Dataset | Transformations Applied | Model Used | Optimizer | Epochs | Testing Accuracy | Precision Score(Macro) | F1-Score(Macro) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Plant_Doc + Plant_Wild** | **Resized to 224, Rotation (0-180°), Histogram Equalization, YOLO v11 cropping, RandomHorizontalFlip** | **Convex-Net Base** | **AdamW (lr = 0.0001)** | **20** | **60.21%** | **0.58** | **0.56** |
| **Rice Disease Dataset** | **Resized to 224, Rotation (0-180°), RandomHorizontalFlip** | **Convex-Net Base** | **AdamW (lr = 0.0001)** | **20** | **94.19%** | **0.95** | **0.93** |

## Notes
- Testing for all the models trained on Plant_Doc/Plant_Wild/New_Plant_Diseases or a combination of these datasets is done for 10 classes for comparison,
    - Potato_Early_blight
    - Potato_healthy
    - Potato_Lateblight
    - Tomato_Bacterial_spot
    - Tomato_Early_blight
    - Tomato_healthy
    - Tomato_Late_blight
    - Tomato_Leaf_mold
    - Tomato_Septoria_leaf_spot
    - Tomato_Tomato_Yellow_Leaf_Curl_Virus

- Testing for the models trained on Plant_Doc+Plant_Wild[27 Classes] is done on the following classes,
    - Apple Scab Leaf
    - Apple leaf
    - Apple rust leaf
    - Bell_pepper leaf
    - Bell_pepper leaf spot
    - Blueberry leaf
    - Cherry leaf
    - Corn Gray leaf spot
    - Corn leaf blight
    - Corn rust leaf
    - Grape leaf
    - Grape leaf black rot
    - Peach leaf
    - Potato leaf early blight
    - Potato leaf late blight
    - Raspberry leaf
    - Soyabean leaf
    - Squash Powdery mildew leaf
    - Strawberry leaf
    - Tomato Early blight leaf
    - Tomato Septoria leaf spot
    - Tomato leaf
    - Tomato leaf bacterial spot
    - Tomato leaf late blight
    - Tomato leaf mosaic virus
    - Tomato leaf yellow virus
    - Tomato mold leaf

- Testing for all the models trained on Rice Disease Dataset is done for all 19 classes.