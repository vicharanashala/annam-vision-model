## Datasets

| Dataset Name | Size (Records) | Classes
| :---: | :---: | :---:
| New Plant Disease Dataset (Augmented) | 87,000 | 38 |
| Plant-Doc | 2,572 | 28 |
| Plant-Wild v1 | 18,542 | 89 |
| Rice Disease Dataset | 45,400 | 19 |

## Results

| Dataset | Transformations Applied | Model Used | Optimizer | Epochs | Testing Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **New Plant Diseases + Plant_Doc + Plant_Wild** | Resized to 224, YOLO v11 pipeline cropping, RandomHorizontalFlip | Efficient Net B4 | AdamW (lr = 0.0001) | 20 | **57.72%** |
| **New Plant Diseases + Plant_Doc + Plant_Wild** | Resized to 380, YOLO v11 pipeline cropping, RandomHorizontalFlip | Efficient Net B4 | AdamW (lr = 0.0001) | 20 | **54.51%** |
| **Plant_Doc + Plant_Wild** | Resized to 224, YOLO v11 pipeline cropping, RandomHorizontalFlip | Efficient Net B4 | AdamW (lr = 0.0001 for 20 epochs, 0.00001 for 20) | 40 | **55.58%** |
| **Plant_Doc + Plant_Wild** | Resized to 224, YOLO v11 pipeline cropping, RandomHorizontalFlip | Efficient Net B7 | AdamW (lr = 0.0001 for 20 epochs, 0.00005 for 20) | 40 | **52.85%** |
| **Plant_Doc + Plant_Wild** | Resized to 224, YOLO v11 pipeline cropping, RandomHorizontalFlip | Convex-Net Large | AdamW (lr = 0.0001) | 10 | **56.77%** |
| **Plant_Doc + Plant_Wild** | Resized to 224, Rotation (0-180°), Histogram Equalization, YOLO v11 cropping, RandomHorizontalFlip | Convex-Net Base | AdamW (lr = 0.0001) | 20 | **60.21%** |
| **Plant_Doc + Plant_Wild** | Resized to 299, Rotation (0-180°), Histogram Equalization, YOLO v11 cropping, RandomHorizontalFlip | Convex-Net Base | AdamW (lr = 0.0001) | 20 | **59.86%** |
| **Plant_Doc + Plant_Wild** | Resized to 224, Rotation (0-180°), Histogram Equalization, YOLO v11 cropping, RandomHorizontalFlip | Convex-Net Small | AdamW (lr = 0.0001) | 20 | **55.34%** |
| **Plant_Doc + Plant_Wild [Potato Only Subclasses]** | Resized to 224, YOLO v11 pipeline cropping, RandomHorizontalFlip | Efficient Net B4 | AdamW (lr = 0.0001) | 20 | **69.29%** |
| **Plant_Doc + Plant_Wild [Potato Only Subclasses]** | Resized to 224, Rotation (0-180°), YOLO v11 pipeline cropping, RandomHorizontalFlip | Convex-Net Base | AdamW (lr = 0.0001) | 20 | **73.62%** |
| **Plant_Doc + Plant_Wild [Tomato Only Subclasses]** | Resized to 224, YOLO v11 pipeline cropping, RandomHorizontalFlip | Efficient Net B7 | AdamW (lr = 0.0001) | 20 | **59.86%** |
| **Plant_Doc + Plant_Wild [Tomato Only Subclasses]** | Resized to 224, YOLO v11 pipeline cropping, RandomHorizontalFlip | Convex-Net Base | AdamW (lr = 0.0001) | 20 | **66.50%** |
| **Rice Disease Dataset** | Resized to 224, Rotation (0-180°), RandomHorizontalFlip | Convex-Net Base | AdamW (lr = 0.0001) | 20 | **94.19%** |
| **Rice Disease Dataset** | Resized to 224, Rotation (0-180°), WeightedRandomSampling, RandomHorizontalFlip | Convex-Net Base | AdamW (lr = 0.0001) | 10 | **93.98%** |
| **Rice Disease Dataset** | Resized to 224, Rotation (0-180°), Histogram Equalization, RandomHorizontalFlip | Convex-Net Base | AdamW (lr = 0.0001) | 20 | **91.40%** |
| **Rice Disease Dataset** | Resized to 224, Rotation (0-180°), WeightedRandomSampling, RandomHorizontalFlip | Convex-Net Base | AdamW (lr = 0.0001) | 20 | **91.40%** |
| **Rice Disease Dataset** | Resized to 224, Rotation (0-180°), RandomHorizontalFlip | Efficient Net B4 | AdamW (lr = 0.0001) | 10/15/20 | **87.10%/87.31%/89.68%** |
| **Rice Disease Dataset** | Resized to 224, Rotation (0-180°), Center Cropping, RandomHorizontalFlip | Convex-Net Base | AdamW (lr = 0.0001) | 20 | **88.39%** |
| **Rice Disease Dataset** | Resized to 224, Rotation (0-180°), RandomHorizontalFlip | Convex-Net Large | AdamW (lr = 0.0001) | 12/15 | **87.31%/87.74%** |

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
- Testing for all the models trained on Rice Disease Dataset is done for all 19 classes.




